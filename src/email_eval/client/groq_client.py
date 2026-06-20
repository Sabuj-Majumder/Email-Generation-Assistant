from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from email_eval.config import Settings

logger = logging.getLogger(__name__)


class GroqAPIError(Exception):
    """Raised when all retry attempts are exhausted or a non-retryable error occurs."""

    def __init__(self, message: str, last_response_body: str | None = None) -> None:
        super().__init__(message)
        self.last_response_body = last_response_body


class _RetryableHTTPError(Exception):
    """Internal sentinel for retryable HTTP status codes (429, 5xx)."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.body = body


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    wait = retry_state.next_action.sleep if retry_state.next_action else 0  # type: ignore[union-attr]
    logger.warning(
        "Groq API retry | attempt=%d next_wait=%.1fs reason=%s",
        retry_state.attempt_number,
        wait,
        str(exc),
    )


class GroqClient:
    """Async Groq API client with exponential-backoff retry."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.groq_base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        temperature: float = 0.7,
    ) -> str:
        """
        Call the Groq chat-completions endpoint and return the assistant reply.

        Retries on 429 and 5xx up to Settings.max_retries times with exponential backoff.
        Raises GroqAPIError after all retries are exhausted or on non-retryable errors.
        """
        settings = self._settings

        @retry(
            retry=retry_if_exception_type(_RetryableHTTPError),
            stop=stop_after_attempt(settings.max_retries),
            wait=wait_exponential(
                multiplier=settings.retry_backoff_seconds, min=1, max=60
            ),
            before_sleep=_log_retry_attempt,
            reraise=False,
        )
        async def _call() -> str:
            return await self._raw_complete(system_prompt, user_prompt, model_name, temperature)

        try:
            result: str = await _call()  # type: ignore[assignment]
            return result
        except _RetryableHTTPError as exc:
            raise GroqAPIError(
                f"Groq API failed after {settings.max_retries} retries: HTTP {exc.status_code}",
                last_response_body=exc.body,
            ) from exc
        except GroqAPIError:
            raise
        except Exception as exc:
            raise GroqAPIError(f"Unexpected error calling Groq API: {exc}") from exc

    async def _raw_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        temperature: float,
    ) -> str:
        """Single (non-retried) HTTP call to the completions endpoint."""
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }

        logger.debug(
            "Groq request | model=%s temperature=%.2f",
            model_name,
            temperature,
        )

        async with httpx.AsyncClient(timeout=self._settings.request_timeout_seconds) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            )

        body = response.text

        if response.status_code in (429,) or response.status_code >= 500:
            raise _RetryableHTTPError(response.status_code, body)

        if response.status_code != 200:
            raise GroqAPIError(
                f"Groq API returned non-retryable status {response.status_code}",
                last_response_body=body,
            )

        data = response.json()
        try:
            content: str = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GroqAPIError(
                f"Unexpected Groq response shape: {exc}",
                last_response_body=body,
            ) from exc

        logger.debug("Groq response | length=%d", len(content))
        return content
