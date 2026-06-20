import streamlit as st
import asyncio
import json
import os
import re
from pathlib import Path
from email_eval.config import get_settings
from email_eval.client.groq_client import GroqClient
from email_eval.generation.generator import generate_email
from email_eval.schemas import Scenario, GeneratedEmail


st.set_page_config(
    page_title="Draftcraft  |  AI Email Assistant",
    page_icon="✉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────
#  Design System — Warm Stone / Slate, Human-Crafted Palette
#  Inspired by: Linear, Raycast, Craft, Notion
#
#  Base:      #111110   (very dark warm gray)
#  Surface:   #1c1b19   (panel)
#  Elevated:  #232220   (raised card)
#  Border:    #2e2c29   (visible but quiet)
#  Muted:     #3d3b37
#  Stone-60:  #78716c
#  Stone-40:  #a8a29e
#  Stone-20:  #d6d3d1
#  Text:      #e8e5e0   (warm off-white)
#  Accent:    #c2a46e   (warm gold/amber — editorial feel)
#  Accent-dk: #8a6e43
#  Green:     #5a7a62   (status dot)
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ===== GLOBAL RESET ===== */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background-color: #111110 !important;
    color: #e8e5e0 !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    letter-spacing: -0.022em;
    color: #f0ede8 !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #111110; }
::-webkit-scrollbar-thumb { background: #2e2c29; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3d3b37; }

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background-color: #171614 !important;
    border-right: 1px solid #252320 !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #78716c;
}

/* ===== FORM LABELS ===== */
label, .stMarkdown p {
    color: #a8a29e !important;
    font-size: 0.835rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.005em;
}

/* ===== INPUT FIELDS ===== */
textarea, input[type="text"], input[type="password"],
.stSelectbox > div > div,
div[data-baseweb="select"] > div {
    background-color: #1c1b19 !important;
    color: #e8e5e0 !important;
    border: 1.5px solid #2e2c29 !important;
    border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
    caret-color: #c2a46e;
}
textarea:focus, input:focus {
    border-color: #7c6a4e !important;
    box-shadow: 0 0 0 3px rgba(194,164,110,0.1) !important;
    outline: none !important;
}

/* ===== SELECTBOX DROPDOWN ===== */
div[data-baseweb="popover"] {
    background-color: #1c1b19 !important;
    border: 1.5px solid #2e2c29 !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important;
}
div[data-baseweb="popover"] li {
    color: #e8e5e0 !important;
    background-color: #1c1b19 !important;
    font-size: 0.875rem !important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: #252320 !important;
    color: #f0ede8 !important;
}

/* ===== TABS ===== */
div[data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 1.5px solid #252320 !important;
    padding-bottom: 0 !important;
    background: transparent !important;
}
button[data-baseweb="tab"] {
    background-color: transparent !important;
    color: #6b6661 !important;
    border: none !important;
    font-size: 0.855rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.6rem 1rem !important;
    border-radius: 0 !important;
    transition: color 0.15s ease !important;
}
button[data-baseweb="tab"]:hover {
    color: #c2a46e !important;
    background-color: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #e8e5e0 !important;
    background-color: transparent !important;
    border-bottom: 2px solid #c2a46e !important;
    font-weight: 600 !important;
}

/* ===== PRIMARY BUTTON ===== */
div.stButton > button {
    background-color: #232220;
    color: #e8e5e0 !important;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.86rem;
    border: 1.5px solid #3d3b37 !important;
    padding: 0.68rem 1.5rem !important;
    border-radius: 7px !important;
    transition: all 0.15s ease !important;
    width: 100%;
    letter-spacing: 0.01em;
}
div.stButton > button:hover {
    background-color: #2a2925 !important;
    border-color: #c2a46e !important;
    color: #f0ede8 !important;
    transform: none;
}
div.stButton > button:active {
    background-color: #1c1b19 !important;
    transform: none;
}

/* ===== SLIDER ===== */
div[data-baseweb="slider"] div[role="slider"] {
    background-color: #c2a46e !important;
    border: 2px solid #e8d5b0 !important;
    width: 16px !important;
    height: 16px !important;
}
div[data-baseweb="slider"] div[role="slider"]:hover {
    box-shadow: 0 0 0 4px rgba(194,164,110,0.15) !important;
}
div[data-testid="stSlider"] > div > div > div > div:first-child {
    background: #c2a46e !important;
}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    background-color: #1c1b19 !important;
    border: 1.5px solid #2e2c29 !important;
    border-radius: 7px !important;
    color: #a8a29e !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
}
.streamlit-expanderHeader:hover {
    border-color: #3d3b37 !important;
    color: #d6d3d1 !important;
}

/* ===== ALERTS ===== */
div[data-testid="stAlert"] {
    border-radius: 7px !important;
    border: 1.5px solid #2e2c29 !important;
    background-color: #1c1b19 !important;
}

/* ===== DIVIDER ===== */
hr { border-color: #252320 !important; }

/* ===== SPINNER ===== */
div[data-testid="stSpinner"] > div {
    border-top-color: #c2a46e !important;
}

/* ================================================================
   CUSTOM COMPONENTS
   ================================================================ */

/* ── Brand header ── */
.brand-header {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.brand-logo {
    width: 40px;
    height: 40px;
    background-color: #232220;
    border: 1.5px solid #3d3b37;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    color: #c2a46e;
    flex-shrink: 0;
}
.brand-name {
    font-size: 1.45rem;
    font-weight: 700;
    color: #f0ede8;
    letter-spacing: -0.035em;
    line-height: 1.1;
}
.brand-tagline {
    color: #6b6661;
    font-size: 0.8rem;
    font-weight: 400;
    margin-top: 0.12rem;
    letter-spacing: 0.008em;
}

/* ── Pane header ── */
.pane-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.1rem;
    padding-bottom: 0.8rem;
    border-bottom: 1.5px solid #252320;
}
.pane-title {
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: #78716c;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.pane-dot {
    width: 5px;
    height: 5px;
    background: #c2a46e;
    border-radius: 50%;
}

/* ── Section label ── */
.section-label {
    font-size: 0.67rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: #4a4744;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e1d1b;
}

/* ── Result card ── */
.result-card {
    background: #171614;
    border: 1.5px solid #2e2c29;
    border-radius: 10px;
    padding: 0;
    color: #e8e5e0;
    min-height: 440px;
    animation: fadeSlide 0.32s ease;
    overflow: hidden;
}
.result-header {
    font-size: 0.76rem;
    font-family: 'JetBrains Mono', monospace;
    color: #6b6661;
    padding: 0.85rem 1.2rem;
    border-bottom: 1.5px solid #252320;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #141312;
}
.result-body {
    font-size: 0.915rem;
    line-height: 1.82;
    white-space: pre-wrap;
    color: #dedad5;
    padding: 1.35rem 1.4rem;
}

/* ── Tag ── */
.tag {
    background-color: #232220;
    border: 1px solid #3d3b37;
    color: #78716c;
    padding: 0.2rem 0.6rem;
    border-radius: 5px;
    font-size: 0.69rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* ── Placeholder box ── */
.placeholder-box {
    border: 1.5px dashed #2e2c29;
    border-radius: 10px;
    padding: 3.5rem 2rem;
    text-align: center;
    min-height: 440px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: #141312;
    gap: 0.5rem;
}
.placeholder-icon {
    width: 52px;
    height: 52px;
    background: #1c1b19;
    border: 1.5px solid #2e2c29;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    margin-bottom: 0.65rem;
    color: #78716c;
}
.placeholder-title {
    font-size: 0.96rem;
    color: #d6d3d1;
    font-weight: 600;
    letter-spacing: -0.01em;
}
.placeholder-sub {
    font-size: 0.8rem;
    color: #6b6661;
    max-width: 290px;
    line-height: 1.65;
}

/* ── Info card (sidebar) ── */
.info-card {
    background-color: #1c1b19;
    border: 1.5px solid #2e2c29;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-top: 0.5rem;
}
.info-card-label {
    font-size: 0.67rem;
    font-family: 'JetBrains Mono', monospace;
    color: #4a4744;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.32rem;
}
.info-card-value {
    font-size: 0.79rem;
    color: #78716c;
    font-family: 'JetBrains Mono', monospace;
    word-break: break-all;
    line-height: 1.55;
}

/* ── Temperature creative guide ── */
.temp-bar-wrap {
    margin-top: 6px;
    border-radius: 4px;
    height: 4px;
    background: linear-gradient(90deg, #4a4744 0%, #7c6a4e 40%, #c2a46e 75%, #d6b97a 100%);
    opacity: 0.65;
}
.temp-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 5px;
}
.temp-label-low  { font-size: 0.67rem; color: #78716c; font-weight: 500; font-family: 'Inter', sans-serif; }
.temp-label-high { font-size: 0.67rem; color: #c2a46e; font-weight: 500; font-family: 'Inter', sans-serif; }
.temp-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    margin-top: 0.55rem;
    padding: 0.32rem 0.8rem;
    border-radius: 5px;
    font-size: 0.77rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.005em;
    width: 100%;
    justify-content: center;
}
.temp-badge.mode-precise {
    background: #1c1b19;
    border: 1px solid #2e2c29;
    color: #a8a29e;
}
.temp-badge.mode-balanced {
    background: #201e1a;
    border: 1px solid #3d3633;
    color: #c2a46e;
}
.temp-badge.mode-creative {
    background: #231f18;
    border: 1px solid #4d4030;
    color: #d6b97a;
}
.temp-hint {
    text-align: center;
    margin-top: 4px;
    font-size: 0.69rem;
    color: #4a4744;
    font-family: 'Inter', sans-serif;
    line-height: 1.5;
}

/* ── How temp works guide ── */
.temp-guide {
    background: #1c1b19;
    border: 1.5px solid #2e2c29;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-top: 0.5rem;
}
.temp-guide-title {
    font-size: 0.67rem;
    font-family: 'JetBrains Mono', monospace;
    color: #78716c;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.6rem;
}
.temp-guide-row {
    font-size: 0.77rem;
    color: #6b6661;
    line-height: 1.6;
    font-family: 'Inter', sans-serif;
    margin-bottom: 0.5rem;
}
.temp-guide-row:last-child { margin-bottom: 0; }
.tgr-range { font-weight: 600; }
.tgr-range.r-low  { color: #a8a29e; }
.tgr-range.r-mid  { color: #c2a46e; }
.tgr-range.r-high { color: #d6b97a; }

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 1.25rem 0 0.5rem 0;
    color: #3d3b37;
    font-size: 0.73rem;
    letter-spacing: 0.03em;
    border-top: 1px solid #1e1d1b;
    margin-top: 1.75rem;
}

/* ── Sidebar brand ── */
.sb-brand {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid #252320;
}
.sb-logo {
    width: 28px; height: 28px;
    background: #232220;
    border: 1px solid #3d3b37;
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.82rem; color: #c2a46e;
}
.sb-name {
    font-size: 1rem;
    font-weight: 700;
    color: #f0ede8;
    letter-spacing: -0.02em;
}

/* ── Animations ── */
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Defaults & scenario helpers
# ---------------------------------------------------------------------------
SCENARIOS_PATH = (
    Path(__file__).resolve().parent
    / "src" / "email_eval" / "data" / "scenarios.json"
)

DEFAULT_NEW_KEY_FACTS = """e.g.
- Draft is attached to the email
- Need feedback by Tuesday afternoon
- Next meeting is scheduled for Friday"""

DEFAULT_REPLY_KEY_FACTS = """e.g.
- Project deliverables are currently 90% complete
- Initial review will be ready by Thursday
- Requesting budget sign-off by next Monday"""

DEFAULT_ORIGINAL_EMAIL = """From: client@company.com
Subject: Project update request

Hi, could you please send over the latest status update for the project deliverables? We need to review the timeline and current budget.

Thanks,
Sarah"""

_SIGNOFF_PATTERN = re.compile(
    r"\n+(?:Best regards|Kind regards|Warm regards|Regards|Sincerely|"
    r"Cheers|Thanks|Thank you|With gratitude|Respectfully|"
    r"Yours truly|Yours sincerely|Best|All the best|"
    r"Cordially|Warmly)[,:\s]*[\s\S]*$",
    re.IGNORECASE,
)


def _strip_model_signoff(email_text: str) -> str:
    return _SIGNOFF_PATTERN.sub("", email_text).rstrip()


@st.cache_data
def get_scenarios_list():
    if SCENARIOS_PATH.exists():
        with SCENARIOS_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return []


scenarios = get_scenarios_list()

# ---------------------------------------------------------------------------
# Brand Header
# ---------------------------------------------------------------------------
col_brand, col_model = st.columns([3, 1])

with col_brand:
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-logo">&#9993;</div>
            <div>
                <div class="brand-name">Draftcraft</div>
                <div class="brand-tagline">AI-powered professional email generation with precision tone control</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_model:
    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)
    strategy_label = st.selectbox(
        "Active Model",
        options=[
            "Model A  (llama-3.3-70b-versatile)",
            "Model B  (openai/gpt-oss-120b)",
        ],
        index=0,
    )
    selected_model_id = "model_a" if "Model A" in strategy_label else "model_b"

st.markdown(
    "<hr style='margin:0.5rem 0 1.4rem 0; border:0; border-top:1.5px solid #252320;'>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Layout: Workspace | Response
# ---------------------------------------------------------------------------
col_input, col_output = st.columns([1, 1], gap="large")

# ─── LEFT PANE — WORKSPACE ───────────────────────────────────────
with col_input:
    st.markdown(
        """
        <div class="pane-header">
            <div class="pane-title">
                <div class="pane-dot"></div>
                Workspace
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_write, tab_reply = st.tabs(["Compose new email", "Reply to email"])

    intent_val = ""
    facts_val = ""
    tone_val = "formal"
    name_val = ""
    recipient_val = ""
    original_email_val = ""
    active_tab = "write"

    # ── Tab: Compose ──
    with tab_write:
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Scenario Template</div>", unsafe_allow_html=True)

        if scenarios:
            selected_scenario_idx = st.selectbox(
                "Load a preset scenario",
                options=[None] + list(range(len(scenarios))),
                format_func=lambda x: (
                    "— Start from scratch —"
                    if x is None
                    else f"[{scenarios[x]['id'].upper()}]  {scenarios[x]['intent']}"
                ),
                key="write_scenario",
            )
        else:
            selected_scenario_idx = None

        if selected_scenario_idx is not None:
            sc = scenarios[selected_scenario_idx]
            default_write_intent = sc["intent"]
            default_write_facts = "\n".join(f"- {f}" for f in sc["key_facts"])
            default_write_tone = sc["tone"]
        else:
            default_write_intent = ""
            default_write_facts = ""
            default_write_tone = "formal"

        st.markdown("<div class='section-label'>Participants</div>", unsafe_allow_html=True)
        col_sender, col_recipient = st.columns(2)
        with col_sender:
            write_sender = st.text_input(
                "Your name (sign-off)",
                value="",
                placeholder="e.g. John Smith",
                key="write_sender",
            )
        with col_recipient:
            write_recipient = st.text_input(
                "Recipient name",
                value="",
                placeholder="e.g. Sarah Chen",
                key="write_recipient",
            )

        st.markdown("<div class='section-label'>Email Details</div>", unsafe_allow_html=True)
        write_intent = st.text_input(
            "Intent / Purpose",
            value=default_write_intent,
            placeholder="e.g. Request project timeline extension",
            key="write_intent",
        )

        write_facts = st.text_area(
            "Key facts to include",
            value=default_write_facts,
            placeholder=DEFAULT_NEW_KEY_FACTS,
            height=155,
            key="write_facts",
            help="Add each fact on a new line, prefixed with a dash (-).",
        )

        write_tone = st.selectbox(
            "Tone",
            options=["formal", "casual", "urgent", "empathetic", "custom"],
            index=(
                ["formal", "casual", "urgent", "empathetic", "custom"].index(
                    default_write_tone
                )
                if default_write_tone in ["formal", "casual", "urgent", "empathetic"]
                else 0
            ),
            key="write_tone",
        )

        if write_tone == "custom":
            custom_write_tone = st.text_input(
                "Specify custom tone",
                value=(
                    default_write_tone
                    if default_write_tone not in ["formal", "casual", "urgent", "empathetic"]
                    else ""
                ),
                placeholder="e.g. diplomatic, warm, assertive",
                key="write_custom_tone",
            )
            final_write_tone = custom_write_tone
        else:
            final_write_tone = write_tone

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        generate_write = st.button("Generate draft →", key="btn_generate_write")

    # ── Tab: Reply ──
    with tab_reply:
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Participants</div>", unsafe_allow_html=True)
        col_reply_sender, col_reply_recipient = st.columns(2)
        with col_reply_sender:
            reply_sender = st.text_input(
                "Your name (sign-off)",
                value="",
                placeholder="e.g. John Smith",
                key="reply_sender",
            )
        with col_reply_recipient:
            reply_recipient = st.text_input(
                "Recipient name",
                value="",
                placeholder="e.g. Sarah Chen",
                key="reply_recipient",
            )

        st.markdown("<div class='section-label'>Original Email</div>", unsafe_allow_html=True)
        original_email = st.text_area(
            "Paste the email you are replying to",
            value=DEFAULT_ORIGINAL_EMAIL,
            height=155,
            key="reply_original_email",
        )

        st.markdown("<div class='section-label'>Reply Details</div>", unsafe_allow_html=True)
        reply_intent = st.text_input(
            "Reply instructions / intent",
            value="",
            placeholder="e.g. Confirm delivery by Wednesday, delay on budget",
            key="reply_intent",
        )

        reply_facts = st.text_area(
            "Key facts to include",
            value="",
            placeholder=DEFAULT_REPLY_KEY_FACTS,
            height=120,
            key="reply_facts",
        )

        reply_tone = st.selectbox(
            "Tone",
            options=["formal", "casual", "urgent", "empathetic", "custom"],
            index=0,
            key="reply_tone",
        )

        if reply_tone == "custom":
            custom_reply_tone = st.text_input(
                "Specify custom tone",
                value="",
                placeholder="e.g. diplomatic, warm, assertive",
                key="reply_custom_tone",
            )
            final_reply_tone = custom_reply_tone
        else:
            final_reply_tone = reply_tone

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        generate_reply = st.button("Generate reply →", key="btn_generate_reply")

# Determine which form was submitted
is_submitted = False
facts_raw = ""
if generate_write:
    active_tab = "write"
    is_submitted = True
    intent_val = write_intent
    facts_raw = write_facts
    tone_val = final_write_tone
    name_val = write_sender
    recipient_val = write_recipient
elif generate_reply:
    active_tab = "reply"
    is_submitted = True
    intent_val = (
        f"You are replying to the following original email:\n---\n"
        f"{original_email}\n---\n"
        f"Your reply instructions/intent: {reply_intent}"
    )
    facts_raw = reply_facts
    tone_val = final_reply_tone
    name_val = reply_sender
    recipient_val = reply_recipient

# ---------------------------------------------------------------------------
# Sidebar — Engine Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sb-brand">
            <div class="sb-logo">&#9993;</div>
            <span class="sb-name">Draftcraft</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-label'>Engine Settings</div>", unsafe_allow_html=True)

    try:
        settings = get_settings()
        default_model = settings.judge_model_name
        default_base_url = settings.groq_base_url
    except Exception:
        default_model = "llama-3.3-70b-versatile"
        default_base_url = "https://api.groq.com/openai/v1"

    model_name = st.text_input(
        "Model ID",
        value=default_model,
        help="LLM model identifier sent to the Groq API",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls how creative vs. predictable the output is.",
    )

    # ── Temperature descriptors ──
    if temperature <= 0.3:
        badge_class = "mode-precise"
        badge_icon  = "○"
        badge_text  = "Precise & Consistent"
        badge_hint  = "Strict, factual — ideal for formal correspondence"
    elif temperature <= 0.6:
        badge_class = "mode-balanced"
        badge_icon  = "◐"
        badge_text  = "Balanced"
        badge_hint  = "Reliable with light variation — great for business email"
    else:
        badge_class = "mode-creative"
        badge_icon  = "●"
        badge_text  = "Expressive & Creative"
        badge_hint  = "More varied language — suited for marketing or creative copy"

    st.markdown(
        f"""
        <div class="temp-bar-wrap"></div>
        <div class="temp-labels">
            <span class="temp-label-low">Low creativity</span>
            <span class="temp-label-high">High creativity</span>
        </div>
        <div class="temp-badge {badge_class}">{badge_icon}&nbsp;&nbsp;{badge_text}</div>
        <div class="temp-hint">{badge_hint}</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Temperature Guide</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="temp-guide">
            <div class="temp-guide-row">
                <span class="tgr-range r-low">0.0 – 0.3 &nbsp;Precise</span><br>
                Outputs the same structure every time. Best for legal, compliance, or official memos.
            </div>
            <div class="temp-guide-row">
                <span class="tgr-range r-mid">0.4 – 0.6 &nbsp;Balanced</span><br>
                Natural phrasing with mild variation. Recommended for most business email.
            </div>
            <div class="temp-guide-row">
                <span class="tgr-range r-high">0.7 – 1.0 &nbsp;Expressive</span><br>
                More inventive, opinionated language. Good for outreach, newsletters, or creative writing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Connection</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-label">API Endpoint</div>
            <div class="info-card-value">{default_base_url}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Core generation (async)
# ---------------------------------------------------------------------------
async def run_generation(
    intent: str,
    key_facts: list[str],
    tone: str,
    model_id: str,
    model_name: str,
    temp: float,
):
    scenario = Scenario(
        id="live-run",
        intent=intent,
        key_facts=key_facts,
        tone=tone,
        human_reference_email="",
    )

    try:
        app_settings = get_settings()
    except Exception:
        api_key = os.environ.get("GROQ_API_KEY")
        from email_eval.config import Settings
        app_settings = Settings(
            groq_api_key=api_key or "DUMMY",
            model_a=model_name if model_id == "model_a" else "llama-3.3-70b-versatile",
            model_b=model_name if model_id == "model_b" else "openai/gpt-oss-120b",
            groq_base_url=default_base_url,
        )

    if not app_settings.groq_api_key or app_settings.groq_api_key == "DUMMY":
        raise ValueError("API key is missing. Set GROQ_API_KEY in your .env file.")

    client = GroqClient(app_settings)
    return await generate_email(scenario, model_id, model_name, client)


# ---------------------------------------------------------------------------
# RIGHT PANE — Response
# ---------------------------------------------------------------------------
with col_output:
    st.markdown(
        """
        <div class="pane-header">
            <div class="pane-title">
                <div class="pane-dot"></div>
                Generated Response
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_submitted:
        facts_list = [
            f.strip().lstrip("- ").strip()
            for f in facts_raw.split("\n")
            if f.strip()
        ]

        if active_tab == "write" and not intent_val:
            st.error("Please provide an intent / purpose for the email.")
        elif active_tab == "reply" and not reply_intent.strip():
            st.error("Please provide reply instructions / intent.")
        elif not facts_list:
            st.error("Please add at least one key fact.")
        elif not tone_val:
            st.error("Please specify a tone.")
        else:
            with st.spinner("Generating your email…"):
                try:
                    effective_intent = intent_val
                    if recipient_val.strip():
                        effective_intent = (
                            f"Address the email to {recipient_val.strip()}. "
                            f"Use their name in the greeting/salutation.\n\n"
                            f"{effective_intent}"
                        )

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    generated: GeneratedEmail = loop.run_until_complete(
                        run_generation(
                            effective_intent,
                            facts_list,
                            tone_val,
                            selected_model_id,
                            model_name,
                            temperature,
                        )
                    )

                    if generated.cot_scratchpad:
                        with st.expander("View planning scratchpad"):
                            st.markdown(
                                f"<div style='font-size:0.81rem; color:#78716c; "
                                f"white-space:pre-wrap; font-family:\"JetBrains Mono\", monospace; "
                                f"line-height:1.65;'>{generated.cot_scratchpad}</div>",
                                unsafe_allow_html=True,
                            )
                    elif generated.parse_fallback_used:
                        st.warning(
                            "Model did not follow the standard reasoning format. "
                            "Output used as-is."
                        )

                    email_body = generated.parsed_email or ""
                    if name_val.strip():
                        email_body = _strip_model_signoff(email_body)
                        email_body = (
                            email_body.rstrip()
                            + f"\n\nBest regards,\n{name_val.strip()}"
                        )

                    strategy_tag = selected_model_id.replace("_", " ").upper()

                    st.markdown(
                        f"""
                        <div class='result-card'>
                            <div class='result-header'>
                                <span>&#128231;&nbsp;&nbsp;generated_draft.eml</span>
                                <span class='tag'>{strategy_tag}</span>
                            </div>
                            <div class='result-body'>{email_body}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.info(
                        "Please verify your GROQ_API_KEY in the .env file "
                        "or check your network connection."
                    )
    else:
        st.markdown(
            """
            <div class='placeholder-box'>
                <div class='placeholder-icon'>&#9998;</div>
                <div class='placeholder-title'>Ready to draft</div>
                <div class='placeholder-sub'>
                    Fill in the workspace on the left, then click
                    <strong style='color:#c2a46e;'>Generate draft</strong>
                    to see your email appear here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class='app-footer'>
        Draftcraft &nbsp;&middot;&nbsp; Powered by Groq &nbsp;&middot;&nbsp; Built by Sabuj Majumder
    </div>
    """,
    unsafe_allow_html=True,
)
