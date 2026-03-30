import os
import sys
import time
import streamlit as st
import mlflow
from llama_cpp import Llama

st.set_page_config(
    page_title="PolicyCortex | AI Cybersecurity Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configure MLflow centralized tracking
@st.cache_resource
def init_mlflow():
    status = "Disconnected"
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("PolicyCortex_LLM")
        # Ping status
        import requests
        requests.get(tracking_uri, timeout=0.5)
        status = "Connected"
    except Exception:
        pass
    return status

MLFLOW_STATUS = init_mlflow()

# Project-root import so policy_engine and rag are reachable
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
RAG_DIR      = os.path.join(PROJECT_ROOT, "rag")
for _p in (PROJECT_ROOT, RAG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from policy_engine.startup_advisor import startup_security_advisor
from policy_engine.roadmap_generator import generate_security_roadmap
from utils.pdf_export import generate_pdf
from security_advisor import ask_security_advisor
from utils.llm_utils import get_unified_llm, GGUF_PATH


# ── LLM Initialisation ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️  Connecting to Intelligent Core...")
def init_llm():
    return get_unified_llm()

LLM_ENGINE = get_unified_llm()
if LLM_ENGINE:
    MODEL_SOURCE = LLM_ENGINE.mode.upper() # "GROQ" or "LOCAL"
else:
    MODEL_SOURCE = "ERROR"


# ── Premium Cybersecurity CSS (Dark Mode) ──────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

<style>
/* ── Global Reset & Base (Dark) ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #0b0e14 !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Scrollbar (Dark) ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0b0e14; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

/* ── Main content padding ── */
[data-testid="stMain"] > div { padding-top: 1rem !important; }

/* ── Sidebar (Dark) ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%) !important;
    border-right: 1px solid #1e293b !important;
    box-shadow: 2px 0 10px rgba(0,0,0,0.3) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Tabs (Dark) ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #111827 !important;
    border-bottom: 2px solid #1e293b !important;
    border-radius: 8px 8px 0 0;
    gap: 8px;
    padding: 4px 4px 0 4px;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: #94a3b8 !important;
    border: none !important;
    border-radius: 6px 6px 0 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: #3b82f6 !important;
    background: #1e293b !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #3b82f6 !important;
    background: #1e293b !important;
    border-bottom: 2px solid #3b82f6 !important;
}
[data-testid="stTabPanel"] {
    background: #151921 !important;
    border: 1px solid #1e293b !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 2rem !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4) !important;
}

/* ── Headings (Dark) ── */
h1 { font-family: 'Inter', sans-serif !important; font-weight: 800 !important; color: #ffffff !important; }
h2, h3 {
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}
h4, h5, h6 { color: #cbd5e1 !important; font-weight: 600 !important; }

/* ── Labels & Captions (Dark) ── */
label, [data-testid="stWidgetLabel"] > p {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
}

/* ── Inputs (selectbox, multiselect, text_area, slider) (Dark) ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #f8fafc !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stMultiSelect"] > div > div:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
}
[data-baseweb="select"] * { color: #f8fafc !important; }
[data-baseweb="popover"] { background: #1e293b !important; border: 1px solid #334155 !important; box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important; }
[data-baseweb="menu"] { background: #1e293b !important; }
[data-baseweb="option"]:hover { background: #334155 !important; }

[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #f8fafc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    caret-color: #3b82f6 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    outline: none !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #3b82f6 !important;
    border-color: #3b82f6 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stTickBar"] { background: #334155 !important; }

/* ── Buttons (Glow) ── */
[data-testid="stButton"] button,
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    padding: 10px 28px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4) !important;
}
[data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.6) !important;
    filter: brightness(1.1) !important;
}

[data-testid="stDownloadButton"] button {
    background: transparent !important;
    border: 1.5px solid #334155 !important;
    border-radius: 8px !important;
    color: #3b82f6 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: #1e293b !important;
    border-color: #3b82f6 !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #151921 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
}
[data-testid="stExpander"] summary {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    padding: 14px 18px !important;
}
[data-testid="stExpander"] summary:hover { background: #334155 !important; }
[data-testid="stExpander"] summary svg { color: #3b82f6 !important; fill: #3b82f6 !important; }

/* ── Info / Success / Warning / Error alerts ── */
[data-testid="stAlert"] {
    background: #1e293b !important;
    border-radius: 8px !important;
    border-left-width: 4px !important;
    color: #f1f5f9 !important;
}

/* ── Divider ── */
hr { border-color: #1e293b !important; }

/* ── Code blocks ── */
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 6px !important;
    color: #38bdf8 !important;
    font-size: 0.82rem !important;
}

/* ── Header brand area ── */
.pc-header {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 2rem;
}
.pc-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3b82f6 0%, #0ea5e9 60%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.pc-tagline {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #94a3b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 8px;
    font-weight: 600;
}
.pc-badge {
    display: inline-flex;
    align-items: center;
    background: #064e3b;
    border: 1px solid #059669;
    border-radius: 6px;
    color: #10b981;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 4px 12px;
    text-transform: uppercase;
    margin-top: 14px;
}

/* ── Section headers ── */
.pc-section-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 6px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.pc-section-title::before {
    content: '';
    display: inline-block;
    width: 4px;
    height: 1.1em;
    background: linear-gradient(180deg, #3b82f6, #0ea5e9);
    border-radius: 2px;
}
.pc-section-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #94a3b8;
    margin-bottom: 2rem;
}

/* ── Card container ── */
.pc-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 20px rgba(0,0,0,0.3);
}
.pc-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #0ea5e9, transparent);
}

/* ── Output / result area ── */
.pc-output {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.5rem 1.8rem;
    margin-top: 1rem;
    color: #f1f5f9;
}
.pc-output-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    color: #3b82f6;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Phase roadmap cards ── */
.pc-phase {
    background: #1e293b;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    border-left: 4px solid;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.pc-phase-1 { border-color: #ef4444; }
.pc-phase-2 { border-color: #f59e0b; }
.pc-phase-3 { border-color: #10b981; }

.pc-phase-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.pc-phase-title {
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.8rem;
}
.pc-phase li {
    color: #cbd5e1;
    font-size: 0.9rem;
    margin-bottom: 0.4rem;
}

/* ── Priority badge ── */
.pc-priority {
    display: inline-block;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 3px 8px;
    text-transform: uppercase;
    margin-left: 8px;
}
.pc-critical { background: #fee2e2; color: #dc2626; }
.pc-high     { background: #fef3c7; color: #d97706; }
.pc-medium   { background: #fef9c3; color: #ca8a04; }
.pc-low      { background: #d1fae5; color: #059669; }

/* ── Sidebar extras ── */
.pc-sidebar-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 800;
    color: #3b82f6;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 1.2rem;
}
.pc-status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 8px;
}
.pc-dot-green { background: #10b981; box-shadow: 0 0 8px rgba(16, 185, 129, 0.6); }
.pc-dot-red   { background: #ef4444; box-shadow: 0 0 8px rgba(239, 68, 68, 0.6); }
.pc-dot-amber { background: #f59e0b; box-shadow: 0 0 8px rgba(245, 158, 11, 0.6); }

/* ── Spinner text ── */
[data-testid="stSpinner"] p {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── GGUF model path ─────────────────────────────────────────────────────────────
# Using unified loader from utils/llm_utils.py (Gemini or Local)
def load_model():
    return LLM_ENGINE


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pc-header">
  <div class="pc-logo">🛡️ PolicyCortex</div>
  <div class="pc-tagline">AI-Powered Cybersecurity Policy Intelligence Platform</div>
  <div class="pc-badge">⬤&nbsp; Model Engine: {MODEL_SOURCE}</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="pc-sidebar-logo">🛡️ PolicyCortex</div>', unsafe_allow_html=True)
    st.markdown("##### Runtime Status")

    if os.path.exists(GGUF_PATH):
        st.markdown(
            '<span class="pc-status-dot pc-dot-green"></span> **Model file found**',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="pc-status-dot pc-dot-red"></span> **Model file missing**',
            unsafe_allow_html=True,
        )

    if MLFLOW_STATUS == "Connected":
        st.markdown(
            '<span class="pc-status-dot pc-dot-green"></span> **MLOps: Connected**',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="pc-status-dot pc-dot-amber"></span> **MLOps: Offline**',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("##### Inference Engine")
    if MODEL_SOURCE == "GROQ":
        st.success("🚀 Running on **Groq Cloud**\n\nModel: **Llama-3.3-70b-versatile**\nInference: **Sub-second Latency**")
    elif MODEL_SOURCE == "LOCAL":
        st.info("💻 Running on **Local CPU** via `llama-cpp-python`\n\nModel: **Gemma-2B Q4_K_M**\nContext: **4096 tokens**")
    else:
        st.error("⚠️ No LLM Engine available. Please check environment variables.")

    st.markdown("---")
    if st.button("⟳  Reload Model", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;'
        'color:#2a4a6a;letter-spacing:0.08em;text-align:center;">'
        "POLICYCORTEX v1.0 · AI RESEARCH BUILD</p>",
        unsafe_allow_html=True,
    )

# ── Tabs ─────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔐  Policy Generator",
    "🧠  Security Advisor",
    "✏️  Policy Refiner",
    "🗺️  Security Roadmap",
])

# ==============================================================================
# TAB 1 — LLM Policy Generator
# ==============================================================================
with tab1:
    st.markdown("""
    <div class="pc-section-title">Generate a Cybersecurity Policy</div>
    <div class="pc-section-desc">
        Configure your organization profile below and click <strong>Generate Policy</strong>
        to produce a structured, actionable policy draft using the on-device Gemma-2B model.
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns(2, gap="large")
        with col1:
            control = st.selectbox(
                "Security Control",
                ["Multi-Factor Authentication", "Access Control",
                 "Logging & Monitoring", "Data Backup", "Encryption",
                 "Incident Response", "Vulnerability Management"],
                key="pg_control",
            )
            company_size = st.selectbox(
                "Company Size", ["Startup", "SME", "Enterprise"],
                key="pg_company_size",
            )
        with col2:
            infra = st.selectbox(
                "Infrastructure", ["Cloud", "On-Prem", "Hybrid"],
                key="pg_infra",
            )
            data = st.selectbox(
                "Data Sensitivity", ["Low", "Medium", "High"],
                key="pg_data",
            )

    max_tokens = st.slider("Max Output Tokens", 100, 500, 300, 50, key="pg_max_tokens")

    if st.button("Generate Policy", type="primary", key="pg_generate"):
        system_msg = (
            "You are a professional cybersecurity policy advisor. "
            "Write clear, structured, actionable cybersecurity policies."
        )
        user_msg = (
            f"Draft a professional cybersecurity policy for the "
            f"'{control}' security control at a {company_size} company.\n"
            f"Infrastructure: {infra}\n"
            f"Data Sensitivity: {data}\n\n"
            "Explain why this control matters and provide clear implementation steps."
        )

        try:
            with st.spinner("Generating policy draft…"):
                start_time = time.time()
                llm = load_model()
                
                # Log to MLflow only if connected
                if MLFLOW_STATUS == "Connected":
                    with mlflow.start_run(run_name="Policy_Generation"):
                        mlflow.log_param("control", control)
                        mlflow.log_param("company_size", company_size)
                        mlflow.log_param("infrastructure", infra)
                        mlflow.log_param("max_tokens", max_tokens)
                        mlflow.log_param("system_prompt", system_msg)
                        mlflow.log_param("model_engine", MODEL_SOURCE)
                        
                        response = llm.create_chat_completion(
                            messages=[
                                {"role": "system", "content": system_msg},
                                {"role": "user",   "content": user_msg},
                            ],
                            max_tokens=max_tokens,
                            temperature=0.7,
                            stop=["</s>"],
                        )
                        
                        policy_text = response["choices"][0]["message"]["content"].strip()
                        if not policy_text:
                            policy_text = "(Model returned an empty response — try again.)"
                            
                        duration = time.time() - start_time
                        mlflow.log_metric("latency_seconds", duration)
                        mlflow.log_text(policy_text, "generated_policy.txt")
                else:
                    # Offline generation
                    response = llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user",   "content": user_msg},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.7,
                        stop=["</s>"],
                    )
                    policy_text = response["choices"][0]["message"]["content"].strip()
                    if not policy_text:
                        policy_text = "(Model returned an empty response — try again.)"

            st.markdown('<div class="pc-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="pc-output-label">Generated Policy Output</div>',
                unsafe_allow_html=True,
            )
            st.markdown(policy_text)
            st.markdown('</div>', unsafe_allow_html=True)

            pdf_path = generate_pdf("Cybersecurity Policy", policy_text)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️  Download Policy as PDF",
                    data=f,
                    file_name="policycortex_policy.pdf",
                    mime="application/pdf",
                    key="download_policy_pdf",
                )

        except Exception as e:
            st.error(f"Error during generation: {e}")

# ==============================================================================
# TAB 2 — RAG-Powered Chat Security Advisor
# ==============================================================================
with tab2:
    st.markdown("""
    <div class="pc-section-title">AI Security Advisor</div>
    <div class="pc-section-desc">
        Chat with the AI security advisor grounded in CIS Controls and OWASP
        best practices. Your conversation history is preserved within this session.
    </div>
    """, unsafe_allow_html=True)

    # ── Session state for chat history ──────────────────────────────────────
    # Each entry: {"user": str, "assistant": str, "docs": list[dict]}
    if "advisor_history" not in st.session_state:
        st.session_state.advisor_history = []

    # ── Clear chat button ────────────────────────────────────────────────────
    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("🗑️  Clear Chat", key="advisor_clear"):
            st.session_state.advisor_history = []
            st.rerun()

    st.markdown("---")

    # ── Render existing conversation ─────────────────────────────────────────
    for turn in st.session_state.advisor_history:
        with st.chat_message("user"):
            st.markdown(turn["user"])

        with st.chat_message("assistant", avatar="🛡️"):
            st.markdown(turn["assistant"])

            # Retrieved source controls for this turn
            docs = turn.get("docs", [])
            if docs:
                with st.expander("📚 Retrieved Reference Controls", expanded=False):
                    for i, doc in enumerate(docs, 1):
                        control_name = doc.get("control", "Unknown Control")
                        source       = doc.get("source",  "Unknown Source")
                        content      = doc.get("content", "")
                        st.markdown(
                            f"**[{i}] {control_name}** "
                            f'<span class="pc-priority pc-low">{source}</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption(content)
                        if i < len(docs):
                            st.markdown("---")

    # ── Chat input (always at the bottom) ───────────────────────────────────
    user_question = st.chat_input(
        "Ask a security question… e.g. How should startups secure their APIs?",
        key="advisor_chat_input",
    )

    if user_question:
        # Show the new user message immediately
        with st.chat_message("user"):
            st.markdown(user_question)

        # Build the history list that the advisor expects
        history_for_advisor = [
            {"user": t["user"], "assistant": t["assistant"]}
            for t in st.session_state.advisor_history
        ]

        try:
            with st.chat_message("assistant", avatar="🛡️"):
                with st.spinner("Retrieving relevant controls and generating answer…"):
                    start_time = time.time()
                    if MLFLOW_STATUS == "Connected":
                        with mlflow.start_run(run_name="Security_Advisor_Chat"):
                            mlflow.log_param("user_question", user_question)
                            mlflow.log_param("history_length", len(history_for_advisor))
                            mlflow.log_param("model_engine", MODEL_SOURCE)
                            
                            answer, retrieved_docs = ask_security_advisor(
                                user_question, history=history_for_advisor
                            )
                            
                            duration = time.time() - start_time
                            mlflow.log_metric("latency_seconds", duration)
                            mlflow.log_metric("retrieved_docs_count", len(retrieved_docs))
                            mlflow.log_text(answer, "advisor_response.txt")
                    else:
                        answer, retrieved_docs = ask_security_advisor(
                            user_question, history=history_for_advisor
                        )

                st.markdown(answer)

                # Show retrieved docs inline
                if retrieved_docs:
                    with st.expander("📚 Retrieved Reference Controls", expanded=False):
                        for i, doc in enumerate(retrieved_docs, 1):
                            control_name = doc.get("control", "Unknown Control")
                            source       = doc.get("source",  "Unknown Source")
                            content      = doc.get("content", "")
                            st.markdown(
                                f"**[{i}] {control_name}** "
                                f'<span class="pc-priority pc-low">{source}</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(content)
                            if i < len(retrieved_docs):
                                st.markdown("---")

            # Persist this turn in session state
            st.session_state.advisor_history.append({
                "user":      user_question,
                "assistant": answer,
                "docs":      retrieved_docs,
            })

        except Exception as e:
            st.error(f"Error running advisor: {e}")

# ==============================================================================
# TAB 3 — Policy Refiner
# ==============================================================================
with tab3:
    st.markdown("""
    <div class="pc-section-title">Policy Refiner</div>
    <div class="pc-section-desc">
        Paste an existing cybersecurity policy below.
        The AI will rewrite it into a <strong>clear, professional, NIST-aligned</strong> document
        with structured sections and compliance references.
    </div>
    """, unsafe_allow_html=True)

    user_policy = st.text_area(
        "Paste your existing policy",
        height=220,
        placeholder="Example: Users should use strong passwords and not share them…",
        key="refiner_input",
    )

    if st.button("Refine Policy", type="primary", key="refiner_run"):
        if not user_policy.strip():
            st.warning("Please paste a policy to refine.")
        else:
            system_msg = (
                "You are a professional cybersecurity policy expert. "
                "Rewrite policies clearly and professionally using structured sections."
            )
            user_msg = f"""Rewrite the following cybersecurity policy professionally.

Structure the output into:

1. Policy Statement
2. Why It Matters
3. Implementation Steps
4. Compliance References

Policy:
{user_policy}
"""
            try:
                with st.spinner("Refining policy…"):
                    start_time = time.time()
                    llm = load_model()
                    
                    if MLFLOW_STATUS == "Connected":
                        with mlflow.start_run(run_name="Policy_Refiner"):
                            mlflow.log_param("input_length", len(user_policy))
                            mlflow.log_text(user_policy, "original_policy.txt")
                            
                            response = llm.create_chat_completion(
                                messages=[
                                    {"role": "system", "content": system_msg},
                                    {"role": "user",   "content": user_msg},
                                ],
                                max_tokens=400,
                                temperature=0.6,
                                stop=["</s>"],
                            )

                            refined_policy = response["choices"][0]["message"]["content"].strip()
                            
                            duration = time.time() - start_time
                            mlflow.log_metric("latency_seconds", duration)
                            mlflow.log_text(refined_policy, "refined_policy.txt")
                    else:
                        response = llm.create_chat_completion(
                            messages=[
                                {"role": "system", "content": system_msg},
                                {"role": "user",   "content": user_msg},
                            ],
                            max_tokens=400,
                            temperature=0.6,
                            stop=["</s>"],
                        )
                        refined_policy = response["choices"][0]["message"]["content"].strip()

                st.markdown('<div class="pc-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="pc-output-label">Refined Policy Output</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(refined_policy)
                st.markdown('</div>', unsafe_allow_html=True)

                pdf_path = generate_pdf("Refined Cybersecurity Policy", refined_policy)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="⬇️  Download Refined Policy as PDF",
                        data=f,
                        file_name="policycortex_refined_policy.pdf",
                        mime="application/pdf",
                        key="download_refined_pdf",
                    )

            except Exception as e:
                st.error(f"Error refining policy: {e}")

# ==============================================================================
# TAB 4 — Security Roadmap Generator
# ==============================================================================
with tab4:
    st.markdown("""
    <div class="pc-section-title">Security Roadmap Generator</div>
    <div class="pc-section-desc">
        Generate a <strong>30–60–90 day phased cybersecurity implementation roadmap</strong>
        automatically prioritised by your Cyber Risk Index and organisational profile.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        roadmap_size = st.selectbox(
            "Startup Size",
            ["small", "medium", "large"],
            key="roadmap_size",
        )
        roadmap_infra = st.selectbox(
            "Infrastructure",
            ["cloud", "on-prem", "hybrid"],
            key="roadmap_infra",
        )
        roadmap_data = st.selectbox(
            "Data Sensitivity",
            ["low", "medium", "high"],
            key="roadmap_data",
        )

    with col2:
        roadmap_maturity = st.selectbox(
            "Security Maturity",
            ["none", "basic", "advanced"],
            key="roadmap_maturity",
        )
        roadmap_systems = st.multiselect(
            "System Types",
            ["database", "api", "web", "mobile", "iot"],
            default=["database", "api", "web"],
            key="roadmap_systems",
        )
        roadmap_cri = st.slider(
            "Cyber Risk Index",
            0, 100, 50,
            key="roadmap_cri",
        )

    if st.button("Generate Roadmap", type="primary", key="roadmap_run"):
        context = {
            "startup_size":      roadmap_size,
            "infrastructure":    roadmap_infra,
            "system_types":      roadmap_systems,
            "data_sensitivity":  roadmap_data,
            "security_maturity": roadmap_maturity,
        }

        recs    = startup_security_advisor(context, cri=roadmap_cri)
        roadmap = generate_security_roadmap(recs, cri=roadmap_cri)

        st.markdown("---")
        st.markdown(
            '<div class="pc-output-label">90-Day Cybersecurity Roadmap</div>',
            unsafe_allow_html=True,
        )

        # ── Phase 1 ──
        items_html = "".join(f"<li>{c}</li>" for c in roadmap["phase1"])
        st.markdown(f"""
        <div class="pc-phase pc-phase-1">
          <div class="pc-phase-label">Phase 1 &nbsp;·&nbsp; Week 1–2</div>
          <div class="pc-phase-title">🔴 Immediate Actions</div>
          <ul>{items_html}</ul>
        </div>
        """, unsafe_allow_html=True)

        # ── Phase 2 ──
        items_html = "".join(f"<li>{c}</li>" for c in roadmap["phase2"])
        st.markdown(f"""
        <div class="pc-phase pc-phase-2">
          <div class="pc-phase-label">Phase 2 &nbsp;·&nbsp; Week 3–6</div>
          <div class="pc-phase-title">🟠 Risk Reduction</div>
          <ul>{items_html}</ul>
        </div>
        """, unsafe_allow_html=True)

        # ── Phase 3 ──
        items_html = "".join(f"<li>{c}</li>" for c in roadmap["phase3"])
        st.markdown(f"""
        <div class="pc-phase pc-phase-3">
          <div class="pc-phase-label">Phase 3 &nbsp;·&nbsp; Week 7–12</div>
          <div class="pc-phase-title">🟢 Security Maturity</div>
          <ul>{items_html}</ul>
        </div>
        """, unsafe_allow_html=True)

        # ── PDF export ──
        roadmap_text = "90-Day Cybersecurity Roadmap\n\n"
        roadmap_text += "Phase 1 — Immediate (Week 1-2)\n"
        for c in roadmap["phase1"]:
            roadmap_text += f"- {c}\n"
        roadmap_text += "\nPhase 2 — Risk Reduction (Week 3-6)\n"
        for c in roadmap["phase2"]:
            roadmap_text += f"- {c}\n"
        roadmap_text += "\nPhase 3 — Security Maturity (Week 7-12)\n"
        for c in roadmap["phase3"]:
            roadmap_text += f"- {c}\n"

        pdf_path = generate_pdf("Security Roadmap", roadmap_text)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇️  Download Roadmap as PDF",
                data=f,
                file_name="policycortex_roadmap.pdf",
                mime="application/pdf",
                key="download_roadmap_pdf",
            )