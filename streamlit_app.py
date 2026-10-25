import html
import sys
from pathlib import Path

import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="SarcHate | NLP Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import SarcasmHatePipeline  # noqa: E402


# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown(
    r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #090d18;
        --panel: #111827;
        --panel2: #151e30;
        --border: rgba(148, 163, 184, 0.17);
        --text: #edf2ff;
        --muted: #aab5cb;
        --purple: #9b8cff;
        --teal: #53e0c0;
    }

    .stApp {
        background:
          radial-gradient(ellipse at 5% 0%, rgba(106, 76, 255, .13), transparent 32%),
          radial-gradient(ellipse at 95% 10%, rgba(30, 210, 180, .08), transparent 30%),
          var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] { background: transparent; }

    .block-container {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -.035em;
    }

    p, label, .stCaption { color: var(--muted); }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        padding: 17px 21px;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: rgba(17, 24, 39, .88);
        margin-bottom: 36px;
    }

    .brand { display: flex; align-items: center; gap: 13px; }

    .brand-icon {
        width: 43px;
        height: 43px;
        border-radius: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 17px;
        font-weight: 700;
        color: #ffffff;
        background: linear-gradient(135deg, #8773ff, #4b43bd);
        box-shadow: 0 0 25px rgba(135, 115, 255, .19);
    }

    .brand-name {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 16px;
        color: #f3f6ff;
    }

    .brand-sub { color: #aab5cb; font-size: 11px; margin-top: 3px; }

    .version-pill {
        border: 1px solid rgba(155, 140, 255, .40);
        background: rgba(155, 140, 255, .12);
        color: #d5cdff;
        padding: 7px 11px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .07em;
        white-space: nowrap;
    }

    .eyebrow {
        color: #71ecd3;
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(32px, 4vw, 49px);
        font-weight: 700;
        letter-spacing: -.055em;
        line-height: 1.12;
        margin-bottom: 15px;
        color: #f0f3ff;
    }

    .gradient-text {
        background: linear-gradient(90deg, #b0a3ff, #63e8ce);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero-description {
        font-size: 15px;
        line-height: 1.8;
        max-width: 690px;
        color: #b2bdd2;
        margin-bottom: 23px;
    }

    .feature-row { display: flex; flex-wrap: wrap; gap: 9px; margin-bottom: 30px; }

    .feature-pill {
        border: 1px solid var(--border);
        background: rgba(17, 24, 39, .78);
        border-radius: 9px;
        padding: 8px 11px;
        color: #dce4f6;
        font-size: 12px;
    }

    .panel {
        background: linear-gradient(150deg, rgba(21, 30, 48, .97), rgba(15, 22, 36, .98));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px;
        height: 100%;
    }

    .panel-heading {
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: 'Space Grotesk', sans-serif;
        color: #f0f3ff;
        font-weight: 600;
        font-size: 17px;
        margin-bottom: 5px;
    }

    .panel-description {
        font-size: 12px;
        color: #b2bdd2;
        line-height: 1.7;
        margin-bottom: 18px;
    }

    .section-label {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
        color: #b9c6e2;
        margin: 18px 0 10px;
    }

    [data-testid="stTextArea"] textarea {
        background: #0c1322 !important;
        color: #f5f7ff !important;
        caret-color: #b5a8ff !important;
        border: 1px solid #3b4965 !important;
        border-radius: 13px !important;
        padding: 16px !important;
        font-family: 'DM Sans', sans-serif !important;
        line-height: 1.7 !important;
        font-size: 14px !important;
        min-height: 190px;
    }

    [data-testid="stTextArea"] textarea::placeholder { color: #8795af !important; opacity: 1; }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #a99aff !important;
        box-shadow: 0 0 0 1px rgba(169, 154, 255, .25) !important;
    }

    /* Standard buttons: strong foreground/background contrast */
    .stButton > button {
        width: 100%;
        min-height: 44px;
        border-radius: 10px !important;
        background: #202b41 !important;
        color: #f8fafc !important;
        border: 1px solid #465673 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        opacity: 1 !important;
        transition: filter .15s ease, border-color .15s ease;
    }

    .stButton > button:hover:not(:disabled) {
        background: #2a3853 !important;
        color: #ffffff !important;
        border-color: #9b8cff !important;
    }

    /* Primary analysis button */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(100deg, #7965ed, #6551d5) !important;
        color: #ffffff !important;
        border: 1px solid #b2a6ff !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        min-height: 48px;
        box-shadow: 0 7px 24px rgba(105, 84, 225, .19);
    }

    .stButton > button[kind="primary"]:hover:not(:disabled),
    .stButton > button[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
        background: linear-gradient(100deg, #8b79ff, #7866ed) !important;
        border-color: #d0c7ff !important;
    }

    /* Disabled buttons remain readable but visibly inactive */
    .stButton > button:disabled {
        color: #f0edff !important;
        opacity: .82 !important;
        filter: saturate(.78);
        cursor: not-allowed !important;
    }

    .result-hero {
        border: 1px solid var(--border);
        background: radial-gradient(circle at 100% 0%, rgba(155, 140, 255, .13), transparent 38%), #111827;
        border-radius: 18px;
        padding: 25px;
        margin: 13px 0 17px;
    }

    .result-label { font-size: 10px; letter-spacing: .15em; color: #b2bdd2; font-weight: 700; text-transform: uppercase; margin-bottom: 12px; }
    .result-title { font-family: 'Space Grotesk', sans-serif; font-size: clamp(24px, 3vw, 34px); font-weight: 700; letter-spacing: -.04em; line-height: 1.2; color: #f4f6ff; overflow-wrap: anywhere; }
    .result-description { font-size: 13px; color: #b9c5da; line-height: 1.75; margin-top: 11px; max-width: 720px; }

    .metric-card {
        background: #111827;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 17px;
        min-height: 105px;
    }

    .metric-title { text-transform: uppercase; color: #b2bdd2; font-size: 10px; letter-spacing: .1em; font-weight: 700; }
    .metric-value { color: #f4f6ff; font-family: 'Space Grotesk', sans-serif; font-size: 21px; font-weight: 600; margin-top: 10px; overflow-wrap: anywhere; }

    .quote-card {
        border-left: 3px solid #9b8cff;
        border-radius: 0 10px 10px 0;
        background: rgba(12, 19, 34, .95);
        padding: 13px 15px;
        color: #f0f4ff;
        font-size: 13px;
        line-height: 1.8;
        overflow-wrap: anywhere;
    }

    /* Accessible class-probability labels and bars */
    .probability-row { margin: 16px 0 20px; }
    .probability-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 9px;
        color: #f2f5ff;
        font-size: 13px;
        font-weight: 600;
    }
    .probability-value { color: #d2caff; font-weight: 700; white-space: nowrap; }
    .probability-track {
        height: 11px;
        overflow: hidden;
        background: #29354b;
        border: 1px solid #465673;
        border-radius: 20px;
    }
    .probability-fill {
        height: 100%;
        border-radius: 20px;
        background: linear-gradient(90deg, #917cff, #55dfc2);
    }

    .footer {
        border-top: 1px solid var(--border);
        margin-top: 40px;
        padding-top: 18px;
        color: #a0acc4;
        font-size: 11px;
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
    }

    @media (max-width: 700px) {
        .block-container { padding: 1rem .8rem 2rem; }
        .topbar { padding: 13px; margin-bottom: 25px; }
        .hero-title { font-size: 34px; }
        .panel { padding: 16px; }
        .result-hero { padding: 18px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Session state and example inputs
# --------------------------------------------------
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_input" not in st.session_state:
    st.session_state.last_input = ""

SAMPLES = {
    "English sarcasm": "I work 40 hours a week for me to be this poor.",
    "Normal English": "I enjoyed watching the movie yesterday.",
    "Hate speech": (
        "People from that ethnic group are inferior "
        "and should not have equal rights."
    ),
    "Hinglish": "Bhai kya mast joke maara hai.",
}


def choose_sample(text: str) -> None:
    st.session_state.input_text = text
    st.session_state.last_result = None
    st.session_state.last_input = ""


@st.cache_resource(show_spinner=False)
def load_pipeline() -> SarcasmHatePipeline:
    return SarcasmHatePipeline()


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-icon">SH</div>
            <div>
                <div class="brand-name">SarcHate Studio</div>
                <div class="brand-sub">Natural Language Classification</div>
            </div>
        </div>
        <div class="version-pill">MODEL V7 · NLP STUDIO</div>
    </div>
    <div class="eyebrow">Text intelligence workspace</div>
    <div class="hero-title">
        Understand the intent.<br>
        <span class="gradient-text">Uncover the undertone.</span>
    </div>
    <div class="hero-description">
        Analyze text for sarcasm, hate speech, and offensive language
        through a unified classification pipeline. Supports English
        sarcasm and romanized Hindi (Hinglish) sarcasm.
    </div>
    <div class="feature-row">
        <div class="feature-pill">English sarcasm</div>
        <div class="feature-pill">Hinglish sarcasm</div>
        <div class="feature-pill">Hate classification</div>
        <div class="feature-pill">Offensive language detection</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Load once per Streamlit process.
try:
    with st.spinner("Initializing the classification engines..."):
        nlp = load_pipeline()
    model_error = None
except Exception as exc:
    nlp = None
    model_error = exc


# --------------------------------------------------
# Main workspace
# --------------------------------------------------
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-heading">01 &nbsp; Text workspace</div>
            <div class="panel-description">
                Enter a sentence or a short passage to begin analysis.
                The system routes text to the appropriate classifier.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_text = st.text_area(
        "Text to analyze",
        key="input_text",
        height=205,
        max_chars=3000,
        placeholder=(
            "Type or paste your text here...\n\n"
            "Example: Oh wonderful, exactly what I needed today."
        ),
        label_visibility="collapsed",
    )

    count_col, clear_col = st.columns([1, 1])
    with count_col:
        st.caption(f"{len(user_text)} / 3000 characters")
    with clear_col:
        if st.button("Clear text", use_container_width=True):
            st.session_state.input_text = ""
            st.session_state.last_result = None
            st.session_state.last_input = ""
            st.rerun()

    analyze_clicked = st.button(
        "Run text analysis",
        type="primary",
        use_container_width=True,
        disabled=(nlp is None),
    )

    if model_error is not None:
        st.error("The NLP pipeline could not be initialized.")
        with st.expander("Technical details"):
            st.exception(model_error)

    if analyze_clicked:
        if not user_text.strip():
            st.warning("Enter some text before starting the analysis.")
        else:
            try:
                with st.spinner("Analyzing text..."):
                    st.session_state.last_result = nlp.predict(user_text.strip())
                    st.session_state.last_input = user_text.strip()
            except Exception as exc:
                st.session_state.last_result = None
                st.error("Classification failed.")
                with st.expander("Technical details"):
                    st.exception(exc)

with right:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-heading">02 &nbsp; Try an example</div>
            <div class="panel-description">
                Load a sample into the workspace, then run the analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for sample_name, sample_text in SAMPLES.items():
        with st.container(border=True):
            st.markdown(f"**{html.escape(sample_name)}**")
            st.caption(sample_text)
            st.button(
                f"Use {sample_name.lower()}",
                key=f"sample_{sample_name}",
                use_container_width=True,
                on_click=choose_sample,
                args=(sample_text,),
            )

    st.markdown(
        """
        <div style="margin-top:17px;padding:17px;border-radius:14px;
                    border:1px solid rgba(83,224,192,.25);
                    background:rgba(83,224,192,.055);">
            <div style="color:#75ead4;font-size:10px;font-weight:700;
                        letter-spacing:.12em;margin-bottom:8px;">
                LANGUAGE SUPPORT
            </div>
            <div style="color:#edf3ff;font-size:13px;line-height:1.9;">
                English: sarcasm and hate/offensive classification<br>
                Hinglish: sarcasm classification<br>
                Devanagari Hindi: currently unsupported
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Results
# --------------------------------------------------
result = st.session_state.last_result

if result:
    st.markdown("---")
    st.markdown(
        """
        <div class="eyebrow">Analysis output</div>
        <h2 style="margin-top:0">Classification report</h2>
        <p style="color:#b2bdd2;font-size:13px">
            Results for the most recently analyzed input.
        </p>
        """,
        unsafe_allow_html=True,
    )

    bucket = result.get("bucket", "unknown")
    descriptions = {
        "hate_sarcastic": "The classifiers flagged both hate-related content and sarcasm in this text.",
        "hate": "The hate classifier selected the hate category for this text.",
        "offensive_sarcastic": "The classifiers flagged offensive language and sarcasm in this text.",
        "offensive": "The hate classifier selected the offensive category for this text.",
        "sarcastic": "The sarcasm classifier detected sarcastic language.",
        "neutral": "The combined pipeline did not select a hate or offensive label, and the sarcasm classifier did not flag sarcasm.",
        "unsupported": "This writing script is not currently supported by the project's sarcasm classifier.",
    }
    result_titles = {
        "hate_sarcastic": "Hate + Sarcasm",
        "hate": "Hate-related content",
        "offensive_sarcastic": "Offensive + Sarcastic",
        "offensive": "Offensive language",
        "sarcastic": "Sarcastic",
        "neutral": "Neutral",
        "unsupported": "Unsupported language script",
    }

    result_title = result_titles.get(bucket, bucket.replace("_", " ").title())
    safe_title = html.escape(result_title)
    safe_description = html.escape(descriptions.get(bucket, "Analysis complete."))

    st.markdown(
        f"""
        <div class="result-hero">
            <div class="result-label">Primary classification</div>
            <div class="result-title">{safe_title}</div>
            <div class="result-description">{safe_description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    safe_input = html.escape(st.session_state.last_input)
    st.markdown(
        f"""
        <div class="section-label">Analyzed text</div>
        <div class="quote-card">{safe_input}</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    metric1, metric2 = st.columns(2)
    language = result.get("language", "unknown").title()
    is_sarcastic = result.get("is_sarcastic")
    hate_label = result.get("hate_label")
    sarcasm_display = (
        "Detected" if is_sarcastic
        else "Not detected" if is_sarcastic is False
        else "Not assessed"
    )
    hate_display = (
        str(hate_label).replace("_", " ").title()
        if hate_label is not None
        else "Not assessed"
    )

    with metric1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Detected language</div>
                <div class="metric-value">{html.escape(language)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Sarcasm detection</div>
                <div class="metric-value">{html.escape(sarcasm_display)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if hate_label is not None:
        st.markdown(
            '<div class="section-label">Hate / offensive classifier</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Selected class</div>
                <div class="metric-value">{html.escape(hate_display)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        probabilities = result.get("hate_class_probabilities") or {}
        if probabilities:
            st.markdown(
                '<div class="section-label">Class probabilities</div>',
                unsafe_allow_html=True,
            )

            for label, probability in sorted(
                probabilities.items(), key=lambda item: float(item[1]), reverse=True
            ):
                numeric_probability = max(0.0, min(1.0, float(probability)))
                percentage = numeric_probability * 100
                safe_label = html.escape(str(label).replace("_", " ").title())
                st.markdown(
                    f"""
                    <div class="probability-row">
                        <div class="probability-header">
                            <span>{safe_label}</span>
                            <span class="probability-value">{percentage:.1f}%</span>
                        </div>
                        <div class="probability-track">
                            <div class="probability-fill" style="width:{percentage:.2f}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    sarcasm_score = result.get("sarcasm_confidence_score")
    if sarcasm_score is not None:
        st.caption(
            f"Sarcasm classifier decision score: {float(sarcasm_score):.3f}. "
            "This is a decision score, not a probability."
        )

    if result.get("note"):
        st.info(result["note"])


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        <span>SarcHate Studio · Text Classification</span>
        <span>Research prototype · Predictions may be incorrect</span>
    </div>
    """,
    unsafe_allow_html=True,
)
