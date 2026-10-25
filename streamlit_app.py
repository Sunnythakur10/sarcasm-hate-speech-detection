import sys
from pathlib import Path

import streamlit as st

# Make the existing src modules importable
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import SarcasmHatePipeline


st.set_page_config(
    page_title="Sarcasm & Hate Speech Detection",
    page_icon=None,
    layout="centered",
)

st.title("Sarcasm & Hate Speech Detection")
st.caption(
    "English and Hinglish sarcasm detection "
    "with English hate-speech classification."
)

st.divider()


@st.cache_resource(show_spinner=False)
def load_pipeline():
    return SarcasmHatePipeline()


# Load the models once and reuse them between interactions
try:
    with st.spinner("Loading NLP models. First startup may take a few minutes..."):
        nlp = load_pipeline()
except Exception as exc:
    st.error("The NLP pipeline could not be loaded.")
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()


with st.form("classification_form"):
    text = st.text_area(
        "Enter text to analyze",
        placeholder="Example: Oh great, another genius decision.",
        height=140,
    )

    submitted = st.form_submit_button(
        "Analyze Text",
        type="primary",
        use_container_width=True,
    )


if submitted:
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            with st.spinner("Analyzing text..."):
                result = nlp.predict(text.strip())

            st.divider()
            st.subheader("Analysis Results")

            st.markdown(
                f"### {result['bucket'].replace('_', ' ').title()}"
            )
            st.write(f"**Detected language:** {result['language'].title()}")

            if result.get("note"):
                st.info(result["note"])

            if result.get("hate_label") is not None:
                st.write(
                    f"**Hate classification:** "
                    f"{result['hate_label'].title()}"
                )

                probabilities = result.get("hate_class_probabilities")

                if probabilities:
                    st.write("**Class probabilities**")

                    for label, probability in sorted(
                        probabilities.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    ):
                        st.progress(
                            float(probability),
                            text=f"{label.title()}: {probability:.1%}",
                        )

            if result.get("is_sarcastic") is not None:
                sarcasm = (
                    "Detected"
                    if result["is_sarcastic"]
                    else "Not detected"
                )
                st.write(f"**Sarcasm:** {sarcasm}")

                score = result.get("sarcasm_confidence_score")
                if score is not None:
                    st.caption(
                        f"Classifier decision score: {score:.3f} "
                        "(not a probability)"
                    )

        except Exception as exc:
            st.error("Classification failed.")
            with st.expander("Technical details"):
                st.exception(exc)