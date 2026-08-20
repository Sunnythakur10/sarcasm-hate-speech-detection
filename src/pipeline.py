"""
Chains classifiers into one pipeline. Input: raw text -> Output.

Sarcasm detection routes by detected language/script:
    English            -> TF-IDF + LinearSVC on Reddit SARC data
    Hinglish (romanized
    Hindi, Latin script) -> TF-IDF (char n-grams) + LinearSVC on
                            human-labeled Hinglish tweet data
    Devanagari script   -> NOT SUPPORTED. The Devanagari Hindi dataset
                            this project first tried had severe label
                            leakage (sarcasm-marking hashtags baked
                            into the text, topic-clustering artifacts)
                            that made the trained model meaningless —
                            see project notes. No trustworthy
                            Devanagari-script model exists in this
                            project as a result. Devanagari input is
                            flagged, not silently misrouted.

Language routing between English and Hinglish is a hand-built
heuristic (word-list overlap), NOT a proper language-ID model — a
real language-ID library was out of scope for a mini project. This
is disclosed here and should be disclosed in the report: it's a
simplification, not a claim of robust language detection.

Hate/offensive/neither is ENGLISH ONLY (models/hate_bert_v2/, BERT
fine-tuned on Davidson + Implicit Hate Corpus). No Hindi/Hinglish hate
speech model exists in this project.
"""
import os
import pickle
import re

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from preprocess import clean_text
from preprocess_hinglish import clean_hinglish_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
HATE_BERT_DIR = os.path.join(MODEL_DIR, "hate_bert_v2")

# Devanagari Unicode block: U+0900–U+097F
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Marker words strongly indicative of Hinglish and NOT common English
# words (deliberately excludes ambiguous overlaps like "the", "to",
# "or", "hi" which are valid English words too).
HINGLISH_MARKERS = {
    "hai", "hain", "kya", "nahi", "nhi", "kyun", "kyu", "yaar", "yr",
    "krte", "krta", "krna", "bhi", "toh", "aur", "mein", "wala",
    "wali", "bhai", "bahut", "acha", "accha", "theek", "matlab",
    "chahiye", "karo", "karta", "karti", "hoga", "hogi", "tha", "thi",
    "koi", "kuch", "sab", "log", "ghar", "paisa", "kaam", "chutiya",
    "yee", "pah", "hona", "aisa", "aaisa", "tumhare", "sath", "uski",
    "iske", "hota", "hoti", "isko", "aaj", "kal", "abhi", "waha",
    "yaha", "kahan", "kaise", "itna", "itni", "sala", "yr",
}


def _detect_language(text: str) -> str:
    """
    Returns 'devanagari', 'hinglish', or 'en'.

    Heuristic, not a trained language-ID model: Devanagari script is
    detected by Unicode range (reliable). English vs Hinglish (both
    Latin script) is decided by what fraction of words match a
    hand-picked Hinglish marker list — a real language-ID library
    would be more robust; this is a scoped-down mini-project choice,
    documented as such.
    """
    if _DEVANAGARI_RE.search(text):
        return "devanagari"

    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return "en"

    hinglish_hits = sum(1 for w in words if w in HINGLISH_MARKERS)
    # Threshold: at least 1 marker word AND at least 12% of words
    # match. The absolute-count floor matters for short sentences
    # (e.g. a 6-word sentence needs 20% = 1.2 words under a pure-ratio
    # rule, which rounds against short inputs) — one strong marker
    # like "hai" or "nahi" in a short sentence is real signal.
    if hinglish_hits >= 1 and hinglish_hits / len(words) >= 0.12:
        return "hinglish"
    return "en"


class SarcasmHatePipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hate/offensive: English only
        self.hate_tokenizer = DistilBertTokenizerFast.from_pretrained(HATE_BERT_DIR)
        self.hate_model = DistilBertForSequenceClassification.from_pretrained(HATE_BERT_DIR)
        self.hate_model.to(self.device)
        self.hate_model.eval()
        with open(os.path.join(HATE_BERT_DIR, "label_encoder.pkl"), "rb") as f:
            self.hate_label_encoder = pickle.load(f)

        # English sarcasm
        with open(os.path.join(MODEL_DIR, "sarcasm_vectorizer.pkl"), "rb") as f:
            self.en_sarcasm_vectorizer = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "sarcasm_classifier.pkl"), "rb") as f:
            self.en_sarcasm_clf = pickle.load(f)

        # Hinglish sarcasm
        with open(os.path.join(MODEL_DIR, "hinglish_sarcasm_vectorizer.pkl"), "rb") as f:
            self.hinglish_sarcasm_vectorizer = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "hinglish_sarcasm_classifier.pkl"), "rb") as f:
            self.hinglish_sarcasm_clf = pickle.load(f)

    def _predict_hate(self, text: str):
        inputs = self.hate_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            logits = self.hate_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_id = probs.argmax()
        label = self.hate_label_encoder.inverse_transform([pred_id])[0]
        proba = {
            cls: round(float(probs[i]), 3)
            for i, cls in enumerate(self.hate_label_encoder.classes_)
        }
        return label, proba

    def predict(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty.")

        language = _detect_language(text)

        if language == "devanagari":
            return {
                "text": text,
                "language": language,
                "bucket": "unsupported",
                "hate_label": None,
                "hate_class_probabilities": None,
                "is_hate": None,
                "is_offensive": None,
                "is_sarcastic": None,
                "sarcasm_confidence_score": None,
                "note": (
                    "Devanagari-script input detected. No trustworthy "
                    "sarcasm model exists for this script in this "
                    "project — see README Known Limitations. Try "
                    "romanized (Hinglish) input instead."
                ),
            }

        if language == "hinglish":
            cleaned = clean_hinglish_text(text)
            vec = self.hinglish_sarcasm_vectorizer.transform([cleaned])
            clf = self.hinglish_sarcasm_clf
        else:
            cleaned = clean_text(text)
            vec = self.en_sarcasm_vectorizer.transform([cleaned])
            clf = self.en_sarcasm_clf

        sarcasm_score = float(clf.decision_function(vec)[0])
        is_sarcastic = sarcasm_score >= 0

        if language == "hinglish":
            # No Hindi/Hinglish hate speech model in this project.
            bucket = "sarcastic" if is_sarcastic else "neutral"
            return {
                "text": text,
                "language": language,
                "bucket": bucket,
                "hate_label": None,
                "hate_class_probabilities": None,
                "is_hate": None,
                "is_offensive": None,
                "is_sarcastic": is_sarcastic,
                "sarcasm_confidence_score": round(sarcasm_score, 3),
                "note": "Hate speech detection is English-only in this project.",
            }

        hate_label, hate_proba = self._predict_hate(text)
        is_hate = hate_label == "hate"
        is_offensive = hate_label == "offensive"

        if is_hate and is_sarcastic:
            bucket = "hate_sarcastic"
        elif is_hate:
            bucket = "hate"
        elif is_offensive and is_sarcastic:
            bucket = "offensive_sarcastic"
        elif is_offensive:
            bucket = "offensive"
        elif is_sarcastic:
            bucket = "sarcastic"
        else:
            bucket = "neutral"

        return {
            "text": text,
            "language": language,
            "bucket": bucket,
            "hate_label": hate_label,
            "hate_class_probabilities": hate_proba,
            "is_hate": is_hate,
            "is_offensive": is_offensive,
            "is_sarcastic": is_sarcastic,
            "sarcasm_confidence_score": round(sarcasm_score, 3),
        }


if __name__ == "__main__":
    pipeline = SarcasmHatePipeline()
    test_sentences = [
        "Oh great, another genius decision from you.",
        "The weather is nice today.",
        "kya chutiya pah h yee media",
        "aachcha hua tumhare sath aaisa hi hona chahie",
        "यह वाक्य बहुत अच्छा है",
    ]
    for s in test_sentences:
        result = pipeline.predict(s)
        print(f"\nText: {s}")
        print(f"  Language: {result['language']}")
        print(f"  Bucket: {result['bucket']}")
        if result.get('is_sarcastic') is not None:
            print(f"  Sarcastic: {result['is_sarcastic']} (score={result['sarcasm_confidence_score']})")
        if result.get('note'):
            print(f"  Note: {result['note']}")