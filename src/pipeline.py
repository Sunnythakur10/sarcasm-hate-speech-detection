"""
Chains classifiers into one pipeline. Input: raw text -> Output.

Sarcasm detection routes by detected language/script:
    English            -> TF-IDF + LinearSVC on Reddit SARC data
    Hinglish (romanized
    Hindi, Latin script) -> TF-IDF (char n-grams) + LinearSVC on
                            human-labeled Hinglish tweet data
    Devanagari script   -> Experimental translation-based sarcasm and
                            hate/offensive classification: Hindi is
                            translated to English with MyMemory and then
                            passed to the existing English classifiers.
                            This is not a Hindi-trained sarcasm model.

Language routing between English and Hinglish is a hand-built
heuristic (word-list overlap), NOT a proper language-ID model — a
real language-ID library was out of scope for a mini project. This
is disclosed here and should be disclosed in the report: it's a
simplification, not a claim of robust language detection.

The hate/offensive/neither and English sarcasm classifiers are English
models. For Devanagari Hindi, text is translated through MyMemory and the
translation is passed to both English classifiers. This is experimental
translation-based support, not Hindi-trained classification. Romanized
Hinglish continues to use the dedicated Hinglish sarcasm model; Hinglish
hate classification is not enabled.
"""
import html
import os
import pickle
import re

import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import clean_text
from preprocess_hinglish import clean_hinglish_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
HATE_MODEL_DIR = os.path.join(MODEL_DIR, "hate_roberta_pretrained")
HATE_MODEL_ID = os.environ.get("HATE_MODEL_ID", "AuricErgeson/hate-speech-detector")

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


def _split_for_translation(text: str, max_bytes: int = 450) -> list[str]:
    """Split longer text into safe UTF-8-sized MyMemory segments."""
    text = text.strip()
    if not text:
        return []
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    # Prefer sentence boundaries, then word boundaries. Keep punctuation
    # attached to its sentence and never cut through a Unicode character.
    sentences = re.split(r"(?<=[.!?।])\s+|\n+", text)
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence.encode("utf-8")) <= max_bytes:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate.encode("utf-8")) <= max_bytes:
                current = candidate
            else:
                flush()
                current = sentence
            continue

        # This sentence alone is too long; split it on whitespace.
        flush()
        for word in sentence.split():
            candidate = f"{current} {word}".strip() if current else word
            if len(candidate.encode("utf-8")) <= max_bytes:
                current = candidate
                continue

            flush()
            if len(word.encode("utf-8")) <= max_bytes:
                current = word
                continue

            # Extremely long single token: split on character boundaries.
            piece = ""
            for char in word:
                if len((piece + char).encode("utf-8")) > max_bytes:
                    if piece:
                        chunks.append(piece)
                    piece = char
                else:
                    piece += char
            current = piece

    flush()
    return chunks


def _translate_hindi_to_english(text: str) -> str:
    """Translate Devanagari Hindi to English with the MyMemory API.

    MyMemory allows up to 500 UTF-8 bytes per query. Long inputs are
    split into smaller text segments. The API's free quota can run out;
    failures raise a clear error instead of classifying untranslated text.
    """
    segments = _split_for_translation(text)
    if not segments:
        raise ValueError("Hindi text cannot be empty.")

    contact_email = os.environ.get("MYMEMORY_EMAIL", "").strip()
    translated_segments: list[str] = []

    for segment in segments:
        params = {
            "q": segment,
            "langpair": "hi|en",
        }
        # Optional: providing a valid contact email increases the published
        # daily quota. Do not set this unless the user chooses to provide it.
        if contact_email:
            params["de"] = contact_email

        try:
            response = requests.get(
                "https://api.mymemory.translated.net/get",
                params=params,
                timeout=12,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Hindi translation is temporarily unavailable. "
                "Check your internet connection or try again later."
            ) from exc
        except ValueError as exc:
            # Covers a non-JSON response from the remote service.
            raise RuntimeError(
                "The translation service returned an unreadable response."
            ) from exc

        if str(payload.get("responseStatus")) != "200":
            detail = payload.get("responseDetails") or "Translation request failed."
            raise RuntimeError(f"Hindi translation failed: {detail}")

        translated = html.unescape(
            str(payload.get("responseData", {}).get("translatedText", ""))
        ).strip()
        if not translated:
            raise RuntimeError("The translation service returned empty text.")
        translated_segments.append(translated)

    return " ".join(translated_segments).strip()


class SarcasmHatePipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pretrained RoBERTa hate/offensive/neither classifier (English only).
        # Auto* classes support the RoBERTa architecture used by this model.
        # Prefer local files for offline/local development. If the model
        # folder is absent or incomplete, fetch it from Hugging Face.
        local_config = os.path.join(HATE_MODEL_DIR, "config.json")
        hate_model_source = HATE_MODEL_DIR if os.path.isfile(local_config) else HATE_MODEL_ID
        print(f"[Pipeline] Loading hate model from: {hate_model_source}")
        self.hate_tokenizer = AutoTokenizer.from_pretrained(hate_model_source)
        self.hate_model = AutoModelForSequenceClassification.from_pretrained(hate_model_source)
        self.hate_model.to(self.device)
        self.hate_model.eval()

        # Translate the model's label names into the stable labels consumed
        # by the rest of this pipeline and the existing Flask frontend.
        self.hate_id2label = self.hate_model.config.id2label
        self.hate_label_by_id = {
            int(label_id): self._canonical_hate_label(label_name)
            for label_id, label_name in self.hate_id2label.items()
        }
        expected = {"hate", "offensive", "neither"}
        found = set(self.hate_label_by_id.values())
        if found != expected:
            raise ValueError(
                "Unexpected labels in the pretrained hate model. "
                f"Expected {sorted(expected)}, found {sorted(found)}. "
                f"Model id2label: {self.hate_id2label}"
            )

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

    @staticmethod
    def _canonical_hate_label(label_name: str) -> str:
        """Map pretrained-model labels to the app's stable class names."""
        normalized = str(label_name).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"hate_speech", "hate", "hateful", "hate_speech_content"}:
            return "hate"
        if normalized in {"offensive", "offensive_language", "abusive"}:
            return "offensive"
        if normalized in {"neither", "non_hate", "not_hate", "non_hateful", "normal"}:
            return "neither"
        raise ValueError(f"Unrecognized label in hate model config: {label_name!r}")

    def _predict_hate(self, text: str):
        inputs = self.hate_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            logits = self.hate_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_id = int(probs.argmax())
        label = self.hate_label_by_id[pred_id]
        # Keep a consistent three-class probability dictionary for the API,
        # regardless of the pretrained model's internal class ordering.
        proba = {"hate": 0.0, "offensive": 0.0, "neither": 0.0}
        for class_id, probability in enumerate(probs):
            canonical_label = self.hate_label_by_id[int(class_id)]
            proba[canonical_label] = round(float(probability), 3)
        return label, proba

    def predict(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty.")

        language = _detect_language(text)

        if language == "devanagari":
            # Experimental translation-based Hindi route:
            # 1) Translate Devanagari Hindi to English.
            # 2) Run the existing English sarcasm and hate classifiers on the
            #    translated text. This does NOT make either classifier Hindi-trained.
            # Translation can lose sarcasm, idioms, slang, or social context.
            translated_text = _translate_hindi_to_english(text)

            # Sarcasm: existing English TF-IDF + LinearSVC model applied to
            # the translated English sentence. LinearSVC score is not a probability.
            cleaned_translation = clean_text(translated_text)
            sarcasm_vec = self.en_sarcasm_vectorizer.transform([cleaned_translation])
            sarcasm_score = float(
                self.en_sarcasm_clf.decision_function(sarcasm_vec)[0]
            )
            is_sarcastic = sarcasm_score >= 0

            # Hate/offensive: existing English three-class classifier on the
            # same translation.
            hate_label, hate_proba = self._predict_hate(translated_text)
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
                "translated_text": translated_text,
                "note": (
                    "Experimental Hindi analysis: the original Hindi was "
                    "translated into English, then the existing English sarcasm "
                    "and hate/offensive classifiers were applied to that translation. "
                    f"English translation: {translated_text} "
                    "Translation may lose sarcasm, idioms, slang, or context, so "
                    "these predictions are not equivalent to a Hindi-trained model."
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