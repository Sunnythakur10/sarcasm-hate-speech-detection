"""
Shared text preprocessing for both the hate-speech and sarcasm classifiers.
Kept intentionally simple and identical across both datasets so the two
models see text in a consistent format before their TF-IDF vectorizers run.
"""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data quietly (no-op if already present)
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_STOPWORDS = set(stopwords.words("english"))
# Negation words are informative for sarcasm/hate signal — never strip these.
_NEGATIONS = {"not", "no", "never", "nor", "none", "cannot", "n't"}
_STOPWORDS -= _NEGATIONS

_LEMMATIZER = WordNetLemmatizer()

_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_SYMBOL_RE = re.compile(r"#")
_NON_ALPHA_RE = re.compile(r"[^a-z\s']")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Normalize one piece of text: lowercase, strip URLs/mentions/punctuation
    noise, keep negation contractions, remove stopwords, lemmatize.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_SYMBOL_RE.sub(" ", text)  # drop '#' but keep the word
    text = text.translate(str.maketrans("", "", string.punctuation.replace("'", "")))
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    tokens = text.split()
    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in _STOPWORDS and len(tok) > 1
    ]
    return " ".join(tokens)


def clean_series(series):
    """Vectorized helper for a pandas Series of raw text."""
    return series.astype(str).apply(clean_text)
