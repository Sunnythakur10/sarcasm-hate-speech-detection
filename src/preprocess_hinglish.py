"""
Preprocessing for the Hinglish (romanized Hindi, Latin script)
sarcasm dataset. Separate from BOTH preprocess.py (English/NLTK) and
preprocess_hindi.py (Devanagari script) — this text looks like
"kya chutiya pah h yee media", which is Hindi vocabulary written in
Latin letters, not English and not Devanagari. Neither of the other
two preprocessors is built for this.

No lemmatization/stemming here — no readily available lightweight
Hinglish lemmatizer, and transliteration spelling is inconsistent
enough (multiple ways to romanize the same Hindi word) that stemming
rules built for English or Devanagari won't transfer.

Also strips emoji-description tokens (e.g. "face_with_tears_of_joy")
that appear as literal text in this dataset's `comment` column,
since they're metadata artifacts, not the user's chosen words —
similar leakage risk to the sarcasm-hashtag issue found in the
Devanagari dataset, worth checking for even though these look like
generic emoji names rather than sarcasm-specific markers.
"""
import re

# Short hand-picked list of extremely common Hinglish function words.
# Not exhaustive — a proper Hinglish stopword corpus would be a
# stretch-goal improvement over this baseline list.
HINGLISH_STOPWORDS = {
    "hai", "hain", "ho", "ka", "ki", "ke", "ko", "se", "me", "mein",
    "par", "or", "aur", "ye", "yeh", "wo", "voh", "to", "toh", "hi",
    "bhi", "nahi", "nhi", "na", "kya", "kyun", "kyu", "jo", "yaar",
    "yr", "the", "tha", "thi", "hoti", "hota", "kar", "krte", "krta",
}

_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_SYMBOL_RE = re.compile(r"#")
# Emoji-description tokens in this dataset look like
# "face_with_tears_of_joy" — underscore-joined lowercase words.
_EMOJI_DESC_RE = re.compile(r"\b[a-z]+(?:_[a-z]+){1,}\b")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_hinglish_text(text: str) -> str:
    """Normalize one piece of Hinglish (romanized Hindi) text."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_SYMBOL_RE.sub(" ", text)
    text = _EMOJI_DESC_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    tokens = text.split()
    tokens = [tok for tok in tokens if tok not in HINGLISH_STOPWORDS and len(tok) > 1]
    return " ".join(tokens)


def clean_hinglish_series(series):
    """Vectorized helper for a pandas Series of raw Hinglish text."""
    return series.astype(str).apply(clean_hinglish_text)