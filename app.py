"""
Flask Application for Sarcasm & Hate Speech Neural Detection Pipeline
Serves modern web interface and provides REST API for dual-engine NLP classification.
"""
import os
import sys
import traceback
from flask import Flask, render_template, request, jsonify

# Add src to sys.path so pipeline and preprocess modules resolve seamlessly
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline import SarcasmHatePipeline, _detect_language, HINGLISH_MARKERS

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize pipeline lazily or at startup
print("[App] Initializing Sarcasm & Hate Speech Pipeline...")
try:
    nlp_pipeline = SarcasmHatePipeline()
    pipeline_ready = True
    init_error = None
    print("[App] Pipeline loaded successfully.")
except Exception as e:
    nlp_pipeline = None
    pipeline_ready = False
    init_error = str(e)
    print(f"[App ERROR] Failed to initialize pipeline: {e}")
    traceback.print_exc()

# Bucket metadata for rich UI presentation
BUCKET_CONFIG = {
    "hate_sarcastic": {
        "title": "Hateful & Sarcastic",
        "badge": "Hate + Sarcasm",
        "category": "High Risk & Sarcastic",
        "description": "Text exhibits both hostile or hateful intent and sarcastic framing (ironic hostility).",
        "color": "#f43f5e",
        "glow": "rgba(244, 63, 94, 0.4)",
        "icon": "flame-slash",
        "severity": "critical"
    },
    "hate": {
        "title": "Hate Speech",
        "badge": "Hate Speech",
        "category": "High Risk",
        "description": "Explicit or implicit hate speech targeting protected groups, identity, or individuals.",
        "color": "#ef4444",
        "glow": "rgba(239, 68, 68, 0.4)",
        "icon": "shield-alert",
        "severity": "high"
    },
    "offensive_sarcastic": {
        "title": "Offensive & Sarcastic",
        "badge": "Offensive + Sarcasm",
        "category": "Medium Risk & Sarcastic",
        "description": "Contains derogatory, vulgar, or offensive phrasing masked with sarcastic tone.",
        "color": "#f97316",
        "glow": "rgba(249, 115, 22, 0.4)",
        "icon": "message-square-warning",
        "severity": "medium"
    },
    "offensive": {
        "title": "Offensive Language",
        "badge": "Offensive",
        "category": "Medium Risk",
        "description": "Profanity, vulgarity, or harsh language without hate speech criteria.",
        "color": "#f59e0b",
        "glow": "rgba(245, 158, 11, 0.4)",
        "icon": "alert-triangle",
        "severity": "medium"
    },
    "sarcastic": {
        "title": "Sarcastic (Non-Hateful)",
        "badge": "Sarcastic",
        "category": "Benign Nuance",
        "description": "Irony, witty mocking, or figurative sarcasm detected with no hate or toxicity.",
        "color": "#a855f7",
        "glow": "rgba(168, 85, 247, 0.4)",
        "icon": "sparkles",
        "severity": "low"
    },
    "neutral": {
        "title": "Neutral / Clean",
        "badge": "Neutral",
        "category": "Safe",
        "description": "Standard declarative statement with neither sarcasm, vulgarity, nor hostility.",
        "color": "#10b981",
        "glow": "rgba(16, 185, 129, 0.4)",
        "icon": "check-circle",
        "severity": "safe"
    },
    "unsupported": {
        "title": "Unsupported Script",
        "badge": "Unsupported",
        "category": "Out of Scope",
        "description": "Devanagari script detected. Excluded due to verified dataset label leakage.",
        "color": "#64748b",
        "glow": "rgba(100, 116, 139, 0.3)",
        "icon": "help-circle",
        "severity": "info"
    }
}

PRESET_SAMPLES = [
    {
        "id": "sample-1",
        "label": "Sarcastic Complaint",
        "lang": "English",
        "tag": "Sarcastic",
        "text": "Oh fantastic, another delayed train right when I have the most important meeting of the year."
    },
    {
        "id": "sample-2",
        "label": "Hostile Hate Speech",
        "lang": "English",
        "tag": "Hate",
        "text": "These people are disgusting parasites and should all be kicked out of the country."
    },
    {
        "id": "sample-3",
        "label": "Sarcastic Toxicity",
        "lang": "English",
        "tag": "Hate + Sarcasm",
        "text": "Oh wow, what a great contribution to humanity you subhuman trash are making."
    },
    {
        "id": "sample-4",
        "label": "Offensive Slang",
        "lang": "English",
        "tag": "Offensive",
        "text": "Shut your damn mouth and stop talking absolute bullshit already."
    },
    {
        "id": "sample-5",
        "label": "Hinglish Sarcasm",
        "lang": "Hinglish",
        "tag": "Sarcastic",
        "text": "kya chutiya pah h yee media, har cheez me inko breaking news chahiye"
    },
    {
        "id": "sample-6",
        "label": "Hinglish Everyday",
        "lang": "Hinglish",
        "tag": "Hinglish",
        "text": "bhai kal subah aana sath me kaam khatam kar lenge"
    },
    {
        "id": "sample-7",
        "label": "Neutral Fact",
        "lang": "English",
        "tag": "Neutral",
        "text": "The international climate conference concluded on Wednesday with a joint statement."
    },
    {
        "id": "sample-8",
        "label": "Devanagari (Flagged)",
        "lang": "Devanagari",
        "tag": "Unsupported",
        "text": "यह नीति वास्तव में देश के लिए बहुत ही अद्भुत और लाभकारी साबित होगी।"
    }
]


def extract_matched_markers(text: str):
    """Find which Hinglish marker words appear in the text."""
    import re
    words = re.findall(r"[a-zA-Z]+", text.lower())
    found = [w for w in set(words) if w in HINGLISH_MARKERS]
    return sorted(found)


@app.route("/")
def index():
    return render_template("index.html", ready=pipeline_ready, error=init_error)


@app.route("/api/health", methods=["GET"])
def health():
    import torch
    return jsonify({
        "status": "ready" if pipeline_ready else "error",
        "error": init_error,
        "device": "CUDA (GPU)" if torch.cuda.is_available() else "CPU",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models": {
            "hate_model": "DistilBERT (fine-tuned on Davidson + Implicit Hate)",
            "english_sarcasm": "TF-IDF + LinearSVC (Reddit SARC 100k)",
            "hinglish_sarcasm": "Char n-gram TF-IDF + LinearSVC (Human-annotated tweets)",
            "devanagari_sarcasm": "Unsupported (Documented dataset label leakage)"
        }
    })


@app.route("/api/samples", methods=["GET"])
def get_samples():
    return jsonify({"samples": PRESET_SAMPLES})


@app.route("/api/classify", methods=["POST"])
def classify():
    if not pipeline_ready or nlp_pipeline is None:
        return jsonify({
            "error": "NLP Pipeline is not ready",
            "details": init_error
        }), 503

    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text cannot be empty."}), 400

    try:
        raw_result = nlp_pipeline.predict(text)
        bucket_key = raw_result.get("bucket", "neutral")
        config = BUCKET_CONFIG.get(bucket_key, BUCKET_CONFIG["neutral"])

        # Detect markers for diagnostics
        matched_markers = []
        if raw_result.get("language") == "hinglish":
            matched_markers = extract_matched_markers(text)

        # Enrich response for frontend visualization
        enriched = {
            "text": text,
            "language": raw_result.get("language"),
            "language_display": {
                "en": "English (Latin)",
                "hinglish": "Hinglish (Romanized Hindi)",
                "devanagari": "Hindi (Devanagari Script)"
            }.get(raw_result.get("language"), raw_result.get("language")),
            "bucket": bucket_key,
            "bucket_info": config,
            "hate": {
                "evaluated": raw_result.get("hate_label") is not None,
                "label": raw_result.get("hate_label"),
                "is_hate": raw_result.get("is_hate", False),
                "is_offensive": raw_result.get("is_offensive", False),
                "probabilities": raw_result.get("hate_class_probabilities") or {},
                "model_name": "DistilBERT Contextual Transformer"
            },
            "sarcasm": {
                "evaluated": raw_result.get("is_sarcastic") is not None,
                "is_sarcastic": raw_result.get("is_sarcastic"),
                "score": raw_result.get("sarcasm_confidence_score"),
                "threshold": 0.0,
                "model_name": (
                    "Hinglish Char-n-gram LinearSVC"
                    if raw_result.get("language") == "hinglish"
                    else "Reddit SARC LinearSVC"
                )
            },
            "matched_markers": matched_markers,
            "note": raw_result.get("note")
        }

        return jsonify(enriched)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch", methods=["POST"])
def batch_classify():
    if not pipeline_ready or nlp_pipeline is None:
        return jsonify({"error": "NLP Pipeline is not ready"}), 503

    payload = request.get_json(silent=True) or {}
    items = payload.get("texts", [])

    if not isinstance(items, list) or not items:
        return jsonify({"error": "Expected a non-empty array of texts"}), 400

    results = []
    for item in items[:25]:  # Safety cap at 25
        t = str(item).strip()
        if not t:
            continue
        try:
            raw = nlp_pipeline.predict(t)
            b = raw.get("bucket", "neutral")
            cfg = BUCKET_CONFIG.get(b, BUCKET_CONFIG["neutral"])
            results.append({
                "text": t,
                "language": raw.get("language"),
                "bucket": b,
                "bucket_info": cfg,
                "hate_label": raw.get("hate_label"),
                "is_sarcastic": raw.get("is_sarcastic"),
                "sarcasm_score": raw.get("sarcasm_confidence_score")
            })
        except Exception as e:
            results.append({
                "text": t,
                "error": str(e),
                "bucket": "error"
            })

    return jsonify({"results": results, "count": len(results)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"  Sarcasm & Hate Speech Neural UI Server")
    print(f"  Local URL: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
