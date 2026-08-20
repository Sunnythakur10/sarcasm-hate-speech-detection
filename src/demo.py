"""
Simple CLI demo. Run: python3 demo.py
Type a sentence, get the pipeline's classification. Type 'quit' to exit.
Handles English, Hinglish (romanized Hindi), and Devanagari input.
"""
from pipeline import SarcasmHatePipeline

BUCKET_LABELS = {
    "hate_sarcastic": "Hate + Sarcastic",
    "hate": "Hate",
    "offensive_sarcastic": "Offensive + Sarcastic",
    "offensive": "Offensive",
    "sarcastic": "Sarcastic (non-hateful)",
    "neutral": "Neutral",
    "unsupported": "Unsupported input",
}


def main():
    print("Loading models...")
    pipeline = SarcasmHatePipeline()
    print("Ready. Type a sentence to classify (or 'quit' to exit).\n")

    while True:
        text = input("> ").strip()
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue

        result = pipeline.predict(text)
        bucket_label = BUCKET_LABELS.get(result["bucket"], result["bucket"])

        print(f"  Language      : {result['language']}")
        print(f"  Bucket        : {bucket_label}")

        if result.get("hate_label") is not None:
            print(f"  Hate label    : {result['hate_label']} {result['hate_class_probabilities']}")

        if result.get("is_sarcastic") is not None:
            print(f"  Sarcastic     : {result['is_sarcastic']} (confidence score {result['sarcasm_confidence_score']})")

        if result.get("note"):
            print(f"  Note          : {result['note']}")

        print()


if __name__ == "__main__":
    main()