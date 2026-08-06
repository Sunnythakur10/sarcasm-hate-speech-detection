"""
Simple CLI demo. Run: python3 demo.py
Type a sentence, get the pipeline's classification. Type 'quit' to exit.
"""
from pipeline import SarcasmHatePipeline

BUCKET_LABELS = {
    "hateful_sarcastic": "Hateful + Sarcastic",
    "hateful": "Hateful / Offensive",
    "sarcastic": "Sarcastic (non-hateful)",
    "neutral": "Neutral",
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
        print(f"  Bucket        : {BUCKET_LABELS[result['bucket']]}")
        print(f"  Hate label    : {result['hate_label']} {result['hate_class_probabilities']}")
        print(f"  Sarcastic     : {result['is_sarcastic']} (confidence score {result['sarcasm_confidence_score']})")
        print()


if __name__ == "__main__":
    main()
