"""
Combined Sarcasm + Hate Speech Detection Pipeline

Models:
    1. Sarcasm:
       TF-IDF + LinearSVC trained on Reddit SARC data.

    2. Hate / Offensive:
       Fine-tuned DistilBERT trained on the Davidson dataset.

Final categories:
    - hate
    - hate + sarcastic
    - offensive
    - offensive + sarcastic
    - sarcastic
    - neutral

Important:
    The two classifiers are trained independently and then chained.
"""

import os
import pickle

import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

from preprocess import clean_text


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models"
)

HATE_BERT_DIR = os.path.join(
    MODEL_DIR,
    "hate_bert"
)

# Sarcasm threshold.
#
# Original LinearSVC prediction effectively uses:
#     score >= 0 -> sarcastic
#
# We use a slightly stricter threshold to reduce false positives.
# This should ideally be tuned later using a validation set.
SARCASM_THRESHOLD = 0.4


class SarcasmHatePipeline:

    def __init__(self):

        # ====================================================
        # DEVICE
        # ====================================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Using device: {self.device}")

        # ====================================================
        # HATE / OFFENSIVE MODEL
        # ====================================================

        print("Loading DistilBERT hate-speech model...")

        self.hate_tokenizer = (
            DistilBertTokenizerFast.from_pretrained(
                HATE_BERT_DIR
            )
        )

        self.hate_model = (
            DistilBertForSequenceClassification.from_pretrained(
                HATE_BERT_DIR
            )
        )

        self.hate_model.to(self.device)
        self.hate_model.eval()

        # ----------------------------------------------------
        # Load label mapping
        # ----------------------------------------------------

        with open(
            os.path.join(
                HATE_BERT_DIR,
                "label_encoder.pkl"
            ),
            "rb"
        ) as f:

            self.hate_label_encoder = pickle.load(f)

        # ====================================================
        # SARCASM MODEL
        # ====================================================

        print("Loading sarcasm model...")

        with open(
            os.path.join(
                MODEL_DIR,
                "sarcasm_vectorizer.pkl"
            ),
            "rb"
        ) as f:

            self.sarcasm_vectorizer = pickle.load(f)

        with open(
            os.path.join(
                MODEL_DIR,
                "sarcasm_classifier.pkl"
            ),
            "rb"
        ) as f:

            self.sarcasm_clf = pickle.load(f)

        print("All models loaded.")
        print()

    # ========================================================
    # HATE / OFFENSIVE PREDICTION
    # ========================================================

    def _predict_hate(self, text: str):

        # IMPORTANT:
        # Use RAW text for DistilBERT.
        # Do not use the TF-IDF preprocessing here.

        inputs = self.hate_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.hate_model(
                **inputs
            )

            logits = outputs.logits

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

        probabilities = (
            probabilities
            .cpu()
            .numpy()[0]
        )

        # Highest-probability class
        predicted_id = int(
            probabilities.argmax()
        )

        # Convert numeric class back to text label
        label = self.hate_label_encoder.inverse_transform(
            [predicted_id]
        )[0]

        # Create readable probability dictionary
        probability_dict = {}

        for index, class_name in enumerate(
            self.hate_label_encoder.classes_
        ):
            probability_dict[class_name] = round(
                float(probabilities[index]),
                3
            )

        return label, probability_dict

    # ========================================================
    # MAIN PREDICTION
    # ========================================================

    def predict(self, text: str) -> dict:

        text = text.strip()

        if not text:
            raise ValueError(
                "Input text cannot be empty."
            )

        # ====================================================
        # HATE / OFFENSIVE
        # ====================================================

        hate_label, hate_probabilities = (
            self._predict_hate(text)
        )

        # ====================================================
        # SARCASM
        # ====================================================

        # TF-IDF model uses the original preprocessing pipeline.
        cleaned_text = clean_text(text)

        sarcasm_vector = (
            self.sarcasm_vectorizer.transform(
                [cleaned_text]
            )
        )

        # LinearSVC decision score.
        sarcasm_score = float(
            self.sarcasm_clf.decision_function(
                sarcasm_vector
            )[0]
        )

        # Use our explicit threshold instead of
        # sklearn's default zero threshold.
        is_sarcastic = (
            sarcasm_score >= SARCASM_THRESHOLD
        )

        # ====================================================
        # HATE / OFFENSIVE FLAGS
        # ====================================================

        is_hate = (
            hate_label == "hate"
        )

        is_offensive = (
            hate_label == "offensive"
        )

        # ====================================================
        # FINAL BUCKET
        # ====================================================

        if is_hate and is_sarcastic:

            bucket = "hate + sarcastic"

        elif is_hate:

            bucket = "hate"

        elif is_offensive and is_sarcastic:

            bucket = "offensive + sarcastic"

        elif is_offensive:

            bucket = "offensive"

        elif is_sarcastic:

            bucket = "sarcastic"

        else:

            bucket = "neutral"

        # ====================================================
        # RETURN
        # ====================================================

        return {
            "text": text,
            "bucket": bucket,

            "hate_label": hate_label,

            "hate_class_probabilities":
                hate_probabilities,

            "is_hate":
                is_hate,

            "is_offensive":
                is_offensive,

            "is_sarcastic":
                is_sarcastic,

            "sarcasm_confidence_score":
                round(
                    sarcasm_score,
                    3
                ),

            "sarcasm_threshold":
                SARCASM_THRESHOLD,
        }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    pipeline = SarcasmHatePipeline()

    test_sentences = [

        # Normal
        "I really enjoyed the movie.",

        # Obvious sarcasm
        "Yeah, because that's exactly what everyone wanted.",

        # Hate / offensive
        "You people are worthless and should disappear.",

        # Offensive
        "You are an idiot.",

        # Sarcasm + negative/offensive
        "Oh great, another genius decision from you.",

        # Strong sarcasm
        "Wow, what a brilliant idea, said no one ever about your plan.",
    ]

    print("=" * 65)

    for text in test_sentences:

        result = pipeline.predict(text)

        print()
        print(f"Text: {result['text']}")
        print(f"Bucket: {result['bucket']}")

        print(
            "Hate label:",
            result["hate_label"]
        )

        print(
            "Hate probabilities:",
            result["hate_class_probabilities"]
        )

        print(
            "Sarcastic:",
            result["is_sarcastic"]
        )

        print(
            "Sarcasm score:",
            result["sarcasm_confidence_score"]
        )

        print(
            "Threshold:",
            result["sarcasm_threshold"]
        )

        print("-" * 65)