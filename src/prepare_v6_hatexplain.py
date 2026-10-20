import json
import os
from collections import Counter

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.join(
    os.path.dirname(__file__),
    ".."
)

HATEXPLAIN_DATA_DIR = os.path.join(
    BASE_DIR,
    "HateXplain",
    "Data"
)

DATASET_PATH = os.path.join(
    HATEXPLAIN_DATA_DIR,
    "dataset.json"
)

DIVISIONS_PATH = os.path.join(
    HATEXPLAIN_DATA_DIR,
    "post_id_divisions.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "v6"
)


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_MAP = {
    "hatespeech": "hate",
    "offensive": "offensive",
    "normal": "neither"
}


# ============================================================
# MAJORITY VOTE
# ============================================================

def majority_label(annotators):
    labels = []

    for annotation in annotators:
        label = annotation.get("label")

        if label is not None:
            labels.append(
                str(label).lower().strip()
            )

    if not labels:
        return None

    counts = Counter(labels)

    return counts.most_common(1)[0][0]


# ============================================================
# BUILD ONE SPLIT
# ============================================================

def build_split(data, post_ids):
    rows = []

    for post_id in post_ids:

        if post_id not in data:
            continue

        record = data[post_id]

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        tokens = record.get(
            "post_tokens",
            []
        )

        text = " ".join(
            str(token)
            for token in tokens
        ).strip()

        if not text:
            continue

        # ----------------------------------------------------
        # Majority annotation
        # ----------------------------------------------------

        annotators = record.get(
            "annotators",
            []
        )

        voted_label = majority_label(
            annotators
        )

        if voted_label is None:
            continue

        final_label = LABEL_MAP.get(
            voted_label
        )

        if final_label is None:
            continue

        rows.append({
            "text": text,
            "label": final_label,
            "source": "hatexplain"
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING V6 — HATEXPLAIN")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    print(
        "Total posts:",
        len(data)
    )

    # --------------------------------------------------------
    # Load official splits
    # --------------------------------------------------------

    with open(
        DIVISIONS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        divisions = json.load(f)

    # --------------------------------------------------------
    # Build official splits
    # --------------------------------------------------------

    train_df = build_split(
        data,
        divisions["train"]
    )

    val_df = build_split(
        data,
        divisions["val"]
    )

    test_df = build_split(
        data,
        divisions["test"]
    )

    # --------------------------------------------------------
    # Remove duplicates INSIDE each split
    # --------------------------------------------------------

    train_df = (
        train_df
        .drop_duplicates(subset=["text"])
        .reset_index(drop=True)
    )

    val_df = (
        val_df
        .drop_duplicates(subset=["text"])
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .drop_duplicates(subset=["text"])
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Cross-split leakage check BEFORE cleanup
    # --------------------------------------------------------

    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    train_val_overlap = train_texts & val_texts
    train_test_overlap = train_texts & test_texts
    val_test_overlap = val_texts & test_texts

    print("\n" + "=" * 70)
    print("LEAKAGE BEFORE CLEANUP")
    print("=" * 70)

    print(
        "Train/Validation overlap:",
        len(train_val_overlap)
    )

    print(
        "Train/Test overlap:",
        len(train_test_overlap)
    )

    print(
        "Validation/Test overlap:",
        len(val_test_overlap)
    )

    # --------------------------------------------------------
    # Remove validation texts that occur in train
    # --------------------------------------------------------

    val_df = val_df[
        ~val_df["text"].isin(train_texts)
    ].copy()

    # --------------------------------------------------------
    # Remove test texts that occur in train
    # --------------------------------------------------------

    test_df = test_df[
        ~test_df["text"].isin(train_texts)
    ].copy()

    # --------------------------------------------------------
    # Remove test texts that occur in validation
    # --------------------------------------------------------

    val_texts = set(val_df["text"])

    test_df = test_df[
        ~test_df["text"].isin(val_texts)
    ].copy()

    # Reset indexes
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # --------------------------------------------------------
    # Final leakage check
    # --------------------------------------------------------

    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    print("\n" + "=" * 70)
    print("LEAKAGE AFTER CLEANUP")
    print("=" * 70)

    print(
        "Train/Validation overlap:",
        len(train_texts & val_texts)
    )

    print(
        "Train/Test overlap:",
        len(train_texts & test_texts)
    )

    print(
        "Validation/Test overlap:",
        len(val_texts & test_texts)
    )

    # --------------------------------------------------------
    # Final distributions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TRAIN DISTRIBUTION")
    print("=" * 70)

    print(
        "Rows:",
        len(train_df)
    )

    print(
        train_df["label"].value_counts()
    )

    print("\n" + "=" * 70)
    print("FINAL VALIDATION DISTRIBUTION")
    print("=" * 70)

    print(
        "Rows:",
        len(val_df)
    )

    print(
        val_df["label"].value_counts()
    )

    print("\n" + "=" * 70)
    print("FINAL TEST DISTRIBUTION")
    print("=" * 70)

    print(
        "Rows:",
        len(test_df)
    )

    print(
        test_df["label"].value_counts()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    train_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "train.csv"
        ),
        index=False
    )

    val_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation.csv"
        ),
        index=False
    )

    test_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test.csv"
        ),
        index=False
    )

    print("\n" + "=" * 70)
    print("V6 DATASET SAVED")
    print("=" * 70)

    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )


if __name__ == "__main__":
    main()