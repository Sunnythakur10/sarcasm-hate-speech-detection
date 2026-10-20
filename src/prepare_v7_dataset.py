import json
import os
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.join(
    os.path.dirname(__file__),
    ".."
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

DAVIDSON_PATH = os.path.join(
    DATA_DIR,
    "hate_speech_raw.csv"
)

HATEXPLAIN_DIR = os.path.join(
    BASE_DIR,
    "HateXplain",
    "Data"
)

HATEXPLAIN_DATASET = os.path.join(
    HATEXPLAIN_DIR,
    "dataset.json"
)

HATEXPLAIN_SPLITS = os.path.join(
    HATEXPLAIN_DIR,
    "post_id_divisions.json"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "v7"
)


# ============================================================
# LABEL MAPPINGS
# ============================================================

DAVIDSON_LABEL_MAP = {
    0: "hate",
    1: "offensive",
    2: "neither",
}

HATEXPLAIN_LABEL_MAP = {
    "hatespeech": "hate",
    "offensive": "offensive",
    "normal": "neither",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Used ONLY for duplicate detection.

    Training text itself is left unchanged.
    """
    return (
        str(text)
        .lower()
        .strip()
        .replace("\n", " ")
        .replace("\r", " ")
    )


# ============================================================
# LOAD DAVIDSON
# ============================================================

def load_davidson():

    print("=" * 70)
    print("LOADING DAVIDSON")
    print("=" * 70)

    df = pd.read_csv(
        DAVIDSON_PATH
    )

    df = df.rename(
        columns={
            "tweet": "text"
        }
    )

    df["label"] = (
        df["class"]
        .map(DAVIDSON_LABEL_MAP)
    )

    df = df.dropna(
        subset=[
            "text",
            "label"
        ]
    )

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["text"].str.len() > 0
    ]

    df["source"] = "davidson"

    result = df[
        [
            "text",
            "label",
            "source"
        ]
    ].copy()

    print(
        "Rows:",
        len(result)
    )

    print(
        result["label"].value_counts()
    )

    return result


# ============================================================
# HATEXPLAIN MAJORITY LABEL
# ============================================================

def majority_label(annotators):

    labels = []

    for annotation in annotators:

        label = annotation.get(
            "label"
        )

        if label is not None:

            labels.append(
                str(label)
                .lower()
                .strip()
            )

    if not labels:
        return None

    counts = Counter(
        labels
    )

    return counts.most_common(1)[0][0]


# ============================================================
# LOAD HATEXPLAIN SPLIT
# ============================================================

def build_hatexplain_split(
    data,
    post_ids,
    split_name
):

    rows = []

    for post_id in post_ids:

        if post_id not in data:
            continue

        record = data[post_id]

        # ----------------------------------------------
        # Reconstruct text
        # ----------------------------------------------

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

        # ----------------------------------------------
        # Majority annotation
        # ----------------------------------------------

        annotators = record.get(
            "annotators",
            []
        )

        voted_label = majority_label(
            annotators
        )

        if voted_label is None:
            continue

        final_label = (
            HATEXPLAIN_LABEL_MAP.get(
                voted_label
            )
        )

        if final_label is None:
            continue

        rows.append(
            {
                "text": text,
                "label": final_label,
                "source": "hatexplain",
            }
        )

    df = pd.DataFrame(
        rows
    )

    # Remove duplicates inside split
    df = (
        df
        .drop_duplicates(
            subset=["text"]
        )
        .reset_index(drop=True)
    )

    print(
        f"\nHateXplain {split_name}:",
        len(df)
    )

    print(
        df["label"].value_counts()
    )

    return df


# ============================================================
# CROSS-SPLIT LEAKAGE CLEANUP
# ============================================================

def remove_cross_split_duplicates(
    train_df,
    val_df,
    test_df
):

    # Normalized text only for duplicate detection
    train_df["_norm"] = (
        train_df["text"]
        .map(normalize_text)
    )

    val_df["_norm"] = (
        val_df["text"]
        .map(normalize_text)
    )

    test_df["_norm"] = (
        test_df["text"]
        .map(normalize_text)
    )

    train_norm = set(
        train_df["_norm"]
    )

    val_norm = set(
        val_df["_norm"]
    )

    test_norm = set(
        test_df["_norm"]
    )

    print("\n" + "=" * 70)
    print("CROSS-SPLIT DUPLICATES BEFORE CLEANUP")
    print("=" * 70)

    print(
        "Train / Validation:",
        len(train_norm & val_norm)
    )

    print(
        "Train / Test:",
        len(train_norm & test_norm)
    )

    print(
        "Validation / Test:",
        len(val_norm & test_norm)
    )

    # --------------------------------------------------------
    # Train has priority.
    # Remove validation/test examples that occur in train.
    # --------------------------------------------------------

    val_df = val_df[
        ~val_df["_norm"].isin(train_norm)
    ].copy()

    test_df = test_df[
        ~test_df["_norm"].isin(train_norm)
    ].copy()

    # Recalculate validation after cleanup
    val_norm = set(
        val_df["_norm"]
    )

    # Remove remaining validation/test overlap
    test_df = test_df[
        ~test_df["_norm"].isin(val_norm)
    ].copy()

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    train_norm = set(
        train_df["_norm"]
    )

    val_norm = set(
        val_df["_norm"]
    )

    test_norm = set(
        test_df["_norm"]
    )

    print("\n" + "=" * 70)
    print("CROSS-SPLIT DUPLICATES AFTER CLEANUP")
    print("=" * 70)

    print(
        "Train / Validation:",
        len(train_norm & val_norm)
    )

    print(
        "Train / Test:",
        len(train_norm & test_norm)
    )

    print(
        "Validation / Test:",
        len(val_norm & test_norm)
    )

    # Remove helper column
    train_df = train_df.drop(
        columns=["_norm"]
    )

    val_df = val_df.drop(
        columns=["_norm"]
    )

    test_df = test_df.drop(
        columns=["_norm"]
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING V7 — DAVIDSON + HATEXPLAIN")
    print("=" * 70)

    # --------------------------------------------------------
    # Load Davidson
    # --------------------------------------------------------

    davidson = load_davidson()

    # --------------------------------------------------------
    # Stratified Davidson split
    # --------------------------------------------------------

    davidson_train, davidson_temp = train_test_split(
        davidson,
        test_size=0.20,
        random_state=42,
        stratify=davidson["label"]
    )

    davidson_val, davidson_test = train_test_split(
        davidson_temp,
        test_size=0.50,
        random_state=42,
        stratify=davidson_temp["label"]
    )

    davidson_train = (
        davidson_train
        .reset_index(drop=True)
    )

    davidson_val = (
        davidson_val
        .reset_index(drop=True)
    )

    davidson_test = (
        davidson_test
        .reset_index(drop=True)
    )

    print("\nDavidson splits:")
    print(
        "Train:",
        len(davidson_train)
    )
    print(
        "Validation:",
        len(davidson_val)
    )
    print(
        "Test:",
        len(davidson_test)
    )

    # --------------------------------------------------------
    # Load HateXplain
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING HATEXPLAIN")
    print("=" * 70)

    with open(
        HATEXPLAIN_DATASET,
        "r",
        encoding="utf-8"
    ) as f:

        hatexplain_data = json.load(f)

    with open(
        HATEXPLAIN_SPLITS,
        "r",
        encoding="utf-8"
    ) as f:

        split_ids = json.load(f)

    print(
        "Total HateXplain posts:",
        len(hatexplain_data)
    )

    # --------------------------------------------------------
    # Official HateXplain split
    # --------------------------------------------------------

    hatexplain_train = build_hatexplain_split(
        hatexplain_data,
        split_ids["train"],
        "train"
    )

    hatexplain_val = build_hatexplain_split(
        hatexplain_data,
        split_ids["val"],
        "validation"
    )

    hatexplain_test = build_hatexplain_split(
        hatexplain_data,
        split_ids["test"],
        "test"
    )

    # --------------------------------------------------------
    # Combine matching splits
    # --------------------------------------------------------

    train_df = pd.concat(
        [
            davidson_train,
            hatexplain_train
        ],
        ignore_index=True
    )

    val_df = pd.concat(
        [
            davidson_val,
            hatexplain_val
        ],
        ignore_index=True
    )

    test_df = pd.concat(
        [
            davidson_test,
            hatexplain_test
        ],
        ignore_index=True
    )

    print("\n" + "=" * 70)
    print("COMBINED BEFORE LEAKAGE CLEANUP")
    print("=" * 70)

    print(
        "Train:",
        len(train_df)
    )

    print(
        "Validation:",
        len(val_df)
    )

    print(
        "Test:",
        len(test_df)
    )

    # --------------------------------------------------------
    # Remove duplicates across splits
    # --------------------------------------------------------

    (
        train_df,
        val_df,
        test_df
    ) = remove_cross_split_duplicates(
        train_df,
        val_df,
        test_df
    )

    # --------------------------------------------------------
    # Shuffle each split
    # --------------------------------------------------------

    train_df = (
        train_df
        .sample(
            frac=1.0,
            random_state=42
        )
        .reset_index(drop=True)
    )

    val_df = (
        val_df
        .sample(
            frac=1.0,
            random_state=42
        )
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .sample(
            frac=1.0,
            random_state=42
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    for name, df in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df)
    ]:

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        print(
            "Rows:",
            len(df)
        )

        print(
            df["label"].value_counts()
        )

        print(
            "\nPercentages:"
        )

        print(
            (
                df["label"]
                .value_counts(
                    normalize=True
                )
                * 100
            ).round(2)
        )

        print(
            "\nSources:"
        )

        print(
            df["source"].value_counts()
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

    # Also save the complete combined dataset
    combined_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="validation"),
            test_df.assign(split="test")
        ],
        ignore_index=True
    )

    combined_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "combined.csv"
        ),
        index=False
    )

    print("\n" + "=" * 70)
    print("V7 DATASET SAVED")
    print("=" * 70)

    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )


if __name__ == "__main__":
    main()