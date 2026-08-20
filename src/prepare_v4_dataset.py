import os
import pandas as pd


DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "v4"
)


def load_davidson():
    path = os.path.join(
        DATA_DIR,
        "hate_speech_raw.csv"
    )

    df = pd.read_csv(path)

    # Davidson:
    # 0 = hate
    # 1 = offensive
    # 2 = neither

    df = df.rename(
        columns={
            "tweet": "text"
        }
    )

    df["label"] = df["class"].map({
        0: "hate",
        1: "offensive",
        2: "neither"
    })

    return df[["text", "label"]].dropna()


def load_implicit_hate():
    path = os.path.join(
        DATA_DIR,
        "implicit_hate_v1_stg1_posts.tsv"
    )

    df = pd.read_csv(
        path,
        sep="\t"
    )

    print("\nImplicit Hate columns:")
    print(df.columns.tolist())

    print("\nImplicit Hate sample:")
    print(df.head())

    # Try to detect the text/label columns.
    text_candidates = [
        "text",
        "tweet",
        "post",
        "content"
    ]

    label_candidates = [
        "label",
        "class",
        "stage",
        "hate_type"
    ]

    text_col = next(
        (c for c in text_candidates if c in df.columns),
        None
    )

    label_col = next(
        (c for c in label_candidates if c in df.columns),
        None
    )

    if text_col is None:
        raise ValueError(
            f"Could not find text column. "
            f"Columns: {df.columns.tolist()}"
        )

    if label_col is None:
        raise ValueError(
            f"Could not find label column. "
            f"Columns: {df.columns.tolist()}"
        )

    print(
        f"\nUsing text column: {text_col}"
    )

    print(
        f"Using label column: {label_col}"
    )

    df = df.rename(
        columns={
            text_col: "text",
            label_col: "source_label"
        }
    )

    # Implicit Hate mapping
    #
    # implicit_hate -> hate
    # explicit_hate -> hate
    # not_hate      -> neither

    df["label"] = df["source_label"].astype(str).str.lower().map({
        "implicit_hate": "hate",
        "explicit_hate": "hate",
        "not_hate": "neither"
    })

    df = df.dropna(
        subset=["text", "label"]
    )

    return df[["text", "label"]]


def load_olid():
    path = os.path.join(
        DATA_DIR,
        "olid",
        "olid-training-v1.0.tsv"
    )

    df = pd.read_csv(
        path,
        sep="\t"
    )

    print("\nOLID columns:")
    print(df.columns.tolist())

    # OLID Level A:
    #
    # OFF = offensive
    # NOT = neither

    df = df.rename(
        columns={
            "tweet": "text",
            "subtask_a": "source_label"
        }
    )

    df["label"] = (
        df["source_label"]
        .astype(str)
        .str.upper()
        .map({
            "OFF": "offensive",
            "NOT": "neither"
        })
    )

    df = df.dropna(
        subset=["text", "label"]
    )

    return df[["text", "label"]]


def normalize_text(series):
    """
    Light normalization ONLY for deduplication.

    We do not modify the actual text used for training.
    """
    return (
        series
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
    )


def main():

    print("=" * 70)
    print("BUILDING V4 COMBINED DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    davidson = load_davidson()

    print(
        f"\nDavidson rows: {len(davidson)}"
    )

    print(
        davidson["label"].value_counts()
    )

    implicit = load_implicit_hate()

    print(
        f"\nImplicit Hate rows: {len(implicit)}"
    )

    print(
        implicit["label"].value_counts()
    )

    olid = load_olid()

    print(
        f"\nOLID rows: {len(olid)}"
    )

    print(
        olid["label"].value_counts()
    )

    # --------------------------------------------------------
    # Add source information
    # --------------------------------------------------------

    davidson["source"] = "davidson"
    implicit["source"] = "implicit_hate"
    olid["source"] = "olid"

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = pd.concat(
        [
            davidson,
            implicit,
            olid
        ],
        ignore_index=True
    )

    print(
        f"\nCombined before deduplication: "
        f"{len(combined)}"
    )

    # --------------------------------------------------------
    # Normalize text for duplicate checking
    # --------------------------------------------------------

    combined["_norm_text"] = normalize_text(
        combined["text"]
    )

    # Remove empty text
    combined = combined[
        combined["_norm_text"].str.len() > 0
    ]

    # --------------------------------------------------------
    # Detect conflicting duplicate labels
    # --------------------------------------------------------

    label_counts = (
        combined
        .groupby("_norm_text")["label"]
        .nunique()
    )

    conflicting_texts = set(
        label_counts[
            label_counts > 1
        ].index
    )

    print(
        f"\nConflicting duplicate texts: "
        f"{len(conflicting_texts)}"
    )

    if conflicting_texts:

        print(
            "\nRemoving conflicting duplicates "
            "to prevent label noise..."
        )

        combined = combined[
            ~combined["_norm_text"].isin(
                conflicting_texts
            )
        ]

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    before = len(combined)

    combined = combined.drop_duplicates(
        subset=["_norm_text"],
        keep="first"
    )

    removed = before - len(combined)

    print(
        f"Exact duplicates removed: {removed}"
    )

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    final_df = combined[
        [
            "text",
            "label",
            "source"
        ]
    ].reset_index(drop=True)

    print(
        f"\nFinal V4 rows: {len(final_df)}"
    )

    print(
        "\nFINAL CLASS DISTRIBUTION:"
    )

    print(
        final_df["label"].value_counts()
    )

    print(
        "\nFINAL CLASS PERCENTAGES:"
    )

    print(
        (
            final_df["label"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    print(
        "\nSOURCE DISTRIBUTION:"
    )

    print(
        final_df["source"].value_counts()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "hate_offensive_v4.csv"
    )

    final_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(output_path)


if __name__ == "__main__":
    main()