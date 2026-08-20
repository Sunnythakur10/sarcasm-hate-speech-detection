import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data"
)

INPUT_PATH = os.path.join(
    DATA_DIR,
    "v4",
    "hate_offensive_v4.csv"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "v4"
)


def split_source(df, source):
    """
    80% train
    10% validation
    10% test

    Stratified by label.
    """

    source_df = df[df["source"] == source].copy()

    train_df, temp_df = train_test_split(
        source_df,
        test_size=0.20,
        random_state=42,
        stratify=source_df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    return train_df, val_df, test_df


def main():

    df = pd.read_csv(INPUT_PATH)

    print("Loaded:", len(df))

    # ----------------------------------------------------
    # Split every source independently
    # ----------------------------------------------------

    train_parts = []
    val_parts = []
    test_parts = []

    for source in df["source"].unique():

        train_df, val_df, test_df = split_source(
            df,
            source
        )

        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)

        print(
            f"\n{source}"
        )

        print(
            "train:",
            len(train_df)
        )

        print(
            "validation:",
            len(val_df)
        )

        print(
            "test:",
            len(test_df)
        )

    train = pd.concat(
        train_parts,
        ignore_index=True
    )

    validation = pd.concat(
        val_parts,
        ignore_index=True
    )

    test = pd.concat(
        test_parts,
        ignore_index=True
    )

    # ----------------------------------------------------
    # Shuffle
    # ----------------------------------------------------

    train = train.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    validation = validation.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    test = test.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # ----------------------------------------------------
    # Print distributions
    # ----------------------------------------------------

    for name, split in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test)
    ]:

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        print("Rows:", len(split))

        print(
            split["label"].value_counts()
        )

        print(
            (
                split["label"]
                .value_counts(normalize=True)
                * 100
            ).round(2)
        )

    # ----------------------------------------------------
    # Save
    # ----------------------------------------------------

    train.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "train.csv"
        ),
        index=False
    )

    validation.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation.csv"
        ),
        index=False
    )

    test.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test.csv"
        ),
        index=False
    )

    print("\nSaved:")
    print("data/v4/train.csv")
    print("data/v4/validation.csv")
    print("data/v4/test.csv")


if __name__ == "__main__":
    main()