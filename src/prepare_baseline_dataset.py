import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.join(
    os.path.dirname(__file__),
    ".."
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "hate_speech_raw.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "baseline"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("BUILDING CLEAN DAVIDSON BASELINE")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print("Original rows:", len(df))
print("Columns:", df.columns.tolist())


# ============================================================
# LABEL MAPPING
# ============================================================

# Davidson:
# 0 = hate
# 1 = offensive
# 2 = neither

label_map = {
    0: "hate",
    1: "offensive",
    2: "neither"
}

df = df.rename(
    columns={
        "tweet": "text"
    }
)

df["label"] = df["class"].map(label_map)

df = df.dropna(
    subset=["text", "label"]
)

df["text"] = (
    df["text"]
    .astype(str)
    .str.strip()
)

df = df[
    df["text"].str.len() > 0
].copy()

# Remove exact duplicate text before splitting
df = (
    df
    .drop_duplicates(
        subset=["text"]
    )
    .reset_index(drop=True)
)

print("\nAfter cleanup:", len(df))

print("\nClass distribution:")
print(
    df["label"].value_counts()
)


# ============================================================
# TRAIN / TEMP
# 80% train, 20% temporary
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)


# ============================================================
# VALIDATION / TEST
# 10% validation, 10% test
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)


# Reset indexes
train_df = train_df.reset_index(
    drop=True
)

val_df = val_df.reset_index(
    drop=True
)

test_df = test_df.reset_index(
    drop=True
)


# ============================================================
# LEAKAGE CHECK
# ============================================================

train_texts = set(
    train_df["text"]
)

val_texts = set(
    val_df["text"]
)

test_texts = set(
    test_df["text"]
)

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print(
    "Train/Validation:",
    len(train_texts & val_texts)
)

print(
    "Train/Test:",
    len(train_texts & test_texts)
)

print(
    "Validation/Test:",
    len(val_texts & test_texts)
)


# ============================================================
# REPORT
# ============================================================

for name, split in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("Rows:", len(split))

    print(
        split["label"].value_counts()
    )

    print(
        (
            split["label"]
            .value_counts(
                normalize=True
            )
            * 100
        ).round(2)
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

train_df[
    ["text", "label"]
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "train.csv"
    ),
    index=False
)

val_df[
    ["text", "label"]
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "validation.csv"
    ),
    index=False
)

test_df[
    ["text", "label"]
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test.csv"
    ),
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)