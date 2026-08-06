"""
Loads and does light structural cleanup on the two raw datasets.

No labeling/annotation here — both datasets already come pre-labeled.
"""

import json
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_hate_speech_df() -> pd.DataFrame:
    """
    Davidson et al. dataset.

    Expected columns in raw CSV:
      - tweet
      - class

    class mapping:
      0 = hate speech
      1 = offensive
      2 = neither

    Returns:
      columns: text, label
    """
    path = os.path.join(DATA_DIR, "hate_speech_raw.csv")
    df = pd.read_csv(path)
    label_map = {0: "hate", 1: "offensive", 2: "neither"}

    df = df.rename(columns={"tweet": "text", "class": "class_id"})
    df["label"] = df["class_id"].map(label_map)

    return df[["text", "label"]].dropna().reset_index(drop=True)


def load_sarcasm_df() -> pd.DataFrame:
    """
    Misra News Headlines dataset (JSON lines).

    Expected fields per JSON line:
      - headline
      - is_sarcastic

    Returns:
      columns: text, label
    """
    path = os.path.join(DATA_DIR, "sarcasm_raw.json")
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    df = df.rename(columns={"headline": "text", "is_sarcastic": "label"})

    return df[["text", "label"]].dropna().reset_index(drop=True)


def load_reddit_sarcasm_df(sample_size: int = 100000) -> pd.DataFrame:
    """
    SARC (Self-Annotated Reddit Corpus), Kaggle CSV version.

    File expected:
      data/train-balanced-sarcasm.csv

    Important columns in this CSV:
      - label
      - comment
      - author
      - subreddit
      - score
      - ups
      - downs
      - date
      - created_utc
      - parent_comment

    Returns:
      columns: text, label
    """
    path = os.path.join(DATA_DIR, "train-balanced-sarcasm.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download train-balanced-sarcasm.csv from Kaggle "
            "and place it in the data/ folder."
        )

    df = pd.read_csv(path, low_memory=False)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Validate required columns
    required = {"label", "comment"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Keep only what we need
    df = df[["label", "comment"]].copy()

    # Clean data
    df = df.dropna(subset=["label", "comment"])
    df["comment"] = df["comment"].astype(str).str.strip()
    df = df[df["comment"].str.len() > 0]

    # Make label numeric and clean invalid values
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Optional balanced sampling
    if sample_size is not None and len(df) > sample_size:
        half = sample_size // 2

        pos = df[df["label"] == 1]
        neg = df[df["label"] == 0]

        pos_n = min(half, len(pos))
        neg_n = min(half, len(neg))

        pos = pos.sample(n=pos_n, random_state=42)
        neg = neg.sample(n=neg_n, random_state=42)

        df = pd.concat([pos, neg], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df.rename(columns={"comment": "text"})[["text", "label"]]


if __name__ == "__main__":
    hate_df = load_hate_speech_df()
    sarcasm_df = load_sarcasm_df()

    print("Hate speech dataset:", hate_df.shape)
    print(hate_df["label"].value_counts())

    print("\nSarcasm dataset:", sarcasm_df.shape)
    print(sarcasm_df["label"].value_counts())

    # Uncomment this if you want to test Reddit sarcasm loading too
    # reddit_df = load_reddit_sarcasm_df(sample_size=10000)
    # print("\nReddit sarcasm dataset:", reddit_df.shape)
    # print(reddit_df["label"].value_counts())