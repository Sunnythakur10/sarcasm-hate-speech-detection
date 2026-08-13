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
    Davidson et al. dataset. class: 0=hate speech, 1=offensive, 2=neither.
    Returns columns: text, label ('hate', 'offensive', 'neither').
    """
    path = os.path.join(DATA_DIR, "hate_speech_raw.csv")
    df = pd.read_csv(path)
    label_map = {0: "hate", 1: "offensive", 2: "neither"}
    df = df.rename(columns={"tweet": "text", "class": "class_id"})
    df["label"] = df["class_id"].map(label_map)
    return df[["text", "label"]].dropna()


def load_sarcasm_df() -> pd.DataFrame:
    """
    Misra News Headlines dataset (JSON lines). is_sarcastic: 0/1.
    Returns columns: text, label (0/1).
    NOTE: kept for reference / comparison only — superseded by
    load_reddit_sarcasm_df() as the primary sarcasm dataset, since
    news headlines don't match the conversational register of the
    hate speech dataset. See README "Known Limitation" section.
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
    return df[["text", "label"]].dropna()


def load_reddit_sarcasm_df(sample_size: int = 100000) -> pd.DataFrame:
    """
    SARC (Self-Annotated Reddit Corpus), Khodak et al. — conversational,
    self-labeled sarcasm from Reddit comments. This is the domain-matched
    replacement for the news-headlines dataset: same register (informal,
    conversational, sometimes hostile) as the Davidson hate speech tweets.

    File: train-balanced-sarcasm.csv, downloaded manually from
    https://www.kaggle.com/datasets/sherinclaudia/sarcastic-comments-on-reddit
    (not auto-downloadable — requires a free Kaggle login) and placed at
    data/train-balanced-sarcasm.csv.

    sample_size: the full file is ~1M balanced rows (255MB) — for a mini
    project, training on a random sample keeps train time reasonable
    without materially hurting a linear model's performance. Pass None
    to use the full file.
    """
    path = os.path.join(DATA_DIR, "train-balanced-sarcasm.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download train-balanced-sarcasm.csv from "
            "https://www.kaggle.com/datasets/sherinclaudia/sarcastic-comments-on-reddit "
            "and place it in the data/ folder."
        )

    # comment/parent_comment can contain stray quotes/newlines from Reddit;
    # on_bad_lines='skip' drops any malformed rows rather than crashing.
    df = pd.read_csv(path, usecols=["label", "comment"], on_bad_lines="skip")
    df = df.rename(columns={"comment": "text"})
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip().str.len() > 0]

    if sample_size is not None and len(df) > sample_size:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(n=sample_size // 2, random_state=42)
        )

    return df[["text", "label"]].reset_index(drop=True)


def load_implicit_hate_df() -> pd.DataFrame:
    """
    Latent Hatred / Implicit Hate Corpus (ElSherief et al., 2021).
    File: data/implicit_hate_v1_stg1_posts.tsv (tab-separated).

    Columns: post, class (implicit_hate / explicit_hate / not_hate).

    Mapped to the same 3-class scheme as Davidson for merging:
      implicit_hate + explicit_hate -> "hate"
      not_hate                      -> "neither"

    No "offensive" examples come from this dataset — it only
    distinguishes hate vs. not-hate, not a third profanity-only class.
    Merging only adds more "hate" and "neither" examples to Davidson;
    it doesn't touch the "offensive" class at all.
    """
    path = os.path.join(DATA_DIR, "implicit_hate_v1_stg1_posts.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download implicit-hate-corpus.zip from "
            "https://github.com/SALT-NLP/implicit-hate (requires the linked "
            "survey form) and place implicit_hate_v1_stg1_posts.tsv in data/."
        )

    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"post": "text", "class": "raw_label"})

    label_map = {
        "implicit_hate": "hate",
        "explicit_hate": "hate",
        "not_hate": "neither",
    }
    df["label"] = df["raw_label"].map(label_map)

    return df[["text", "label"]].dropna().reset_index(drop=True)


def load_combined_hate_df() -> pd.DataFrame:
    """
    Davidson (Twitter, slur-based hate/offensive/neither) + Implicit
    Hate Corpus (Twitter, extremist-group implicit/explicit hate vs.
    not-hate), concatenated into one training set.

    Purpose: Davidson alone only teaches the model to recognize hate
    speech that contains slurs. This adds ~8.2k more "hate" examples
    (implicit_hate + explicit_hate) that DON'T rely on slurs, so the
    combined model should generalize beyond lexical hate markers.

    Returns the same columns as load_hate_speech_df(): text, label.
    """
    davidson_df = load_hate_speech_df()
    implicit_df = load_implicit_hate_df()

    combined = pd.concat([davidson_df, implicit_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)

    return combined


if __name__ == "__main__":
    hate_df = load_hate_speech_df()
    sarcasm_df = load_sarcasm_df()
    print("Hate speech dataset:", hate_df.shape)
    print(hate_df["label"].value_counts())
    print("\nSarcasm dataset:", sarcasm_df.shape)
    print(sarcasm_df["label"].value_counts())