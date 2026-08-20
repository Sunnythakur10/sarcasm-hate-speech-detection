import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from torch.utils.data import Dataset


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "train-balanced-sarcasm.csv",
)

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "sarcasm_roberta",
)

BASE_MODEL = "roberta-base"

# 100k balanced samples
SAMPLE_SIZE = 100000

MAX_LENGTH = 128


# ============================================================
# DATASET
# ============================================================

class SarcasmDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro"
        ),
        "sarcastic_f1": f1_score(
            labels,
            predictions,
            pos_label=1
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ROBERTA SARCASM CLASSIFIER")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nDevice:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    df = pd.read_csv(
        DATA_PATH,
        usecols=["label", "comment"],
        on_bad_lines="skip",
    )

    df = df.dropna(
        subset=["label", "comment"]
    )

    df = df.rename(
        columns={"comment": "text"}
    )

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["text"].str.len() > 0
    ]

    # ========================================================
    # BALANCED 100K SAMPLE
    # ========================================================

    if (
        SAMPLE_SIZE is not None
        and len(df) > SAMPLE_SIZE
    ):

        per_class = SAMPLE_SIZE // 2

        df = (
            df.groupby(
                "label",
                group_keys=False
            )
            .sample(
                n=per_class,
                random_state=42
            )
            .reset_index(drop=True)
        )

    print("\nDataset size:", len(df))

    print("\nClass distribution:")
    print(df["label"].value_counts())

    # ========================================================
    # LABELS
    # ========================================================

    df["label"] = df["label"].astype(int)

    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # ========================================================
    # TOKENIZER
    # ========================================================

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("Loading RoBERTa...")

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={
            0: "not_sarcastic",
            1: "sarcastic",
        },
        label2id={
            "not_sarcastic": 0,
            "sarcastic": 1,
        },
    )

    # ========================================================
    # DATASET OBJECTS
    # ========================================================

    train_dataset = SarcasmDataset(
        X_train,
        y_train,
        tokenizer,
    )

    test_dataset = SarcasmDataset(
        X_test,
        y_test,
        tokenizer,
    )

    # ========================================================
    # TRAINING ARGUMENTS
    # ========================================================

    training_args = TrainingArguments(

        output_dir=os.path.join(
            MODEL_DIR,
            "checkpoints"
        ),

        num_train_epochs=3,

        # RTX 4090 has 24GB VRAM
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,

        learning_rate=2e-5,

        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="macro_f1",

        greater_is_better=True,

        logging_steps=100,

        report_to="none",

        fp16=True,

        dataloader_num_workers=4,

        save_total_limit=2,
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print("\n" + "=" * 70)
    print("STARTING ROBERTA TRAINING")
    print("=" * 70)

    trainer.train()

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    result = trainer.predict(
        test_dataset
    )

    y_pred = np.argmax(
        result.predictions,
        axis=-1
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    sarcastic_f1 = f1_score(
        y_test,
        y_pred,
        pos_label=1
    )

    print(
        f"\nAccuracy     : {accuracy:.4f}"
    )

    print(
        f"Macro F1     : {macro_f1:.4f}"
    )

    print(
        f"Sarcastic F1 : {sarcastic_f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "not_sarcastic",
                "sarcastic",
            ],
            digits=4,
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    trainer.save_model(
        MODEL_DIR
    )

    tokenizer.save_pretrained(
        MODEL_DIR
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 70)

    print("\nLocation:")
    print(MODEL_DIR)


if __name__ == "__main__":
    main()



