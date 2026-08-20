"""
V4 Hate Speech / Offensive Language Classifier
------------------------------------------------
Model:
    DeBERTa-v3-base

Dataset:
    data/v4/train.csv
    data/v4/validation.csv
    data/v4/test.csv

Classes:
    hate
    offensive
    neither

Purpose:
    Improve the separation between hate speech and offensive language
    compared with the previous DistilBERT model.

Hardware:
    Optimized for an NVIDIA RTX 4090 (24 GB VRAM).
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.join(
    os.path.dirname(__file__),
    ".."
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "v4"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "hate_deberta_v4"
)

BASE_MODEL = "microsoft/deberta-v3-base"

MAX_LENGTH = 128

NUM_EPOCHS = 3

TRAIN_BATCH_SIZE = 16

EVAL_BATCH_SIZE = 32

LEARNING_RATE = 2e-5


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 70)
print("DEVICE")
print("=" * 70)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )

print()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING V4 DATASET")
print("=" * 70)

train_df = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv")
)

val_df = pd.read_csv(
    os.path.join(DATA_DIR, "validation.csv")
)

test_df = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv")
)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

print("\nTrain distribution:")
print(train_df["label"].value_counts())

print()


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

# Fixed order for reproducibility
class_names = [
    "hate",
    "neither",
    "offensive",
]

label_to_id = {
    label: i
    for i, label in enumerate(class_names)
}

id_to_label = {
    i: label
    for label, i in label_to_id.items()
}

for df in [train_df, val_df, test_df]:
    df["label_id"] = df["label"].map(label_to_id)

print("=" * 70)
print("LABEL MAPPING")
print("=" * 70)

print(label_to_id)

print()


# ============================================================
# TEXT DATA
# ============================================================

X_train = train_df["text"].astype(str).tolist()
y_train = train_df["label_id"].astype(int).tolist()

X_val = val_df["text"].astype(str).tolist()
y_val = val_df["label_id"].astype(int).tolist()

X_test = test_df["text"].astype(str).tolist()
y_test = test_df["label_id"].astype(int).tolist()


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=np.array(y_train),
)

class_weights = torch.tensor(
    weights,
    dtype=torch.float
)

for i, weight in enumerate(class_weights):
    print(
        f"{id_to_label[i]:12} -> {weight:.4f}"
    )

print()


# ============================================================
# TOKENIZER
# ============================================================

print("=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

print("Tokenizer loaded.")
print()


# ============================================================
# DATASET CLASS
# ============================================================

class HateDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length=128
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length
        )

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item


train_dataset = HateDataset(
    X_train,
    y_train,
    tokenizer,
    MAX_LENGTH
)

val_dataset = HateDataset(
    X_val,
    y_val,
    tokenizer,
    MAX_LENGTH
)

test_dataset = HateDataset(
    X_test,
    y_test,
    tokenizer,
    MAX_LENGTH
)

print("Train dataset:", len(train_dataset))
print("Validation dataset:", len(val_dataset))
print("Test dataset:", len(test_dataset))

print()


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING DeBERTa")
print("=" * 70)

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=3,
    id2label=id_to_label,
    label2id=label_to_id,
)

print("DeBERTa loaded.")
print()


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),
        "f1_macro": f1_score(
            labels,
            predictions,
            average="macro"
        ),
        "f1_hate": f1_score(
            labels,
            predictions,
            labels=[0],
            average="macro"
        ),
    }


# ============================================================
# WEIGHTED TRAINER
# ============================================================

class WeightedTrainer(Trainer):

    def __init__(
        self,
        class_weights=None,
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):

        labels = inputs.pop("labels")

        outputs = model(
            **inputs
        )

        logits = outputs.logits

        loss_function = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(
                logits.device
            )
        )

        loss = loss_function(
            logits,
            labels
        )

        if return_outputs:
            return loss, outputs

        return loss


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir=MODEL_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    weight_decay=0.01,

    warmup_ratio=0.1,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1_macro",

    greater_is_better=True,

    logging_steps=100,

    save_total_limit=2,

    report_to="none",

    fp16=torch.cuda.is_available(),

    dataloader_num_workers=4,
)


# ============================================================
# TRAINER
# ============================================================

trainer = WeightedTrainer(
    class_weights=class_weights,

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    compute_metrics=compute_metrics,
)


# ============================================================
# TRAIN
# ============================================================

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

trainer.train()

print()
print("TRAINING COMPLETE")
print()


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

prediction_output = trainer.predict(
    test_dataset
)

test_predictions = np.argmax(
    prediction_output.predictions,
    axis=-1
)

print()

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=class_names,
        digits=3
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)

print()

print(
    "Test Accuracy:",
    f"{accuracy_score(y_test, test_predictions):.4f}"
)

print(
    "Test Macro F1:",
    f"{f1_score(y_test, test_predictions, average='macro'):.4f}"
)

print(
    "Test Hate F1:",
    f"{f1_score(y_test, test_predictions, labels=[0], average='macro'):.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("=" * 70)
print("SAVING MODEL")
print("=" * 70)

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

# Save class mapping
with open(
    os.path.join(
        MODEL_DIR,
        "label_mapping.pkl"
    ),
    "wb"
) as f:
    pickle.dump(
        label_to_id,
        f
    )

print()
print("Model saved to:")
print(MODEL_DIR)