"""
Fine-tunes DistilBERT for the hate speech classifier — replaces the
TF-IDF + Logistic Regression baseline for this specific task.

Why this exists: the linear baseline plateaus at F1=0.44 on the "hate"
class no matter how features/thresholds are tuned (see project notes) —
that ceiling comes from the model only seeing surface word patterns.
A transformer can pick up contextual/semantic signal instead.

CPU WARNING: without a GPU this will be slow — realistically 1-3+ hours
for the full dataset on CPU. Strongly recommend running this on Google
Colab (free GPU: Runtime > Change runtime type > T4 GPU), where it takes
roughly 10-15 minutes. Set SUBSET_SIZE below to None for the full run,
or a small number to sanity-check the script first.
"""
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from load_data import load_combined_hate_df

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "hate_bert")
BASE_MODEL = "distilbert-base-uncased"

# Set to a small int (e.g. 500) to sanity-check the pipeline quickly.
# Set to None to train on the full dataset.
SUBSET_SIZE = None  # set to a small int to sanity-check quickly, None for full run


class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True, max_length=max_length
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score, accuracy_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    df = load_combined_hate_df()
    # NOTE: raw text here, NOT the cleaned/lemmatized text used for TF-IDF.
    # BERT's subword tokenizer handles raw text better than pre-stripped
    # text — stripping stopwords/punctuation throws away context BERT
    # would otherwise use.
    df = df.dropna(subset=["text", "label"])

    if SUBSET_SIZE is not None and len(df) > SUBSET_SIZE:
        per_class = SUBSET_SIZE // df["label"].nunique()
        df = (
            df.groupby("label", group_keys=False)[df.columns.tolist()]
            .apply(lambda g: g.sample(n=min(len(g), per_class), random_state=42))
            .reset_index(drop=True)
        )
        print(f"[subset mode] training on {len(df)} rows for a quick sanity check")

    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["label"])
    print("Label mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["label_id"].tolist(),
        test_size=0.2, random_state=42, stratify=df["label_id"]
    )

    print("Loading tokenizer + base model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(label_encoder.classes_)
    )

    train_dataset = HateSpeechDataset(X_train, y_train, tokenizer)
    test_dataset = HateSpeechDataset(X_test, y_test, tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="none",  # no wandb/etc — keeps this self-contained
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("\nFinal evaluation:")
    preds_output = trainer.predict(test_dataset)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    print(classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, digits=3
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    import pickle
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"\nSaved fine-tuned model to {MODEL_DIR}")


if __name__ == "__main__":
    main()