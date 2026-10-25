"""Fair final comparison of DistilBERT hate-speech models.

Runs three candidates with the SAME Davidson train/validation/test split:
  1. baseline: Davidson training split only
  2. implicit: Davidson train + mapped Implicit Hate examples
  3. hatexplain: Davidson train + HateXplain's prebuilt V7 training-only rows

Validation/test remain Davidson-only and are identical for all candidates.
The test set is evaluated once after selecting the checkpoint by validation Macro F1.

Expected project layout (run from any directory):
  data/baseline/{train,validation,test}.csv
  data/implicit_hate_v1_stg1_posts.tsv
  data/v7/train.csv  (must contain text,label,source columns)

This script writes new outputs under artifacts/final_comparison and does NOT
overwrite the models currently used by app.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import unicodedata
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SPLIT_DIR = DATA_DIR / "baseline"
OUTPUT_DIR = BASE_DIR / "artifacts" / "final_comparison"
BASE_MODEL = "distilbert-base-uncased"
CLASS_NAMES = ["hate", "neither", "offensive"]  # fixed, reproducible order
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for name, i in CLASS_TO_ID.items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility can be slightly slower; supported operations are deterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_text(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value)).lower()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def load_split(name: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run src/prepare_baseline_dataset.py first."
        )
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} must contain text,label; missing {sorted(missing)}")
    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[(df["text"] != "") & df["label"].isin(CLASS_TO_ID)]
    return df.reset_index(drop=True)


def load_implicit_supplement(davidson_norms: set[str]) -> pd.DataFrame:
    path = DATA_DIR / "implicit_hate_v1_stg1_posts.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset: {path}")
    raw = pd.read_csv(path, sep="\t")
    if not {"post", "class"}.issubset(raw.columns):
        raise ValueError(f"Unexpected Implicit Hate columns: {raw.columns.tolist()}")
    mapping = {
        "implicit_hate": "hate",
        "explicit_hate": "hate",
        "not_hate": "neither",
    }
    df = raw.rename(columns={"post": "text", "class": "source_label"})
    df["label"] = df["source_label"].astype(str).str.lower().str.strip().map(mapping)
    df = df.dropna(subset=["text", "label"])[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df["_norm"] = df["text"].map(normalize_text)
    before = len(df)
    df = df[~df["_norm"].isin(davidson_norms)].drop_duplicates("_norm")
    print(f"Implicit Hate supplement: {len(df):,} rows retained; {before-len(df):,} duplicates/overlaps removed.")
    return df[["text", "label"]].reset_index(drop=True)


def load_hatexplain_supplement(davidson_norms: set[str]) -> pd.DataFrame:
    # prepare_v7_dataset.py writes source='hatexplain' for these official train-split rows.
    path = DATA_DIR / "v7" / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/prepare_v7_dataset.py first.")
    raw = pd.read_csv(path)
    required = {"text", "label", "source"}
    if not required.issubset(raw.columns):
        raise ValueError(
            f"{path} must contain {sorted(required)}. Columns found: {raw.columns.tolist()}"
        )
    df = raw[raw["source"].astype(str).str.lower().eq("hatexplain")][["text", "label"]].copy()
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[(df["text"] != "") & df["label"].isin(CLASS_TO_ID)]
    df["_norm"] = df["text"].map(normalize_text)
    before = len(df)
    df = df[~df["_norm"].isin(davidson_norms)].drop_duplicates("_norm")
    print(f"HateXplain train supplement: {len(df):,} rows retained; {before-len(df):,} duplicates/overlaps removed.")
    return df[["text", "label"]].reset_index(drop=True)


class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(values[idx], dtype=torch.long) for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_class_weights(labels: List[int], device: torch.device) -> torch.Tensor:
    """Mild square-root inverse-frequency weights, mean-normalized to 1."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(CLASS_NAMES)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training data is missing at least one class. Class counts: {counts.tolist()}")
    weights = np.sqrt(len(labels) / (len(CLASS_NAMES) * counts))
    weights = weights / weights.mean()
    print("Class counts:", {ID_TO_CLASS[i]: int(counts[i]) for i in range(len(CLASS_NAMES))})
    print("Mild square-root class weights:", {ID_TO_CLASS[i]: round(float(weights[i]), 4) for i in range(len(CLASS_NAMES))})
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model, loader, device, use_amp: bool) -> Tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(**batch)
                logits = output.logits
                loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        preds = torch.argmax(logits, dim=-1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        total_loss += float(loss.item()) * len(labels)
        total_n += len(labels)
    metrics = {
        "loss": total_loss / max(total_n, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "hate_f1": f1_score(y_true, y_pred, labels=[CLASS_TO_ID["hate"]], average="macro", zero_division=0),
    }
    return metrics, np.asarray(y_true), np.asarray(y_pred)


def train_candidate(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    args,
    device: torch.device,
) -> dict:
    print("\n" + "=" * 76)
    print(f"TRAINING CANDIDATE: {name}")
    print("=" * 76)
    seed_everything(args.seed)

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    for split in (train_df, val_df, test_df):
        split["label_id"] = split["label"].map(CLASS_TO_ID)
        if split["label_id"].isna().any():
            raise ValueError("Found unmapped labels.")

    train_ds = TextClassificationDataset(train_df["text"].tolist(), train_df["label_id"].astype(int).tolist(), tokenizer, args.max_length)
    val_ds = TextClassificationDataset(val_df["text"].tolist(), val_df["label_id"].astype(int).tolist(), tokenizer, args.max_length)
    test_ds = TextClassificationDataset(test_df["text"].tolist(), test_df["label_id"].astype(int).tolist(), tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(CLASS_NAMES),
        label2id=CLASS_TO_ID,
        id2label=ID_TO_CLASS,
    ).to(device)
    class_weights = make_class_weights(train_df["label_id"].astype(int).tolist(), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_macro_f1 = -1.0
    best_epoch = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**batch).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), labels, weight=class_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.item()) * len(labels)
            seen += len(labels)

        val_metrics, _, _ = evaluate(model, val_loader, device, use_amp)
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_hate_f1": val_metrics["hate_f1"],
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={epoch_record['train_loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_hate_f1={val_metrics['hate_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            model.save_pretrained(out_dir / "best_checkpoint", safe_serialization=True)
            tokenizer.save_pretrained(out_dir / "best_checkpoint")
            encoder = LabelEncoder().fit(CLASS_NAMES)
            with open(out_dir / "best_checkpoint" / "label_encoder.pkl", "wb") as f:
                pickle.dump(encoder, f)
            with open(out_dir / "best_checkpoint" / "training_config.json", "w", encoding="utf-8") as f:
                json.dump({
                    "candidate": name,
                    "base_model": BASE_MODEL,
                    "epochs_requested": args.epochs,
                    "best_epoch_by_validation_macro_f1": best_epoch,
                    "seed": args.seed,
                    "max_length": args.max_length,
                    "train_batch_size": args.train_batch_size,
                    "eval_batch_size": args.eval_batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "warmup_ratio": args.warmup_ratio,
                    "class_weighting": "sqrt inverse frequency; normalized to mean 1",
                    "label_order": CLASS_NAMES,
                }, f, indent=2)

    # Reload the best validation-selected checkpoint before touching the test set.
    model = AutoModelForSequenceClassification.from_pretrained(out_dir / "best_checkpoint").to(device)
    test_metrics, y_true, y_pred = evaluate(model, test_loader, device, use_amp)
    report = classification_report(
        y_true, y_pred,
        labels=[CLASS_TO_ID[name] for name in CLASS_NAMES],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[CLASS_TO_ID[name] for name in CLASS_NAMES])
    result = {
        "candidate": name,
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "best_epoch": int(best_epoch),
        "best_validation_macro_f1": float(best_val_macro_f1),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_hate_f1": float(test_metrics["hate_f1"]),
        "classification_report": report,
        "confusion_matrix_labels_order": CLASS_NAMES,
        "confusion_matrix": matrix.tolist(),
        "epoch_history": history,
        "model_directory": str((out_dir / "best_checkpoint").resolve()),
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(history).to_csv(out_dir / "epoch_history.csv", index=False)
    print("\nTEST RESULTS (common Davidson holdout; evaluated after model selection)")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, digits=4, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted; order=hate, neither, offensive):")
    print(matrix)
    print("Saved candidate outputs to:", out_dir)

    del model, optimizer, scheduler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["baseline", "implicit", "hatexplain"], default=["baseline", "implicit", "hatexplain"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision even on CUDA")
    args = parser.parse_args()

    for dependency_path in [
        SPLIT_DIR / "train.csv", SPLIT_DIR / "validation.csv", SPLIT_DIR / "test.csv"
    ]:
        if not dependency_path.exists():
            raise FileNotFoundError(f"Missing {dependency_path}; prepare the Davidson baseline splits first.")

    train_df = load_split("train")
    val_df = load_split("validation")
    test_df = load_split("test")

    # Use identical Davidson validation/test examples for each model.
    davidson_norms = set()
    for frame in (train_df, val_df, test_df):
        davidson_norms.update(frame["text"].map(normalize_text).tolist())

    val_norm = set(val_df["text"].map(normalize_text))
    test_norm = set(test_df["text"].map(normalize_text))
    train_norm = set(train_df["text"].map(normalize_text))
    if train_norm & val_norm or train_norm & test_norm or val_norm & test_norm:
        raise ValueError("Davidson baseline split has normalized text overlap. Re-run preparation and inspect duplicates.")

    print("Common Davidson splits:")
    print(f"  train={len(train_df):,}, validation={len(val_df):,}, test={len(test_df):,}")
    print("  train label counts:", train_df["label"].value_counts().to_dict())
    print("  validation label counts:", val_df["label"].value_counts().to_dict())
    print("  test label counts:", test_df["label"].value_counts().to_dict())
    print("\nDevice:", "CUDA / " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    if not torch.cuda.is_available():
        print("WARNING: CPU training for three DistilBERT candidates may be very slow. Prefer a Colab GPU.")

    supplements: Dict[str, pd.DataFrame] = {}
    if "implicit" in args.models:
        supplements["implicit"] = load_implicit_supplement(davidson_norms)
    if "hatexplain" in args.models:
        supplements["hatexplain"] = load_hatexplain_supplement(davidson_norms)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    results = []
    for candidate in args.models:
        candidate_train = train_df.copy()
        if candidate in supplements:
            candidate_train = pd.concat([candidate_train, supplements[candidate]], ignore_index=True)
        # Preserve the Davidson label if an overlapping supplementary row is encountered.
        candidate_train["_norm"] = candidate_train["text"].map(normalize_text)
        candidate_train = candidate_train.drop_duplicates("_norm", keep="first").drop(columns="_norm").reset_index(drop=True)
        results.append(train_candidate(candidate, candidate_train, val_df, test_df, tokenizer, args, torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{k: r[k] for k in ["candidate", "training_rows", "validation_rows", "test_rows", "best_epoch", "best_validation_macro_f1", "test_accuracy", "test_macro_f1", "test_hate_f1", "model_directory"]} for r in results])
    summary_path = OUTPUT_DIR / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    with open(OUTPUT_DIR / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 76)
    print("FINAL COMPARISON — SAME DAVIDSON TEST SET")
    print("=" * 76)
    print(summary.to_string(index=False))
    print("\nSummary saved to:", summary_path)
    print("IMPORTANT: choose the winner based on the common test metrics and intended use. This test measures Davidson-domain performance; it does not establish Hinglish hate-speech performance.")


if __name__ == "__main__":
    main()
