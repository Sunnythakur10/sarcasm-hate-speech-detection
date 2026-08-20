# Sarcasm Detection in Hate Speech — NLP Mini Project

## Problem Statement

Detect whether text is sarcastic, hateful/offensive, both, or neither.
This is framed as **two independently trained classifiers chained
together**, not a single jointly-labeled model — no dataset exists that
labels the same text for both sarcasm and hate speech simultaneously, so
building one was out of scope for a mini project.

## Datasets

| Dataset | Source | Size | Labels |
|---|---|---|---|
| Hate speech | Davidson et al. (2017), ICWSM — Twitter | 24,783 tweets | hate / offensive / neither |
| Sarcasm | Misra & Arora, News Headlines Dataset — TheOnion vs HuffPost | 28,619 headlines | sarcastic / not sarcastic |

Both are pre-labeled, publicly available, and required no manual
annotation.

## Pipeline

```
raw text
   │
   ├──► [Hate Speech Classifier] ──► hate / offensive / neither
   │        TF-IDF + Logistic Regression (class_weight=balanced)
   │
   └──► [Sarcasm Classifier] ──────► sarcastic / not sarcastic
            TF-IDF + Linear SVM

Combined into 4 buckets:
  hateful_sarcastic | hateful | sarcastic | neutral
```

Preprocessing (`src/preprocess.py`) is shared and identical across both
classifiers: lowercase, strip URLs/mentions/punctuation, remove
stopwords (negation words like "not", "never" are deliberately kept —
they carry sarcasm/hate signal), lemmatize.

## Results

**Hate speech classifier** (test set, n=4957):
- Accuracy: 86.9%
- F1 — hate: 0.44, offensive: 0.92, neither: 0.86
- **Known weakness**: "hate" class has only 0.34 precision. The
  dataset is heavily imbalanced (only 5.8% of samples are "hate"),
  and the model frequently confuses hate with offensive. This is
  reported honestly rather than hidden — a known, documented
  limitation of this dataset in published literature as well.

**Sarcasm classifier** (test set, n=5724):
- Accuracy: 78.7%, F1 ≈ 0.79 for both classes (balanced dataset,
  no class-imbalance issue here).

## Known Limitation — Domain Mismatch

The sarcasm classifier is trained on **formal news headlines**
(Onion/HuffPost), while the hate classifier is trained on
**conversational Twitter text**. When chained on hostile/conversational
input, the sarcasm model does not reliably transfer — e.g. it can
mislabel plain hateful statements as non-sarcastic-neutral, or flag
neutral statements as sarcastic, because it never saw that register of
text during training. This is disclosed here rather than hidden, and is
a direct consequence of using two disjoint datasets instead of one
jointly-annotated corpus — the tradeoff made deliberately to keep the
project achievable without manual annotation.

## Setup & Run

```bash
pip install pandas scikit-learn nltk

# Train both classifiers (run once, saves to models/)
cd src
python3 train_hate_classifier.py
python3 train_sarcasm_classifier.py

# Try the pipeline
python3 pipeline.py

# Interactive demo
python3 demo.py
```

## Possible Future Work (stretch, not required for this mini project)

- Fine-tune DistilBERT on one or both tasks to compare against the
  TF-IDF baseline.
- Find or construct a jointly-labeled sarcasm+hate corpus to train a
  single model instead of chaining two.
- Add context-incongruity features (sentiment shift between clauses)
  as explicit inputs, which the sarcasm literature shows helps.

## Project Structure

```
sarcasm-hate-project/
├── data/
│   ├── hate_speech_raw.csv
│   └── sarcasm_raw.json
├── models/              (generated after training)
│   ├── hate_vectorizer.pkl / hate_classifier.pkl
│   └── sarcasm_vectorizer.pkl / sarcasm_classifier.pkl
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_hate_classifier.py
│   ├── train_sarcasm_classifier.py
│   ├── pipeline.py
│   └── demo.py
└── README.md
```
