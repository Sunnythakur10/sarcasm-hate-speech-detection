---
language:
- en
license: mit
tags:
- text-classification
- hate-speech-detection
- content-moderation
- nlp
- roberta
- twitter
- safety
- offensive-language
- transformers
- pytorch
datasets:
- tdavidson/hate_speech_offensive
- tasksource/implicit-hate-stg1
- Hate-speech-CNERG/hatexplain
- manueltonneau/hateday
metrics:
- f1
- accuracy
pipeline_tag: text-classification
model-index:
- name: roberta-hate-speech-detector
  results:
  - task:
      type: text-classification
      name: Hate Speech Detection
    dataset:
      name: held-out test set (stratified 10% split)
      type: mixed
    metrics:
    - type: f1
      value: 0.8489
      name: Weighted F1
    - type: accuracy
      value: 0.8434
      name: Accuracy
---

# roberta-hate-speech-detector

Twitter-native RoBERTa fine-tuned on 110K+ examples from four public hate speech datasets, augmented with targeted neo-Nazi codes and antisemitic dog whistles that general Twitter corpora miss.

Three output classes:

| ID | Label | Meaning |
|----|-------|---------|
| 0  | `neither` | Clean — not hateful or offensive |
| 1  | `offensive` | Crude, profane, or offensive — but not hate speech |
| 2  | `hate_speech` | Hate speech — slurs, coded language, dog whistles, white nationalist symbols |

## Quick Start

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="AuricErgeson/roberta-hate-speech-detector",
)

classifier("they control the media")
# [{'label': 'hate_speech', 'score': 0.87}]

classifier("this movie sucked ass")
# [{'label': 'offensive', 'score': 0.91}]

classifier("I really enjoyed the concert")
# [{'label': 'neither', 'score': 0.97}]
```

Batch inference:

```python
texts = ["heil hitler", "what a stupid film", "the weather is nice today"]
results = classifier(texts)
```

## Performance

Evaluated on a stratified held-out test set of **11,059 examples**.

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| neither | 0.955 | 0.822 | 0.884 | 6,514 |
| offensive | 0.830 | 0.913 | 0.870 | 2,703 |
| hate_speech | 0.607 | 0.817 | 0.697 | 1,842 |
| **weighted avg** | **0.867** | **0.843** | **0.849** | **11,059** |

- Val weighted F1: **0.852** (best checkpoint, epoch 3)
- Test accuracy: **0.843**
- Passed 7/8 probe cases (see Limitations)

## Training Data

110,585 examples fused from four public datasets plus targeted augmentation:

| Dataset | Examples | What it covers |
|---------|----------|----------------|
| [Davidson et al. 2017](https://huggingface.co/datasets/tdavidson/hate_speech_offensive) | 24,783 | Explicit Twitter slurs and offensive language |
| [ImplicitHate](https://huggingface.co/datasets/tasksource/implicit-hate-stg1) | 21,480 | Coded language and dog whistles |
| [HateXplain](https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain) | 19,229 | Multi-annotator hate speech with rationales |
| [HateDay 2025](https://huggingface.co/datasets/manueltonneau/hateday) (English only) | 45,000 | Large-scale contemporary Twitter hate |
| Targeted augmentation | 93 | Neo-Nazi codes (1488, heil hitler, 14 words), antisemitic dog whistles (ZOG, "they control the media"), white nationalist phrases |

**Label harmonization** — all datasets mapped to the unified 3-class scheme. Training set oversampled to 21,621 examples per class (64,863 total) to counter the heavy `neither` majority.

## Training Details

- **Base model:** [cardiffnlp/twitter-roberta-base-2022-154m](https://huggingface.co/cardiffnlp/twitter-roberta-base-2022-154m) — RoBERTa trained on 154M tweets through 2022
- **Epochs:** 3 | **LR:** 2e-5 | **Batch:** 32 | **Grad accum:** 2 | **Warmup:** 200 steps
- **Precision:** bf16 | **Hardware:** NVIDIA A100 | **Training time:** ~6 minutes
- **Framework:** HuggingFace Transformers + Trainer API

One non-obvious implementation detail: `cardiffnlp/twitter-roberta-base-2022-154m` uses legacy TF-style LayerNorm parameter names (`gamma`/`beta`) that transformers ≥5.0 no longer maps automatically. The checkpoint weights were reloaded manually with `gamma→weight` and `beta→bias` renaming before training.

## Limitations

- **Bare numeric codes** — `"1488"` as a standalone string is predicted `neither`. The model correctly classifies it in context ("1488 white power") but the 4-digit code alone is below threshold. This is the only known failed probe case.
- **Post-2022 slang** — base model training data ends 2022; emerging coded language coined after that date may not be recognized.
- **Context dependence** — academic discussion of hate speech ("researchers studying the n-word...") and quotation of hateful phrases to condemn them can produce false positives. The model sees tokens, not intent.
- **English only** — trained exclusively on English-language data. Do not use on multilingual content.
- **Short bare tokens** — very short inputs (1–3 tokens) are unreliable; the model needs minimal context.

## Intended Use

- Content moderation pipelines needing a fast, lightweight classifier
- Research on hate speech detection and NLP fairness
- Dataset annotation assistance (human-in-the-loop review)
- Building safety filters for social media applications

## Out-of-Scope Use

This model **should not** be used:
- As the sole arbiter for automated account suspension or content removal without human review
- To profile or surveil individuals or communities
- In contexts outside English-language social media text
- As a substitute for human moderators on high-stakes decisions

## Citation

If you use this model in research, please cite the four underlying datasets:

```
Davidson et al. (2017) — Automated Hate Speech Detection and the Problem of Offensive Language
El-Shirbeeny et al. (2021) — ImplicitHate
Mathew et al. (2021) — HateXplain
Tonneau et al. (2025) — HateDay
```
## Live Demo

https://huggingface.co/spaces/AuricErgeson/hate-speech-detector
