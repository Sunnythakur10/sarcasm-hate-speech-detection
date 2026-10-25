# Sarcasm and Hate Speech Detection

An NLP web application that combines sarcasm detection with English hate-speech/offensive-language classification. The project includes separate sarcasm classifiers for English and Romanized Hindi (Hinglish), plus a three-class English text classifier for **hate**, **offensive**, and **neither**.

The current demo interface is built with Streamlit. The repository also retains the Flask application and its HTML/CSS/JavaScript interface.

> **Scope note:** This is an academic/demo project. Predictions are model estimates, not definitive judgments about a speaker's intent or the social meaning of a sentence.

## Features

- **English sarcasm detection:** TF-IDF text features with a Linear SVM classifier.
- **Hinglish sarcasm detection:** Character-level TF-IDF features with a Linear SVM classifier, intended for Hindi written in Latin script.
- **English hate/offensive classification:** Three labels: `hate`, `offensive`, and `neither`.
- **Combined result categories:** The pipeline combines the sarcasm and hate/offensive decisions into categories such as `hate_sarcastic`, `offensive_sarcastic`, `hate`, `offensive`, `sarcastic`, and `neutral`.
- **Language/script routing:** A small rule-based heuristic routes text to the English or Hinglish sarcasm model. Devanagari-script input is identified and reported as unsupported rather than silently sent to the wrong sarcasm classifier.
- **Interactive web interface:** A Streamlit interface for entering text and viewing the prediction and class probabilities.

## How the pipeline works

1. The user enters text in the web interface.
2. The pipeline checks the input and determines whether it appears to be English, Romanized Hinglish, or Devanagari script.
3. For English input, the English TF-IDF + Linear SVM classifier predicts sarcasm. The English hate classifier predicts `hate`, `offensive`, or `neither`.
4. For Romanized Hinglish input, the Hinglish TF-IDF + Linear SVM classifier predicts sarcasm. Hate/offensive classification is not run for Hinglish.
5. The application combines the available predictions into a result category and displays the output.

### Important interpretation detail

The sarcasm classifier uses a Linear SVM decision score. That score is **not a calibrated probability**. Hate-class probabilities are produced by the three-class transformer classifier; confidence values should still be interpreted cautiously.

## Model components

### English sarcasm

- **Features:** TF-IDF word n-grams.
- **Classifier:** Linear SVM (`LinearSVC`).
- **Training data used in project experiments:** A working subset of Reddit's SARC / `train-balanced-sarcasm` data (approximately 100,000 examples).
- **Reported experiment result:** Accuracy around 65.1%; class F1 scores around 0.79 and 0.77. These are experiment results, not a guarantee of performance on new domains.

### Hinglish sarcasm

- **Features:** Character-level TF-IDF n-grams.
- **Classifier:** Linear SVM (`LinearSVC`).
- **Training data used in project experiments:** A human-labelled Romanized Hinglish dataset with more than 10,000 examples.
- **Reported experiment result:** Accuracy 66.2%; F1 scores of approximately 0.705 for non-sarcastic and 0.603 for sarcastic.

### Hate/offensive classification — V7

The application uses the **V7** version label for its selected three-class hate/offensive classifier. The current integrated checkpoint is the published RoBERTa-based model [`AuricErgeson/hate-speech-detector`](https://huggingface.co/AuricErgeson/hate-speech-detector), loaded through the Hugging Face Transformers library. V7 is the version used by this application; the currently integrated checkpoint should not be described as independently trained by this project unless a separate fine-tuning run and checkpoint for that exact model are available.

The output labels are normalized to:

- `hate_speech` → `hate`
- `offensive` → `offensive`
- `neither` → `neither`

A small, manually prepared set of example sentences was used for a functional sanity check. That check showed useful detections on several explicit hate examples, but also false positives and missed examples. It is **not** a formal benchmark and should not be reported as an accuracy evaluation.

## Research and experiments

The `v4` branch preserves the research history, datasets, training scripts, experimental checkpoints, and evaluation outputs. Experiments included Davidson, Implicit Hate, and HateXplain data and compared candidate DistilBERT models on a common Davidson test split.

One common-test comparison produced the following results:

| Candidate experiment | Accuracy | Macro F1 | Hate-class F1 |
|---|---:|---:|---:|
| Davidson-only baseline | 91.40% | 0.7838 | 0.5180 |
| Davidson + Implicit Hate | 90.07% | 0.7540 | 0.4695 |
| Davidson + HateXplain | 90.67% | 0.7841 | 0.5316 |

These metrics belong to those **experimental DistilBERT candidates**, not the currently integrated V7 checkpoint. The test set contained 2,477 examples, including 143 hate examples. This evaluation mainly measures performance on the Davidson dataset domain and does not establish performance on Hinglish hate speech or all real-world text.

## Repository structure

The deployment branch (`main`) should contain the application code and only the model assets needed at runtime. The research branch (`v4`) preserves the broader experiments.

```text
.
├── app.py                     # Flask application (retained)
├── streamlit_app.py           # Streamlit entry point used for the hosted demo
├── requirements.txt            # Python dependencies
├── README.md
├── src/
│   ├── pipeline.py             # Loads models and combines predictions
│   ├── preprocess.py           # English text preprocessing
│   └── preprocess_hinglish.py  # Hinglish text preprocessing
├── models/
│   ├── sarcasm_vectorizer.pkl
│   ├── sarcasm_classifier.pkl
│   ├── hinglish_sarcasm_vectorizer.pkl
│   └── hinglish_sarcasm_classifier.pkl
├── templates/                  # Flask HTML templates
└── static/                     # Flask CSS and JavaScript
```

The exact files in a checkout can differ as the project evolves. Large transformer weights need not be committed to `main` if the application downloads the configured public checkpoint at startup. The four `.pkl` files above are required for the two TF-IDF sarcasm models.

## Run locally

Use Python 3.12 for the closest match to the cloud deployment configuration.

### 1. Clone and enter the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <REPOSITORY_FOLDER>
```

### 2. Create and activate a virtual environment

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start Streamlit

From the repository root:

```bash
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

### Optional: run the Flask app

If you are working on the Flask interface/API instead of Streamlit, use:

```bash
python app.py
```

Use the Flask routes and port configured in `app.py`; the Streamlit interface is started separately with `streamlit run streamlit_app.py`.

## Deploy on Streamlit Community Cloud

1. Push the application code on the `main` branch to GitHub.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and select the repository and `main` branch.
4. Set the app entry point to `streamlit_app.py`.
5. Choose a Python version compatible with the project's dependencies (Python 3.12 is the current project configuration).
6. Deploy and review the app logs if startup fails.

The first startup may take longer because dependencies are installed and the transformer checkpoint may need to be downloaded. Memory availability on shared hosting can vary; a model-loading memory error may require a more memory-capable host or a separate inference service.

## Dependencies

The main technologies used are:

- Python
- Streamlit for the current demo UI
- Flask for the existing web application/API
- scikit-learn for TF-IDF and Linear SVM sarcasm classifiers
- Transformers and PyTorch for the three-class hate/offensive classifier
- NLTK for English text preprocessing
- Git and Git LFS for version control and large research/model assets

The `requirements.txt` file is the authoritative dependency list. When changing model libraries or scikit-learn versions, retest loading the serialized `.pkl` vectorizers and classifiers because pickle artifacts can be version-sensitive.

## Limitations

- Hate/offensive classification is intended for **English text only**.
- Hinglish support is for **sarcasm detection only**; the application does not provide Hinglish hate-speech predictions.
- Devanagari-script sarcasm is currently unsupported.
- English-versus-Hinglish routing uses a hand-built marker-word heuristic, not a trained language identification model.
- Sarcasm is highly context-dependent; short text, missing conversation context, slang, reclaimed terms, and cultural references can cause errors.
- Hate/offensive labels can be confused, and false positives/false negatives are possible.
- The reported model metrics come from particular datasets and splits and should not be generalized to every population or platform.
- Do not use this demo as the sole basis for moderation, disciplinary decisions, or other high-impact actions.

## Team workflow and branches

- **`main`:** deployment-ready application code and runtime assets only.
- **`v4`:** research datasets, notebooks, training and preparation scripts, experimental checkpoints, metrics, and comparison outputs.

Before deleting research files or rewriting model assets, ensure the corresponding changes are committed and pushed to the appropriate branch. Use Git LFS for large files that must be versioned; avoid committing secrets, local virtual environments, or `.env` files.

## Contributors

Add the names and responsibilities of the team members here before the final submission.

## Acknowledgements and model attribution

- Reddit SARC / `train-balanced-sarcasm` for English sarcasm research data.
- Davidson hate-speech dataset, Implicit Hate Corpus, and HateXplain for hate/offensive classification experiments.
- [`AuricErgeson/hate-speech-detector`](https://huggingface.co/AuricErgeson/hate-speech-detector) for the current V7 hate/offensive checkpoint integrated into the application.

Review the applicable dataset and model licenses and terms before redistributing data or model weights.
