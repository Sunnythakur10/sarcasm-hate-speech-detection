"""
Trains the Hinglish (romanized Hindi) sarcasm classifier: TF-IDF
(char n-grams) + Linear SVM. Same architecture family as the English
and Devanagari-Hindi sarcasm models, applied to a third, differently
human-labeled dataset — this one replaced the earlier Devanagari
Hindi dataset after it turned out to have severe label leakage
(sarcasm-marking hashtags and topic-clustering artifacts baked into
the text itself, producing a meaningless 1.000 accuracy).
"""
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from load_data import load_hinglish_sarcasm_df
from preprocess_hinglish import clean_hinglish_series

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    df = load_hinglish_sarcasm_df()
    df["clean_text"] = clean_hinglish_series(df["text"])
    df = df[df["clean_text"].str.len() > 0]

    # Check for exact-duplicate rows before splitting — the last
    # Hindi dataset had massive duplication (same viral tweet repeated
    # via retweets) that inflated accuracy by letting identical text
    # land on both sides of the train/test split.
    dup_count = df.duplicated(subset=["clean_text"]).sum()
    print(f"Duplicate cleaned-text rows: {dup_count} / {len(df)} "
          f"({100 * dup_count / len(df):.1f}%)")
    if dup_count / len(df) > 0.05:
        print("WARNING: >5% duplicate rows — dropping duplicates before "
              "splitting to avoid train/test leakage via repeated text.")
        df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        max_features=10000, analyzer="char_wb", ngram_range=(2, 4), min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LinearSVC(random_state=42, max_iter=5000)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    print("\n=== Hinglish Sarcasm Classifier ===")
    print(classification_report(y_test, y_pred, digits=3, target_names=["not_sarcastic", "sarcastic"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    if classification_report(y_test, y_pred, output_dict=True)["accuracy"] > 0.97:
        print(
            "\nWARNING: accuracy is suspiciously high (>97%). Before "
            "trusting this, check for label-leakage artifacts the same "
            "way diagnose_leak.py did for the Devanagari dataset — "
            "print top predictive words and check they're plausible "
            "sarcasm signal, not topic/collection artifacts."
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "hinglish_sarcasm_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, "hinglish_sarcasm_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved model + vectorizer to {MODEL_DIR}")


if __name__ == "__main__":
    main()