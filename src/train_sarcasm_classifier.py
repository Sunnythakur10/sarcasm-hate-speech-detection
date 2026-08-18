"""
Trains the sarcasm classifier: TF-IDF + Linear SVM.
Binary problem, roughly balanced classes (52/48), so no class weighting
needed here — a meaningful difference from the hate classifier, and worth
noting in the report as to why the two models use different settings.
"""
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from load_data import load_reddit_sarcasm_df
from preprocess import clean_series

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    # Using the Reddit SARC dataset (conversational, domain-matched to the
    # hate speech tweets) instead of news headlines. See load_data.py.
    df = load_reddit_sarcasm_df(sample_size=100000)
    df["clean_text"] = clean_series(df["text"])
    df = df[df["clean_text"].str.len() > 0]

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LinearSVC(random_state=42, max_iter=5000)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    print("=== Sarcasm Classifier ===")
    print(classification_report(y_test, y_pred, digits=3, target_names=["not_sarcastic", "sarcastic"]))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "sarcasm_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, "sarcasm_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved model + vectorizer to {MODEL_DIR}")


if __name__ == "__main__":
    main()