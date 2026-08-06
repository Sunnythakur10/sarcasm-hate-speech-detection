"""
Trains the hate-speech classifier: TF-IDF + Logistic Regression.
3-class problem (hate / offensive / neither), heavily imbalanced —
class_weight='balanced' compensates instead of letting the model
default to predicting the majority class.
"""
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from load_data import load_hate_speech_df
from preprocess import clean_series

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    df = load_hate_speech_df()
    df["clean_text"] = clean_series(df["text"])
    df = df[df["clean_text"].str.len() > 0]  # drop rows that cleaned to empty

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    print("=== Hate Speech Classifier ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix (rows=true, cols=pred), labels order:", sorted(y_test.unique()))
    print(confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique())))

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "hate_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, "hate_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved model + vectorizer to {MODEL_DIR}")


if __name__ == "__main__":
    main()
