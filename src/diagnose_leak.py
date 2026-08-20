"""
Diagnostic: find what's actually separating the two classes so
perfectly. Run this locally and paste me the output.
"""
import sys
sys.path.insert(0, '.')
from load_data import load_hindi_sarcasm_df
from preprocess_hindi import clean_hindi_series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

df = load_hindi_sarcasm_df()
df['clean_text'] = clean_hindi_series(df['text'])
df = df[df['clean_text'].str.len() > 0]

print("Row counts:", df['label'].value_counts().to_dict())
print()
print("Average cleaned text length by class:")
print(df.groupby('label')['clean_text'].apply(lambda s: s.str.len().mean()))
print()
print("Sample cleaned SARCASTIC rows:")
for t in df[df['label']==1]['clean_text'].head(10):
    print(' ', repr(t[:80]))
print()
print("Sample cleaned NOT-SARCASTIC rows:")
for t in df[df['label']==0]['clean_text'].head(10):
    print(' ', repr(t[:80]))
print()

# Fit a quick model and print the highest-weight words per class —
# this shows EXACTLY what the model is keying off.
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)
vec = TfidfVectorizer(max_features=5000, min_df=2)
Xtr = vec.fit_transform(X_train)
clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr, y_train)

feature_names = np.array(vec.get_feature_names_out())
coefs = clf.coef_[0]
top_sarcastic_idx = np.argsort(coefs)[-20:][::-1]
top_not_idx = np.argsort(coefs)[:20]

print("Top words pushing toward SARCASTIC:")
for i in top_sarcastic_idx:
    print(f"  {feature_names[i]}: {coefs[i]:.3f}")
print()
print("Top words pushing toward NOT-SARCASTIC:")
for i in top_not_idx:
    print(f"  {feature_names[i]}: {coefs[i]:.3f}")