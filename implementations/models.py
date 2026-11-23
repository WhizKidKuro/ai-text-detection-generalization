"""Detector A: TF-IDF + logistic regression. Detector B: 6 stylometric features + random forest.
Detector L: text length only."""
import re, numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

FEATURES = ["avg sentence length", "sentence length variation", "avg word length",
            "vocabulary variety (TTR)", "comma/punct rate", "long-word share"]
GROUPS = {"sentence length": (0, 1), "vocabulary variety": (3,), "punctuation": (4,), "word length": (2, 5)}

def stylo(texts):
    rows = []
    for t in texts:
        s = [len(x.split()) for x in re.split(r"[.!?]+\s|[.!?]+$", t) if x.strip()] or [0]
        w = re.findall(r"[A-Za-z']+", t) or ["a"]
        L = [len(x) for x in w]
        rows.append([np.mean(s), np.std(s), np.mean(L), len(set(x.lower() for x in w)) / len(w),
                     len(re.findall(r"[,;:]", t)) / len(w), np.mean([l >= 7 for l in L])])
    return np.array(rows)

class Stylo(BaseEstimator, TransformerMixin):
    def __init__(self, drop=()): self.drop = drop
    def fit(self, X, y=None): return self
    def transform(self, X):
        F = stylo(list(X)); return F[:, [i for i in range(6) if i not in self.drop]]

class Length(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return np.log1p([[len(t.split())] for t in X])

def make(det, drop=(), seed=0):
    if det == "A":
        return make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True, max_features=50000),
                             LogisticRegression(max_iter=1000))
    if det == "B":
        return make_pipeline(Stylo(drop), RandomForestClassifier(300, random_state=seed, n_jobs=-1))
    return make_pipeline(Length(), LogisticRegression())
