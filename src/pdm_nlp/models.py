"""Four ways to rank assets by breakdown risk, from simplest to most capable."""
from __future__ import annotations

import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import NUMERIC

_TOKEN = re.compile(r"(?u)\b[a-z][a-z/]+\b")


def entry_ngrams(text: str) -> list[str]:
    """Unigrams and bigrams built inside each log entry (entries are joined with '|'),
    so the last word of one entry never pairs with the first word of the next."""
    out = []
    for entry in text.split("|"):
        toks = _TOKEN.findall(entry)
        out += toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    return out


def _tfidf():
    return TfidfVectorizer(analyzer=entry_ngrams, min_df=25, sublinear_tf=True)

# 1. What a reliability engineer might do with a spreadsheet filter: count warning words, recent ones double.
WARNING_WORDS = re.compile(
    r"vibration|leak|weeping|drip|noise|noisy|rising|increasing|hot|high|low|sticking|hunting|drift|"
    r"fluctuating|discoloured|burning|imbalance|plugged|frozen|bad pv|deviation|rough|metal")


def keyword_score(df: pd.DataFrame) -> np.ndarray:
    return (2 * df.text_recent.str.count(WARNING_WORDS) + df.text_older.str.count(WARNING_WORDS)).values


# 2. Structured history only, no text: activity counts, time since last breakdown, age.
def history_model() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["asset_type", "criticality"]),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])


class TextLogReg:
    """3. TF-IDF on the last 14 days and on days 15-60 separately (so 'rising' recently weighs differently),
    plus the history features, into a class-balanced logistic regression. Coefficients are readable."""

    def __init__(self, C: float = 0.5):
        self.vec_recent = _tfidf()
        self.vec_older = _tfidf()
        self.hist = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["asset_type", "criticality"]),
        ])
        self.clf = LogisticRegression(C=C, class_weight="balanced", max_iter=4000)

    def _x(self, df, fit=False):
        f = (lambda v, s: v.fit_transform(s)) if fit else (lambda v, s: v.transform(s))
        return hstack([f(self.vec_recent, df.text_recent), f(self.vec_older, df.text_older),
                       f(self.hist, df)]).tocsr()

    def fit(self, df):
        self.clf.fit(self._x(df, fit=True), df.label)
        return self

    def predict_proba(self, df):
        return self.clf.predict_proba(self._x(df))[:, 1]

    def top_terms(self, n=15):
        names = ([f"recent: {t}" for t in self.vec_recent.get_feature_names_out()]
                 + [f"days 15-60: {t}" for t in self.vec_older.get_feature_names_out()])
        coef = self.clf.coef_[0][: len(names)]
        order = np.argsort(coef)
        return ([(names[i], coef[i]) for i in order[::-1][:n]], [(names[i], coef[i]) for i in order[:n]])

    def recent_term_weights(self) -> dict:
        names = self.vec_recent.get_feature_names_out()
        return dict(zip(names, self.clf.coef_[0][: len(names)]))


class TextLightGBM:
    """4. The same text compressed with SVD (latent 'symptom themes'), plus history, into gradient-boosted trees.
    Trees can learn interactions such as 'warning words AND no corrective work order since'."""

    def __init__(self, n_comp: int = 60, seed: int = 0):
        self.vec_recent = _tfidf()
        self.vec_older = _tfidf()
        self.svd_recent = TruncatedSVD(n_comp, random_state=seed)
        self.svd_older = TruncatedSVD(n_comp, random_state=seed)
        self.model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                        scale_pos_weight=5, random_state=seed, verbose=-1)

    def _x(self, df, fit=False):
        if fit:
            r = self.svd_recent.fit_transform(self.vec_recent.fit_transform(df.text_recent))
            o = self.svd_older.fit_transform(self.vec_older.fit_transform(df.text_older))
        else:
            r = self.svd_recent.transform(self.vec_recent.transform(df.text_recent))
            o = self.svd_older.transform(self.vec_older.transform(df.text_older))
        h = df[NUMERIC].values
        cat = pd.get_dummies(df.asset_type).reindex(columns=self.types_, fill_value=0).values
        return np.hstack([r, o, h, cat])

    def fit(self, df):
        self.types_ = sorted(df.asset_type.unique())
        self.model.fit(self._x(df, fit=True), df.label)
        return self

    def predict_proba(self, df):
        return self.model.predict_proba(self._x(df))[:, 1]
