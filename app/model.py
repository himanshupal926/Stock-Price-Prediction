"""Training, honest evaluation and prediction.

Engine loads data, scores sentiment, builds features, trains two models on the EARLY part of
the timeline (price-only baseline vs price + sentiment) and compares them on the LAST part.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .data import ensure_data, is_synthetic
from .features import ALL_FEATURES, PRICE_FEATURES, build_dataset
from .leakage import check_features_use_only_past, check_label_shuffle, check_time_order
from .sentiment import score_headlines


def make_model(kind: str, seed: int):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=25,
                                      random_state=seed, n_jobs=1)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))


def fit(kind, X, y, seed):
    model = make_model(kind, seed)
    model.fit(X, y)
    return model


def time_split(dates, test_fraction):
    """Chronological split. One day between train and test is dropped, because the label of
    the last training day looks one day ahead (into the first test day)."""
    u = pd.DatetimeIndex(dates.unique()).sort_values()
    cut = int(len(u) * (1 - test_fraction))
    return u[:cut - 1], u[cut:]


def cv_accuracy(train: pd.DataFrame, features, kind, seed, n_splits=4) -> float:
    """Expanding-window cross-validation inside the training period only."""
    u = pd.DatetimeIndex(train["date"].unique()).sort_values()
    chunks = np.array_split(np.arange(len(u)), n_splits + 1)
    scores = []
    for k in range(1, n_splits + 1):
        tr_dates = u[np.concatenate(chunks[:k])[:-1]]
        va_dates = u[chunks[k]]
        tr, va = train[train["date"].isin(tr_dates)], train[train["date"].isin(va_dates)]
        model = fit(kind, tr[features], tr["target"], seed)
        scores.append(accuracy_score(va["target"], model.predict(va[features])))
    return float(np.mean(scores))


def label(score: float) -> str:
    return "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"


class Engine:
    def __init__(self):
        self.kind, self.seed = config.MODEL_KIND, config.SEED
        self.synthetic = is_synthetic()

        self.prices, self.news = ensure_data()
        scores, self.backend_used = score_headlines(self.news["headline"].tolist(), config.SENTIMENT_BACKEND)
        self.news["score"] = self.news["headline"].map(scores)

        # A headline dated D counts for the first trading day on/after D (assumed published
        # before that day's close). Headlines after the last trading day are dropped.
        cal = pd.DatetimeIndex(self.prices["date"].unique()).sort_values()
        pos = cal.searchsorted(self.news["date"])
        self.news["trade_date"] = pd.NaT
        ok = pos < len(cal)
        self.news.loc[ok, "trade_date"] = cal[pos[ok]]
        self.daily = (
            self.news.dropna(subset=["trade_date"])
            .groupby(["ticker", "trade_date"])
            .agg(sent_day=("score", "mean"), n_news=("score", "size"))
            .reset_index().rename(columns={"trade_date": "date"})
        )

        self.df = build_dataset(self.prices, self.daily)
        self._train_and_evaluate()

    # ------------------------------------------------------------------ training
    def _train_and_evaluate(self):
        self.train_dates, self.test_dates = time_split(self.df["date"], config.TEST_FRACTION)
        labelled = self.df.dropna(subset=["target"])
        train = labelled[labelled["date"].isin(self.train_dates)]
        test = labelled[labelled["date"].isin(self.test_dates)]
        y_test = test["target"].astype(int).to_numpy()

        self.models = {
            "price_only": fit(self.kind, train[PRICE_FEATURES], train["target"], self.seed),
            "combined": fit(self.kind, train[ALL_FEATURES], train["target"], self.seed),
        }
        feats = {"price_only": PRICE_FEATURES, "combined": ALL_FEATURES}
        pred = {k: self.models[k].predict(test[feats[k]]).astype(int) for k in self.models}

        def metrics(k):
            return {
                "accuracy": float(accuracy_score(y_test, pred[k])),
                "f1": float(f1_score(y_test, pred[k], zero_division=0)),
                "cv_accuracy": cv_accuracy(train, feats[k], self.kind, self.seed),
            }

        m = {k: metrics(k) for k in self.models}
        majority = float(max(y_test.mean(), 1 - y_test.mean()))

        # Bootstrap CI for the accuracy gain (approximate: treats test rows as independent)
        rng = np.random.default_rng(self.seed)
        hit_b, hit_c = (pred["price_only"] == y_test), (pred["combined"] == y_test)
        idx = rng.integers(0, len(y_test), (2000, len(y_test)))
        gains = hit_c[idx].mean(axis=1) - hit_b[idx].mean(axis=1)
        gain = m["combined"]["accuracy"] - m["price_only"]["accuracy"]
        lo, hi = (float(x) for x in np.percentile(gains, [2.5, 97.5]))
        better = gain >= config.MIN_GAIN and lo > 0 and m["combined"]["f1"] >= m["price_only"]["f1"]

        shuffled = fit(self.kind, train[ALL_FEATURES], rng.permutation(train["target"].to_numpy()), self.seed)
        acc_shuffled = float(accuracy_score(y_test, shuffled.predict(test[ALL_FEATURES])))

        self.report = {
            "data_source": "synthetic" if self.synthetic else "user data",
            "sentiment_backend": self.backend_used,
            "model_kind": self.kind,
            "split": {
                "train_start": f"{self.train_dates.min():%Y-%m-%d}", "train_end": f"{self.train_dates.max():%Y-%m-%d}",
                "test_start": f"{self.test_dates.min():%Y-%m-%d}", "test_end": f"{self.test_dates.max():%Y-%m-%d}",
                "n_train": int(len(train)), "n_test": int(len(test)),
            },
            "metrics": {"majority": {"accuracy": majority}, **m},
            "comparison": {
                "accuracy_gain": float(gain), "ci_low": lo, "ci_high": hi, "verdict": "better" if better else "not_better",
                "message": (
                    "Price + sentiment beat the price-only baseline on the unseen period."
                    if better else
                    "Not better: price + sentiment did not clearly beat the price-only baseline on the unseen period."
                ),
            },
            "leakage": [
                check_time_order(self.train_dates, self.test_dates),
                check_features_use_only_past(self.prices, self.daily, build_dataset),
                check_label_shuffle(acc_shuffled, majority),
            ],
        }

    # ------------------------------------------------------------------ queries
    @property
    def tickers(self):
        return sorted(self.df["ticker"].unique().tolist())

    def ticker_frame(self, ticker: str) -> pd.DataFrame:
        g = self.df[self.df["ticker"] == ticker.upper()]
        if g.empty:
            raise KeyError(ticker)
        return g

    def predict(self, ticker: str, date: str | None = None) -> dict:
        g = self.ticker_frame(ticker)
        if date:
            g = g[g["date"] <= pd.Timestamp(date)]
            if g.empty:
                raise ValueError("No data on or before that date")
        row = g.iloc[[-1]]
        d = row["date"].iloc[0]
        p_comb = float(self.models["combined"].predict_proba(row[ALL_FEATURES])[0, 1])
        p_base = float(self.models["price_only"].predict_proba(row[PRICE_FEATURES])[0, 1])
        period = "train" if d <= self.train_dates.max() else "test" if d >= self.test_dates.min() else "gap"

        day_news = self.news[(self.news["ticker"] == ticker.upper()) & (self.news["trade_date"] == d)]
        return {
            "ticker": ticker.upper(),
            "as_of": f"{d:%Y-%m-%d}",
            "is_latest": bool(d == self.df["date"].max()),
            "period": period,
            "trend": "up" if p_comb >= 0.5 else "down",
            "prob_up": p_comb,
            "prob_up_baseline": p_base,
            "sent_day": float(row["sent_day"].iloc[0]),
            "sent_3d": float(row["sent_3d"].iloc[0]),
            "headlines": [{"headline": h, "score": float(s), "label": label(s)}
                          for h, s in zip(day_news["headline"], day_news["score"])],
            "disclaimer": config.DISCLAIMER,
        }

    def sentiment_table(self, ticker: str, limit: int = 30) -> list[dict]:
        n = self.news[self.news["ticker"] == ticker.upper()].sort_values("date", ascending=False).head(limit)
        return [{"date": f"{d:%Y-%m-%d}", "headline": h, "score": float(s), "label": label(s)}
                for d, h, s in zip(n["date"], n["headline"], n["score"])]
