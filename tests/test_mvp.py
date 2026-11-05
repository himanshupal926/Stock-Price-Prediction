"""Tests mirror the MVP acceptance checklist.

1. Baseline and combined model are compared
2. Leakage test passes
3. Sentiment table is visible
4. Disclaimer banner is shown
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.features import ALL_FEATURES, build_dataset
from app.leakage import check_features_use_only_past
from app.main import app
from app.model import Engine, time_split
from app.sentiment import lexicon_score


@pytest.fixture(scope="module")
def engine():
    return Engine()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---- sentiment -------------------------------------------------------------
def test_lexicon_signs():
    assert lexicon_score("ACME beats estimates as demand surges") > 0
    assert lexicon_score("ACME cuts outlook amid weak demand") < 0
    assert lexicon_score("ACME schedules investor day") == 0
    assert lexicon_score("ACME does not beat estimates") < 0   # negation flips the sign


# ---- 1. baseline vs combined ------------------------------------------------
def test_baseline_and_combined_are_compared(engine):
    m = engine.report["metrics"]
    for key in ("price_only", "combined"):
        assert 0 <= m[key]["accuracy"] <= 1 and 0 <= m[key]["f1"] <= 1 and 0 <= m[key]["cv_accuracy"] <= 1
    assert engine.report["comparison"]["verdict"] in ("better", "not_better")


# ---- 2. leakage -------------------------------------------------------------
def test_split_is_chronological(engine):
    assert engine.train_dates.max() < engine.test_dates.min()


def test_all_leakage_checks_pass(engine):
    assert all(c["passed"] for c in engine.report["leakage"]), engine.report["leakage"]


def test_leakage_check_catches_a_leaky_feature(engine):
    """Sanity-check the checker itself: a feature that peeks at tomorrow must be flagged."""
    def leaky_build(prices, daily):
        df = build_dataset(prices, daily)
        df["ret_1"] = df.groupby("ticker")["close"].transform(lambda s: np.log(s.shift(-1) / s))
        return df.fillna(0)

    assert not check_features_use_only_past(engine.prices, engine.daily, leaky_build)["passed"]


def test_time_split_drops_gap_day():
    dates = pd.Series(pd.bdate_range("2022-01-03", periods=100))
    train, test = time_split(dates, 0.3)
    assert len(train) == 69 and len(test) == 30


# ---- 3 & 4. API + dashboard ---------------------------------------------------
def test_dashboard_has_disclaimer_and_sentiment_table(client):
    html = client.get("/").text
    assert 'id="disclaimer"' in html and 'id="sentiment-table"' in html


def test_endpoints(client):
    t = client.get("/api/tickers").json()
    ticker = t["tickers"][0]
    assert t["disclaimer"]

    d = client.get("/api/data", params={"ticker": ticker}).json()
    assert len(d["dates"]) == len(d["close"]) == len(d["sent_day"])

    s = client.get("/api/sentiment", params={"ticker": ticker, "limit": 5}).json()
    assert len(s["rows"]) == 5 and {"date", "headline", "score", "label"} <= set(s["rows"][0])

    p = client.get("/api/predict", params={"ticker": ticker}).json()
    assert 0 <= p["prob_up"] <= 1 and p["trend"] in ("up", "down") and p["disclaimer"]

    assert client.get("/api/evaluate").json()["leakage"]


def test_predict_edge_cases(client):
    ticker = client.get("/api/tickers").json()["tickers"][0]
    # a weekend rolls back to the previous trading day
    p = client.get("/api/predict", params={"ticker": ticker, "date": "2023-07-15"}).json()
    assert p["as_of"] == "2023-07-14"
    assert client.get("/api/predict", params={"ticker": "NOPE"}).status_code == 404
    assert client.get("/api/predict", params={"ticker": ticker, "date": "1999-01-01"}).status_code == 422
    assert client.get("/api/predict", params={"ticker": ticker, "date": "not-a-date"}).status_code == 422
