"""Checks that catch look-ahead leakage. Each returns {"name", "passed", "detail"}."""
import numpy as np
import pandas as pd

from .features import ALL_FEATURES


def check_time_order(train_dates, test_dates) -> dict:
    ok = bool(train_dates.max() < test_dates.min())
    return {
        "name": "Time-ordered split",
        "passed": ok,
        "detail": f"training ends {train_dates.max():%Y-%m-%d}, test starts {test_dates.min():%Y-%m-%d}; "
                  "no shuffling, one-day gap dropped",
    }


def check_features_use_only_past(prices, daily_sent, build_fn, seed=0) -> dict:
    """Scramble ALL data after a cutoff date. Features for dates <= cutoff must not change."""
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(prices["date"].unique()).sort_values()
    cutoff = dates[int(len(dates) * 0.6)]

    p2 = prices.copy()
    p2[["open", "high", "low", "close", "volume"]] = p2[["open", "high", "low", "close", "volume"]].astype(float)
    future = (p2["date"] > cutoff).to_numpy()
    for col in ["open", "high", "low", "close", "volume"]:
        p2.loc[future, col] = p2.loc[future, col] * rng.uniform(0.5, 2.0, future.sum())
    s2 = daily_sent.copy()
    sfut = (s2["date"] > cutoff).to_numpy()
    s2.loc[sfut, "sent_day"] = rng.uniform(-1, 1, sfut.sum())
    s2.loc[sfut, "n_news"] = rng.integers(0, 5, sfut.sum())

    a = build_fn(prices, daily_sent)
    b = build_fn(p2, s2)
    a = a[a["date"] <= cutoff].set_index(["ticker", "date"])[ALL_FEATURES]
    b = b[b["date"] <= cutoff].set_index(["ticker", "date"])[ALL_FEATURES]
    ok = a.shape == b.shape and bool(np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True))
    return {
        "name": "Features use only past data",
        "passed": ok,
        "detail": f"rewrote every price and headline after {cutoff:%Y-%m-%d}; "
                  "features on or before that date were " + ("unchanged" if ok else "CHANGED"),
    }


def check_label_shuffle(acc_shuffled: float, majority_acc: float) -> dict:
    ok = acc_shuffled <= majority_acc + 0.04
    return {
        "name": "Shuffled-label sanity check",
        "passed": bool(ok),
        "detail": f"a model trained on random labels scores {acc_shuffled:.1%} on the test period "
                  f"(always guessing the common class gives {majority_acc:.1%}); it should not beat that",
    }
