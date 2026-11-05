"""Feature engineering. Every feature at date t uses data up to and including t only."""
import numpy as np
import pandas as pd

PRICE_FEATURES = ["ret_1", "ret_5", "ret_10", "vol_10", "ma_gap_10", "vol_chg"]
SENT_FEATURES = ["sent_day", "sent_3d", "n_news"]
ALL_FEATURES = PRICE_FEATURES + SENT_FEATURES


def build_dataset(prices: pd.DataFrame, daily_sent: pd.DataFrame) -> pd.DataFrame:
    """prices: ticker,date,close,volume,...   daily_sent: ticker,date,sent_day,n_news

    `target` = 1 if the NEXT trading day's close is higher than today's (NaN on the last row).
    """
    frames = []
    for ticker, g in prices.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        close, vol = g["close"], g["volume"]
        logc = np.log(close)
        r = logc.diff()
        nxt = close.shift(-1)
        frames.append(pd.DataFrame({
            "ticker": ticker,
            "date": g["date"],
            "close": close,
            "ret_1": r,
            "ret_5": logc.diff(5),
            "ret_10": logc.diff(10),
            "vol_10": r.rolling(10).std(),
            "ma_gap_10": close / close.rolling(10).mean() - 1,
            "vol_chg": vol / vol.rolling(10).mean() - 1,
            "target": (nxt > close).astype(float).where(nxt.notna()),
        }))
    df = pd.concat(frames, ignore_index=True)

    df = df.merge(daily_sent, on=["ticker", "date"], how="left")
    df[["sent_day", "n_news"]] = df[["sent_day", "n_news"]].fillna(0.0)
    df["sent_3d"] = df.groupby("ticker")["sent_day"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return df.dropna(subset=PRICE_FEATURES).reset_index(drop=True)
