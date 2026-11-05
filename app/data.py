"""Load prices and headlines from CSV (generating sample data on first run)."""
import pandas as pd

from . import sample_data
from .config import DATA_DIR

PRICE_COLS = {"date", "ticker", "open", "high", "low", "close", "volume"}
NEWS_COLS = {"date", "ticker", "headline"}


def is_synthetic() -> bool:
    return (DATA_DIR / "SYNTHETIC_DATA.txt").exists()


def ensure_data():
    prices_path, news_path = DATA_DIR / "prices.csv", DATA_DIR / "news.csv"
    if not (prices_path.exists() and news_path.exists()):
        sample_data.generate()

    prices = pd.read_csv(prices_path, parse_dates=["date"])
    news = pd.read_csv(news_path, parse_dates=["date"])
    for name, df, cols in (("prices.csv", prices, PRICE_COLS), ("news.csv", news, NEWS_COLS)):
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    news = news.dropna(subset=["headline"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    return prices, news
