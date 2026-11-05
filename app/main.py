"""FastAPI app: /api/data, /api/predict, /api/evaluate (+ sentiment table) and the dashboard."""
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .model import Engine

STATIC = config.ROOT / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = Engine()   # loads data, scores sentiment, trains both models (a few seconds)
    yield


app = FastAPI(title="Stock Price Prediction - ML + LLM Sentiment (MVP)", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def engine(request: Request) -> Engine:
    return request.app.state.engine


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/tickers")
def tickers(request: Request):
    e = engine(request)
    return {
        "tickers": e.tickers,
        "first_date": f"{e.df['date'].min():%Y-%m-%d}",
        "last_date": f"{e.df['date'].max():%Y-%m-%d}",
        "data_source": e.report["data_source"],
        "disclaimer": config.DISCLAIMER,
    }


@app.get("/api/data")
def data(request: Request, ticker: str):
    e = engine(request)
    try:
        g = e.ticker_frame(ticker)
    except KeyError:
        raise HTTPException(404, f"Unknown ticker {ticker}")
    return {
        "ticker": ticker.upper(),
        "dates": g["date"].dt.strftime("%Y-%m-%d").tolist(),
        "close": g["close"].round(2).tolist(),
        "sent_day": g["sent_day"].round(3).tolist(),
        "sent_3d": g["sent_3d"].round(3).tolist(),
        "test_start": e.report["split"]["test_start"],
    }


@app.get("/api/sentiment")
def sentiment(request: Request, ticker: str, limit: int = Query(30, ge=1, le=200)):
    e = engine(request)
    try:
        e.ticker_frame(ticker)
    except KeyError:
        raise HTTPException(404, f"Unknown ticker {ticker}")
    return {"ticker": ticker.upper(), "backend": e.backend_used, "rows": e.sentiment_table(ticker, limit)}


@app.get("/api/predict")
def predict(request: Request, ticker: str, date: str | None = None):
    e = engine(request)
    try:
        if date:
            pd.Timestamp(date)
        return e.predict(ticker, date)
    except KeyError:
        raise HTTPException(404, f"Unknown ticker {ticker}")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get("/api/evaluate")
def evaluate(request: Request):
    return engine(request).report
