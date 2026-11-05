# Stock Trend Predictor — ML + News Sentiment

A small machine learning project that explores whether news sentiment can improve next-day stock trend predictions.

The project compares two models:

- **Price-only model:** uses historical market data to predict whether the next trading day's closing price will be higher.
- **Price + sentiment model:** uses the same market features, along with sentiment scores generated from news headlines.

Both models are trained on the earlier part of the timeline and evaluated on later dates they have not seen during training. The goal is to make the comparison fair. If adding sentiment does not produce a meaningful improvement, the app reports **“Not better”** instead of presenting it as a successful result.

> **Note:** This is an educational project, not financial advice. It does not place trades or provide investment recommendations.

## Getting started

### Requirements

- Python 3.10 or newer (Python 3.12 is recommended)
- pip

### 1. Clone or download the project

Open the project folder in your terminal or VS Code.

### 2. Create and activate a virtual environment

On Windows CMD:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

```bash
python run.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

To run the tests:

```bash
python -m pytest
```

## What the app includes

### Prediction dashboard

Choose a ticker and date to see the estimated probability that the next trading day will close higher. The dashboard displays the combined model alongside the price-only baseline, plus price and sentiment charts and the relevant news headlines.

### Model evaluation

The evaluation view compares the two models using accuracy and F1 score on the held-out period. It also includes time-series cross-validation, an approximate 95% interval for the accuracy difference, and checks intended to catch data leakage.

### Synthetic demo data

On its first run, the project uses a generated dataset containing four fictional tickers, around 900 trading days, and sample headlines. The dashboard labels this clearly as **synthetic demo data**.

The sample data includes a deliberately weak relationship between headline mood and the following day's return. This makes it useful for testing the workflow, but results from this dataset should not be treated as evidence that sentiment predicts real stock movements.

## How it works

The application follows a chronological workflow so that information from the future is not used to predict the past.

| Stage | What happens | Main file |
|---|---|---|
| Data split | Dates are kept in order. The final 30% is held out for testing, with a one-day gap. | `app/model.py` |
| Price features | Calculates returns, volatility, moving-average gaps, and volume changes. | `app/features.py` |
| Sentiment scoring | Converts headlines into daily sentiment scores and caches the results. | `app/sentiment.py` |
| Model comparison | Compares the price-only model with the model that also uses sentiment. | `app/model.py` |
| API and dashboard | Serves predictions, evaluation results, charts, and the disclaimer. | `app/main.py`, `static/index.html` |

The combined model is only considered better when it clears the project's improvement checks: at least a one-point accuracy gain, an estimated interval above zero, and no reduction in F1 score.

## API endpoints

The FastAPI application exposes these endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /api/tickers` | Lists available tickers. |
| `GET /api/data?ticker=` | Returns price data for a ticker. |
| `GET /api/sentiment?ticker=` | Returns sentiment information for a ticker. |
| `GET /api/predict?ticker=&date=` | Returns a prediction for the selected ticker and date. |
| `GET /api/evaluate` | Returns model evaluation results. |
| `/docs` | Opens the interactive API documentation. |

## Data leakage checks

The project includes checks to reduce the risk of accidentally using future information during training or evaluation:

1. The train/test split follows date order rather than randomly shuffling observations, and includes a gap day.
2. Prices and headlines after a chosen cutoff are changed to check that features calculated on or before the cutoff remain unchanged.
3. A model trained on shuffled labels is checked against a baseline that always predicts the most common class.

The test suite also checks that the second leakage check catches a feature that incorrectly uses information from the following day.

## Using your own data

You can replace the sample dataset with two CSV files in the `data/` directory. Remove the generated-data marker and old sentiment cache so the app does not keep using the demo data or stale scores:

- `data/SYNTHETIC_DATA.txt`
- `data/sentiment_cache.json`

The CSV files should use the following columns.

**`prices.csv`**

```csv
date,ticker,open,high,low,close,volume
```

**`news.csv`**

```csv
date,ticker,headline
```

A headline dated on day D is assigned to the first trading day on or after D. This assumes the headline was available before that day's market close. If it was published after the close, date it to the next trading day to avoid introducing future information.

## Configuration

The application can be configured with environment variables:

| Variable | Default | Description |
|---|---|---|
| `SENTIMENT_BACKEND` | `lexicon` | Uses an offline word list by default. Set to `llm` to score headlines with Claude. |
| `SENTIMENT_MODEL` | `claude-haiku-5-5` | Model name used by the optional LLM backend. |
| `MODEL_KIND` | `logreg` | Selects `logreg` or `rf` (a small random forest). |
| `DATA_DIR` | `./data` | Directory used for input data and the sentiment cache. |

## Limitations

- The demo dataset is synthetic. It is only intended to demonstrate the pipeline.
- The accuracy-gain interval uses a simple bootstrap that treats test observations as independent, so it is approximate and does not account for all time-series effects.
- Predicting the direction of a real stock's next-day movement is difficult; results may be close to chance.
- Models are trained at application startup using the training period only. They are not refitted on the held-out test period.
- The optional LLM sentiment path requires an API key and may incur API costs.

## Project structure

```text
stock_sentiment_mvp/
├── app/
│   ├── main.py          # FastAPI routes
│   ├── model.py         # Training, splitting, and evaluation
│   ├── features.py      # Price feature engineering
│   ├── sentiment.py     # Headline sentiment scoring
│   ├── leakage.py       # Data leakage checks
│   ├── data.py          # Data loading
│   ├── sample_data.py   # Synthetic demo data generation
│   └── config.py        # Application settings
├── data/                # Price data, news, and sentiment cache
├── static/
│   └── index.html       # Dashboard and evaluation UI
├── tests/
│   └── test_mvp.py      # Project tests
├── requirements.txt
└── run.py               # Application entry point
```

---

Built as a learning project to explore time-series machine learning, news sentiment, model evaluation, and the importance of avoiding data leakage.
