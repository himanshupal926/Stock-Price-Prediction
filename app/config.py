"""Central settings. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))

# "lexicon" = offline word-list scorer (default, free, deterministic)
# "llm"     = Claude scores each headline (needs `pip install anthropic` + ANTHROPIC_API_KEY)
SENTIMENT_BACKEND = os.getenv("SENTIMENT_BACKEND", "lexicon").lower()
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "claude-haiku-5-5")

# "logreg" (default, simple) or "rf" (small random forest)
MODEL_KIND = os.getenv("MODEL_KIND", "logreg").lower()

TEST_FRACTION = 0.30   # last 30% of dates are held out, never shuffled
MIN_GAIN = 0.01        # combined model must beat the baseline by >= 1 point of accuracy ...
SEED = 42              # ... and the bootstrap CI of the gain must sit above zero

DISCLAIMER = (
    "Educational demo only - not financial advice. Predictions are probabilistic, "
    "often wrong, and must not be used to make trading decisions."
)
