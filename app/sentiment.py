"""Headline sentiment: one score in [-1, 1] per headline, cached on disk.

Two backends:
  lexicon  offline word list (default)
  llm      Claude via the Anthropic API (set SENTIMENT_BACKEND=llm)
Scores are cached per (backend, headline), so each unique headline is scored once.
"""
import hashlib
import json
import logging
import re

from . import config

log = logging.getLogger(__name__)

POS_WORDS = {
    "beats", "beat", "surges", "surge", "raises", "strong", "upgrade", "upgrades", "upgraded",
    "record", "growth", "wins", "win", "rally", "rallies", "profit", "profits", "jump", "jumps",
    "expands", "expand", "cheer", "soar", "soars", "gains", "gain", "boost", "boosts",
    "outperform", "bullish", "improves", "improved", "breakthrough", "approval", "approved",
    "success", "exceeds",
}
NEG_WORDS = {
    "cuts", "cut", "weak", "weaker", "misses", "miss", "downgrade", "downgrades", "downgraded",
    "slowing", "slow", "slows", "investigation", "concerns", "concern", "layoffs", "loss",
    "losses", "tumble", "tumbles", "recall", "falls", "fall", "drops", "plunge", "plunges",
    "lawsuit", "fraud", "bearish", "decline", "declines", "warning", "warns", "probe",
    "bankruptcy", "default", "fails", "failed",
}
NEGATORS = {"not", "no", "never", "without"}


def lexicon_score(text: str) -> float:
    words = re.findall(r"[a-z']+", text.lower())
    pos = neg = 0
    for i, w in enumerate(words):
        sign = 1 if w in POS_WORDS else -1 if w in NEG_WORDS else 0
        if sign and i > 0 and words[i - 1] in NEGATORS:
            sign = -sign
        pos += sign > 0
        neg += sign < 0
    return round((pos - neg) / (pos + neg + 1), 3)


def llm_scores(headlines: list[str]) -> list[float]:
    import anthropic  # optional dependency

    client = anthropic.Anthropic()
    out: list[float] = []
    for i in range(0, len(headlines), 25):
        chunk = headlines[i:i + 25]
        numbered = "\n".join(f"{j + 1}. {h}" for j, h in enumerate(chunk))
        prompt = (
            "Rate the likely short-term stock-price sentiment of each headline from -1.0 "
            "(very negative) to 1.0 (very positive); 0 means neutral. Reply with ONLY a JSON "
            f"array of {len(chunk)} numbers, in the same order as the headlines.\n\n{numbered}"
        )
        msg = client.messages.create(
            model=config.SENTIMENT_MODEL, max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        values = json.loads(re.search(r"\[.*\]", msg.content[0].text, re.S).group(0))
        if len(values) != len(chunk):
            raise ValueError("LLM returned the wrong number of scores")
        out += [max(-1.0, min(1.0, float(v))) for v in values]
    return out


def _cache_path():
    return config.DATA_DIR / "sentiment_cache.json"


def _key(backend: str, headline: str) -> str:
    return f"{backend}:{hashlib.sha1(headline.encode()).hexdigest()}"


def score_headlines(headlines, backend: str):
    """Return ({headline: score}, backend_label_actually_used)."""
    path = _cache_path()
    cache = json.loads(path.read_text()) if path.exists() else {}
    unique = sorted(set(headlines))
    missing = [h for h in unique if _key(backend, h) not in cache]
    used = backend

    if missing:
        try:
            values = llm_scores(missing) if backend == "llm" else [lexicon_score(h) for h in missing]
            cache.update({_key(backend, h): v for h, v in zip(missing, values)})
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache))
        except Exception as exc:  # LLM unavailable: fall back, but say so and don't cache it
            log.warning("LLM scoring failed (%s); using lexicon for uncached headlines", exc)
            used = "lexicon (LLM unavailable)"
            return {h: cache.get(_key(backend, h), lexicon_score(h)) for h in unique}, used

    return {h: cache[_key(backend, h)] for h in unique}, used
