"""Headline sentiment scoring and lagged sector sentiment signals.

The text model is run only in the batch build script. The Streamlit app should
read the CSV outputs and never score headlines at runtime.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

TEXT_COLUMNS = ("title", "headline", "text")
TOKEN_PATTERN = re.compile(r"[a-z][a-z'-]{2,}")
STOPWORDS = {
    "about",
    "above",
    "according",
    "after",
    "again",
    "against",
    "ahead",
    "also",
    "amid",
    "among",
    "and",
    "announces",
    "announced",
    "announcing",
    "are",
    "around",
    "before",
    "below",
    "between",
    "but",
    "can",
    "could",
    "daily",
    "day",
    "days",
    "during",
    "for",
    "from",
    "has",
    "have",
    "here",
    "how",
    "into",
    "its",
    "last",
    "may",
    "might",
    "month",
    "months",
    "more",
    "most",
    "new",
    "news",
    "next",
    "not",
    "now",
    "over",
    "per",
    "q1",
    "q2",
    "q3",
    "q4",
    "report",
    "reports",
    "said",
    "says",
    "sector",
    "shares",
    "stock",
    "stocks",
    "than",
    "that",
    "the",
    "their",
    "this",
    "through",
    "today",
    "update",
    "upon",
    "via",
    "was",
    "week",
    "weeks",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "why",
    "will",
    "with",
    "year",
    "years",
}
MONTH_WORDS = {
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}
TICKER_LIKE_WORDS = {
    "aapl",
    "abbv",
    "abt",
    "adbe",
    "adm",
    "agnc",
    "alpn",
    "amgn",
    "aytu",
    "cat",
    "cht",
    "cof",
    "cost",
    "cvx",
    "dow",
    "dxc",
    "epzm",
    "eqwl",
    "hbi",
    "hst",
    "iaa",
    "ibio",
    "itw",
    "jnj",
    "ko",
    "logi",
    "mdt",
    "mnk",
    "mrk",
    "msft",
    "myl",
    "nobl",
    "nrg",
    "nke",
    "onct",
    "pep",
    "pld",
    "pypl",
    "qqq",
    "rare",
    "run",
    "seel",
    "slb",
    "swx",
    "tgt",
    "vici",
    "xom",
}
COMPANY_WORDS = {
    "abbott",
    "abbvie",
    "adobe",
    "allergan",
    "amazon",
    "apple",
    "boeing",
    "chevron",
    "comcast",
    "disney",
    "exxon",
    "gilead",
    "goldman",
    "intel",
    "merck",
    "morgan",
    "netflix",
    "nike",
    "nvidia",
    "pfizer",
    "qualcomm",
    "starbucks",
    "visa",
    "walmart",
}
FINANCE_LEXICON_SEED = {
    "beat": (0.45, "positive", "Earnings or sales beating expectations is usually favourable."),
    "beats": (0.45, "positive", "Plural form of beat; commonly signals results above estimates."),
    "raises": (0.35, "positive", "Often appears in price-target or guidance increases."),
    "raised": (0.30, "positive", "Often appears in price-target or guidance increases."),
    "upgrade": (0.45, "positive", "Analyst upgrades are favourable recommendation changes."),
    "upgraded": (0.45, "positive", "Past-tense analyst upgrade signal."),
    "outperform": (0.35, "positive", "Positive analyst recommendation language."),
    "bullish": (0.45, "positive", "Directly positive market-view language."),
    "rally": (0.35, "positive", "Describes a favourable price move."),
    "rallies": (0.35, "positive", "Describes favourable price moves."),
    "rebound": (0.25, "positive", "Describes recovery after weakness."),
    "rebounds": (0.25, "positive", "Describes recovery after weakness."),
    "surge": (0.35, "positive", "Describes a strong positive move."),
    "surges": (0.35, "positive", "Describes strong positive moves."),
    "inflow": (0.25, "positive", "ETF or fund inflows indicate demand."),
    "inflows": (0.25, "positive", "ETF or fund inflows indicate demand."),
    "buy": (
        0.20,
        "positive",
        "Common positive analyst/action word; score kept low because it can be generic.",
    ),
    "buys": (
        0.20,
        "positive",
        "Institutional or insider buying can be favourable; score kept low.",
    ),
    "buying": (
        0.20,
        "positive",
        "Buying can be favourable; score kept low because context varies.",
    ),
    "tailwinds": (0.25, "positive", "Business tailwinds usually describe supportive conditions."),
    "dividend": (
        0.15,
        "positive",
        "Dividend language can indicate shareholder return; conservative score.",
    ),
    "dividends": (
        0.15,
        "positive",
        "Dividend language can indicate shareholder return; conservative score.",
    ),
    "yield": (
        0.15,
        "positive",
        "Income yield can be attractive; conservative because context varies.",
    ),
    "yields": (
        0.15,
        "positive",
        "Income yield can be attractive; conservative because context varies.",
    ),
    "downgrade": (-0.45, "negative", "Analyst downgrades are unfavourable recommendation changes."),
    "downgraded": (-0.45, "negative", "Past-tense analyst downgrade signal."),
    "downgrades": (-0.45, "negative", "Plural analyst downgrade signal."),
    "underperform": (-0.35, "negative", "Negative analyst recommendation language."),
    "underweight": (-0.25, "negative", "Negative analyst allocation/recommendation language."),
    "bearish": (-0.45, "negative", "Directly negative market-view language."),
    "outflow": (-0.25, "negative", "ETF or fund outflows indicate weaker demand."),
    "outflows": (-0.25, "negative", "ETF or fund outflows indicate weaker demand."),
    "sell": (-0.25, "negative", "Sell recommendation/action word; score kept moderate."),
    "sells": (-0.25, "negative", "Selling can be unfavourable; score kept moderate."),
    "selling": (-0.25, "negative", "Selling can be unfavourable; score kept moderate."),
    "plunge": (-0.45, "negative", "Describes a sharp negative move."),
    "plunges": (-0.45, "negative", "Describes sharp negative moves."),
    "slump": (-0.35, "negative", "Describes a negative price or business move."),
    "slumps": (-0.35, "negative", "Describes negative price or business moves."),
    "decline": (-0.25, "negative", "Describes deterioration or falling values."),
    "declines": (-0.25, "negative", "Describes deterioration or falling values."),
    "declined": (-0.25, "negative", "Describes deterioration or falling values."),
    "headwinds": (-0.25, "negative", "Business headwinds describe adverse conditions."),
    "default": (-0.50, "negative", "Credit default language is clearly adverse."),
    "defaults": (-0.50, "negative", "Credit default language is clearly adverse."),
    "writedown": (-0.45, "negative", "Write-downs usually signal asset impairment."),
    "layoff": (-0.35, "negative", "Layoffs suggest business stress or cost pressure."),
    "layoffs": (-0.35, "negative", "Layoffs suggest business stress or cost pressure."),
}
POSITIVE_WORDS = {
    "beat",
    "beats",
    "benefit",
    "boost",
    "bullish",
    "buy",
    "gain",
    "gains",
    "growth",
    "higher",
    "improve",
    "improves",
    "outperform",
    "positive",
    "profit",
    "rally",
    "record",
    "rise",
    "rises",
    "strong",
    "upgrade",
    "upside",
}
NEGATIVE_WORDS = {
    "bearish",
    "concern",
    "cut",
    "decline",
    "downgrade",
    "drop",
    "falls",
    "fear",
    "loss",
    "lower",
    "miss",
    "misses",
    "negative",
    "risk",
    "sell",
    "slump",
    "weak",
    "warning",
    "worse",
}


def _text_column(panel: pd.DataFrame) -> str:
    for column in TEXT_COLUMNS:
        if column in panel.columns:
            return column
    raise ValueError(f"headline data needs one text column from {TEXT_COLUMNS}")


def _normalise_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "sector"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"headline data is missing required columns: {sorted(missing)}")

    text_col = _text_column(panel)
    out = panel.copy()
    out["headline_date"] = (
        pd.to_datetime(out["date"], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.normalize()
    )
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["sector"] = out["sector"].astype(str).str.strip()
    out["headline"] = out[text_col].fillna("").astype(str)
    out = out.loc[out["headline_date"].notna() & out["headline"].str.strip().ne("")]
    out = out.drop_duplicates(subset=["ticker", "headline_date", "headline"])
    return out.reset_index(drop=True)


def align_headlines_to_trading_dates(
    panel: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Roll each headline to the next available equity trading date.

    A headline aligned to Monday is first usable for Tuesday's trade because the
    portfolio signal is shifted by one trading day after this alignment.
    """
    out = _normalise_headlines(panel)
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).sort_values().unique()
    if calendar.empty:
        raise ValueError("trading_calendar must contain at least one date")

    headline_dates = out["headline_date"].to_numpy(dtype="datetime64[ns]")
    calendar_values = calendar.to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(calendar_values, headline_dates, side="left")
    keep = positions < len(calendar_values)
    out = out.loc[keep].copy()
    out["date"] = pd.to_datetime(calendar_values[positions[keep]])
    return out.reset_index(drop=True)


def _vader_scorer():
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
    except Exception:
        return None

    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        try:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
        except Exception:
            return None


def vader_lexicon_terms() -> set[str]:
    """Return the standard VADER lexicon terms when VADER is available."""
    scorer = _vader_scorer()
    return set(scorer.lexicon) if scorer is not None else set()


def finance_tokens(text: str) -> list[str]:
    """Tokenise headline text for explainable finance-lexicon checks."""
    return [token.strip("'-") for token in TOKEN_PATTERN.findall(str(text).lower())]


def _candidate_exclusions(extra_stopwords: set[str] | None = None) -> set[str]:
    exclusions = set(STOPWORDS) | MONTH_WORDS | TICKER_LIKE_WORDS | COMPANY_WORDS
    if extra_stopwords:
        exclusions.update(str(word).lower() for word in extra_stopwords)
    return exclusions


def finance_lexicon_candidate_terms(
    headlines: pd.DataFrame,
    max_terms: int = 250,
    min_frequency: int = 20,
    extra_stopwords: set[str] | None = None,
) -> pd.DataFrame:
    """Rank frequent headline words that are missing from standard VADER."""
    normalised = _normalise_headlines(headlines)
    vader_terms = vader_lexicon_terms()
    exclusions = _candidate_exclusions(extra_stopwords)
    selected_terms = set(FINANCE_LEXICON_SEED)

    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for headline in normalised["headline"]:
        seen_in_headline = set()
        for token in finance_tokens(headline):
            if (
                token in seen_in_headline
                or token in vader_terms
                or token in exclusions
                or token.isnumeric()
            ):
                continue
            counts[token] += 1
            seen_in_headline.add(token)
            if len(examples[token]) < 2:
                examples[token].append(str(headline))

    rows = []
    for term, frequency in counts.most_common():
        if frequency < min_frequency:
            break
        example_values = examples[term]
        rows.append(
            {
                "term": term,
                "frequency": int(frequency),
                "example_headline_1": example_values[0] if example_values else "",
                "example_headline_2": example_values[1] if len(example_values) > 1 else "",
                "in_vader_lexicon": term in vader_terms,
                "selected_for_custom_lexicon": term in selected_terms,
            }
        )
        if len(rows) >= max_terms:
            break
    return pd.DataFrame(rows)


def custom_finance_lexicon(headlines: pd.DataFrame) -> pd.DataFrame:
    """Create a small custom finance lexicon filtered to the actual headlines."""
    normalised = _normalise_headlines(headlines)
    present_terms = {token for text in normalised["headline"] for token in finance_tokens(text)}
    vader_terms = vader_lexicon_terms()

    rows = []
    for term, (score, direction, reason) in sorted(FINANCE_LEXICON_SEED.items()):
        if term not in present_terms or term in vader_terms:
            continue
        rows.append(
            {
                "term": term,
                "finance_sentiment_score": float(score),
                "direction": direction,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _fallback_compound(text: str) -> float:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    positives = sum(word in POSITIVE_WORDS for word in words)
    negatives = sum(word in NEGATIVE_WORDS for word in words)
    total = positives + negatives
    if total == 0:
        return 0.0
    return float(np.clip((positives - negatives) / total, -1.0, 1.0))


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Score each headline, preserving casing and punctuation for VADER.

    VADER is used when available. The small word-list fallback exists only so
    the batch build remains reproducible in minimal environments.
    """
    out = _normalise_headlines(panel) if "headline" not in panel.columns else panel.copy()
    scorer = _vader_scorer()
    if scorer is not None:
        out["compound"] = out["headline"].map(lambda text: scorer.polarity_scores(text)["compound"])
        out["sentiment_model"] = "vader"
    else:
        out["compound"] = out["headline"].map(_fallback_compound)
        out["sentiment_model"] = "fallback_word_list"

    out["is_positive"] = out["compound"] > 0.05
    out["is_negative"] = out["compound"] < -0.05
    out["is_neutral"] = ~(out["is_positive"] | out["is_negative"])
    return out


def score_headlines_finance_adjusted(
    panel: pd.DataFrame,
    finance_lexicon: pd.DataFrame,
    adjustment_strength: float = 0.20,
) -> pd.DataFrame:
    """Score headlines with VADER plus a conservative finance-term adjustment."""
    out = score_headlines(panel).rename(columns={"compound": "vader_compound"})
    lexicon = dict(
        zip(
            finance_lexicon["term"].astype(str),
            finance_lexicon["finance_sentiment_score"].astype(float),
            strict=True,
        )
    )

    matches = []
    adjustments = []
    adjusted_scores = []
    for headline, vader_score in zip(out["headline"], out["vader_compound"], strict=True):
        tokens = finance_tokens(headline)
        matched_terms = sorted({token for token in tokens if token in lexicon})
        average_score = (
            float(np.mean([lexicon[token] for token in matched_terms]))
            if matched_terms
            else 0.0
        )
        adjustment = adjustment_strength * average_score
        matches.append("; ".join(matched_terms))
        adjustments.append(float(adjustment))
        adjusted_scores.append(float(np.clip(float(vader_score) + adjustment, -1.0, 1.0)))

    out["finance_term_matches"] = matches
    out["finance_adjustment"] = adjustments
    out["finance_adjusted_score"] = adjusted_scores
    out["is_positive_finance_adjusted"] = out["finance_adjusted_score"] > 0.05
    out["is_negative_finance_adjusted"] = out["finance_adjusted_score"] < -0.05
    out["is_neutral_finance_adjusted"] = ~(
        out["is_positive_finance_adjusted"] | out["is_negative_finance_adjusted"]
    )
    return out


def sector_sentiment_index(scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate headline scores into a daily sector sentiment index.

    The main index equal-weights ticker-day sentiment within each sector-date.
    Headline counts and positive/negative/neutral shares are retained for
    transparency.
    """
    required = {"date", "ticker", "sector", "compound"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"sentiment scores are missing required columns: {sorted(missing)}")

    scored = scores.copy()
    scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()
    headline_stats = (
        scored.groupby(["date", "sector"], as_index=False)
        .agg(
            headline_count=("compound", "size"),
            mean_compound=("compound", "mean"),
            positive_share=("is_positive", "mean"),
            negative_share=("is_negative", "mean"),
            neutral_share=("is_neutral", "mean"),
        )
    )
    ticker_day = (
        scored.groupby(["date", "sector", "ticker"], as_index=False)["compound"]
        .mean()
        .rename(columns={"compound": "ticker_day_compound"})
    )
    sector_index = (
        ticker_day.groupby(["date", "sector"], as_index=False)["ticker_day_compound"]
        .mean()
        .rename(columns={"ticker_day_compound": "sentiment_index"})
    )
    out = headline_stats.merge(sector_index, on=["date", "sector"], how="left")
    return out.sort_values(["date", "sector"]).reset_index(drop=True)


def sector_sentiment_index_finance_adjusted(scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate VADER and finance-adjusted scores into sector-date indexes."""
    required = {"date", "ticker", "sector", "vader_compound", "finance_adjusted_score"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"finance-adjusted scores are missing required columns: {sorted(missing)}")

    scored = scores.copy()
    scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()
    headline_stats = (
        scored.groupby(["date", "sector"], as_index=False)
        .agg(
            headline_count=("finance_adjusted_score", "size"),
            finance_adjustment_mean=("finance_adjustment", "mean"),
            positive_share_finance_adjusted=("is_positive_finance_adjusted", "mean"),
            negative_share_finance_adjusted=("is_negative_finance_adjusted", "mean"),
            neutral_share_finance_adjusted=("is_neutral_finance_adjusted", "mean"),
        )
    )
    ticker_day = (
        scored.groupby(["date", "sector", "ticker"], as_index=False)
        .agg(
            vader_ticker_day=("vader_compound", "mean"),
            finance_adjusted_ticker_day=("finance_adjusted_score", "mean"),
        )
    )
    sector_index = (
        ticker_day.groupby(["date", "sector"], as_index=False)
        .agg(
            vader_sentiment_index=("vader_ticker_day", "mean"),
            finance_adjusted_sentiment_index=("finance_adjusted_ticker_day", "mean"),
        )
    )
    out = headline_stats.merge(sector_index, on=["date", "sector"], how="left")
    return out.sort_values(["date", "sector"]).reset_index(drop=True)


def build_sector_sentiment_index(
    headlines: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create the daily sector sentiment index from raw headline data."""
    aligned = align_headlines_to_trading_dates(headlines, trading_calendar)
    scores = score_headlines(aligned)
    return sector_sentiment_index(scores)


def build_finance_adjusted_sentiment_outputs(
    headlines: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build candidate, lexicon, headline-level and sector-level adjusted outputs."""
    candidate_terms = finance_lexicon_candidate_terms(headlines)
    custom_lexicon = custom_finance_lexicon(headlines)
    aligned = align_headlines_to_trading_dates(headlines, trading_calendar)
    adjusted_scores = score_headlines_finance_adjusted(aligned, custom_lexicon)
    adjusted_index = sector_sentiment_index_finance_adjusted(adjusted_scores)
    return candidate_terms, custom_lexicon, adjusted_scores, adjusted_index


def build_sector_sentiment_signal(
    sentiment_index: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    rolling_window: int = 21,
) -> pd.DataFrame:
    """Create a one-trading-day-lagged sector signal for portfolio weights."""
    required = {"date", "sector", "sentiment_index"}
    missing = required.difference(sentiment_index.columns)
    if missing:
        raise ValueError(f"sentiment_index is missing required columns: {sorted(missing)}")

    dates = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).sort_values().unique()
    sectors = sorted(sentiment_index["sector"].dropna().unique())
    grid = pd.MultiIndex.from_product([dates, sectors], names=["date", "sector"]).to_frame(
        index=False
    )

    values = sentiment_index.copy()
    values["date"] = pd.to_datetime(values["date"]).dt.normalize()
    signal = grid.merge(
        values[["date", "sector", "sentiment_index"]],
        on=["date", "sector"],
        how="left",
    )
    signal["sentiment_index"] = signal["sentiment_index"].fillna(0.0)
    signal = signal.sort_values(["sector", "date"]).reset_index(drop=True)
    grouped = signal.groupby("sector", observed=True)["sentiment_index"]
    signal["lagged_sentiment_index"] = grouped.shift(1).fillna(0.0)
    signal["rolling_lagged_sentiment"] = (
        signal.groupby("sector", observed=True)["lagged_sentiment_index"]
        .rolling(window=rolling_window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    return signal.sort_values(["date", "sector"]).reset_index(drop=True)


def build_finance_adjusted_sentiment_signal(
    finance_adjusted_index: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    rolling_window: int = 21,
) -> pd.DataFrame:
    """Create a lagged signal from the finance-adjusted sector sentiment index."""
    required = {"date", "sector", "finance_adjusted_sentiment_index"}
    missing = required.difference(finance_adjusted_index.columns)
    if missing:
        raise ValueError(
            f"finance_adjusted_index is missing required columns: {sorted(missing)}"
        )

    signal_input = finance_adjusted_index.rename(
        columns={"finance_adjusted_sentiment_index": "sentiment_index"}
    )[["date", "sector", "sentiment_index"]]
    signal = build_sector_sentiment_signal(
        signal_input,
        trading_calendar,
        rolling_window=rolling_window,
    )
    signal = signal.rename(
        columns={
            "sentiment_index": "finance_adjusted_sentiment_index",
            "lagged_sentiment_index": "lagged_finance_adjusted_sentiment",
            "rolling_lagged_sentiment": "rolling_lagged_finance_sentiment",
        }
    )
    return signal
