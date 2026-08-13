"""Simple look-ahead-safe fusion of sector sentiment and portfolio weights."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _asset_class_for(asset: str) -> str:
    return "Crypto" if str(asset).startswith("CR_") else "Equity"


def _ticker_for_asset(asset: str) -> str:
    text = str(asset)
    if text.startswith(("EQ_", "CR_")):
        return text[3:]
    return text


def latest_sector_signal_before(
    sentiment: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    signal_col: str = "rolling_lagged_sentiment",
) -> pd.Series:
    """Return the latest sector signal strictly before a rebalance date."""
    required = {"date", "sector", signal_col}
    missing = required.difference(sentiment.columns)
    if missing:
        raise ValueError(f"sentiment data is missing required columns: {sorted(missing)}")

    values = sentiment.copy()
    values["date"] = pd.to_datetime(values["date"]).dt.normalize()
    available = values.loc[values["date"] < pd.Timestamp(rebalance_date).normalize()]
    if available.empty:
        return pd.Series(dtype=float)
    latest_rows = available.sort_values("date").groupby("sector", observed=True).tail(1)
    return latest_rows.set_index("sector")[signal_col].astype(float)


def _normalise_with_crypto_cap(weights: pd.Series, crypto_cap: float | None) -> pd.Series:
    out = weights.clip(lower=0.0).astype(float)
    total = float(out.sum())
    if total <= 0:
        return out
    out = out / total

    if crypto_cap is None:
        return out

    crypto_assets = [asset for asset in out.index if _asset_class_for(asset) == "Crypto"]
    equity_assets = [asset for asset in out.index if _asset_class_for(asset) == "Equity"]
    crypto_weight = float(out.loc[crypto_assets].sum()) if crypto_assets else 0.0
    if crypto_weight <= crypto_cap + 1e-12 or not crypto_assets or not equity_assets:
        return out

    out.loc[crypto_assets] *= crypto_cap / crypto_weight
    equity_weight = float(out.loc[equity_assets].sum())
    if equity_weight > 0:
        out.loc[equity_assets] *= (1.0 - crypto_cap) / equity_weight
    return out / float(out.sum())


def apply_sentiment(
    weights: pd.Series,
    sentiment: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    sector_map: dict[str, str] | pd.Series,
    tilt_strength: float = 0.10,
    crypto_cap: float | None = 0.20,
    signal_col: str = "rolling_lagged_sentiment",
) -> pd.Series:
    """Apply a small sector sentiment tilt to equity weights.

    Positive lagged sector sentiment scales equity weights up slightly, negative
    lagged sector sentiment scales them down slightly, and missing sectors are
    treated as neutral. Crypto weights are not sentiment-tilted.
    """
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series indexed by return-panel asset name")

    base = weights.astype(float).copy()
    signals = latest_sector_signal_before(sentiment, rebalance_date, signal_col=signal_col)
    sectors = dict(sector_map)

    multipliers = pd.Series(1.0, index=base.index, dtype=float)
    for asset in base.index:
        if _asset_class_for(asset) != "Equity":
            continue
        ticker = _ticker_for_asset(asset)
        sector = sectors.get(ticker)
        sector_signal = float(signals.get(sector, 0.0)) if sector is not None else 0.0
        multipliers.loc[asset] = max(0.0, 1.0 + tilt_strength * sector_signal)

    tilted = base * multipliers
    if float(tilted.sum()) <= 0 or not np.isfinite(tilted.to_numpy(dtype=float)).all():
        tilted = base
    return _normalise_with_crypto_cap(tilted, crypto_cap)
