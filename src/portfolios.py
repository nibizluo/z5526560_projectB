"""Systematic fund construction and walk-forward backtesting for Part B."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

METHOD_EQUAL_WEIGHT = "equal_weight"
METHOD_MIN_VARIANCE = "min_variance"
METHOD_MAX_SHARPE = "max_sharpe"


def asset_class_for(asset: str) -> str:
    """Infer asset class from the column prefix used in the return panels."""
    if asset.startswith("CR_"):
        return "Crypto"
    return "Equity"


def display_asset_name(asset: str) -> str:
    """Remove internal return-panel prefixes for tables and app output."""
    if asset.startswith(("EQ_", "CR_")):
        return asset[3:]
    return asset


def equal_weight(assets: list[str]) -> pd.Series:
    """Long-only equal weights across all included assets."""
    if not assets:
        raise ValueError("at least one asset is required")
    return pd.Series(1.0 / len(assets), index=assets, dtype=float)


def _covariance_matrix(window: pd.DataFrame) -> np.ndarray:
    """Estimate a numerically stable covariance matrix from past returns."""
    cov = window.cov().to_numpy(dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = (cov + cov.T) / 2
    cov += np.eye(cov.shape[0]) * 1e-8
    return cov


def _constraints(assets: list[str], crypto_cap: float | None = None) -> list[dict]:
    constraints: list[dict] = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
    ]
    if crypto_cap is not None:
        crypto_idx = np.array([asset_class_for(asset) == "Crypto" for asset in assets])
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights, mask=crypto_idx: crypto_cap - np.sum(weights[mask]),
            }
        )
    return constraints


def _solve_weights(
    window: pd.DataFrame,
    method: str,
    crypto_cap: float | None = None,
) -> pd.Series:
    """Solve one rebalance portfolio using only the supplied past window."""
    assets = list(window.columns)
    initial = equal_weight(assets)

    if method == METHOD_EQUAL_WEIGHT:
        if crypto_cap is None:
            return initial
        return _cap_crypto_equal_weight(initial, crypto_cap)

    window = window.fillna(0.0)
    cov = _covariance_matrix(window)
    mean_returns = window.mean().to_numpy(dtype=float)
    bounds = [(0.0, 1.0)] * len(assets)

    def variance_objective(weights: np.ndarray) -> float:
        return float(weights @ cov @ weights)

    def negative_sharpe_objective(weights: np.ndarray) -> float:
        portfolio_return = float(weights @ mean_returns)
        portfolio_volatility = float(np.sqrt(weights @ cov @ weights))
        if portfolio_volatility <= 0:
            return 1e6
        return -portfolio_return / portfolio_volatility

    if method == METHOD_MIN_VARIANCE:
        objective = variance_objective
    elif method == METHOD_MAX_SHARPE:
        objective = negative_sharpe_objective
    else:
        raise ValueError(f"unknown portfolio method: {method}")

    result = minimize(
        objective,
        initial.to_numpy(),
        method="SLSQP",
        bounds=bounds,
        constraints=_constraints(assets, crypto_cap),
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    if not result.success or not np.all(np.isfinite(result.x)):
        warnings.warn(
            f"Optimisation failed for method={method}; falling back to equal weight. "
            f"Solver message: {result.message}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _cap_crypto_equal_weight(initial, crypto_cap) if crypto_cap is not None else initial

    weights = np.clip(result.x, 0.0, 1.0)
    weights = weights / weights.sum()
    series = pd.Series(weights, index=assets, dtype=float)
    if crypto_cap is not None and _crypto_weight(series) > crypto_cap + 1e-6:
        warnings.warn(
            "Optimiser returned weights above the crypto cap; applying capped "
            "equal-weight fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _cap_crypto_equal_weight(initial, crypto_cap)
    return series


def _crypto_weight(weights: pd.Series) -> float:
    crypto_assets = [asset for asset in weights.index if asset_class_for(asset) == "Crypto"]
    return float(weights.loc[crypto_assets].sum()) if crypto_assets else 0.0


def _cap_crypto_equal_weight(weights: pd.Series, crypto_cap: float) -> pd.Series:
    """Apply a simple aggregate crypto cap to equal weights."""
    crypto_assets = [asset for asset in weights.index if asset_class_for(asset) == "Crypto"]
    equity_assets = [asset for asset in weights.index if asset_class_for(asset) == "Equity"]
    if not crypto_assets or not equity_assets:
        return weights

    out = pd.Series(0.0, index=weights.index, dtype=float)
    out.loc[crypto_assets] = crypto_cap / len(crypto_assets)
    out.loc[equity_assets] = (1.0 - crypto_cap) / len(equity_assets)
    return out


def monthly_rebalance_dates(returns: pd.DataFrame, estimation_window: int) -> pd.DatetimeIndex:
    """First available date in each month after the estimation window."""
    dates = pd.DatetimeIndex(returns.index).sort_values()
    if len(dates) <= estimation_window:
        raise ValueError("return panel is shorter than the estimation window")
    live_dates = dates[estimation_window:]
    return (
        pd.Series(live_dates, index=live_dates)
        .groupby(live_dates.to_period("M"))
        .first()
        .pipe(pd.DatetimeIndex)
    )


def oos_backtest(
    returns: pd.DataFrame,
    fund_name: str,
    method: str = METHOD_MIN_VARIANCE,
    estimation_window: int = 252,
    crypto_cap: float | None = None,
    weight_adjuster=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward out-of-sample backtest with monthly rebalancing.

    At each rebalance date, the optimiser sees only rows before that date. The
    resulting weights are applied from that live date until the next rebalance.
    """
    panel = returns.copy()
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index().apply(pd.to_numeric, errors="coerce")
    panel = panel.dropna(how="all")
    panel = panel.loc[:, panel.notna().any()]

    rebalances = monthly_rebalance_dates(panel, estimation_window)
    rebalance_set = set(rebalances)
    rows = []
    weight_rows = []
    current_weights: pd.Series | None = None

    for pos, date in enumerate(panel.index):
        if pos < estimation_window:
            continue

        if date in rebalance_set or current_weights is None:
            past_window = panel.iloc[pos - estimation_window:pos]
            active_assets = list(past_window.columns[past_window.notna().any()])
            current_weights = _solve_weights(
                past_window[active_assets],
                method=method,
                crypto_cap=crypto_cap,
            )
            if weight_adjuster is not None:
                adjusted = weight_adjuster(current_weights.copy(), date)
                adjusted = pd.Series(adjusted, index=current_weights.index, dtype=float)
                adjusted = adjusted.reindex(current_weights.index).fillna(0.0).clip(lower=0.0)
                if float(adjusted.sum()) > 0:
                    current_weights = adjusted / float(adjusted.sum())
            for asset, weight in current_weights.items():
                weight_rows.append(
                    {
                        "rebalance_date": date,
                        "fund_name": fund_name,
                        "asset": display_asset_name(asset),
                        "weight": float(weight),
                        "asset_class": asset_class_for(asset),
                    }
                )

        live_returns = panel.loc[date, current_weights.index].fillna(0.0)
        daily_return = float(live_returns @ current_weights)
        rows.append(
            {
                "date": date,
                "fund_name": fund_name,
                "daily_return": daily_return,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Calculate fund performance metrics from out-of-sample daily returns."""
    returns = pd.Series(daily_returns).dropna().astype(float)
    if returns.empty:
        raise ValueError("daily_returns must contain at least one observation")

    growth = (1.0 + returns).cumprod()
    cumulative_return = float(growth.iloc[-1] - 1.0)
    annualised_return = float(growth.iloc[-1] ** (periods_per_year / len(returns)) - 1.0)
    annualised_volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    daily_volatility = float(returns.std(ddof=1))
    sharpe_ratio = (
        float(np.sqrt(periods_per_year) * returns.mean() / daily_volatility)
        if daily_volatility > 0
        else np.nan
    )
    drawdown = growth / growth.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    return {
        "cumulative_return": cumulative_return,
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }


def weight_summary(weights: pd.DataFrame) -> dict:
    """Average equity and crypto allocation across rebalance dates."""
    if weights.empty:
        return {
            "average_equity_weight": np.nan,
            "average_crypto_weight": np.nan,
            "rebalance_count": 0,
        }

    by_rebalance = (
        weights.groupby(["rebalance_date", "asset_class"], as_index=False)["weight"]
        .sum()
        .pivot(index="rebalance_date", columns="asset_class", values="weight")
        .fillna(0.0)
    )
    return {
        "average_equity_weight": float(by_rebalance.get("Equity", pd.Series(0.0)).mean()),
        "average_crypto_weight": float(by_rebalance.get("Crypto", pd.Series(0.0)).mean()),
        "rebalance_count": int(by_rebalance.shape[0]),
    }
