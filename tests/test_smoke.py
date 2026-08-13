"""Smoke test: imports resolve and the data loads.

    python tests/test_smoke.py
"""
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import data_access
from src.portfolios import performance_metrics
from src.sentiment import custom_finance_lexicon, vader_lexicon_terms
from streamlit_app import calculate_portfolio_metrics, fund_return_path, latest_fund_holdings


def test_imports():
    assert hasattr(data_access, "load_equity_prices")


def test_data_loads():
    eq = data_access.load_equity_prices()
    assert eq.shape[0] > 0
    assert {"ticker", "date", "adjClose", "sector"}.issubset(eq.columns)


def test_performance_metrics_uses_standard_zero_rf_sharpe():
    returns = [0.02, -0.01, 0.015, 0.005]
    metrics = performance_metrics(returns, periods_per_year=252)
    expected = (252 ** 0.5) * (
        sum(returns) / len(returns)
    ) / pd.Series(returns).std(ddof=1)
    assert abs(metrics["sharpe_ratio"] - expected) < 1e-12


def test_app_custom_allocation_metrics_uses_standard_zero_rf_sharpe():
    returns = [0.02, -0.01, 0.015, 0.005]
    metrics = calculate_portfolio_metrics(returns, periods_per_year=252)
    expected = (252 ** 0.5) * (
        sum(returns) / len(returns)
    ) / pd.Series(returns).std(ddof=1)
    assert abs(metrics["sharpe_ratio"] - expected) < 1e-12


def test_custom_finance_lexicon_filters_to_non_vader_terms():
    headlines = pd.DataFrame(
        {
            "date": ["2021-01-04", "2021-01-04"],
            "ticker": ["AAA", "BBB"],
            "sector": ["Tech", "Energy"],
            "title": ["AAA beats estimates after analyst upgrade", "BBB plunges after downgrade"],
        }
    )
    lexicon = custom_finance_lexicon(headlines)
    vader_terms = vader_lexicon_terms()
    assert {"beats", "upgrade", "plunges", "downgrade"}.issubset(set(lexicon["term"]))
    assert not set(lexicon["term"]).intersection(vader_terms)


def test_fund_fact_sheet_return_path_calculates_growth_and_drawdown():
    returns = pd.DataFrame(
        {
            "date": ["2021-01-02", "2021-01-01", "2021-01-03"],
            "fund_name": ["Test Fund", "Test Fund", "Other Fund"],
            "daily_return": [0.05, 0.10, 0.20],
        }
    )
    path = fund_return_path(returns, "Test Fund")
    assert path.index.is_monotonic_increasing
    assert abs(path["growth_of_1"].iloc[-1] - 1.155) < 1e-12
    assert path["drawdown"].iloc[-1] == 0.0


def test_latest_fund_holdings_uses_latest_rebalance_date():
    weights = pd.DataFrame(
        {
            "rebalance_date": ["2021-01-01", "2021-02-01", "2021-02-01"],
            "fund_name": ["Test Fund", "Test Fund", "Test Fund"],
            "asset": ["AAA", "BBB", "CCC"],
            "asset_class": ["Equity", "Equity", "Crypto"],
            "weight": [1.0, 0.7, 0.3],
        }
    )
    latest, latest_date = latest_fund_holdings(weights, "Test Fund")
    assert latest_date == pd.Timestamp("2021-02-01")
    assert latest["asset"].tolist() == ["BBB", "CCC"]


if __name__ == "__main__":
    test_imports()
    print("imports OK")
    try:
        test_data_loads()
        print("data load OK")
    except Exception as e:
        print("data load skipped/failed (need network or FINS_DATA_ZIP):", e)
