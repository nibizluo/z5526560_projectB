"""Return features used by the Part B portfolio backtests.

The important rule is to compute returns inside each raw price panel before any
equity/crypto calendar merge. Crypto is then aligned to the equity calendar only
after the crypto daily returns already exist.
"""
import pandas as pd


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Compute simple daily returns within each ticker using adjusted close."""
    required = {"ticker", "date", price_col}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {sorted(missing)}")

    out = prices.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out["return"] = out.groupby("ticker", observed=True)[price_col].pct_change()
    return out


def returns_wide(
    returns: pd.DataFrame,
    prefix: str,
    value_col: str = "return",
) -> pd.DataFrame:
    """Convert long ticker-date returns to a date x asset return panel."""
    wide = (
        returns.pivot(index="date", columns="ticker", values=value_col)
        .sort_index()
        .add_prefix(prefix)
    )
    wide.index.name = "date"
    return wide


def build_return_panels(
    equity_prices: pd.DataFrame,
    crypto_prices: pd.DataFrame,
    crypto_end: str = "2023-12-31",
) -> dict[str, pd.DataFrame]:
    """Build equity, crypto, and combined return panels for Part B.

    Returns
    -------
    dict
        Contains ``equity``, ``crypto``, and ``combined`` wide return panels.
        The combined panel uses the equity trading calendar after both asset
        classes have already had returns computed on their own calendars.
    """
    eq_ret = daily_returns(equity_prices)

    cr_prices = crypto_prices.copy()
    cr_prices["date"] = pd.to_datetime(cr_prices["date"]).dt.tz_localize(None).dt.normalize()
    cr_prices = cr_prices.loc[cr_prices["date"] <= pd.Timestamp(crypto_end)]
    cr_ret = daily_returns(cr_prices)

    equity_panel = returns_wide(eq_ret, "EQ_").dropna(how="all")
    crypto_panel = returns_wide(cr_ret, "CR_").dropna(how="all")

    crypto_on_equity_calendar = crypto_panel.reindex(equity_panel.index)
    combined_panel = pd.concat([equity_panel, crypto_on_equity_calendar], axis=1)

    return {
        "equity": equity_panel,
        "crypto": crypto_panel,
        "combined": combined_panel,
    }


def assemble_headline_panel(headlines: pd.DataFrame) -> pd.DataFrame:
    """Assemble the headlines into a daily panel per ticker and sector.

    Station 2 is assembly only: structure the text and date-align it to the
    trading calendar. Scoring the text - and lagging the signal - is the
    Station 3 model.
    """
    raise NotImplementedError
