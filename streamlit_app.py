"""MarketBridge Streamlit dashboard.

The app is intentionally results-only: it reads precomputed CSV files and
figures from results/ and does not run optimisation, VADER, transaction-cost,
or backtest code at runtime.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA_DIR = RESULTS / "data"
TABLES_DIR = RESULTS / "tables"
FIGURES_DIR = RESULTS / "figures"

PERCENT_COLUMNS = {
    "cumulative_return",
    "annualised_return",
    "annualised_volatility",
    "max_drawdown",
    "average_equity_weight",
    "average_crypto_weight",
    "gross_cumulative_return",
    "net_cumulative_return",
    "gross_annualised_return",
    "net_annualised_return",
    "gross_max_drawdown",
    "net_max_drawdown",
    "average_turnover",
    "total_turnover",
    "estimated_cost_drag",
    "transaction_cost_rate",
    "positive_share",
    "negative_share",
    "neutral_share",
    "weight",
    "entered_weight",
    "normalised_weight",
    "turnover",
}

DISPLAY_NAMES = {
    "fund_name": "Fund",
    "date": "Date",
    "rebalance_date": "Rebalance date",
    "asset": "Asset",
    "asset_class": "Asset class",
    "sector": "Sector",
    "headline_count": "Headlines",
    "mean_compound": "Mean compound",
    "sentiment_index": "Sentiment index",
    "lagged_sentiment_index": "Lagged sentiment",
    "rolling_lagged_sentiment": "21-day lagged sentiment",
    "cumulative_return": "Cumulative return",
    "annualised_return": "Annualised return",
    "annualised_volatility": "Annualised volatility",
    "sharpe_ratio": "Sharpe ratio",
    "max_drawdown": "Maximum drawdown",
    "average_equity_weight": "Average equity weight",
    "average_crypto_weight": "Average crypto weight",
    "rebalance_count": "Rebalances",
    "gross_cumulative_return": "Gross cumulative return",
    "net_cumulative_return": "Net cumulative return",
    "gross_annualised_return": "Gross annualised return",
    "net_annualised_return": "Net annualised return",
    "gross_sharpe_ratio": "Gross Sharpe ratio",
    "net_sharpe_ratio": "Net Sharpe ratio",
    "gross_max_drawdown": "Gross max drawdown",
    "net_max_drawdown": "Net max drawdown",
    "average_turnover": "Average turnover",
    "total_turnover": "Total turnover",
    "estimated_cost_drag": "Estimated cost drag",
    "transaction_cost_rate": "Cost rate",
    "entered_weight": "Entered weight",
    "normalised_weight": "Allocation used",
    "custom_allocation": "Custom allocation",
    "term": "Term",
    "finance_sentiment_score": "Finance sentiment score",
    "direction": "Direction",
    "reason": "Reason",
    "average_vader_sentiment": "Average VADER sentiment",
    "average_finance_adjusted_sentiment": "Average finance-adjusted sentiment",
    "average_difference": "Average difference",
    "correlation_between_old_and_new": "Correlation",
}


st.set_page_config(
    page_title="MarketBridge Dashboard",
    page_icon="MB",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1180px;
    }
    .app-note {
        border-left: 4px solid #426b69;
        padding: 0.65rem 0.9rem;
        background: #f5f8f7;
        color: #1f2f2e;
        margin-bottom: 1rem;
    }
    .small-muted {
        color: #5f6b6b;
        font-size: 0.92rem;
    }
    div[data-testid="stMetric"] {
        background: #f8faf9;
        border: 1px solid #d9e1df;
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=86_400, show_spinner=False)
def load_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative_path)


def optional_csv(relative_path: str) -> pd.DataFrame | None:
    path = ROOT / relative_path
    if not path.exists():
        st.warning(f"Missing expected file: {relative_path}")
        return None
    try:
        return load_csv(relative_path)
    except Exception as exc:
        st.error(f"Could not read {relative_path}: {exc}")
        return None


def first_available_csv(paths: list[str]) -> tuple[pd.DataFrame | None, str | None]:
    for relative_path in paths:
        path = ROOT / relative_path
        if path.exists():
            return optional_csv(relative_path), relative_path
    st.warning("None of these files were found: " + ", ".join(paths))
    return None, None


def optional_image(relative_path: str, caption: str | None = None) -> None:
    path = ROOT / relative_path
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Optional figure not found: {relative_path}")


def first_available_image(paths: list[str], caption: str | None = None) -> None:
    for relative_path in paths:
        if (ROOT / relative_path).exists():
            optional_image(relative_path, caption=caption)
            return
    st.info("Optional figure not found: " + " or ".join(paths))


def format_percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def format_date(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return pd.to_datetime(value).date().isoformat()


def display_table(df: pd.DataFrame, preferred_columns: list[str] | None = None) -> None:
    if df is None or df.empty:
        st.info("No rows available for this view.")
        return

    out = df.copy()
    if preferred_columns is not None:
        columns = [column for column in preferred_columns if column in out.columns]
        out = out[columns]

    for column in out.columns:
        if "date" in column:
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.date

    formatters = {
        column: "{:.2%}"
        for column in out.columns
        if column in PERCENT_COLUMNS and pd.api.types.is_numeric_dtype(out[column])
    }
    for column in out.columns:
        if column not in formatters and pd.api.types.is_numeric_dtype(out[column]):
            formatters[column] = "{:.3f}"

    out = out.rename(columns=DISPLAY_NAMES)
    st.dataframe(
        out.style.format(formatters, na_rep=""),
        hide_index=True,
        width="stretch",
    )


def metric_strip(metrics: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, help_text) in zip(columns, metrics, strict=True):
        column.metric(label, value, help=help_text)


def data_span(df: pd.DataFrame, date_column: str = "date") -> str:
    if df is None or df.empty or date_column not in df.columns:
        return "n/a"
    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    if dates.empty:
        return "n/a"
    return f"{dates.min().date()} to {dates.max().date()}"


def compact_data_span(df: pd.DataFrame, date_column: str = "date") -> str:
    if df is None or df.empty or date_column not in df.columns:
        return "n/a"
    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    if dates.empty:
        return "n/a"
    return f"{dates.min():%Y-%m} to {dates.max():%Y-%m}"


def infer_periods_per_year(dates: pd.Series) -> int:
    parsed = pd.to_datetime(dates, errors="coerce").dropna()
    if parsed.empty:
        return 252
    yearly_counts = parsed.dt.year.value_counts()
    if yearly_counts.empty:
        return 252
    return 365 if yearly_counts.median() > 300 else 252


def calculate_portfolio_metrics(returns: pd.Series, periods_per_year: int) -> dict[str, float]:
    clean_returns = pd.Series(returns).dropna().astype(float)
    if clean_returns.empty:
        return {
            "cumulative_return": float("nan"),
            "annualised_return": float("nan"),
            "annualised_volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
        }

    growth = (1.0 + clean_returns).cumprod()
    cumulative_return = float(growth.iloc[-1] - 1.0)
    annualised_return = float(growth.iloc[-1] ** (periods_per_year / len(clean_returns)) - 1.0)
    annualised_volatility = float(clean_returns.std(ddof=1) * math.sqrt(periods_per_year))
    daily_volatility = float(clean_returns.std(ddof=1))
    sharpe_ratio = (
        float(math.sqrt(periods_per_year) * clean_returns.mean() / daily_volatility)
        if daily_volatility > 0
        else float("nan")
    )
    drawdown = growth / growth.cummax() - 1.0
    return {
        "cumulative_return": cumulative_return,
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": float(drawdown.min()),
    }


def selected_fund_returns(
    fund_returns_data: pd.DataFrame,
    selected_funds: list[str],
) -> pd.DataFrame:
    returns = fund_returns_data.copy()
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
    returns = returns.loc[
        returns["fund_name"].isin(selected_funds),
        ["date", "fund_name", "daily_return"],
    ].dropna()
    return (
        returns.pivot_table(
            index="date",
            columns="fund_name",
            values="daily_return",
            aggfunc="mean",
        )
        .sort_index()
        .dropna(subset=selected_funds)
    )


def asset_class_weight_timeseries(fund_weights_data: pd.DataFrame, fund_name: str) -> pd.DataFrame:
    weights = fund_weights_data.copy()
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"], errors="coerce")
    weights = weights.loc[weights["fund_name"] == fund_name]
    if weights.empty:
        return pd.DataFrame()
    return (
        weights.groupby(["rebalance_date", "asset_class"], as_index=False)["weight"]
        .sum()
        .pivot(index="rebalance_date", columns="asset_class", values="weight")
        .fillna(0.0)
        .sort_index()
    )


def fund_return_path(fund_returns_data: pd.DataFrame, fund_name: str) -> pd.DataFrame:
    returns = fund_returns_data.copy()
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
    returns["daily_return"] = pd.to_numeric(returns["daily_return"], errors="coerce")
    returns = (
        returns.loc[returns["fund_name"] == fund_name, ["date", "daily_return"]]
        .dropna()
        .sort_values("date")
    )
    if returns.empty:
        return pd.DataFrame()

    returns["growth_of_1"] = (1.0 + returns["daily_return"]).cumprod()
    returns["drawdown"] = returns["growth_of_1"] / returns["growth_of_1"].cummax() - 1.0
    return returns.set_index("date")


def latest_fund_holdings(
    fund_weights_data: pd.DataFrame,
    fund_name: str,
) -> tuple[pd.DataFrame, object]:
    weights = fund_weights_data.copy()
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"], errors="coerce")
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")
    weights = weights.loc[weights["fund_name"] == fund_name].dropna(
        subset=["rebalance_date", "weight"]
    )
    if weights.empty:
        return pd.DataFrame(), None

    latest_date = weights["rebalance_date"].max()
    latest = weights.loc[weights["rebalance_date"] == latest_date].sort_values(
        "weight",
        ascending=False,
    )
    return latest, latest_date


def sector_sentiment_view(sector_sentiment_data: pd.DataFrame) -> pd.DataFrame:
    data = sector_sentiment_data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "sector", "sentiment_index"])
    data = data.sort_values(["sector", "date"])
    data["21-day average"] = (
        data.groupby("sector", observed=True)["sentiment_index"]
        .rolling(window=21, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return data


st.title("MarketBridge: Fund Performance, Risk and News Sentiment Dashboard")
st.caption("Systematic equity, crypto and combined fund results from the Project B pipeline.")

st.sidebar.header("Dashboard")
st.sidebar.write("All analytics are loaded from committed `results/` artifacts.")
st.sidebar.markdown(
    """
    **No live recomputation**

    The dashboard does not run portfolio optimisation, VADER sentiment scoring,
    transaction-cost calculations, or backtests during app use.
    """
)

performance, performance_source = first_available_csv(
    [
        "results/tables/performance_metrics_common_sample.csv",
        "results/tables/performance_metrics.csv",
    ]
)
fund_weights = optional_csv("results/data/fund_weights.csv")
fund_returns = optional_csv("results/data/fund_returns.csv")
net_returns = optional_csv("results/data/fund_returns_net.csv")
cost_metrics = optional_csv("results/tables/transaction_cost_metrics.csv")
fusion_comparison = optional_csv("results/tables/fusion_comparison.csv")
finance_lexicon = optional_csv("results/tables/custom_finance_lexicon.csv")
lexicon_comparison = optional_csv("results/tables/sentiment_lexicon_comparison.csv")
finance_fusion_comparison = optional_csv(
    "results/tables/finance_lexicon_fusion_comparison.csv"
)
sector_sentiment = optional_csv("results/data/sector_sentiment_index.csv")
sentiment_signal = optional_csv("results/data/sector_sentiment_signal.csv")

with st.sidebar.expander("Loaded Artifacts", expanded=False):
    artifact_status = []
    for relative_path in [
        performance_source or "results/tables/performance_metrics_common_sample.csv",
        "results/data/fund_weights.csv",
        "results/data/fund_returns_net.csv",
        "results/tables/transaction_cost_metrics.csv",
        "results/tables/fusion_comparison.csv",
        "results/tables/custom_finance_lexicon.csv",
        "results/tables/sentiment_lexicon_comparison.csv",
        "results/tables/finance_lexicon_fusion_comparison.csv",
        "results/data/sector_sentiment_index.csv",
        "results/tables/weights_over_time_summary.csv",
        "results/figures/portfolio_weights_over_time.png",
        "results/figures/sector_sentiment_top6_grid.png",
        "results/figures/sector_sentiment_index.png",
    ]:
        artifact_status.append(
            {
                "Artifact": relative_path,
                "Found": (ROOT / relative_path).exists(),
            }
        )
    st.dataframe(pd.DataFrame(artifact_status), hide_index=True, width="stretch")

st.markdown(
    """
    <div class="app-note">
    MarketBridge is an academic prototype for comparing systematic fund backtests,
    implementation costs and headline sentiment. It is not financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)

(
    overview_tab,
    performance_tab,
    fact_sheet_tab,
    allocation_tab,
    costs_tab,
    sentiment_tab,
    finance_lexicon_tab,
    fusion_tab,
) = st.tabs(
    [
        "Overview",
        "Fund Performance",
        "Fund Fact Sheet",
        "Allocation and weights",
        "Transaction Costs",
        "Sector Sentiment",
        "Finance Lexicon",
        "Sentiment Fusion",
    ]
)

with overview_tab:
    st.subheader("Overview")
    st.write(
        "MarketBridge compares systematic equity-only, crypto-only and combined funds "
        "using precomputed Project B outputs. The dashboard focuses on investor-facing "
        "questions: return, drawdown risk, crypto exposure, trading-cost drag and news "
        "sentiment context."
    )
    st.write(
        "The results are generated outside the app by the project pipeline and saved "
        "under `results/`. This keeps the deployed dashboard light and reproducible."
    )

    if performance is not None and not performance.empty:
        best_sharpe = performance.loc[performance["sharpe_ratio"].idxmax()]
        lowest_drawdown = performance.loc[performance["max_drawdown"].idxmax()]
        full_performance_span = data_span(fund_returns)
        metric_strip(
            [
                ("Funds compared", str(performance["fund_name"].nunique()), None),
                (
                    "Best Sharpe",
                    format_number(best_sharpe["sharpe_ratio"]),
                    best_sharpe["fund_name"],
                ),
                (
                    "Lowest drawdown",
                    format_percent(lowest_drawdown["max_drawdown"]),
                    lowest_drawdown["fund_name"],
                ),
                (
                    "Performance sample",
                    compact_data_span(fund_returns),
                    full_performance_span,
                ),
            ]
        )
        st.caption(f"Full performance sample: {full_performance_span}.")

    col1, col2 = st.columns(2)
    with col1:
        optional_image(
            "results/figures/growth_of_1_common_sample.png",
            "Growth of $1 across funds, common sample",
        )
    with col2:
        optional_image(
            "results/figures/drawdowns_common_sample.png",
            "Drawdowns across funds, common sample",
        )

with performance_tab:
    st.subheader("Fund Performance")
    if performance_source:
        st.caption(f"Source: `{performance_source}`")
    st.write(
        "The comparison uses cumulative return, annualised return, volatility, "
        "Sharpe ratio and maximum drawdown. The common-sample table is preferred "
        "when available so funds are compared over the same live dates."
    )

    display_table(
        performance,
        [
            "fund_name",
            "cumulative_return",
            "annualised_return",
            "annualised_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "average_equity_weight",
            "average_crypto_weight",
            "rebalance_count",
        ],
    )

    col1, col2 = st.columns(2)
    with col1:
        optional_image("results/figures/growth_of_1_common_sample.png")
    with col2:
        optional_image("results/figures/drawdowns_common_sample.png")

    first_available_image(
        [
            "results/figures/sharpe_comparison_common_sample.png",
            "results/figures/sharpe_comparison.png",
        ]
    )

with fact_sheet_tab:
    st.subheader("Fund Fact Sheet")
    st.write(
        "Open one MarketBridge fund to inspect its return, risk, downside path, "
        "latest holdings and estimated implementation cost in one view."
    )

    if performance is None or performance.empty:
        st.info("Performance metrics are unavailable, so the fund fact sheet cannot load.")
    else:
        available_funds = sorted(performance["fund_name"].dropna().unique())
        preferred_default = "Combined Maximum Sharpe"
        default_index = (
            available_funds.index(preferred_default)
            if preferred_default in available_funds
            else 0
        )
        fact_fund = st.selectbox(
            "Select fund",
            available_funds,
            index=default_index,
            key="fact_sheet_fund",
        )

        fund_row = performance.loc[performance["fund_name"] == fact_fund].iloc[0]
        metric_strip(
            [
                (
                    "Cumulative return",
                    format_percent(fund_row["cumulative_return"]),
                    "Growth over the out-of-sample comparison period.",
                ),
                (
                    "Annualised return",
                    format_percent(fund_row["annualised_return"]),
                    "Compounded annual return from daily fund returns.",
                ),
                (
                    "Annualised volatility",
                    format_percent(fund_row["annualised_volatility"]),
                    "Annualised standard deviation of daily fund returns.",
                ),
                (
                    "Sharpe ratio",
                    format_number(fund_row["sharpe_ratio"]),
                    "Zero-risk-free-rate Sharpe ratio.",
                ),
                (
                    "Maximum drawdown",
                    format_percent(fund_row["max_drawdown"]),
                    "Largest peak-to-trough loss in the backtest.",
                ),
            ]
        )

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Average equity weight",
            format_percent(fund_row.get("average_equity_weight")),
        )
        col2.metric(
            "Average crypto weight",
            format_percent(fund_row.get("average_crypto_weight")),
        )
        col3.metric("Rebalance count", format_number(fund_row.get("rebalance_count"), digits=0))

        if cost_metrics is not None and not cost_metrics.empty:
            cost_row = cost_metrics.loc[cost_metrics["fund_name"] == fact_fund]
            if not cost_row.empty:
                cost_row = cost_row.iloc[0]
                cost_col1, cost_col2, cost_col3 = st.columns(3)
                cost_col1.metric(
                    "Net cumulative return",
                    format_percent(cost_row["net_cumulative_return"]),
                    help="After the assumed 0.10% cost per dollar traded.",
                )
                cost_col2.metric(
                    "Estimated cost drag",
                    format_percent(cost_row["estimated_cost_drag"]),
                    help="Gross cumulative return minus net cumulative return.",
                )
                cost_col3.metric(
                    "Average turnover",
                    format_percent(cost_row["average_turnover"]),
                    help="Average one-way turnover at rebalance dates.",
                )

        if fund_returns is not None and not fund_returns.empty:
            path = fund_return_path(fund_returns, fact_fund)
            if path.empty:
                st.info("No return path is available for the selected fund.")
            else:
                st.caption(
                    "Fact-sheet sample: "
                    f"{path.index.min().date()} to {path.index.max().date()} "
                    f"({len(path):,} daily observations)."
                )
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.caption("Growth of $1")
                    st.line_chart(path[["growth_of_1"]].rename(columns={"growth_of_1": fact_fund}))
                with chart_col2:
                    st.caption("Drawdown")
                    st.line_chart(path[["drawdown"]].rename(columns={"drawdown": fact_fund}))
        else:
            st.info("Fund return data is unavailable.")

        if fund_weights is not None and not fund_weights.empty:
            weight_path = asset_class_weight_timeseries(fund_weights, fact_fund)
            holdings, latest_date = latest_fund_holdings(fund_weights, fact_fund)
            if not weight_path.empty:
                st.caption("Asset-class weights through monthly rebalances")
                st.line_chart(weight_path)
            if not holdings.empty:
                st.caption(f"Current target holdings at {format_date(latest_date)}")
                display_table(holdings.head(15), ["asset", "asset_class", "weight"])
        else:
            st.info("Fund weight data is unavailable.")

with allocation_tab:
    st.subheader("Build your own fund allocation")
    st.write(
        "Set an investor-facing blend across existing funds. The calculator uses "
        "precomputed daily fund returns only; it is not a new optimisation or "
        "backtest."
    )

    if fund_returns is not None and not fund_returns.empty:
        available_funds = sorted(fund_returns["fund_name"].dropna().unique())
        preferred_funds = [
            "Combined Maximum Sharpe",
            "Combined Minimum Variance",
            "Guarded Minimum Variance",
        ]
        default_funds = [fund for fund in preferred_funds if fund in available_funds]
        if len(default_funds) < 2:
            default_funds = available_funds[: min(3, len(available_funds))]

        selected_funds = st.multiselect(
            "Funds to include",
            available_funds,
            default=default_funds,
        )

        if selected_funds:
            base_default_weight = round(100.0 / len(selected_funds), 2)
            default_weights = [base_default_weight] * len(selected_funds)
            default_weights[-1] = round(100.0 - sum(default_weights[:-1]), 2)
            weight_values = {}
            columns = st.columns(min(3, len(selected_funds)))
            for i, fund_name in enumerate(selected_funds):
                with columns[i % len(columns)]:
                    weight_values[fund_name] = st.number_input(
                        fund_name,
                        min_value=0.0,
                        max_value=100.0,
                        value=default_weights[i],
                        step=1.0,
                        format="%.2f",
                        key=f"allocation_weight_{fund_name}",
                    )

            entered_total = sum(weight_values.values())
            if entered_total <= 0:
                st.error("Enter at least one positive allocation weight.")
            else:
                if abs(entered_total - 100.0) > 0.01:
                    st.warning(
                        f"Entered weights sum to {entered_total:.2f}%. "
                        "The calculator normalises them to 100% for the blend."
                    )

                normalised_weights = {
                    fund_name: weight / entered_total
                    for fund_name, weight in weight_values.items()
                }
                selected_returns = selected_fund_returns(fund_returns, selected_funds)

                if selected_returns.empty:
                    st.info("No overlapping return dates are available for the selected funds.")
                else:
                    weight_series = pd.Series(normalised_weights).reindex(selected_funds)
                    blended_returns = (
                        selected_returns[selected_funds]
                        .mul(weight_series, axis=1)
                        .sum(axis=1)
                    )
                    growth = (1.0 + blended_returns).cumprod()
                    periods_per_year = infer_periods_per_year(pd.Series(selected_returns.index))
                    custom_metrics = calculate_portfolio_metrics(
                        blended_returns,
                        periods_per_year=periods_per_year,
                    )

                    metric_strip(
                        [
                            (
                                "Cumulative return",
                                format_percent(custom_metrics["cumulative_return"]),
                                None,
                            ),
                            (
                                "Annualised return",
                                format_percent(custom_metrics["annualised_return"]),
                                f"Uses {periods_per_year} observations per year.",
                            ),
                            (
                                "Annualised volatility",
                                format_percent(custom_metrics["annualised_volatility"]),
                                None,
                            ),
                            (
                                "Sharpe ratio",
                                format_number(custom_metrics["sharpe_ratio"]),
                                "Assumes a zero risk-free rate.",
                            ),
                            (
                                "Maximum drawdown",
                                format_percent(custom_metrics["max_drawdown"]),
                                None,
                            ),
                        ]
                    )

                    st.caption(
                        "Custom allocation sample: "
                        f"{selected_returns.index.min().date()} to "
                        f"{selected_returns.index.max().date()} "
                        f"({len(selected_returns):,} common daily observations)."
                    )
                    growth_frame = pd.DataFrame(
                        {"Custom allocation": growth},
                        index=selected_returns.index,
                    )
                    st.line_chart(growth_frame)

                    allocation_table = pd.DataFrame(
                        {
                            "fund_name": selected_funds,
                            "entered_weight": [
                                weight_values[fund] / 100.0 for fund in selected_funds
                            ],
                            "normalised_weight": [
                                normalised_weights[fund] for fund in selected_funds
                            ],
                        }
                    )
                    display_table(
                        allocation_table,
                        ["fund_name", "entered_weight", "normalised_weight"],
                    )

                    if performance is not None and not performance.empty:
                        st.caption("Reference metrics for the selected funds")
                        display_table(
                            performance.loc[performance["fund_name"].isin(selected_funds)],
                            [
                                "fund_name",
                                "cumulative_return",
                                "annualised_return",
                                "annualised_volatility",
                                "sharpe_ratio",
                                "max_drawdown",
                            ],
                        )
        else:
            st.info("Select at least one fund to build an allocation.")
    else:
        st.info("Fund return data is unavailable, so the allocation calculator cannot run.")

    st.divider()
    st.subheader("Portfolio weights over time")
    st.write(
        "Fund weights describe the target allocation set at each rebalance date. "
        "These monthly rebalance weights are precomputed by the Project B pipeline "
        "and loaded from `results/data/fund_weights.csv`."
    )

    optional_image(
        "results/figures/portfolio_weights_over_time.png",
        "Asset-class weights at monthly rebalance dates",
    )

    if fund_weights is not None and not fund_weights.empty:
        weights = fund_weights.copy()
        weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"], errors="coerce")
        allocation = (
            weights.groupby(["fund_name", "asset_class"], as_index=False)["weight"]
            .mean()
            .pivot(index="fund_name", columns="asset_class", values="weight")
            .fillna(0.0)
            .reset_index()
            .rename_axis(None, axis=1)
        )
        allocation["average_equity_weight"] = allocation.get("Equity", 0.0)
        allocation["average_crypto_weight"] = allocation.get("Crypto", 0.0)
        display_table(
            allocation,
            ["fund_name", "average_equity_weight", "average_crypto_weight"],
        )

        chart_frame = allocation.set_index("fund_name")[
            ["average_equity_weight", "average_crypto_weight"]
        ].rename(
            columns={
                "average_equity_weight": "Equity",
                "average_crypto_weight": "Crypto",
            }
        )
        st.bar_chart(chart_frame)

        funds = sorted(weights["fund_name"].dropna().unique())
        selected_fund = st.selectbox("Inspect one fund's asset-class weights", funds, index=0)
        selected_weight_path = asset_class_weight_timeseries(weights, selected_fund)
        if not selected_weight_path.empty:
            st.line_chart(selected_weight_path)

        latest_date = weights.loc[
            weights["fund_name"] == selected_fund, "rebalance_date"
        ].max()
        latest_holdings = weights.loc[
            (weights["fund_name"] == selected_fund)
            & (weights["rebalance_date"] == latest_date)
        ].sort_values("weight", ascending=False)
        st.caption(f"Latest holdings at rebalance date: {format_date(latest_date)}")
        display_table(latest_holdings, ["asset", "asset_class", "weight"])
    else:
        st.info("Fund weight data is unavailable.")

with costs_tab:
    st.subheader("Transaction-Cost Transparency")
    st.write(
        "Transaction costs are estimated with an assumed 0.10% cost per dollar "
        "traded at rebalance dates. This is a simple implementation-cost model, "
        "not an exact broker-fee schedule."
    )

    if cost_metrics is not None and not cost_metrics.empty:
        display_table(
            cost_metrics,
            [
                "fund_name",
                "gross_cumulative_return",
                "net_cumulative_return",
                "estimated_cost_drag",
                "average_turnover",
                "total_turnover",
                "gross_sharpe_ratio",
                "net_sharpe_ratio",
                "transaction_cost_rate",
            ],
        )

        highest_drag = cost_metrics.loc[cost_metrics["estimated_cost_drag"].idxmax()]
        highest_turnover = cost_metrics.loc[cost_metrics["average_turnover"].idxmax()]
        metric_strip(
            [
                (
                    "Largest cost drag",
                    format_percent(highest_drag["estimated_cost_drag"]),
                    highest_drag["fund_name"],
                ),
                (
                    "Highest average turnover",
                    format_percent(highest_turnover["average_turnover"]),
                    highest_turnover["fund_name"],
                ),
                ("Assumed cost rate", "0.10%", "Per dollar traded at rebalance"),
            ]
        )
    else:
        st.info("Transaction-cost metrics are unavailable.")

    col1, col2 = st.columns(2)
    with col1:
        optional_image("results/figures/turnover_by_fund.png")
    with col2:
        optional_image("results/figures/cost_drag_by_fund.png")

with sentiment_tab:
    st.subheader("Sector Sentiment")
    st.write(
        "Headline sentiment is scored outside the app using VADER compound "
        "sentiment, aligned to equity trading dates and aggregated by sector-date. "
        "The figure shows the sector sentiment index over time for the six sectors "
        "with the most headline coverage."
    )

    first_available_image(
        [
            "results/figures/sector_sentiment_top6_grid.png",
            "results/figures/sector_sentiment_index_small_multiples_preview.png",
            "results/figures/sector_sentiment_index.png",
        ],
        "Top-six equity sector sentiment index; each panel is a separate sector view.",
    )

    if sector_sentiment is not None and not sector_sentiment.empty:
        sentiment_view = sector_sentiment_view(sector_sentiment)
        top_sectors = (
            sentiment_view.groupby("sector", observed=True)["headline_count"]
            .sum()
            .sort_values(ascending=False)
            .head(6)
            .index
            .tolist()
        )
        selected_sector = st.selectbox("Inspect one sector", top_sectors, index=0)
        sector_plot = sentiment_view.loc[
            sentiment_view["sector"] == selected_sector
        ].set_index("date")
        st.line_chart(
            sector_plot[["sentiment_index", "21-day average"]].rename(
                columns={"sentiment_index": "Daily sentiment index"}
            )
        )

        latest_sentiment = sentiment_view.copy()
        latest_date = latest_sentiment["date"].max()
        latest_rows = latest_sentiment.loc[latest_sentiment["date"] == latest_date].sort_values(
            "sentiment_index",
            ascending=False,
        )
        st.caption(f"Latest sector-date preview: {format_date(latest_date)}")
        display_table(
            latest_rows,
            [
                "date",
                "sector",
                "headline_count",
                "sentiment_index",
                "positive_share",
                "negative_share",
                "neutral_share",
            ],
        )

        st.caption("First rows of the full sector sentiment file")
        display_table(
            sector_sentiment.head(20),
            [
                "date",
                "sector",
                "headline_count",
                "mean_compound",
                "sentiment_index",
                "positive_share",
                "negative_share",
                "neutral_share",
            ],
        )
    else:
        st.info("Sector sentiment data is unavailable.")

    if sentiment_signal is not None and not sentiment_signal.empty:
        st.caption("Lagged signal preview used by the fusion portfolio")
        display_table(
            sentiment_signal.head(20),
            [
                "date",
                "sector",
                "sentiment_index",
                "lagged_sentiment_index",
                "rolling_lagged_sentiment",
            ],
        )

with finance_lexicon_tab:
    st.subheader("Finance Lexicon Extension")
    st.write(
        "The finance lexicon extension is a robustness test of the VADER sentiment "
        "baseline. It uses precomputed outputs to compare the original sector "
        "sentiment index with a small finance-adjusted version, without running "
        "sentiment scoring inside the app."
    )

    if lexicon_comparison is not None and not lexicon_comparison.empty:
        metric_strip(
            [
                (
                    "Sectors compared",
                    str(lexicon_comparison["sector"].nunique()),
                    None,
                ),
                (
                    "Average correlation",
                    format_number(
                        lexicon_comparison[
                            "correlation_between_old_and_new"
                        ].mean(),
                        digits=3,
                    ),
                    "VADER versus finance-adjusted index",
                ),
                (
                    "Largest average adjustment",
                    format_number(
                        lexicon_comparison["average_difference"].abs().max(),
                        digits=3,
                    ),
                    lexicon_comparison.loc[
                        lexicon_comparison["average_difference"].abs().idxmax(),
                        "sector",
                    ],
                ),
            ]
        )
        display_table(
            lexicon_comparison,
            [
                "sector",
                "headline_count",
                "average_vader_sentiment",
                "average_finance_adjusted_sentiment",
                "average_difference",
                "correlation_between_old_and_new",
            ],
        )
    else:
        st.info("Finance lexicon comparison data is unavailable.")

    col1, col2 = st.columns(2)
    with col1:
        optional_image("results/figures/custom_finance_lexicon_scores.png")
    with col2:
        optional_image("results/figures/sentiment_adjustment_by_sector.png")

    if finance_lexicon is not None and not finance_lexicon.empty:
        st.caption("Selected finance-specific terms and conservative scores")
        display_table(
            finance_lexicon.sort_values(
                "finance_sentiment_score",
                key=lambda series: series.abs(),
                ascending=False,
            ).head(20),
            ["term", "finance_sentiment_score", "direction", "reason"],
        )

    st.divider()
    st.write(
        "The finance-adjusted signal was also tested as a separate sentiment tilt. "
        "In this sample it did not outperform the original Guarded Minimum "
        "Variance fund, although it was marginally above the plain VADER tilt."
    )
    if finance_fusion_comparison is not None and not finance_fusion_comparison.empty:
        display_table(
            finance_fusion_comparison,
            [
                "fund_name",
                "cumulative_return",
                "annualised_return",
                "annualised_volatility",
                "sharpe_ratio",
                "max_drawdown",
                "average_turnover",
                "estimated_cost_drag",
            ],
        )
    else:
        st.info("Finance lexicon fusion comparison data is unavailable.")

    optional_image("results/figures/finance_lexicon_fusion_metrics_comparison.png")

with fusion_tab:
    st.subheader("Sentiment Fusion")
    st.write(
        "The fusion test applies the lagged sector sentiment signal to the equity "
        "sleeve of the Guarded Minimum Variance fund. In this sample, the tilt did "
        "not materially improve performance, but it demonstrates a look-ahead-safe "
        "combination of structured returns and unstructured headline data."
    )

    if fusion_comparison is not None and not fusion_comparison.empty:
        display_table(
            fusion_comparison,
            [
                "fund_name",
                "cumulative_return",
                "annualised_return",
                "annualised_volatility",
                "sharpe_ratio",
                "max_drawdown",
                "average_turnover",
                "estimated_cost_drag",
                "average_crypto_weight",
            ],
        )
    else:
        st.info("Fusion comparison data is unavailable.")

    optional_image("results/figures/fusion_metrics_comparison.png")

    col1, col2 = st.columns(2)
    with col1:
        optional_image("results/figures/fusion_growth_comparison.png")
    with col2:
        optional_image("results/figures/fusion_drawdown_comparison.png")

st.divider()
st.caption(
    "Academic prototype only. Results are historical backtests from precomputed "
    "Project B artifacts and are not investment advice."
)
