"""Reproduce MarketBridge Part B portfolio results. Run from the project root:

    python scripts/run_part_b.py
"""
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from src import data_access
from src.features import build_return_panels
from src.fusion import apply_sentiment
from src.portfolios import (
    METHOD_EQUAL_WEIGHT,
    METHOD_MAX_SHARPE,
    METHOD_MIN_VARIANCE,
    oos_backtest,
    performance_metrics,
    weight_summary,
)
from src.sentiment import (
    build_finance_adjusted_sentiment_outputs,
    build_finance_adjusted_sentiment_signal,
    build_sector_sentiment_index,
    build_sector_sentiment_signal,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA_DIR = RESULTS / "data"
TABLES_DIR = RESULTS / "tables"
FIGURES_DIR = RESULTS / "figures"

ESTIMATION_WINDOW = 252
CRYPTO_CAP = 0.20
TRANSACTION_COST_RATE = 0.001
SENTIMENT_TILT_STRENGTH = 0.10
BASE_FUSION_FUND = "Guarded Minimum Variance"
SENTIMENT_TILT_FUND = "Guarded Minimum Variance + Sentiment Tilt"
FINANCE_SENTIMENT_TILT_FUND = "Guarded Minimum Variance + Finance Lexicon Sentiment Tilt"
FUSION_FUNDS = [BASE_FUSION_FUND, SENTIMENT_TILT_FUND]
FINANCE_FUSION_FUNDS = [
    BASE_FUSION_FUND,
    SENTIMENT_TILT_FUND,
    FINANCE_SENTIMENT_TILT_FUND,
]
SELECTED_COST_FUNDS = [
    "Combined Maximum Sharpe",
    "Combined Minimum Variance",
    "Guarded Minimum Variance",
    "Crypto Minimum Variance",
]
WEIGHTS_FIGURE_FUNDS = [
    "Combined Maximum Sharpe",
    "Combined Minimum Variance",
    "Guarded Minimum Variance",
    "Guarded Minimum Variance + Sentiment Tilt",
]


def ensure_dirs() -> None:
    for path in (DATA_DIR, TABLES_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def fund_specs() -> list[dict]:
    """Funds included in the first MarketBridge backtest stage."""
    return [
        {
            "fund_name": "Equity Equal Weight",
            "panel": "equity",
            "method": METHOD_EQUAL_WEIGHT,
            "periods_per_year": 252,
        },
        {
            "fund_name": "Equity Minimum Variance",
            "panel": "equity",
            "method": METHOD_MIN_VARIANCE,
            "periods_per_year": 252,
        },
        {
            "fund_name": "Equity Maximum Sharpe",
            "panel": "equity",
            "method": METHOD_MAX_SHARPE,
            "periods_per_year": 252,
        },
        {
            "fund_name": "Crypto Equal Weight",
            "panel": "crypto",
            "method": METHOD_EQUAL_WEIGHT,
            "periods_per_year": 365,
        },
        {
            "fund_name": "Crypto Minimum Variance",
            "panel": "crypto",
            "method": METHOD_MIN_VARIANCE,
            "periods_per_year": 365,
        },
        {
            "fund_name": "Crypto Maximum Sharpe",
            "panel": "crypto",
            "method": METHOD_MAX_SHARPE,
            "periods_per_year": 365,
        },
        {
            "fund_name": "Combined Equal Weight",
            "panel": "combined",
            "method": METHOD_EQUAL_WEIGHT,
            "periods_per_year": 252,
        },
        {
            "fund_name": "Combined Minimum Variance",
            "panel": "combined",
            "method": METHOD_MIN_VARIANCE,
            "periods_per_year": 252,
        },
        {
            "fund_name": "Combined Maximum Sharpe",
            "panel": "combined",
            "method": METHOD_MAX_SHARPE,
            "periods_per_year": 252,
        },
        {
            "fund_name": "Guarded Minimum Variance",
            "panel": "combined",
            "method": METHOD_MIN_VARIANCE,
            "periods_per_year": 252,
            "crypto_cap": CRYPTO_CAP,
        },
        {
            "fund_name": SENTIMENT_TILT_FUND,
            "panel": "combined",
            "method": METHOD_MIN_VARIANCE,
            "periods_per_year": 252,
            "crypto_cap": CRYPTO_CAP,
            "sentiment_tilt": True,
            "tilt_strength": SENTIMENT_TILT_STRENGTH,
        },
        {
            "fund_name": FINANCE_SENTIMENT_TILT_FUND,
            "panel": "combined",
            "method": METHOD_MIN_VARIANCE,
            "periods_per_year": 252,
            "crypto_cap": CRYPTO_CAP,
            "finance_sentiment_tilt": True,
            "tilt_strength": SENTIMENT_TILT_STRENGTH,
        },
    ]


def build_metrics(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    spec: dict,
) -> dict:
    fund_name = spec["fund_name"]
    returns = fund_returns.loc[fund_returns["fund_name"] == fund_name, "daily_return"]
    weights = fund_weights.loc[fund_weights["fund_name"] == fund_name]
    metrics = {
        "fund_name": fund_name,
        **performance_metrics(returns, periods_per_year=spec["periods_per_year"]),
        **weight_summary(weights),
    }
    return metrics


def save_figures(
    fund_returns: pd.DataFrame,
    metrics: pd.DataFrame,
    filename_suffix: str = "",
    title_suffix: str = "Out-of-Sample",
    include_sharpe: bool = True,
) -> None:
    """Create simple required comparison figures for the report/app."""
    plt.style.use("seaborn-v0_8-whitegrid")

    returns = fund_returns.copy()
    returns["date"] = pd.to_datetime(returns["date"])
    returns["growth_of_1"] = returns.groupby("fund_name", observed=True)["daily_return"].transform(
        lambda series: (1.0 + series).cumprod()
    )
    returns["drawdown"] = returns.groupby("fund_name", observed=True)["growth_of_1"].transform(
        lambda series: series / series.cummax() - 1.0
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for fund_name, group in returns.groupby("fund_name", observed=True):
        ax.plot(group["date"], group["growth_of_1"], label=fund_name, linewidth=1.3)
    ax.set_title(f"MarketBridge Fund Growth of $1, {title_suffix}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"growth_of_1{filename_suffix}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    selected = [
        "Combined Minimum Variance",
        "Combined Maximum Sharpe",
        "Guarded Minimum Variance",
    ]
    for fund_name, group in returns.loc[returns["fund_name"].isin(selected)].groupby(
        "fund_name",
        observed=True,
    ):
        ax.plot(group["date"], group["drawdown"], label=fund_name, linewidth=1.4)
    ax.set_title(f"MarketBridge Combined-Fund Drawdowns, {title_suffix}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"drawdowns{filename_suffix}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    if include_sharpe:
        plot_metrics = metrics.sort_values("sharpe_ratio")
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        ax.barh(plot_metrics["fund_name"], plot_metrics["sharpe_ratio"], color="#2f6f73")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("MarketBridge Fund Sharpe Ratios, Out-of-Sample")
        ax.set_xlabel("Sharpe ratio")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(
            FIGURES_DIR / f"sharpe_comparison{filename_suffix}.png",
            bbox_inches="tight",
            dpi=220,
        )
        plt.close(fig)


def calculate_turnover(fund_weights: pd.DataFrame) -> pd.DataFrame:
    """Calculate one-way portfolio turnover at each rebalance date.

    Turnover uses 0.5 * sum(abs(new_weight - previous_weight)). Missing asset
    weights are treated as zero, including the first live rebalance.
    """
    weights = fund_weights.copy()
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])
    weights["asset_key"] = weights["asset_class"].astype(str) + "::" + weights["asset"].astype(str)

    turnover_rows = []
    for fund_name, group in weights.groupby("fund_name", observed=True):
        wide = (
            group.pivot_table(
                index="rebalance_date",
                columns="asset_key",
                values="weight",
                aggfunc="sum",
                fill_value=0.0,
            )
            .sort_index()
            .astype(float)
        )
        previous = wide.shift(1).fillna(0.0)
        turnover = 0.5 * (wide - previous).abs().sum(axis=1)
        turnover_rows.append(
            pd.DataFrame(
                {
                    "rebalance_date": turnover.index,
                    "fund_name": fund_name,
                    "turnover": turnover.to_numpy(dtype=float),
                }
            )
        )

    if not turnover_rows:
        return pd.DataFrame(columns=["rebalance_date", "fund_name", "turnover"])
    return pd.concat(turnover_rows, ignore_index=True).sort_values(["fund_name", "rebalance_date"])


def apply_transaction_costs(
    fund_returns: pd.DataFrame,
    turnover: pd.DataFrame,
    transaction_cost_rate: float,
) -> pd.DataFrame:
    """Create daily net returns after assumed rebalance-date trading costs."""
    returns = fund_returns.copy()
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.rename(columns={"daily_return": "gross_daily_return"})

    costs = turnover.copy()
    costs["rebalance_date"] = pd.to_datetime(costs["rebalance_date"])
    costs = costs.rename(columns={"rebalance_date": "date"})

    net_returns = returns.merge(costs, on=["date", "fund_name"], how="left")
    net_returns["turnover"] = net_returns["turnover"].fillna(0.0)
    net_returns["transaction_cost_rate"] = transaction_cost_rate
    net_returns["estimated_transaction_cost"] = (
        net_returns["turnover"] * net_returns["transaction_cost_rate"]
    )
    net_returns["net_daily_return"] = (
        net_returns["gross_daily_return"] - net_returns["estimated_transaction_cost"]
    )
    return net_returns.sort_values(["fund_name", "date"])


def transaction_cost_metrics(
    net_returns: pd.DataFrame,
    turnover: pd.DataFrame,
    specs: list[dict],
    transaction_cost_rate: float,
) -> pd.DataFrame:
    """Summarise gross and net performance under the assumed cost rate."""
    rows = []
    for spec in specs:
        fund_name = spec["fund_name"]
        fund = net_returns.loc[net_returns["fund_name"] == fund_name].sort_values("date")
        fund_turnover = turnover.loc[turnover["fund_name"] == fund_name, "turnover"]

        gross = performance_metrics(
            fund["gross_daily_return"],
            periods_per_year=spec["periods_per_year"],
        )
        net = performance_metrics(
            fund["net_daily_return"],
            periods_per_year=spec["periods_per_year"],
        )

        rows.append(
            {
                "fund_name": fund_name,
                "gross_cumulative_return": gross["cumulative_return"],
                "net_cumulative_return": net["cumulative_return"],
                "gross_annualised_return": gross["annualised_return"],
                "net_annualised_return": net["annualised_return"],
                "gross_sharpe_ratio": gross["sharpe_ratio"],
                "net_sharpe_ratio": net["sharpe_ratio"],
                "gross_max_drawdown": gross["max_drawdown"],
                "net_max_drawdown": net["max_drawdown"],
                "average_turnover": float(fund_turnover.mean()),
                "total_turnover": float(fund_turnover.sum()),
                "estimated_cost_drag": gross["cumulative_return"] - net["cumulative_return"],
                "transaction_cost_rate": transaction_cost_rate,
            }
        )
    return pd.DataFrame(rows).sort_values("fund_name")


def save_transaction_cost_figures(
    net_returns: pd.DataFrame,
    turnover: pd.DataFrame,
) -> None:
    """Create transparency figures for gross/net performance and turnover."""
    plt.style.use("seaborn-v0_8-whitegrid")

    plot_returns = net_returns.copy()
    plot_returns["date"] = pd.to_datetime(plot_returns["date"])
    plot_returns["gross_growth_of_1"] = plot_returns.groupby("fund_name", observed=True)[
        "gross_daily_return"
    ].transform(lambda series: (1.0 + series).cumprod())
    plot_returns["net_growth_of_1"] = plot_returns.groupby("fund_name", observed=True)[
        "net_daily_return"
    ].transform(lambda series: (1.0 + series).cumprod())

    fund_order = sorted(plot_returns["fund_name"].unique())
    colors = plt.cm.tab10.colors
    color_map = {fund_name: colors[i % len(colors)] for i, fund_name in enumerate(fund_order)}

    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    for fund_name, group in plot_returns.groupby("fund_name", observed=True):
        color = color_map[fund_name]
        ax.plot(group["date"], group["gross_growth_of_1"], color=color, linewidth=1.2)
        ax.plot(group["date"], group["net_growth_of_1"], color=color, linewidth=1.2, linestyle="--")
    ax.plot([], [], color="black", linewidth=1.4, label="Gross")
    ax.plot([], [], color="black", linewidth=1.4, linestyle="--", label="Net after assumed costs")
    ax.set_title("Gross vs Net Growth of $1 After Assumed Transaction Costs")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    fund_handles = [
        plt.Line2D([], [], color=color_map[fund_name], linewidth=1.4, label=fund_name)
        for fund_name in fund_order
    ]
    legend_1 = ax.legend(handles=fund_handles, fontsize=7, ncols=2, loc="upper left")
    ax.add_artist(legend_1)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gross_vs_net_performance.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    turnover_plot = turnover.sort_values(["fund_name", "rebalance_date"])
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.barh(
        turnover_plot.groupby("fund_name", observed=True)["turnover"].mean().sort_values().index,
        turnover_plot.groupby("fund_name", observed=True)["turnover"].mean().sort_values().values,
        color="#4c78a8",
    )
    ax.set_title("Average Rebalance Turnover by Fund")
    ax.set_xlabel("Average turnover")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "turnover_by_fund.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def save_report_transaction_cost_outputs(
    net_returns: pd.DataFrame,
    cost_metrics: pd.DataFrame,
    common_start: pd.Timestamp,
) -> None:
    """Save clearer report-focused transaction-cost tables and figures."""
    plt.style.use("seaborn-v0_8-whitegrid")

    drag = cost_metrics.sort_values("estimated_cost_drag", ascending=False).copy()
    drag["cost_drag_percentage_points"] = drag["estimated_cost_drag"] * 100

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.barh(
        drag["fund_name"],
        drag["cost_drag_percentage_points"],
        color="#8c564b",
    )
    ax.invert_yaxis()
    ax.set_title("Estimated Transaction-Cost Drag by Fund")
    ax.set_xlabel("Estimated cost drag (percentage points)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.1f}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cost_drag_by_fund.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    selected_summary = (
        cost_metrics.loc[cost_metrics["fund_name"].isin(SELECTED_COST_FUNDS)]
        .set_index("fund_name")
        .loc[SELECTED_COST_FUNDS]
        .reset_index()
    )
    selected_summary = selected_summary[
        [
            "fund_name",
            "gross_cumulative_return",
            "net_cumulative_return",
            "estimated_cost_drag",
            "average_turnover",
            "total_turnover",
        ]
    ]
    selected_summary.to_csv(TABLES_DIR / "transaction_cost_summary_selected.csv", index=False)

    plot_returns = net_returns.loc[
        net_returns["fund_name"].isin(SELECTED_COST_FUNDS)
    ].copy()
    plot_returns["date"] = pd.to_datetime(plot_returns["date"])
    plot_returns = plot_returns.loc[plot_returns["date"] >= common_start]
    plot_returns["gross_growth_of_1"] = plot_returns.groupby("fund_name", observed=True)[
        "gross_daily_return"
    ].transform(lambda series: (1.0 + series).cumprod())
    plot_returns["net_growth_of_1"] = plot_returns.groupby("fund_name", observed=True)[
        "net_daily_return"
    ].transform(lambda series: (1.0 + series).cumprod())

    fund_colors = {
        "Combined Maximum Sharpe": "#1f77b4",
        "Combined Minimum Variance": "#2ca02c",
        "Guarded Minimum Variance": "#17becf",
        "Crypto Minimum Variance": "#8c564b",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), sharex=True)
    for ax, fund_name in zip(axes.ravel(), SELECTED_COST_FUNDS, strict=True):
        group = plot_returns.loc[plot_returns["fund_name"] == fund_name]
        ax.plot(
            group["date"],
            group["gross_growth_of_1"],
            color=fund_colors[fund_name],
            linewidth=2.0,
            label="Gross",
        )
        ax.plot(
            group["date"],
            group["net_growth_of_1"],
            color="black",
            linewidth=1.6,
            linestyle="--",
            label="Net after assumed costs",
        )
        ax.set_title(fund_name, fontsize=10)
        ax.grid(True, alpha=0.35)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axes.ravel()[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Gross vs Net Growth of $1, Selected Funds", y=0.99)
    fig.supxlabel(f"Date, common sample from {common_start.date()}")
    fig.supylabel("Growth of $1")
    fig.tight_layout(rect=(0.04, 0.04, 1.0, 0.95))
    fig.savefig(FIGURES_DIR / "gross_vs_net_selected_funds.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def build_common_sample_outputs(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    specs: list[dict],
) -> pd.Timestamp:
    """Save comparison metrics and figures from the latest first live fund date."""
    returns = fund_returns.copy()
    weights = fund_weights.copy()
    returns["date"] = pd.to_datetime(returns["date"])
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])

    common_start = returns.groupby("fund_name", observed=True)["date"].min().max()
    common_returns = returns.loc[returns["date"] >= common_start].copy()
    common_weights = weights.loc[weights["rebalance_date"] >= common_start].copy()
    common_metrics = pd.DataFrame(
        build_metrics(common_returns, common_weights, spec)
        for spec in specs
    ).sort_values("fund_name")

    common_metrics.to_csv(TABLES_DIR / "performance_metrics_common_sample.csv", index=False)
    save_figures(
        common_returns,
        common_metrics,
        filename_suffix="_common_sample",
        title_suffix=f"Common Sample from {common_start.date()}",
        include_sharpe=True,
    )
    return common_start


def safe_filename(label: str) -> str:
    """Convert a display label into a simple artifact filename component."""
    cleaned = "".join(char if char.isalnum() else "_" for char in str(label))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "sector"


def save_portfolio_weights_figure(fund_weights: pd.DataFrame) -> None:
    """Save a readable asset-class weights-over-time exhibit and summary table."""
    plt.style.use("seaborn-v0_8-whitegrid")

    weights = fund_weights.copy()
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])
    class_weights = (
        weights.groupby(["fund_name", "rebalance_date", "asset_class"], as_index=False)[
            "weight"
        ]
        .sum()
        .sort_values(["fund_name", "rebalance_date", "asset_class"])
    )

    rows = []
    for fund_name, group in class_weights.groupby("fund_name", observed=True):
        wide = (
            group.pivot(index="rebalance_date", columns="asset_class", values="weight")
            .fillna(0.0)
            .sort_index()
        )
        latest = wide.iloc[-1]
        row = {
            "fund_name": fund_name,
            "first_rebalance_date": wide.index.min().date().isoformat(),
            "latest_rebalance_date": wide.index.max().date().isoformat(),
            "rebalance_count": int(wide.shape[0]),
        }
        for asset_class in ["Equity", "Crypto"]:
            if asset_class in wide.columns:
                series = wide[asset_class]
            else:
                series = pd.Series(0.0, index=wide.index)
            key = asset_class.lower()
            row[f"average_{key}_weight"] = float(series.mean())
            row[f"minimum_{key}_weight"] = float(series.min())
            row[f"maximum_{key}_weight"] = float(series.max())
            row[f"latest_{key}_weight"] = float(latest.get(asset_class, 0.0))
        rows.append(row)

    pd.DataFrame(rows).sort_values("fund_name").to_csv(
        TABLES_DIR / "weights_over_time_summary.csv",
        index=False,
    )

    available_funds = set(class_weights["fund_name"])
    plot_funds = [fund for fund in WEIGHTS_FIGURE_FUNDS if fund in available_funds]
    if not plot_funds:
        plot_funds = sorted(available_funds)[:1]

    fig, axes = plt.subplots(
        len(plot_funds),
        1,
        figsize=(10.0, max(3.0, 2.15 * len(plot_funds))),
        sharex=True,
        squeeze=False,
    )
    colors = {"Equity": "#2f6f73", "Crypto": "#f58518"}

    for ax, fund_name in zip(axes.ravel(), plot_funds, strict=True):
        group = class_weights.loc[class_weights["fund_name"] == fund_name]
        wide = (
            group.pivot(index="rebalance_date", columns="asset_class", values="weight")
            .fillna(0.0)
            .sort_index()
        )
        for asset_class in ["Equity", "Crypto"]:
            if asset_class in wide.columns:
                ax.plot(
                    wide.index,
                    wide[asset_class],
                    label=asset_class,
                    color=colors[asset_class],
                    linewidth=1.8,
                )
        ax.set_title(fund_name, fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    axes[-1, 0].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Portfolio Weights Over Time at Monthly Rebalance Dates", y=0.995)
    fig.supxlabel("Rebalance date")
    fig.supylabel("Asset-class weight")
    fig.tight_layout(rect=(0.04, 0.04, 0.88, 0.96))
    fig.savefig(FIGURES_DIR / "portfolio_weights_over_time.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def save_sector_sentiment_figure(sector_index: pd.DataFrame, max_sectors: int = 6) -> None:
    """Plot readable sector sentiment paths for the sectors with most headlines."""
    plt.style.use("seaborn-v0_8-whitegrid")
    data = sector_index.copy()
    data["date"] = pd.to_datetime(data["date"])
    top_sectors = (
        data.groupby("sector", observed=True)["headline_count"]
        .sum()
        .sort_values(ascending=False)
        .head(max_sectors)
        .index
    )
    plot_data = data.loc[data["sector"].isin(top_sectors)].sort_values(["sector", "date"])
    plot_data["sentiment_21d"] = (
        plot_data.groupby("sector", observed=True)["sentiment_index"]
        .rolling(window=21, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    fig, axes = plt.subplots(3, 2, figsize=(11.2, 8.4), sharex=True, sharey=True)
    sector_colors = plt.cm.Set2.colors
    flat_axes = axes.ravel()
    for ax, (i, sector) in zip(flat_axes, enumerate(top_sectors)):
        group = plot_data.loc[plot_data["sector"] == sector]
        color = sector_colors[i % len(sector_colors)]
        ax.plot(
            group["date"],
            group["sentiment_index"],
            color=color,
            alpha=0.22,
            linewidth=0.7,
        )
        ax.plot(
            group["date"],
            group["sentiment_21d"],
            color=color,
            linewidth=1.9,
        )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(f"{sector} ({int(group['headline_count'].sum()):,} headlines)", fontsize=10)
        ax.grid(True, alpha=0.35)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in flat_axes[len(top_sectors):]:
        ax.axis("off")

    fig.suptitle("Sector Headline Sentiment Index, Top 6 Sectors by Coverage", y=0.995)
    fig.supxlabel("Date")
    fig.supylabel("Sentiment index; faint daily score, solid 21-day average")
    fig.tight_layout(rect=(0.04, 0.04, 1.0, 0.96))
    fig.savefig(FIGURES_DIR / "sector_sentiment_top6_grid.png", bbox_inches="tight", dpi=220)
    fig.savefig(FIGURES_DIR / "sector_sentiment_index.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    for i, sector in enumerate(top_sectors):
        group = plot_data.loc[plot_data["sector"] == sector]
        color = sector_colors[i % len(sector_colors)]
        fig, ax = plt.subplots(figsize=(9.4, 4.8))
        ax.plot(
            group["date"],
            group["sentiment_index"],
            color=color,
            alpha=0.25,
            linewidth=0.8,
            label="Daily sector sentiment",
        )
        ax.plot(
            group["date"],
            group["sentiment_21d"],
            color=color,
            linewidth=2.0,
            label="21-day average",
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{sector} Headline Sentiment Index")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sentiment index")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            FIGURES_DIR / f"sector_sentiment_{safe_filename(sector)}.png",
            bbox_inches="tight",
            dpi=220,
        )
        plt.close(fig)


def build_sentiment_lexicon_comparison(finance_index: pd.DataFrame) -> pd.DataFrame:
    """Compare standard VADER and finance-adjusted sector sentiment indexes."""
    data = finance_index.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["difference"] = (
        data["finance_adjusted_sentiment_index"] - data["vader_sentiment_index"]
    )
    rows = []
    for sector, group in data.groupby("sector", observed=True):
        largest = group.loc[group["difference"].abs().idxmax()]
        rows.append(
            {
                "sector": sector,
                "headline_count": int(group["headline_count"].sum()),
                "average_vader_sentiment": float(group["vader_sentiment_index"].mean()),
                "average_finance_adjusted_sentiment": float(
                    group["finance_adjusted_sentiment_index"].mean()
                ),
                "average_difference": float(group["difference"].mean()),
                "correlation_between_old_and_new": float(
                    group["vader_sentiment_index"].corr(
                        group["finance_adjusted_sentiment_index"]
                    )
                ),
                "largest_absolute_difference_date": (
                    pd.Timestamp(largest["date"]).date().isoformat()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("headline_count", ascending=False)


def save_finance_lexicon_figures(
    finance_index: pd.DataFrame,
    custom_lexicon: pd.DataFrame,
    max_sectors: int = 6,
) -> None:
    """Create figures comparing standard VADER and finance-adjusted sentiment."""
    plt.style.use("seaborn-v0_8-whitegrid")
    data = finance_index.copy()
    data["date"] = pd.to_datetime(data["date"])
    top_sectors = (
        data.groupby("sector", observed=True)["headline_count"]
        .sum()
        .sort_values(ascending=False)
        .head(max_sectors)
        .index
    )
    plot_data = data.loc[data["sector"].isin(top_sectors)].sort_values(["sector", "date"])
    for source_col, target_col in [
        ("vader_sentiment_index", "vader_21d"),
        ("finance_adjusted_sentiment_index", "finance_adjusted_21d"),
    ]:
        plot_data[target_col] = (
            plot_data.groupby("sector", observed=True)[source_col]
            .rolling(window=21, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    fig, axes = plt.subplots(3, 2, figsize=(11.2, 8.4), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    for ax, sector in zip(flat_axes, top_sectors, strict=False):
        group = plot_data.loc[plot_data["sector"] == sector]
        ax.plot(
            group["date"],
            group["vader_21d"],
            color="#4c78a8",
            linewidth=1.7,
            label="VADER",
        )
        ax.plot(
            group["date"],
            group["finance_adjusted_21d"],
            color="#f58518",
            linewidth=1.7,
            label="Finance-adjusted",
        )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(f"{sector} ({int(group['headline_count'].sum()):,} headlines)", fontsize=10)
        ax.grid(True, alpha=0.35)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in flat_axes[len(top_sectors):]:
        ax.axis("off")
    flat_axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("VADER vs Finance-Adjusted Sector Sentiment, Top 6 Sectors", y=0.995)
    fig.supxlabel("Date")
    fig.supylabel("21-day average sentiment index")
    fig.tight_layout(rect=(0.04, 0.04, 1.0, 0.96))
    fig.savefig(
        FIGURES_DIR / "sentiment_vader_vs_finance_adjusted_top6.png",
        bbox_inches="tight",
        dpi=220,
    )
    plt.close(fig)

    by_sector = (
        data.assign(
            difference=lambda frame: frame["finance_adjusted_sentiment_index"]
            - frame["vader_sentiment_index"]
        )
        .groupby("sector", observed=True)["difference"]
        .mean()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    colors = ["#d62728" if value < 0 else "#2f6f73" for value in by_sector]
    ax.barh(by_sector.index, by_sector.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Average Finance-Lexicon Sentiment Adjustment by Sector")
    ax.set_xlabel("Finance-adjusted minus VADER sentiment index")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sentiment_adjustment_by_sector.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    lexicon_plot = custom_lexicon.sort_values("finance_sentiment_score")
    fig, ax = plt.subplots(figsize=(9.6, max(5.4, 0.22 * len(lexicon_plot))))
    colors = [
        "#d62728" if value < 0 else "#2f6f73"
        for value in lexicon_plot["finance_sentiment_score"]
    ]
    ax.barh(
        lexicon_plot["term"],
        lexicon_plot["finance_sentiment_score"],
        color=colors,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Custom Finance Lexicon Scores")
    ax.set_xlabel("Finance-specific sentiment score")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "custom_finance_lexicon_scores.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def build_fusion_comparison(
    metrics: pd.DataFrame,
    cost_metrics: pd.DataFrame,
    fund_names: list[str] | None = None,
) -> pd.DataFrame:
    """Create the before-versus-after fusion comparison table."""
    selected_funds = fund_names or FUSION_FUNDS
    selected_metrics = metrics.loc[metrics["fund_name"].isin(selected_funds)].copy()
    selected_costs = cost_metrics.loc[
        cost_metrics["fund_name"].isin(selected_funds),
        ["fund_name", "average_turnover", "estimated_cost_drag"],
    ]
    comparison = selected_metrics.merge(selected_costs, on="fund_name", how="left")
    comparison = comparison[
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
        ]
    ]
    return comparison.set_index("fund_name").loc[selected_funds].reset_index()


def save_fusion_figures(fund_returns: pd.DataFrame, fusion_comparison: pd.DataFrame) -> None:
    """Save growth, drawdown and scorecard figures for the fusion test."""
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {BASE_FUSION_FUND: "#4c78a8", SENTIMENT_TILT_FUND: "#f58518"}
    returns = fund_returns.loc[fund_returns["fund_name"].isin(FUSION_FUNDS)].copy()
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.sort_values(["fund_name", "date"])
    returns["growth_of_1"] = returns.groupby("fund_name", observed=True)["daily_return"].transform(
        lambda series: (1.0 + series).cumprod()
    )
    returns["drawdown"] = returns.groupby("fund_name", observed=True)["growth_of_1"].transform(
        lambda series: series / series.cummax() - 1.0
    )

    fig, ax = plt.subplots(figsize=(9.4, 5.3))
    for fund_name, group in returns.groupby("fund_name", observed=True):
        ax.plot(group["date"], group["growth_of_1"], label=fund_name, color=colors[fund_name])
    ax.set_title("Growth of $1: Base Fund vs Lagged Sentiment Tilt")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fusion_growth_comparison.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.3))
    for fund_name, group in returns.groupby("fund_name", observed=True):
        ax.plot(group["date"], group["drawdown"], label=fund_name, color=colors[fund_name])
    ax.set_title("Drawdowns: Base Fund vs Lagged Sentiment Tilt")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fusion_drawdown_comparison.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    labels = {
        "annualised_return": "Annualised return",
        "annualised_volatility": "Annualised volatility",
        "sharpe_ratio": "Sharpe ratio",
        "max_drawdown": "Max drawdown",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8))
    for ax, metric in zip(axes.ravel(), labels, strict=True):
        plot_values = fusion_comparison[["fund_name", metric]].copy()
        ax.bar(
            plot_values["fund_name"],
            plot_values[metric],
            color=[colors[name] for name in plot_values["fund_name"]],
        )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(labels[metric], fontsize=10)
        ax.tick_params(axis="x", labelrotation=10, labelsize=8)
        if metric != "sharpe_ratio":
            ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.suptitle("Fusion Test Metrics: Guarded Minimum Variance Before and After Tilt", y=0.99)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))
    fig.savefig(FIGURES_DIR / "fusion_metrics_comparison.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def save_finance_lexicon_fusion_figure(finance_fusion_comparison: pd.DataFrame) -> None:
    """Save a scorecard for the base, VADER tilt and finance-lexicon tilt."""
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        BASE_FUSION_FUND: "#4c78a8",
        SENTIMENT_TILT_FUND: "#f58518",
        FINANCE_SENTIMENT_TILT_FUND: "#2ca02c",
    }
    labels = {
        "annualised_return": "Annualised return",
        "annualised_volatility": "Annualised volatility",
        "sharpe_ratio": "Sharpe ratio",
        "max_drawdown": "Max drawdown",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
    for ax, metric in zip(axes.ravel(), labels, strict=True):
        plot_values = finance_fusion_comparison[["fund_name", metric]].copy()
        ax.bar(
            plot_values["fund_name"],
            plot_values[metric],
            color=[colors[name] for name in plot_values["fund_name"]],
        )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(labels[metric], fontsize=10)
        ax.tick_params(axis="x", labelrotation=12, labelsize=7)
        if metric != "sharpe_ratio":
            ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.suptitle("Finance-Lexicon Fusion Metrics Comparison", y=0.99)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))
    fig.savefig(
        FIGURES_DIR / "finance_lexicon_fusion_metrics_comparison.png",
        bbox_inches="tight",
        dpi=220,
    )
    plt.close(fig)


def run_output_checks(
    sector_index: pd.DataFrame,
    sentiment_signal: pd.DataFrame,
    sentiment_usage: pd.DataFrame,
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    cost_metrics: pd.DataFrame,
) -> None:
    """Print the requested implementation checks clearly."""
    tilted_weights = fund_weights.loc[fund_weights["fund_name"] == SENTIMENT_TILT_FUND].copy()
    by_rebalance = tilted_weights.groupby("rebalance_date", observed=True)["weight"].sum()
    crypto_by_rebalance = (
        tilted_weights.loc[tilted_weights["asset_class"] == "Crypto"]
        .groupby("rebalance_date", observed=True)["weight"]
        .sum()
    )
    negative_count = int((tilted_weights["weight"] < -1e-10).sum())
    max_crypto = float(crypto_by_rebalance.max()) if not crypto_by_rebalance.empty else 0.0
    lagged_columns_present = {"lagged_sentiment_index", "rolling_lagged_sentiment"}.issubset(
        sentiment_signal.columns
    )
    usage_lagged = (
        not sentiment_usage.empty
        and (sentiment_usage["latest_signal_date"] < sentiment_usage["rebalance_date"]).all()
    )

    checks = [
        (
            "sector_sentiment_index.csv exists",
            (DATA_DIR / "sector_sentiment_index.csv").exists() and not sector_index.empty,
        ),
        (
            f"{SENTIMENT_TILT_FUND} appears in fund_returns.csv",
            SENTIMENT_TILT_FUND in set(fund_returns["fund_name"]),
        ),
        (
            f"{SENTIMENT_TILT_FUND} appears in fund_weights.csv",
            SENTIMENT_TILT_FUND in set(fund_weights["fund_name"]),
        ),
        (
            "sentiment-tilted weights sum to 1 at each rebalance",
            bool((by_rebalance.sub(1.0).abs() <= 1e-8).all()),
        ),
        ("sentiment-tilted weights are non-negative", negative_count == 0),
        (
            f"sentiment-tilted crypto weight never exceeds {CRYPTO_CAP:.0%}",
            max_crypto <= CRYPTO_CAP + 1e-8,
        ),
        (
            "sentiment signal is lagged before portfolio use",
            lagged_columns_present and usage_lagged,
        ),
        (
            "transaction-cost metrics include the sentiment-tilted fund",
            SENTIMENT_TILT_FUND in set(cost_metrics["fund_name"]),
        ),
    ]

    print("\nImplementation checks:")
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print(f"  Max sentiment-tilted crypto weight: {max_crypto:.4f}")


def run_finance_lexicon_checks(
    candidate_terms: pd.DataFrame,
    custom_lexicon: pd.DataFrame,
    finance_index: pd.DataFrame,
    finance_signal: pd.DataFrame,
    sentiment_comparison: pd.DataFrame,
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    cost_metrics: pd.DataFrame,
    finance_fusion_comparison: pd.DataFrame,
) -> None:
    """Print transparent checks for the finance-lexicon extension."""
    finance_weights = fund_weights.loc[
        fund_weights["fund_name"] == FINANCE_SENTIMENT_TILT_FUND
    ].copy()
    by_rebalance = finance_weights.groupby("rebalance_date", observed=True)["weight"].sum()
    negative_count = (
        0
        if finance_weights.empty
        else int((finance_weights["weight"] < -1e-10).sum())
    )
    crypto_by_rebalance = (
        finance_weights.loc[finance_weights["asset_class"] == "Crypto"]
        .groupby("rebalance_date", observed=True)["weight"]
        .sum()
    )
    max_crypto = float(crypto_by_rebalance.max()) if not crypto_by_rebalance.empty else 0.0
    lag_columns = {
        "lagged_finance_adjusted_sentiment",
        "rolling_lagged_finance_sentiment",
    }.issubset(finance_signal.columns)

    merged = finance_index[
        ["date", "sector", "vader_sentiment_index", "finance_adjusted_sentiment_index"]
    ].dropna()
    overall_corr = float(
        merged["vader_sentiment_index"].corr(merged["finance_adjusted_sentiment_index"])
    )
    adjustment_by_sector = sentiment_comparison.assign(
        abs_average_difference=lambda frame: frame["average_difference"].abs()
    ).sort_values("abs_average_difference", ascending=False)

    print("\nFinance lexicon extension checks:")
    print(f"  Missing candidate terms found: {len(candidate_terms):,}")
    print(f"  Custom finance lexicon terms selected: {len(custom_lexicon):,}")
    print("  Top 20 custom lexicon terms by absolute score:")
    top_terms = custom_lexicon.assign(
        absolute_score=lambda frame: frame["finance_sentiment_score"].abs()
    ).sort_values(["absolute_score", "term"], ascending=[False, True]).head(20)
    for row in top_terms.itertuples(index=False):
        print(f"    {row.term}: {row.finance_sentiment_score:+.2f} ({row.direction})")
    vader_files_exist = (
        (DATA_DIR / "sector_sentiment_index.csv").exists()
        and (DATA_DIR / "sector_sentiment_signal.csv").exists()
    )
    finance_index_exists = (
        (DATA_DIR / "sector_sentiment_index_finance_adjusted.csv").exists()
        and not finance_index.empty
    )
    print(f"  Original VADER files preserved: {vader_files_exist}")
    print(f"  Finance-adjusted sentiment index created: {finance_index_exists}")
    print(
        "  Lagged finance-adjusted signal created: "
        f"{(DATA_DIR / 'sector_sentiment_signal_finance_adjusted.csv').exists() and lag_columns}"
    )
    print(f"  Correlation between old VADER and finance-adjusted index: {overall_corr:.4f}")
    print("  Sectors with largest average sentiment adjustment:")
    for row in adjustment_by_sector.head(5).itertuples(index=False):
        print(f"    {row.sector}: {row.average_difference:+.4f}")

    if not finance_fusion_comparison.empty:
        comparison = finance_fusion_comparison.set_index("fund_name")
        base = comparison.loc[BASE_FUSION_FUND]
        finance = comparison.loc[FINANCE_SENTIMENT_TILT_FUND]
        print("  Finance-adjusted fusion vs base guarded fund:")
        for metric in [
            "cumulative_return",
            "sharpe_ratio",
            "max_drawdown",
            "estimated_cost_drag",
        ]:
            delta = float(finance[metric] - base[metric])
            if metric == "max_drawdown":
                if delta > 0:
                    verdict = "improved"
                elif delta < 0:
                    verdict = "worsened"
                else:
                    verdict = "unchanged"
            elif metric == "estimated_cost_drag":
                if delta < 0:
                    verdict = "improved/lower"
                elif delta > 0:
                    verdict = "worsened/higher"
                else:
                    verdict = "unchanged"
            else:
                if delta > 0:
                    verdict = "improved"
                elif delta < 0:
                    verdict = "worsened"
                else:
                    verdict = "unchanged"
            print(f"    {metric}: {verdict} by {delta:+.6f}")
    print(
        "  Finance-adjusted tilted weights sum to 1: "
        f"{bool((by_rebalance.sub(1.0).abs() <= 1e-8).all()) if not by_rebalance.empty else False}"
    )
    print(f"  Finance-adjusted tilted weights are non-negative: {negative_count == 0}")
    print(
        f"  Finance-adjusted tilted crypto weight never exceeds {CRYPTO_CAP:.0%}: "
        f"{max_crypto <= CRYPTO_CAP + 1e-8}"
    )
    print(
        "  Look-ahead check: finance signal uses one-day lag and "
        "rebalance-time lookup before date."
    )


def main():
    warnings.simplefilter("always", RuntimeWarning)
    ensure_dirs()
    specs = fund_specs()

    eq = data_access.load_equity_prices()
    cr = data_access.load_crypto_prices()
    panels = build_return_panels(eq, cr)
    sector_map = data_access.load_sector_universe().set_index("ticker")["sector"].to_dict()

    print("Building sector sentiment index...")
    headlines = data_access.load_news_headlines()
    sector_index = build_sector_sentiment_index(headlines, panels["equity"].index)
    sentiment_signal = build_sector_sentiment_signal(sector_index, panels["equity"].index)
    sector_index.to_csv(DATA_DIR / "sector_sentiment_index.csv", index=False)
    sentiment_signal.to_csv(DATA_DIR / "sector_sentiment_signal.csv", index=False)
    save_sector_sentiment_figure(sector_index)

    print("Building finance-adjusted sentiment index...")
    (
        candidate_terms,
        custom_finance_lexicon,
        finance_adjusted_scores,
        finance_adjusted_index,
    ) = build_finance_adjusted_sentiment_outputs(headlines, panels["equity"].index)
    finance_adjusted_signal = build_finance_adjusted_sentiment_signal(
        finance_adjusted_index,
        panels["equity"].index,
    )
    sentiment_lexicon_comparison = build_sentiment_lexicon_comparison(finance_adjusted_index)

    candidate_terms.to_csv(TABLES_DIR / "finance_lexicon_candidate_terms.csv", index=False)
    custom_finance_lexicon.to_csv(TABLES_DIR / "custom_finance_lexicon.csv", index=False)
    sample_scores = finance_adjusted_scores.loc[
        finance_adjusted_scores["finance_term_matches"].astype(str).ne(""),
        [
            "date",
            "ticker",
            "sector",
            "headline",
            "vader_compound",
            "finance_term_matches",
            "finance_adjustment",
            "finance_adjusted_score",
        ],
    ].head(2000)
    sample_scores.to_csv(DATA_DIR / "headline_sentiment_finance_adjusted_sample.csv", index=False)
    finance_adjusted_index.to_csv(
        DATA_DIR / "sector_sentiment_index_finance_adjusted.csv",
        index=False,
    )
    finance_adjusted_signal.to_csv(
        DATA_DIR / "sector_sentiment_signal_finance_adjusted.csv",
        index=False,
    )
    sentiment_lexicon_comparison.to_csv(
        TABLES_DIR / "sentiment_lexicon_comparison.csv",
        index=False,
    )
    save_finance_lexicon_figures(finance_adjusted_index, custom_finance_lexicon)

    returns_outputs = []
    weights_outputs = []
    sentiment_usage_rows = []

    for spec in specs:
        print(f"Backtesting {spec['fund_name']}...")
        weight_adjuster = None
        if spec.get("sentiment_tilt"):
            def weight_adjuster(weights, date, spec=spec):
                latest_signal_date = sentiment_signal.loc[
                    sentiment_signal["date"] < pd.Timestamp(date),
                    "date",
                ].max()
                sentiment_usage_rows.append(
                    {
                        "rebalance_date": pd.Timestamp(date),
                        "latest_signal_date": latest_signal_date,
                    }
                )
                return apply_sentiment(
                    weights,
                    sentiment_signal,
                    rebalance_date=date,
                    sector_map=sector_map,
                    tilt_strength=spec.get("tilt_strength", SENTIMENT_TILT_STRENGTH),
                    crypto_cap=spec.get("crypto_cap"),
                )
        elif spec.get("finance_sentiment_tilt"):
            def weight_adjuster(weights, date, spec=spec):
                latest_signal_date = finance_adjusted_signal.loc[
                    finance_adjusted_signal["date"] < pd.Timestamp(date),
                    "date",
                ].max()
                sentiment_usage_rows.append(
                    {
                        "rebalance_date": pd.Timestamp(date),
                        "latest_signal_date": latest_signal_date,
                    }
                )
                return apply_sentiment(
                    weights,
                    finance_adjusted_signal,
                    rebalance_date=date,
                    sector_map=sector_map,
                    tilt_strength=spec.get("tilt_strength", SENTIMENT_TILT_STRENGTH),
                    crypto_cap=spec.get("crypto_cap"),
                    signal_col="rolling_lagged_finance_sentiment",
                )

        daily_returns, weights = oos_backtest(
            panels[spec["panel"]],
            fund_name=spec["fund_name"],
            method=spec["method"],
            estimation_window=ESTIMATION_WINDOW,
            crypto_cap=spec.get("crypto_cap"),
            weight_adjuster=weight_adjuster,
        )
        returns_outputs.append(daily_returns)
        weights_outputs.append(weights)

    fund_returns = pd.concat(returns_outputs, ignore_index=True).sort_values(["fund_name", "date"])
    fund_weights = pd.concat(weights_outputs, ignore_index=True).sort_values(
        ["fund_name", "rebalance_date", "asset_class", "asset"]
    )
    metrics = pd.DataFrame(
        build_metrics(fund_returns, fund_weights, spec)
        for spec in specs
    ).sort_values("fund_name")

    fund_returns.to_csv(DATA_DIR / "fund_returns.csv", index=False)
    fund_weights.to_csv(DATA_DIR / "fund_weights.csv", index=False)
    metrics.to_csv(TABLES_DIR / "performance_metrics.csv", index=False)
    save_figures(fund_returns, metrics)
    save_portfolio_weights_figure(fund_weights)

    turnover = calculate_turnover(fund_weights)
    net_returns = apply_transaction_costs(
        fund_returns,
        turnover,
        transaction_cost_rate=TRANSACTION_COST_RATE,
    )
    cost_metrics = transaction_cost_metrics(
        net_returns,
        turnover,
        specs,
        transaction_cost_rate=TRANSACTION_COST_RATE,
    )
    fusion_comparison = build_fusion_comparison(metrics, cost_metrics)
    finance_fusion_comparison = build_fusion_comparison(
        metrics,
        cost_metrics,
        FINANCE_FUSION_FUNDS,
    )
    common_start = build_common_sample_outputs(fund_returns, fund_weights, specs)

    net_returns.to_csv(DATA_DIR / "fund_returns_net.csv", index=False)
    cost_metrics.to_csv(TABLES_DIR / "transaction_cost_metrics.csv", index=False)
    fusion_comparison.to_csv(TABLES_DIR / "fusion_comparison.csv", index=False)
    finance_fusion_comparison.to_csv(
        TABLES_DIR / "finance_lexicon_fusion_comparison.csv",
        index=False,
    )
    save_transaction_cost_figures(net_returns, turnover)
    save_report_transaction_cost_outputs(net_returns, cost_metrics, common_start)
    save_fusion_figures(fund_returns, fusion_comparison)
    save_finance_lexicon_fusion_figure(finance_fusion_comparison)

    sentiment_usage = pd.DataFrame(sentiment_usage_rows)
    run_output_checks(
        sector_index,
        sentiment_signal,
        sentiment_usage,
        fund_returns,
        fund_weights,
        cost_metrics,
    )
    run_finance_lexicon_checks(
        candidate_terms,
        custom_finance_lexicon,
        finance_adjusted_index,
        finance_adjusted_signal,
        sentiment_lexicon_comparison,
        fund_returns,
        fund_weights,
        cost_metrics,
        finance_fusion_comparison,
    )

    print(
        f"Equity return panel: {panels['equity'].shape[0]:,} dates x "
        f"{panels['equity'].shape[1]} assets"
    )
    print(
        f"Crypto return panel: {panels['crypto'].shape[0]:,} dates x "
        f"{panels['crypto'].shape[1]} assets"
    )
    print(
        f"Combined return panel: {panels['combined'].shape[0]:,} dates x "
        f"{panels['combined'].shape[1]} assets"
    )
    print(f"Wrote {DATA_DIR / 'sector_sentiment_index.csv'}")
    print(f"Wrote {DATA_DIR / 'sector_sentiment_signal.csv'}")
    print(f"Wrote {DATA_DIR / 'sector_sentiment_index_finance_adjusted.csv'}")
    print(f"Wrote {DATA_DIR / 'sector_sentiment_signal_finance_adjusted.csv'}")
    print(f"Wrote {DATA_DIR / 'fund_returns.csv'}")
    print(f"Wrote {DATA_DIR / 'fund_weights.csv'}")
    print(f"Wrote {DATA_DIR / 'fund_returns_net.csv'}")
    print(f"Wrote {TABLES_DIR / 'finance_lexicon_candidate_terms.csv'}")
    print(f"Wrote {TABLES_DIR / 'custom_finance_lexicon.csv'}")
    print(f"Wrote {TABLES_DIR / 'sentiment_lexicon_comparison.csv'}")
    print(f"Wrote {TABLES_DIR / 'performance_metrics.csv'}")
    print(f"Wrote {TABLES_DIR / 'performance_metrics_common_sample.csv'}")
    print(f"Wrote {TABLES_DIR / 'weights_over_time_summary.csv'}")
    print(f"Wrote {TABLES_DIR / 'transaction_cost_metrics.csv'}")
    print(f"Wrote {TABLES_DIR / 'transaction_cost_summary_selected.csv'}")
    print(f"Wrote {TABLES_DIR / 'fusion_comparison.csv'}")
    print(f"Wrote {TABLES_DIR / 'finance_lexicon_fusion_comparison.csv'}")
    print(f"Common-sample start date: {common_start.date()}")
    print(f"Wrote figures under {FIGURES_DIR}")


if __name__ == "__main__":
    main()
