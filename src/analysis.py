"""Run the named SQL queries and build the README charts.

Run from the project root:  python -m src.analysis
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, never open a window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from src.common import (
    ASSETS_DIR,
    QUERIES_PATH,
    REPORTS_DIR,
    TABLES_DIR,
    ensure_dirs,
    read_sql,
)

BLUE = "#2a78d6"
INK = "#16202c"
MUTED = "#5a6672"
SURFACE = "#f7f9fc"
AXIS = "#d5dce4"
WHITE = "#ffffff"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
    }
)


def style_axes(ax, title: str, subtitle: str | None = None) -> None:
    """Finding-shaped title plus a quiet subtitle, no gridlines."""
    ax.set_title(title, loc="left", fontsize=13.5, fontweight="semibold", color=INK, pad=24)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            color=MUTED,
        )
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def save(fig, name: str) -> None:
    path = ASSETS_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"chart: {path.relative_to(ASSETS_DIR.parents[0])}")


def chart_churn_by_contract(tables: dict[str, pd.DataFrame]) -> None:
    data = tables["churn_by_contract"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(
        data["Contract"],
        data["churn_rate_pct"],
        width=0.52,
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=2,
    )
    for bar, value in zip(bars, data["churn_rate_pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color=INK,
        )
    ax.set_ylim(0, 50)
    ax.set_ylabel("Churn rate (%)", fontsize=10)
    style_axes(
        ax,
        "Month-to-month churn is 15x the two-year rate",
        "Churn rate by contract type · all 7,043 customers",
    )
    save(fig, "churn_by_contract.png")


def chart_churn_by_tenure(tables: dict[str, pd.DataFrame]) -> None:
    data = tables["churn_by_tenure"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(
        data["tenure_group"],
        data["churn_rate_pct"],
        width=0.52,
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=2,
    )
    for bar, value in zip(bars, data["churn_rate_pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color=INK,
        )
    ax.set_ylim(0, 55)
    ax.set_xlabel("Tenure (months)", fontsize=10)
    ax.set_ylabel("Churn rate (%)", fontsize=10)
    style_axes(
        ax,
        "Most churn happens in the first year",
        "Churn rate by tenure group · all 7,043 customers",
    )
    save(fig, "churn_by_tenure.png")


def chart_contract_tenure_heatmap(tables: dict[str, pd.DataFrame]) -> None:
    data = tables["contract_by_tenure"]
    matrix = data.pivot(index="Contract", columns="tenure_group", values="churn_rate_pct")
    matrix = matrix.reindex(
        index=["Month-to-month", "One year", "Two year"],
        columns=["0-12", "13-24", "25-48", "49-72"],
    )
    values = matrix.to_numpy()

    cmap = LinearSegmentedColormap.from_list(
        "blue_ramp", ["#eef4fc", "#a9cbf0", "#5d9ce3", BLUE, "#0e3f7a"]
    )
    fig, ax = plt.subplots(figsize=(8, 4.2))
    mesh = ax.pcolormesh(
        values,
        cmap=cmap,
        vmin=0,
        vmax=60,
        edgecolors=SURFACE,
        linewidth=3,
    )
    ax.set_xticks(np.arange(values.shape[1]) + 0.5, labels=matrix.columns)
    ax.set_yticks(np.arange(values.shape[0]) + 0.5, labels=matrix.index)
    ax.invert_yaxis()
    ax.tick_params(length=0, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(
                col + 0.5,
                row + 0.5,
                f"{value:.2f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="semibold",
                color=WHITE if value >= 45 else INK,
            )

    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Churn rate (%)", fontsize=9, color=MUTED)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=9, length=0)

    style_axes(
        ax,
        "Month-to-month churn stays high at every tenure",
        "Churn rate by contract × tenure group · all 7,043 customers",
    )
    save(fig, "contract_tenure_heatmap.png")


def chart_feature_importance() -> None:
    path = REPORTS_DIR / "permutation_importance.csv"
    if not path.exists():
        print(f"skip feature importance chart: {path.name} not found (run src.train first)")
        return
    data = pd.read_csv(path).head(10).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(
        data["feature"],
        data["importance_mean"],
        height=0.56,
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=2,
    )
    for bar, value in zip(bars, data["importance_mean"]):
        ax.text(
            value + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="semibold",
            color=INK,
        )
    ax.set_xlim(0, data["importance_mean"].max() * 1.18)
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled", fontsize=9.5)
    style_axes(
        ax,
        "Tenure is the signal the model relies on most",
        "Permutation importance, top 10 features · test set",
    )
    save(fig, "feature_importance.png")


def chart_gains_curve() -> None:
    path = REPORTS_DIR / "gains_curve.csv"
    if not path.exists():
        print(f"skip gains chart: {path.name} not found (run src.train first)")
        return
    data = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        data["share_contacted"] * 100,
        data["random_baseline"] * 100,
        color="#9aa6b2",
        linewidth=2,
        linestyle=(0, (5, 4)),
        label="Random targeting",
    )
    ax.plot(
        data["share_contacted"] * 100,
        data["share_of_churners_caught"] * 100,
        color=BLUE,
        linewidth=2.4,
        solid_capstyle="round",
        label="Model targeting",
    )

    marker = data.loc[(data["share_contacted"] - 0.2).abs().idxmin()]
    x, y = marker["share_contacted"] * 100, marker["share_of_churners_caught"] * 100
    ax.scatter([x], [y], s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=5)
    ax.annotate(
        f"Top 20% → {y:.2f}% of churners",
        xy=(x, y),
        xytext=(3, 88),
        fontsize=10.5,
        fontweight="semibold",
        color=INK,
        ha="left",
        va="top",
    )

    ax.plot([0, 100], [0, 100], color=AXIS, linewidth=0.8, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Share of customers contacted, riskiest first (%)", fontsize=10)
    ax.set_ylabel("Share of churners reached (%)", fontsize=10)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5, labelcolor=INK)
    style_axes(
        ax,
        "Targeting the riskiest 20% reaches half of all churners",
        "Gains curve on the hold-out test set · random targeting for comparison",
    )
    save(fig, "gains_curve.png")


def load_queries(path: Path) -> dict[str, str]:
    """Split queries.sql on its '-- name: <name>' markers."""
    queries: dict[str, str] = {}
    name: str | None = None
    lines: list[str] = []

    for line in path.read_text().splitlines():
        if line.strip().startswith("-- name:"):
            if name is not None:
                queries[name] = "\n".join(lines).strip()
            name = line.split("-- name:", 1)[1].strip()
            lines = []
        elif name is not None:
            lines.append(line)

    if name is not None:
        queries[name] = "\n".join(lines).strip()
    return queries


def run_queries() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name, sql in load_queries(QUERIES_PATH).items():
        result = read_sql(sql)
        result.to_csv(TABLES_DIR / f"{name}.csv", index=False)
        tables[name] = result
        print(f"\n=== {name} ===")
        print(result.to_string(index=False))
    return tables


def main() -> None:
    ensure_dirs()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    tables = run_queries()
    print(f"\nsaved {len(tables)} tables to {TABLES_DIR.relative_to(TABLES_DIR.parents[1])}")

    chart_churn_by_contract(tables)
    chart_churn_by_tenure(tables)
    chart_contract_tenure_heatmap(tables)
    chart_feature_importance()
    chart_gains_curve()


if __name__ == "__main__":
    main()
