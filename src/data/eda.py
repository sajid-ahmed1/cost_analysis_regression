"""Simple EDA plotting helpers."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_value_counts(
    df: pd.DataFrame,
    column: str,
    top_n: int = 15,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> None:
    """Horizontal bar chart of the counts of each distinct value in a column."""
    series = df[column]
    counts = series.value_counts(dropna=False).head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=figsize or (8, max(3, 0.4 * len(counts))))
    ax.barh(counts.index.astype(str), counts.values, color="#2a78d6")

    ax.set_title(title or series.name)
    ax.set_xlabel("count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    plt.tight_layout()
    plt.show()
