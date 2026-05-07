"""
visualization/plots.py
----------------------
Matplotlib-based charts for the Analytics page of the Streamlit app.

All functions return a matplotlib Figure object so Streamlit can render
them with st.pyplot(fig).
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

_BG   = "#0e0e0e"
_CARD = "#1a1a2e"
_ACC  = "#e50914"
_TXT  = "#ffffff"
_GRY  = "#888888"
_PALETTE = [
    "#e50914", "#1db954", "#0070f3", "#f5a623",
    "#bd10e0", "#50e3c2", "#ff6b6b", "#feca57",
    "#48dbfb", "#ff9ff3", "#54a0ff", "#5f27cd",
]


def _apply_dark_style(fig: plt.Figure, ax) -> None:
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TXT, labelsize=9)
    ax.xaxis.label.set_color(_TXT)
    ax.yaxis.label.set_color(_TXT)
    ax.title.set_color(_TXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRY)


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------

def plot_rating_distribution(ratings_df: pd.DataFrame) -> plt.Figure:
    """Histogram of all ratings in the dataset."""
    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_dark_style(fig, ax)

    bins = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25]
    counts, edges, patches = ax.hist(
        ratings_df["rating"], bins=bins, color=_ACC, edgecolor=_BG, linewidth=0.5
    )
    # Gradient fill
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.Reds(0.4 + 0.5 * i / len(patches)))

    ax.set_xlabel("Star Rating", fontsize=11)
    ax.set_ylabel("Number of Ratings", fontsize=11)
    ax.set_title("Rating Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    total = len(ratings_df)
    for patch, count in zip(patches, counts):
        if count > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                count + total * 0.005,
                f"{int(count)}",
                ha="center", va="bottom", color=_TXT, fontsize=8,
            )

    fig.tight_layout()
    return fig


def plot_genre_distribution(movies_df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of genre frequencies."""
    genre_counts: dict = {}
    for g_str in movies_df["genres"]:
        if g_str != "(no genres listed)":
            for g in g_str.split("|"):
                genre_counts[g.strip()] = genre_counts.get(g.strip(), 0) + 1

    genre_series = pd.Series(genre_counts).sort_values()
    top_genres = genre_series.tail(20)

    fig, ax = plt.subplots(figsize=(8, max(5, len(top_genres) * 0.35)))
    _apply_dark_style(fig, ax)

    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(top_genres))]
    bars = ax.barh(top_genres.index, top_genres.values, color=colors, edgecolor=_BG, height=0.72)

    ax.set_xlabel("Number of Movies", fontsize=11)
    ax.set_title("Genre Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, top_genres.max() * 1.15)

    for bar, val in zip(bars, top_genres.values):
        ax.text(val + top_genres.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                str(int(val)), va="center", color=_TXT, fontsize=9)

    fig.tight_layout()
    return fig


def plot_top_movies(movie_stats: pd.DataFrame, n: int = 15) -> plt.Figure:
    """Horizontal bar chart of most-rated movies."""
    top = (
        movie_stats[movie_stats["num_ratings"] > 0]
        .sort_values("num_ratings", ascending=True)
        .tail(n)
    )
    # Short titles
    labels = top["title"].apply(lambda t: t[:35] + "…" if len(t) > 35 else t)

    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.42)))
    _apply_dark_style(fig, ax)

    bar_colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(top))]
    bars = ax.barh(labels, top["num_ratings"], color=bar_colors, edgecolor=_BG, height=0.72)

    ax.set_xlabel("Number of Ratings", fontsize=11)
    ax.set_title(f"Top {n} Most Rated Movies", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, top["num_ratings"].max() * 1.15)

    for bar, val in zip(bars, top["num_ratings"]):
        ax.text(val + top["num_ratings"].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                str(int(val)), va="center", color=_TXT, fontsize=9)

    fig.tight_layout()
    return fig


def plot_top_rated_movies(movie_stats: pd.DataFrame, min_ratings: int = 5, n: int = 15) -> plt.Figure:
    """Bar chart of highest average-rated movies."""
    top = (
        movie_stats[movie_stats["num_ratings"] >= min_ratings]
        .sort_values("avg_rating", ascending=True)
        .tail(n)
    )
    labels = top["title"].apply(lambda t: t[:35] + "…" if len(t) > 35 else t)

    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.42)))
    _apply_dark_style(fig, ax)

    norm_ratings = (top["num_ratings"] - top["num_ratings"].min()) / (top["num_ratings"].max() - top["num_ratings"].min() + 1)
    bar_colors = plt.cm.YlOrRd(0.35 + 0.55 * norm_ratings.values)

    bars = ax.barh(labels, top["avg_rating"], color=bar_colors, edgecolor=_BG, height=0.72)
    ax.set_xlim(0, 5.5)
    ax.set_xlabel("Average Rating (★)", fontsize=11)
    ax.set_title(f"Top {n} Highest Rated Movies (≥{min_ratings} ratings)", fontsize=13, fontweight="bold", pad=12)

    for bar, val in zip(bars, top["avg_rating"]):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}★", va="center", color=_TXT, fontsize=9)

    fig.tight_layout()
    return fig


def plot_user_activity(ratings_df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """Bar chart of most-active users by number of ratings given."""
    user_counts = ratings_df.groupby("userId").size().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    _apply_dark_style(fig, ax)

    bar_colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(user_counts))]
    ax.bar(
        [f"User {u}" for u in user_counts.index],
        user_counts.values,
        color=bar_colors,
        edgecolor=_BG,
        width=0.72,
    )
    ax.set_xlabel("User ID", fontsize=11)
    ax.set_ylabel("Ratings Given", fontsize=11)
    ax.set_title(f"Top {top_n} Most Active Users", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_ratings_over_time(ratings_df: pd.DataFrame) -> plt.Figure:
    """Line chart of rating volume per month."""
    if "timestamp" not in ratings_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        _apply_dark_style(fig, ax)
        ax.text(0.5, 0.5, "No timestamp data", transform=ax.transAxes,
                ha="center", va="center", color=_TXT, fontsize=14)
        return fig

    df = ratings_df.copy()
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df = df.dropna(subset=["date"])
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("year_month").size()

    fig, ax = plt.subplots(figsize=(10, 4))
    _apply_dark_style(fig, ax)

    x = range(len(monthly))
    ax.fill_between(x, monthly.values, alpha=0.3, color=_ACC)
    ax.plot(x, monthly.values, color=_ACC, linewidth=2)

    step = max(1, len(monthly) // 10)
    ax.set_xticks(list(x)[::step])
    ax.set_xticklabels([str(monthly.index[i]) for i in range(0, len(monthly), step)],
                       rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Number of Ratings", fontsize=11)
    ax.set_title("Ratings Volume Over Time", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_hybrid_weight_comparison(recommender, user_id: int, n: int = 10) -> plt.Figure:
    """Bar chart comparing how alpha shifts the recommendation list."""
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = [f"α={a}" for a in alphas]

    # Collect unique top-5 movie titles per alpha
    all_movies: set = set()
    rec_sets = {}
    for a in alphas:
        recs = recommender.recommend_for_user(user_id, n=n, alpha=a)
        if not recs.empty:
            movie_set = set(recs["title"].str[:25].tolist())
            rec_sets[a] = movie_set
            all_movies.update(movie_set)

    movies_list = sorted(all_movies)[:15]
    matrix = np.zeros((len(alphas), len(movies_list)))
    for i, a in enumerate(alphas):
        for j, m in enumerate(movies_list):
            if a in rec_sets and m in rec_sets[a]:
                matrix[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(max(8, len(movies_list) * 0.55), 4))
    _apply_dark_style(fig, ax)

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks(range(len(movies_list)))
    ax.set_xticklabels(movies_list, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Recommendation Coverage by Alpha (User {user_id})", fontsize=13, fontweight="bold", pad=10)

    green_patch = mpatches.Patch(color="green", label="Recommended")
    red_patch   = mpatches.Patch(color="red",   label="Not Recommended")
    ax.legend(handles=[green_patch, red_patch], loc="upper right",
              facecolor=_CARD, labelcolor=_TXT, fontsize=9)

    fig.tight_layout()
    return fig
