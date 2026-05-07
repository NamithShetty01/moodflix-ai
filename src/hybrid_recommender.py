"""
hybrid_recommender.py
---------------------
Combines Collaborative Filtering (SVD) and Content-Based Filtering (genre
cosine similarity) using the weighted formula:

    Final Score = α × CF_score + (1 − α) × CB_score

Both component scores are normalised to [0, 1] before blending.
"""

import pandas as pd
import numpy as np

from src.collaborative_filtering import SVDCollaborativeFilter
from src.content_filtering import ContentBasedFilter
from src.data_preprocessing import preprocess_pipeline


class HybridRecommender:
    """Hybrid movie recommender combining SVD-CF and genre-CB filtering."""

    def __init__(self, n_factors: int = 50):
        self._cf = SVDCollaborativeFilter(n_factors=n_factors)
        self._cb = ContentBasedFilter()
        self._data: dict | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, data: dict) -> "HybridRecommender":
        """Train both sub-models on the preprocessed data bundle.

        Parameters
        ----------
        data : output of ``preprocess_pipeline()``
        """
        self._data = data
        self._cf.fit(data["user_item_matrix"])
        self._cb.fit(data["movies"])
        return self

    # ------------------------------------------------------------------
    # Core recommendation methods
    # ------------------------------------------------------------------

    def recommend_for_user(
        self,
        user_id: int,
        n: int = 10,
        alpha: float = 0.5,
    ) -> pd.DataFrame:
        """Return top-N hybrid recommendations for *user_id*.

        Parameters
        ----------
        user_id : int
        n       : number of movies to return
        alpha   : weight for CF score (0 = pure CB, 1 = pure CF)

        Returns a DataFrame with columns:
            [movieId, title, genres, avg_rating, num_ratings,
             cf_score, cb_score, hybrid_score]
        """
        if self._data is None:
            raise RuntimeError("Call fit() before recommend_for_user().")

        movie_stats = self._data["movie_stats"]
        ratings_df  = self._data["ratings"]

        # --- CF scores for all movies for this user ----------------------
        cf_all = self._cf.get_all_scores_for_user(user_id)

        if cf_all.empty:
            # Cold-start: fall back to popularity + CB
            alpha = 0.0
            cf_series = pd.Series(dtype=float)
        else:
            # Normalise
            mn, mx = cf_all.min(), cf_all.max()
            cf_series = (cf_all - mn) / (mx - mn) if mx > mn else pd.Series(1.0, index=cf_all.index)

        # --- CB profile: movies user rated ≥ 3.5 ------------------------
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        liked_ids = user_ratings[user_ratings["rating"] >= 3.5]["movieId"].tolist()
        if not liked_ids:
            liked_ids = user_ratings["movieId"].tolist()

        seen_ids = set(user_ratings["movieId"].tolist())

        cb_df = self._cb.get_scores_for_profile(liked_ids, n=len(movie_stats))
        cb_series = cb_df.set_index("movieId")["cb_score"]

        # --- Merge into candidate pool -----------------------------------
        all_movie_ids = movie_stats["movieId"].tolist()
        rows = []
        for mid in all_movie_ids:
            if mid in seen_ids:
                continue
            cf_s = float(cf_series.get(mid, 0.0))
            cb_s = float(cb_series.get(mid, 0.0))
            hybrid = alpha * cf_s + (1.0 - alpha) * cb_s
            rows.append({"movieId": mid, "cf_score": cf_s, "cb_score": cb_s, "hybrid_score": hybrid})

        if not rows:
            return pd.DataFrame()

        scores_df = pd.DataFrame(rows).sort_values("hybrid_score", ascending=False).head(n)
        result = scores_df.merge(movie_stats[["movieId", "title", "genres", "avg_rating", "num_ratings"]], on="movieId", how="left")
        return result.reset_index(drop=True)

    def recommend_similar_to_movie(
        self,
        movie_id: int,
        n: int = 10,
        alpha: float = 0.3,
    ) -> pd.DataFrame:
        """Return movies similar to *movie_id* using a CB-dominant hybrid.

        Parameters
        ----------
        movie_id : seed movie
        n        : number of results
        alpha    : weight for CF (kept low since we're doing item-similarity)
        """
        if self._data is None:
            raise RuntimeError("Call fit() before recommend_similar_to_movie().")

        movie_stats = self._data["movie_stats"]

        # CB scores
        cb_df = self._cb.get_similar_movies(movie_id, n=len(movie_stats))
        cb_series = cb_df.set_index("movieId")["cb_score"]

        # CF: average predicted rating across all users as a proxy global score
        cf_global = self._cf._predicted_df.mean(axis=0) if self._cf._predicted_df is not None else pd.Series(dtype=float)
        if not cf_global.empty:
            mn, mx = cf_global.min(), cf_global.max()
            cf_global_norm = (cf_global - mn) / (mx - mn) if mx > mn else pd.Series(1.0, index=cf_global.index)
        else:
            cf_global_norm = pd.Series(dtype=float)

        rows = []
        for mid in movie_stats["movieId"].tolist():
            if mid == movie_id:
                continue
            cb_s = float(cb_series.get(mid, 0.0))
            cf_s = float(cf_global_norm.get(mid, 0.0))
            hybrid = alpha * cf_s + (1.0 - alpha) * cb_s
            rows.append({"movieId": mid, "cf_score": cf_s, "cb_score": cb_s, "hybrid_score": hybrid})

        scores_df = pd.DataFrame(rows).sort_values("hybrid_score", ascending=False).head(n)
        result = scores_df.merge(movie_stats[["movieId", "title", "genres", "avg_rating", "num_ratings"]], on="movieId", how="left")
        return result.reset_index(drop=True)

    def get_trending(self, n: int = 20) -> pd.DataFrame:
        """Return trending movies ranked by popularity_score."""
        if self._data is None:
            raise RuntimeError("Call fit() first.")
        return (
            self._data["movie_stats"]
            .sort_values("popularity_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def get_top_rated(self, min_ratings: int = 5, n: int = 20) -> pd.DataFrame:
        """Return top-rated movies with at least *min_ratings* ratings."""
        if self._data is None:
            raise RuntimeError("Call fit() first.")
        filtered = self._data["movie_stats"][self._data["movie_stats"]["num_ratings"] >= min_ratings]
        return filtered.sort_values("avg_rating", ascending=False).head(n).reset_index(drop=True)

    def get_popular(self, n: int = 20) -> pd.DataFrame:
        """Return movies sorted by number of ratings."""
        if self._data is None:
            raise RuntimeError("Call fit() first.")
        return (
            self._data["movie_stats"]
            .sort_values("num_ratings", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def search_movies(self, query: str) -> pd.DataFrame:
        """Case-insensitive substring search on movie titles."""
        if self._data is None:
            raise RuntimeError("Call fit() first.")
        mask = self._data["movie_stats"]["title"].str.contains(query, case=False, na=False)
        return self._data["movie_stats"][mask].reset_index(drop=True)

    def get_movie_detail(self, movie_id: int) -> dict | None:
        """Return detail dict for one movie."""
        if self._data is None:
            return None
        row = self._data["movie_stats"][self._data["movie_stats"]["movieId"] == movie_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    @property
    def valid_user_ids(self) -> list:
        return self._cf.valid_user_ids

    @property
    def data(self) -> dict | None:
        return self._data
