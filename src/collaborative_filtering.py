"""
collaborative_filtering.py
--------------------------
SVD-based Matrix-Factorisation Collaborative Filtering.

Uses scipy.sparse.linalg.svds to decompose the user-item matrix and predicts
ratings for all user-movie pairs that were not yet seen.
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


class SVDCollaborativeFilter:
    """Matrix-factorisation recommender using truncated SVD."""

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self._predicted_df: pd.DataFrame | None = None
        self._user_item: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, user_item_matrix: pd.DataFrame) -> "SVDCollaborativeFilter":
        """Decompose the user-item matrix with truncated SVD.

        Parameters
        ----------
        user_item_matrix : DataFrame[userId × movieId]
            NaN cells represent unseen (userId, movieId) pairs.
        """
        self._user_item = user_item_matrix.copy()

        # Fill NaN with each user's mean rating (mean-centring approach)
        matrix = user_item_matrix.copy()
        user_means = matrix.mean(axis=1)
        matrix_filled = matrix.T.fillna(user_means).T

        R = matrix_filled.values.astype(float)
        user_ratings_mean = np.nanmean(R, axis=1)
        R_demeaned = R - user_ratings_mean[:, np.newaxis]

        # k must be < min(rows, cols)
        k = min(self.n_factors, min(R_demeaned.shape) - 1)
        U, sigma, Vt = svds(R_demeaned, k=k)
        sigma_diag = np.diag(sigma)

        predicted = np.dot(np.dot(U, sigma_diag), Vt) + user_ratings_mean[:, np.newaxis]
        self._predicted_df = pd.DataFrame(
            predicted,
            index=user_item_matrix.index,
            columns=user_item_matrix.columns,
        )
        return self

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Return predicted rating for (user_id, movie_id); 0.0 if unknown."""
        if self._predicted_df is None:
            raise RuntimeError("Call fit() before predict_rating().")
        if user_id not in self._predicted_df.index:
            return 0.0
        if movie_id not in self._predicted_df.columns:
            return 0.0
        return float(self._predicted_df.loc[user_id, movie_id])

    def get_user_recommendations(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """Return top-N movies for a user sorted by predicted rating.

        Returns a DataFrame with columns [movieId, cf_score].
        """
        if self._predicted_df is None:
            raise RuntimeError("Call fit() before get_user_recommendations().")

        if user_id not in self._predicted_df.index:
            # Cold-start: return empty
            return pd.DataFrame(columns=["movieId", "cf_score"])

        scores = self._predicted_df.loc[user_id].copy()

        if exclude_seen and self._user_item is not None:
            seen_mask = self._user_item.loc[user_id].notna()
            scores = scores[~seen_mask]

        scores = scores.sort_values(ascending=False).head(n)

        result = pd.DataFrame({"movieId": scores.index, "cf_score": scores.values})
        # Normalise to [0, 1]
        mn, mx = result["cf_score"].min(), result["cf_score"].max()
        if mx > mn:
            result["cf_score"] = (result["cf_score"] - mn) / (mx - mn)
        else:
            result["cf_score"] = 1.0
        return result.reset_index(drop=True)

    def get_all_scores_for_user(self, user_id: int) -> pd.Series:
        """Return the full predicted-ratings series for a user (all movies)."""
        if self._predicted_df is None:
            raise RuntimeError("Call fit() before get_all_scores_for_user().")
        if user_id not in self._predicted_df.index:
            return pd.Series(dtype=float)
        return self._predicted_df.loc[user_id].copy()

    @property
    def valid_user_ids(self) -> list:
        if self._predicted_df is None:
            return []
        return list(self._predicted_df.index)
