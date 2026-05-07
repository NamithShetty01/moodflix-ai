"""
content_filtering.py
--------------------
Genre-based Content-Based Filtering using TF-IDF vectorisation and
cosine similarity.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedFilter:
    """Recommends movies with similar genre profiles."""

    def __init__(self):
        self._movies_df: pd.DataFrame | None = None
        self._tfidf_matrix = None
        self._idx_map: dict | None = None   # movieId → matrix row index

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, movies_df: pd.DataFrame) -> "ContentBasedFilter":
        """Build sparse TF-IDF features from genre strings.

        Parameters
        ----------
        movies_df : DataFrame with columns ['movieId', 'title', 'genres']
            genres values like "Action|Comedy|Thriller"
        """
        self._movies_df = movies_df.reset_index(drop=True).copy()

        # Replace '|' with space so each genre becomes a token
        genre_docs = self._movies_df["genres"].str.replace("|", " ", regex=False)

        tfidf = TfidfVectorizer(token_pattern=r"[^\s]+")
        self._tfidf_matrix = tfidf.fit_transform(genre_docs)

        self._idx_map = {
            int(row["movieId"]): idx
            for idx, row in self._movies_df.iterrows()
        }
        return self

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_similar_movies(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """Return the top-N most genre-similar movies.

        Returns a DataFrame with columns [movieId, cb_score].
        """
        if self._tfidf_matrix is None:
            raise RuntimeError("Call fit() before get_similar_movies().")

        if movie_id not in self._idx_map:
            return pd.DataFrame(columns=["movieId", "cb_score"])

        idx = self._idx_map[movie_id]
        # Compute only one row of similarities on demand.
        sim_scores = cosine_similarity(
            self._tfidf_matrix[idx],
            self._tfidf_matrix,
            dense_output=False,
        ).toarray().ravel()
        sim_scores[idx] = 0.0  # exclude the query movie itself

        top_indices = np.argsort(sim_scores)[::-1][:n]
        result = pd.DataFrame(
            {
                "movieId": self._movies_df.iloc[top_indices]["movieId"].values,
                "cb_score": sim_scores[top_indices],
            }
        )
        return result.reset_index(drop=True)

    def get_score(self, movie_id_a: int, movie_id_b: int) -> float:
        """Return the cosine similarity between two movies."""
        if self._tfidf_matrix is None:
            raise RuntimeError("Call fit() before get_score().")
        if movie_id_a not in self._idx_map or movie_id_b not in self._idx_map:
            return 0.0

        ia = self._idx_map[movie_id_a]
        ib = self._idx_map[movie_id_b]
        score = cosine_similarity(
            self._tfidf_matrix[ia],
            self._tfidf_matrix[ib],
            dense_output=False,
        )
        return float(score.toarray().ravel()[0])

    def get_scores_for_profile(
        self, liked_movie_ids: list, n: int = 20
    ) -> pd.DataFrame:
        """Aggregate CB scores for a user whose liked movies are known.

        Averages similarity scores across all liked movies, then returns
        the top-N candidates (excluding the liked movies themselves).

        Returns a DataFrame with columns [movieId, cb_score].
        """
        if self._tfidf_matrix is None or self._movies_df is None:
            raise RuntimeError("Call fit() first.")

        valid_ids = [m for m in liked_movie_ids if m in self._idx_map]
        if not valid_ids:
            return pd.DataFrame(columns=["movieId", "cb_score"])

        indices = [self._idx_map[m] for m in valid_ids]

        # Build a sparse user-profile vector and compute one-vs-all similarities.
        profile_vec = np.asarray(self._tfidf_matrix[indices].mean(axis=0)).reshape(1, -1)
        agg_scores = cosine_similarity(
            profile_vec,
            self._tfidf_matrix,
            dense_output=False,
        )
        if hasattr(agg_scores, "toarray"):
            agg_scores = agg_scores.toarray().ravel()
        else:
            agg_scores = np.asarray(agg_scores).ravel()

        # Zero out the seed movies
        for idx in indices:
            agg_scores[idx] = 0.0

        top_indices = np.argsort(agg_scores)[::-1][:n]
        result = pd.DataFrame(
            {
                "movieId": self._movies_df.iloc[top_indices]["movieId"].values,
                "cb_score": agg_scores[top_indices],
            }
        )
        return result.reset_index(drop=True)

    @property
    def movie_ids(self) -> list:
        if self._movies_df is None:
            return []
        return self._movies_df["movieId"].tolist()
