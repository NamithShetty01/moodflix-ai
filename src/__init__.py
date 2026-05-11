"""Movie Recommendation System source package.

The original project imported these as separate modules:
`src.collaborative_filtering`, `src.content_filtering`, and
`src.hybrid_recommender`. Some copies of the workspace no longer contain those
files, so we register compatible in-memory modules here to keep imports working
without changing the app code.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SVDCollaborativeFilter:
	def __init__(self, n_factors: int = 50, random_state: int = 42):
		self.n_factors = int(n_factors)
		self.random_state = int(random_state)
		self.user_index = None
		self.item_index = None
		self.user_factors = None
		self.item_factors = None
		self.item_means = None

	def fit(self, user_item_matrix: pd.DataFrame) -> None:
		if user_item_matrix.empty:
			self.user_index = pd.Index([])
			self.item_index = pd.Index([])
			self.user_factors = np.empty((0, 0))
			self.item_factors = np.empty((0, 0))
			self.item_means = np.array([])
			return

		self.user_index = user_item_matrix.index
		self.item_index = user_item_matrix.columns
		values = user_item_matrix.to_numpy(dtype=float)
		means = np.nanmean(values, axis=0)
		self.item_means = np.where(np.isfinite(means), means, np.nanmean(values))
		centered = np.nan_to_num(values - self.item_means, nan=0.0)
		components = max(1, min(centered.shape[0] - 1, centered.shape[1] - 1, self.n_factors))
		svd = TruncatedSVD(n_components=components, random_state=self.random_state)
		self.user_factors = svd.fit_transform(centered)
		self.item_factors = svd.components_.T * svd.singular_values_

	def predict_rating(self, user_id: int, movie_id: int) -> float:
		if self.user_factors is None or self.item_factors is None:
			return 0.0
		try:
			uidx = int(self.user_index.get_loc(user_id))
			iidx = int(self.item_index.get_loc(movie_id))
		except Exception:
			return 0.0
		score = float(np.dot(self.user_factors[uidx], self.item_factors[iidx]) + self.item_means[iidx])
		return float(np.clip(score, 0.5, 5.0))


class ContentFilter:
	def __init__(self):
		self.movie_ids = None
		self.matrix = None

	def fit(self, movies_df: pd.DataFrame) -> None:
		if movies_df.empty:
			self.movie_ids = pd.Index([])
			self.matrix = None
			return
		self.movie_ids = movies_df["movieId"].astype(int).reset_index(drop=True)
		docs = movies_df["genres"].fillna("").astype(str).str.replace("|", " ", regex=False)
		self.matrix = TfidfVectorizer(stop_words="english").fit_transform(docs)

	def similar_movies(self, movie_id: int, top_n: int = 10) -> list[int]:
		if self.matrix is None or self.movie_ids is None:
			return []
		matches = self.movie_ids[self.movie_ids == movie_id]
		if matches.empty:
			return []
		idx = int(matches.index[0])
		sims = cosine_similarity(self.matrix[idx], self.matrix).ravel()
		sims[idx] = -1.0
		best = np.argsort(-sims)[:top_n]
		return [int(self.movie_ids.iloc[i]) for i in best]


class HybridRecommender:
	def __init__(self, n_factors: int = 50):
		self.n_factors = int(n_factors)
		self._cf = SVDCollaborativeFilter(n_factors=n_factors)
		self._content = ContentFilter()
		self.data = {}
		self.valid_user_ids = []

	def fit(self, data: dict) -> None:
		self.data = data
		self._cf.fit(data.get("user_item_matrix", pd.DataFrame()))
		self._content.fit(data.get("movies", pd.DataFrame()))
		ratings = data.get("ratings", pd.DataFrame())
		if not ratings.empty:
			self.valid_user_ids = sorted(ratings["userId"].astype(int).unique().tolist())

	def _movies(self) -> pd.DataFrame:
		return self.data.get("movies", pd.DataFrame())

	def _stats(self) -> pd.DataFrame:
		return self.data.get("movie_stats", pd.DataFrame())

	def search_movies(self, token: str, limit: int = 20) -> pd.DataFrame:
		movies = self._movies()
		if movies.empty or not token:
			return pd.DataFrame()
		return movies[movies["title"].str.contains(token, case=False, na=False)].head(limit).copy()

	def recommend_for_user(self, user_id: int, n: int = 10, alpha: float = 0.5) -> pd.DataFrame:
		movies = self._movies()
		if movies.empty:
			return pd.DataFrame()
		stats = self._stats().set_index("movieId") if not self._stats().empty else pd.DataFrame()
		rows = []
		for _, movie in movies.iterrows():
			mid = int(movie["movieId"])
			cf_score = self._cf.predict_rating(int(user_id), mid)
			pop_score = float(stats.loc[mid, "popularity_score"]) if not stats.empty and mid in stats.index else 0.0
			rows.append({**movie.to_dict(), "score": (1.0 - alpha) * cf_score + alpha * pop_score})
		return pd.DataFrame(rows).sort_values("score", ascending=False).head(n).reset_index(drop=True)

	def get_trending(self, n: int = 10) -> pd.DataFrame:
		stats = self._stats()
		return stats.sort_values("popularity_score", ascending=False).head(n).reset_index(drop=True) if not stats.empty else pd.DataFrame()

	def get_popular(self, n: int = 10) -> pd.DataFrame:
		return self.get_trending(n)

	def get_top_rated(self, min_ratings: int = 3, n: int = 10) -> pd.DataFrame:
		stats = self._stats()
		if stats.empty:
			return pd.DataFrame()
		return stats[stats["num_ratings"] >= min_ratings].sort_values(["avg_rating", "num_ratings"], ascending=False).head(n).reset_index(drop=True)

	def recommend_similar_to_movie(self, movie_id: int, n: int = 10, alpha: float = 0.3) -> pd.DataFrame:
		movies = self._movies().set_index("movieId")
		stats = self._stats().set_index("movieId") if not self._stats().empty else pd.DataFrame()
		rows = []
		for mid in self._content.similar_movies(int(movie_id), top_n=n * 3):
			if mid not in movies.index:
				continue
			row = movies.loc[mid].to_dict()
			row["score"] = float(stats.loc[mid, "popularity_score"]) if not stats.empty and mid in stats.index else 0.0
			rows.append(row)
		return pd.DataFrame(rows).sort_values("score", ascending=False).head(n).reset_index(drop=True)

	def get_movie_detail(self, movie_id: int) -> dict:
		movies = self._movies()
		if movies.empty:
			return {}
		row = movies[movies["movieId"] == int(movie_id)]
		return row.iloc[0].to_dict() if not row.empty else {}


def _register_module(name: str, attrs: dict) -> None:
	module = types.ModuleType(name)
	module.__dict__.update(attrs)
	sys.modules[name] = module


_register_module(__name__ + ".collaborative_filtering", {"SVDCollaborativeFilter": SVDCollaborativeFilter})
_register_module(__name__ + ".content_filtering", {"ContentFilter": ContentFilter})
_register_module(__name__ + ".hybrid_recommender", {"HybridRecommender": HybridRecommender})

