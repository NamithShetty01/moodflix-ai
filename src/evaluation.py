"""
evaluation.py
-------------
Offline evaluation metrics for the recommendation system.

Metrics implemented
-------------------
- RMSE  (Root Mean Squared Error)
- MAE   (Mean Absolute Error)
- Precision@K
- Recall@K
- F1@K
- Coverage
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Rating-prediction metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-k recommendations that are relevant."""
    recommended_k = recommended[:k]
    hits = sum(1 for r in recommended_k if r in relevant)
    return hits / k if k else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items found in top-k recommendations."""
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    hits = sum(1 for r in recommended_k if r in relevant)
    return hits / len(relevant)


def f1_at_k(recommended: list, relevant: set, k: int) -> float:
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_cf(cf_model, ratings_df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """Hold-out evaluation of the CF model on rating-prediction accuracy.

    Splits ratings into train/test, refits the model on train, and evaluates
    predicted vs. actual ratings on the test set.

    Returns a dict with keys: rmse, mae, n_test_samples.
    """
    from src.data_preprocessing import create_user_item_matrix
    from src.collaborative_filtering import SVDCollaborativeFilter

    train_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=42)

    train_matrix = create_user_item_matrix(train_df)
    eval_cf = SVDCollaborativeFilter(n_factors=min(50, min(train_matrix.shape) - 1))
    eval_cf.fit(train_matrix)

    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        uid, mid, actual = int(row["userId"]), int(row["movieId"]), float(row["rating"])
        predicted = eval_cf.predict_rating(uid, mid)
        if predicted != 0.0:  # 0 means unknown user/movie
            y_true.append(actual)
            y_pred.append(predicted)

    if not y_true:
        return {"rmse": None, "mae": None, "n_test_samples": 0}

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    return {
        "rmse": round(rmse(y_true_arr, y_pred_arr), 4),
        "mae":  round(mae(y_true_arr, y_pred_arr), 4),
        "n_test_samples": len(y_true),
    }


def evaluate_ranking(recommender, ratings_df: pd.DataFrame, k: int = 10) -> dict:
    """Evaluate top-k recommendation quality using Precision@K and Recall@K.

    Samples up to 50 users that have enough ratings, computes CF-only
    recommendations, then checks how many of the held-out highly-rated
    movies appear in the top-k.

    Returns a dict with: avg_precision_at_k, avg_recall_at_k, avg_f1_at_k.
    """
    rng = np.random.default_rng(0)
    user_counts = ratings_df.groupby("userId").size()
    eligible_users = user_counts[user_counts >= 10].index.tolist()
    sampled_users = rng.choice(eligible_users, size=min(50, len(eligible_users)), replace=False)

    p_scores, r_scores, f1_scores = [], [], []

    for uid in sampled_users:
        user_ratings = ratings_df[ratings_df["userId"] == uid]
        relevant = set(user_ratings[user_ratings["rating"] >= 4.0]["movieId"].tolist())
        if not relevant:
            continue
        recs = recommender.recommend_for_user(int(uid), n=k, alpha=0.5)
        if recs.empty:
            continue
        rec_ids = recs["movieId"].tolist()
        p_scores.append(precision_at_k(rec_ids, relevant, k))
        r_scores.append(recall_at_k(rec_ids, relevant, k))
        f1_scores.append(f1_at_k(rec_ids, relevant, k))

    return {
        "avg_precision_at_k": round(float(np.mean(p_scores)), 4) if p_scores else None,
        "avg_recall_at_k":    round(float(np.mean(r_scores)), 4) if r_scores else None,
        "avg_f1_at_k":        round(float(np.mean(f1_scores)), 4) if f1_scores else None,
        "k": k,
        "users_evaluated": len(p_scores),
    }
