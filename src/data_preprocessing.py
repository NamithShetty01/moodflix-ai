"""
data_preprocessing.py
---------------------
Handles loading, downloading, and preprocessing the MovieLens dataset.
Provides utility functions consumed by all other modules.
"""

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MOVIELENS_URL = (
    "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
)


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def download_movielens(data_dir: Path = DATA_DIR) -> bool:
    """Download the latest MovieLens dataset if not already present.

    Returns True on success, False on failure.
    """
    try:
        import requests  # noqa: PLC0415

        data_dir.mkdir(parents=True, exist_ok=True)
        print("Downloading MovieLens dataset …")
        resp = requests.get(MOVIELENS_URL, stream=True, timeout=60)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            for name in z.namelist():
                fname = os.path.basename(name)
                if fname in ("movies.csv", "ratings.csv", "links.csv"):
                    dest = data_dir / fname
                    dest.write_bytes(z.read(name))
                    print(f"  ✓ Extracted {fname}")

        print("Download complete.")
        return True

    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {exc}")
        return False


def _pick_latest_csv(data_dir: Path, base_name: str) -> Path:
    """Return newest CSV matching a base name pattern.

    Examples for base_name='movies': movies.csv, movies_latest.csv, movies_2026.csv
    """
    candidates = sorted(data_dir.glob(f"{base_name}*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return data_dir / f"{base_name}.csv"
    return candidates[0]


def create_sample_data(data_dir: Path = DATA_DIR) -> None:
    """Generate a realistic MovieLens-format sample dataset for offline use."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # 110 well-known movies
    _movies = [
        (1,  "Toy Story (1995)",                               "Adventure|Animation|Children|Comedy|Fantasy"),
        (2,  "Jumanji (1995)",                                 "Adventure|Children|Fantasy"),
        (3,  "Grumpier Old Men (1995)",                        "Comedy|Romance"),
        (4,  "Waiting to Exhale (1995)",                       "Comedy|Drama|Romance"),
        (5,  "Father of the Bride Part II (1995)",             "Comedy"),
        (6,  "Heat (1995)",                                    "Action|Crime|Thriller"),
        (7,  "Sabrina (1995)",                                 "Comedy|Romance"),
        (8,  "Tom and Huck (1995)",                            "Adventure|Children"),
        (9,  "Sudden Death (1995)",                            "Action"),
        (10, "GoldenEye (1995)",                               "Action|Adventure|Thriller"),
        (11, "The American President (1995)",                  "Comedy|Drama|Romance"),
        (12, "Dracula: Dead and Loving It (1995)",             "Comedy|Horror"),
        (13, "Balto (1995)",                                   "Adventure|Animation|Children"),
        (14, "Nixon (1995)",                                   "Drama"),
        (15, "Cutthroat Island (1995)",                        "Action|Adventure|Romance"),
        (16, "Casino (1995)",                                  "Crime|Drama"),
        (17, "Sense and Sensibility (1995)",                   "Drama|Romance"),
        (18, "Four Rooms (1995)",                              "Comedy|Mystery"),
        (19, "Ace Ventura: When Nature Calls (1995)",          "Comedy"),
        (20, "Money Train (1995)",                             "Action|Comedy|Crime"),
        (21, "Get Shorty (1995)",                              "Comedy|Crime|Thriller"),
        (22, "Copycat (1995)",                                 "Crime|Drama|Thriller"),
        (23, "Assassins (1995)",                               "Action|Crime|Thriller"),
        (24, "Powder (1995)",                                  "Drama|Sci-Fi"),
        (25, "Leaving Las Vegas (1995)",                       "Drama|Romance"),
        (26, "Othello (1995)",                                 "Drama|Romance"),
        (27, "Now and Then (1995)",                            "Children|Comedy|Drama"),
        (28, "Persuasion (1995)",                              "Drama|Romance"),
        (29, "City of Lost Children, The (1995)",              "Adventure|Drama|Fantasy|Mystery|Sci-Fi"),
        (30, "Shanghai Triad (1995)",                          "Crime|Drama"),
        (31, "Dangerous Minds (1995)",                         "Drama"),
        (32, "Twelve Monkeys (1995)",                          "Mystery|Sci-Fi|Thriller"),
        (33, "Babe (1995)",                                    "Children|Comedy|Drama|Fantasy"),
        (34, "Dead Man (1995)",                                "Drama|Mystery|Western"),
        (35, "Clueless (1995)",                                "Comedy|Romance"),
        (36, "Dead Presidents (1995)",                         "Action|Crime|Drama"),
        (37, "Se7en (1995)",                                   "Crime|Mystery|Thriller"),
        (38, "Pocahontas (1995)",                              "Animation|Children|Musical|Romance"),
        (39, "Usual Suspects, The (1995)",                     "Crime|Mystery|Thriller"),
        (40, "Mighty Aphrodite (1995)",                        "Comedy|Drama|Romance"),
        (41, "From Dusk Till Dawn (1996)",                     "Action|Comedy|Horror|Thriller"),
        (42, "Strange Days (1995)",                            "Action|Crime|Drama|Mystery|Sci-Fi|Thriller"),
        (43, "Friday (1995)",                                  "Comedy|Crime"),
        (44, "Mortal Kombat (1995)",                           "Action|Adventure"),
        (45, "To Die For (1995)",                              "Comedy|Drama|Thriller"),
        (46, "Die Hard: With a Vengeance (1995)",              "Action|Crime|Thriller"),
        (47, "Batman Forever (1995)",                          "Action|Adventure|Comedy"),
        (48, "Waterworld (1995)",                              "Action|Adventure|Sci-Fi"),
        (49, "Species (1995)",                                 "Horror|Sci-Fi"),
        (50, "Braveheart (1995)",                              "Action|Drama|War"),
        (51, "Apollo 13 (1995)",                               "Adventure|Drama"),
        (52, "Rob Roy (1995)",                                 "Action|Drama|Romance|War"),
        (53, "Natural Born Killers (1994)",                    "Action|Crime|Thriller"),
        (54, "Leon: The Professional (1994)",                  "Action|Crime|Drama|Thriller"),
        (55, "Pulp Fiction (1994)",                            "Comedy|Crime|Drama|Thriller"),
        (56, "Shawshank Redemption, The (1994)",               "Crime|Drama"),
        (57, "Forrest Gump (1994)",                            "Comedy|Drama|Romance|War"),
        (58, "Schindler's List (1993)",                        "Drama|War"),
        (59, "Silence of the Lambs, The (1991)",               "Crime|Horror|Thriller"),
        (60, "Goodfellas (1990)",                              "Crime|Drama"),
        (61, "Rain Man (1988)",                                "Drama"),
        (62, "E.T. the Extra-Terrestrial (1982)",              "Children|Drama|Sci-Fi"),
        (63, "Raiders of the Lost Ark (1981)",                 "Action|Adventure"),
        (64, "Star Wars: Episode IV - A New Hope (1977)",      "Action|Adventure|Sci-Fi"),
        (65, "Godfather, The (1972)",                          "Crime|Drama"),
        (66, "Citizen Kane (1941)",                            "Drama|Mystery"),
        (67, "Casablanca (1942)",                              "Drama|Romance|War"),
        (68, "Wizard of Oz, The (1939)",                       "Adventure|Children|Fantasy|Musical"),
        (69, "2001: A Space Odyssey (1968)",                   "Adventure|Drama|Sci-Fi"),
        (70, "Matrix, The (1999)",                             "Action|Sci-Fi|Thriller"),
        (71, "Fight Club (1999)",                              "Action|Crime|Drama|Thriller"),
        (72, "American Beauty (1999)",                         "Drama|Romance"),
        (73, "Sixth Sense, The (1999)",                        "Drama|Horror|Mystery"),
        (74, "Magnolia (1999)",                                "Drama"),
        (75, "Being John Malkovich (1999)",                    "Comedy|Drama|Fantasy"),
        (76, "Toy Story 2 (1999)",                             "Adventure|Animation|Children|Comedy|Fantasy"),
        (77, "Iron Giant, The (1999)",                         "Action|Adventure|Animation|Children|Drama|Sci-Fi"),
        (78, "American Pie (1999)",                            "Comedy|Romance"),
        (79, "Blair Witch Project, The (1999)",                "Horror|Mystery|Thriller"),
        (80, "Gladiator (2000)",                               "Action|Adventure|Drama"),
        (81, "Cast Away (2000)",                               "Drama"),
        (82, "Crouching Tiger, Hidden Dragon (2000)",          "Action|Drama|Romance"),
        (83, "Almost Famous (2000)",                           "Comedy|Drama|Romance"),
        (84, "Snatch (2000)",                                  "Comedy|Crime"),
        (85, "Traffic (2000)",                                 "Crime|Drama|Thriller"),
        (86, "Requiem for a Dream (2000)",                     "Drama"),
        (87, "Memento (2000)",                                 "Mystery|Thriller"),
        (88, "O Brother, Where Art Thou? (2000)",              "Adventure|Comedy|Crime"),
        (89, "Fellowship of the Ring, The (2001)",             "Adventure|Fantasy"),
        (90, "Harry Potter and the Sorcerer's Stone (2001)",   "Adventure|Children|Fantasy"),
        (91, "A Beautiful Mind (2001)",                        "Drama|Romance"),
        (92, "Moulin Rouge! (2001)",                           "Drama|Musical|Romance"),
        (93, "Ocean's Eleven (2001)",                          "Crime|Thriller"),
        (94, "Shrek (2001)",                                   "Adventure|Animation|Children|Comedy|Fantasy|Romance"),
        (95, "Monsters Inc. (2001)",                           "Adventure|Animation|Children|Comedy|Fantasy"),
        (96, "Black Hawk Down (2001)",                         "Action|Drama|War"),
        (97, "Training Day (2001)",                            "Crime|Drama|Thriller"),
        (98, "Amelie (2001)",                                  "Comedy|Romance"),
        (99, "Dark Knight, The (2008)",                        "Action|Crime|Drama|Thriller"),
        (100,"Inception (2010)",                               "Action|Adventure|Sci-Fi|Thriller"),
        (101,"Interstellar (2014)",                            "Adventure|Drama|Sci-Fi"),
        (102,"Avengers, The (2012)",                           "Action|Adventure|Fantasy|Sci-Fi"),
        (103,"Guardians of the Galaxy (2014)",                 "Action|Adventure|Comedy|Sci-Fi"),
        (104,"Mad Max: Fury Road (2015)",                      "Action|Adventure|Sci-Fi|Thriller"),
        (105,"Revenant, The (2015)",                           "Adventure|Drama|Western"),
        (106,"Spotlight (2015)",                               "Drama|Thriller"),
        (107,"Martian, The (2015)",                            "Drama|Sci-Fi"),
        (108,"Ex Machina (2015)",                              "Drama|Sci-Fi|Thriller"),
        (109,"Get Out (2017)",                                 "Horror|Mystery|Thriller"),
        (110,"Parasite (2019)",                                "Comedy|Drama|Thriller"),
    ]

    movies_df = pd.DataFrame(_movies, columns=["movieId", "title", "genres"])

    # Simulate realistic ratings (popular movies attract more ratings)
    rng = np.random.default_rng(42)
    n_users = 150
    movie_ids = movies_df["movieId"].tolist()
    popularity_weights = rng.exponential(1.0, len(movie_ids))
    popularity_weights /= popularity_weights.sum()

    seen: set = set()
    rows = []
    while len(rows) < 6000:
        uid = rng.integers(1, n_users + 1)
        mid = rng.choice(movie_ids, p=popularity_weights)
        if (uid, mid) not in seen:
            seen.add((uid, mid))
            raw = rng.normal(3.5, 0.9)
            rating = float(np.clip(round(raw * 2) / 2, 0.5, 5.0))
            ts = int(rng.integers(964_980_868, 1_700_000_000))
            rows.append({"userId": uid, "movieId": mid, "rating": rating, "timestamp": ts})

    ratings_df = pd.DataFrame(rows)
    movies_df.to_csv(data_dir / "movies.csv", index=False)
    ratings_df.to_csv(data_dir / "ratings.csv", index=False)
    print(
        f"Sample data created: {len(movies_df)} movies, "
        f"{len(ratings_df)} ratings, {n_users} users."
    )


# ---------------------------------------------------------------------------
# Loading / preprocessing
# ---------------------------------------------------------------------------

def load_data(data_dir: Path = DATA_DIR):
    """Load movies.csv, ratings.csv, and links.csv (if available)."""
    movies_path = _pick_latest_csv(data_dir, "movies")
    ratings_path = _pick_latest_csv(data_dir, "ratings")
    links_path = _pick_latest_csv(data_dir, "links")

    if not movies_path.exists() or not ratings_path.exists():
        print("Data files not found — attempting download …")
        if not download_movielens(data_dir):
            print("Download failed — generating sample data …")
            create_sample_data(data_dir)
        # Re-resolve paths in case download or sample generation created files.
        movies_path = _pick_latest_csv(data_dir, "movies")
        ratings_path = _pick_latest_csv(data_dir, "ratings")
        links_path = _pick_latest_csv(data_dir, "links")

    movies_df  = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    print(f"Using movies file: {movies_path.name}")
    print(f"Using ratings file: {ratings_path.name}")

    # Load links (movieId → imdbId, tmdbId)
    if links_path.exists():
        links_df = pd.read_csv(links_path)
        links_df["movieId"] = links_df["movieId"].astype(int)
        # tmdbId may have NaN floats — convert to nullable int string
        links_df["tmdbId"] = pd.to_numeric(links_df["tmdbId"], errors="coerce")
        links_df["imdbId"] = pd.to_numeric(links_df["imdbId"], errors="coerce")
    else:
        links_df = pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"])

    # Normalise
    movies_df["title"] = movies_df["title"].str.strip()
    movies_df["genres"] = movies_df["genres"].fillna("(no genres listed)")
    movies_df = movies_df.drop_duplicates("movieId").reset_index(drop=True)
    movies_df["movieId"] = movies_df["movieId"].astype(int)

    ratings_df = ratings_df.drop_duplicates(["userId", "movieId"]).reset_index(drop=True)
    ratings_df["userId"] = ratings_df["userId"].astype(int)
    ratings_df["movieId"] = ratings_df["movieId"].astype(int)
    ratings_df["rating"] = ratings_df["rating"].astype(float)

    return movies_df, ratings_df, links_df


def create_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Return a user × movie pivot table (NaN where unrated).

    For very large datasets, this function automatically downsizes the matrix
    to active users and popular movies to avoid out-of-memory errors.
    Tune using environment variables:
      - CF_MAX_USERS (default: 12000)
      - CF_MAX_MOVIES (default: 4000)
      - CF_MIN_USER_RATINGS (default: 20)
      - CF_MIN_MOVIE_RATINGS (default: 50)
    """

    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name, str(default)).strip()
        try:
            val = int(raw)
            return max(1, val)
        except ValueError:
            return default

    max_users = _env_int("CF_MAX_USERS", 12000)
    max_movies = _env_int("CF_MAX_MOVIES", 4000)
    min_user_ratings = _env_int("CF_MIN_USER_RATINGS", 20)
    min_movie_ratings = _env_int("CF_MIN_MOVIE_RATINGS", 50)

    # Keep only active users first.
    user_counts = ratings_df.groupby("userId")["rating"].count()
    active_users = user_counts[user_counts >= min_user_ratings].sort_values(ascending=False).head(max_users).index
    ratings_filtered = ratings_df[ratings_df["userId"].isin(active_users)]

    # Keep only popular movies within active-user slice.
    movie_counts = ratings_filtered.groupby("movieId")["rating"].count()
    popular_movies = movie_counts[movie_counts >= min_movie_ratings].sort_values(ascending=False).head(max_movies).index
    ratings_filtered = ratings_filtered[ratings_filtered["movieId"].isin(popular_movies)]

    if ratings_filtered.empty:
        # Fallback: small direct sample to ensure matrix can still be built.
        ratings_filtered = ratings_df.head(min(len(ratings_df), 200_000)).copy()

    print(
        "CF matrix source ratings:",
        f"{len(ratings_filtered):,}/{len(ratings_df):,}",
        f"users={ratings_filtered['userId'].nunique():,}",
        f"movies={ratings_filtered['movieId'].nunique():,}",
    )

    return ratings_filtered.pivot_table(
        index="userId", columns="movieId", values="rating", aggfunc="mean"
    )


def get_movie_stats(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich movies_df with avg_rating, num_ratings, and popularity_score."""
    stats = (
        ratings_df.groupby("movieId")
        .agg(avg_rating=("rating", "mean"), num_ratings=("rating", "count"))
        .reset_index()
    )
    stats["avg_rating"] = stats["avg_rating"].round(2)
    stats["popularity_score"] = (stats["num_ratings"] * stats["avg_rating"]).rank(pct=True).round(4)

    merged = movies_df.merge(stats, on="movieId", how="left")
    merged["avg_rating"] = merged["avg_rating"].fillna(0.0)
    merged["num_ratings"] = merged["num_ratings"].fillna(0).astype(int)
    merged["popularity_score"] = merged["popularity_score"].fillna(0.0)
    return merged


def get_all_genres(movies_df: pd.DataFrame) -> list:
    """Return sorted list of unique genres across the dataset."""
    genres = set()
    for g_str in movies_df["genres"]:
        if g_str != "(no genres listed)":
            genres.update(g.strip() for g in g_str.split("|"))
    return sorted(genres)


def preprocess_pipeline(data_dir: Path = DATA_DIR) -> dict:
    """Run the full preprocessing pipeline and return a data bundle."""
    movies_df, ratings_df, links_df = load_data(data_dir)
    user_item_matrix = create_user_item_matrix(ratings_df)
    movie_stats = get_movie_stats(movies_df, ratings_df)
    genres = get_all_genres(movies_df)

    # Build movieId → tmdbId lookup dict
    tmdb_map: dict = {}
    if not links_df.empty:
        for _, row in links_df.iterrows():
            tmdb_val = row["tmdbId"]
            if pd.notna(tmdb_val) and tmdb_val > 0:
                tmdb_map[int(row["movieId"])] = int(tmdb_val)

    return {
        "movies": movies_df,
        "ratings": ratings_df,
        "links": links_df,
        "tmdb_map": tmdb_map,
        "user_item_matrix": user_item_matrix,
        "movie_stats": movie_stats,
        "genres": genres,
        "n_users": ratings_df["userId"].nunique(),
        "n_movies": movies_df["movieId"].nunique(),
        "n_ratings": len(ratings_df),
        "cf_n_users": user_item_matrix.shape[0],
        "cf_n_movies": user_item_matrix.shape[1],
    }
