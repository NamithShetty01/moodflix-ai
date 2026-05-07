"""
poster_service.py
-----------------
Fetches real movie poster URLs from The Movie Database (TMDB) API.

How it works
------------
1.  The MovieLens `links.csv` maps each `movieId` to a `tmdbId`.
2.  We call  GET https://api.themoviedb.org/3/movie/{tmdbId}?api_key=KEY
    and read the `poster_path` field from the response.
3.  The full poster URL is:
        https://image.tmdb.org/t/p/w300{poster_path}
4.  Fetched URLs are stored in a simple in-memory dict (cache) so every
    movie is only looked up once per session.

Getting a free TMDB API key
---------------------------
1.  Go to  https://www.themoviedb.org/signup  and create a free account.
2.  Visit  https://www.themoviedb.org/settings/api  and request an API key
    (choose "Developer / personal use").
3.  Copy the **API Key (v3 auth)** string into the sidebar field in the app.
"""

import requests
import re
from pathlib import Path
import csv
import pandas as pd

TMDB_BASE   = "https://api.themoviedb.org/3/movie/{tmdb_id}"
TMDB_SEARCH = "https://api.themoviedb.org/3/search/movie"
TMDB_IMG    = "https://image.tmdb.org/t/p/w300{poster_path}"
WIKI_API    = "https://en.wikipedia.org/w/api.php"
TIMEOUT_SEC = 5

# In-process cache  { tmdbId → poster_url | None }
_cache: dict = {}
_search_cache: dict = {}


def get_poster_url(tmdb_id: int, api_key: str) -> str | None:
    """Return the TMDB poster URL for *tmdb_id*, or None on failure.

    Results are cached in-memory so repeated calls are free.
    """
    if not api_key or not api_key.strip():
        return None

    if tmdb_id in _cache:
        return _cache[tmdb_id]

    try:
        url = TMDB_BASE.format(tmdb_id=tmdb_id)
        resp = requests.get(
            url,
            params={"api_key": api_key.strip(), "language": "en-US"},
            timeout=TIMEOUT_SEC,
        )
        if resp.status_code == 200:
            data = resp.json()
            poster_path = data.get("poster_path")
            if poster_path:
                result = TMDB_IMG.format(poster_path=poster_path)
                _cache[tmdb_id] = result
                return result

        _cache[tmdb_id] = None
        return None

    except Exception:  # noqa: BLE001  — network errors are non-fatal
        _cache[tmdb_id] = None
        return None


def _normalize_title(title: str) -> str:
    """Strip trailing year markers and excess whitespace for TMDB search."""
    if not title:
        return ""
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    return re.sub(r"\s+", " ", t)


def get_poster_url_by_title(title: str, api_key: str) -> str | None:
    """Search TMDB by movie title and return best-match poster URL."""
    if not api_key or not api_key.strip():
        return None

    query = _normalize_title(title)
    if not query:
        return None

    key = query.lower()
    if key in _search_cache:
        return _search_cache[key]

    try:
        resp = requests.get(
            TMDB_SEARCH,
            params={"api_key": api_key.strip(), "query": query, "include_adult": "false"},
            timeout=TIMEOUT_SEC,
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                poster_path = results[0].get("poster_path")
                if poster_path:
                    url = TMDB_IMG.format(poster_path=poster_path)
                    _search_cache[key] = url
                    return url

        _search_cache[key] = None
        return None
    except Exception:  # noqa: BLE001
        _search_cache[key] = None
        return None


def get_poster_url_from_wikipedia(title: str) -> str | None:
    """Fetch a movie thumbnail from Wikipedia by title, or None.

    This is a no-key fallback source for cases where TMDB API key is not set.
    """
    query = _normalize_title(title)
    if not query:
        return None

    cache_key = f"wiki::{query.lower()}"
    if cache_key in _search_cache:
        return _search_cache[cache_key]

    try:
        # 1) Find likely page title
        resp = requests.get(
            WIKI_API,
            params={
                "action": "query",
                "format": "json",
                "list": "search",
                "srlimit": 1,
                "srsearch": f"{query} film",
            },
            timeout=TIMEOUT_SEC,
        )
        if resp.status_code != 200:
            _search_cache[cache_key] = None
            return None

        data = resp.json()
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            _search_cache[cache_key] = None
            return None

        page_title = search_results[0].get("title")
        if not page_title:
            _search_cache[cache_key] = None
            return None

        # 2) Fetch page thumbnail
        thumb_resp = requests.get(
            WIKI_API,
            params={
                "action": "query",
                "format": "json",
                "prop": "pageimages",
                "piprop": "thumbnail",
                "pithumbsize": 500,
                "titles": page_title,
            },
            timeout=TIMEOUT_SEC,
        )
        if thumb_resp.status_code != 200:
            _search_cache[cache_key] = None
            return None

        pages = thumb_resp.json().get("query", {}).get("pages", {})
        for _, page in pages.items():
            thumb = page.get("thumbnail", {})
            source = thumb.get("source")
            if source:
                _search_cache[cache_key] = source
                return source

        _search_cache[cache_key] = None
        return None
    except Exception:  # noqa: BLE001
        _search_cache[cache_key] = None
        return None


def batch_fetch_posters(tmdb_ids: list[int], api_key: str) -> dict:
    """Fetch poster URLs for a list of TMDB IDs.

    Returns a dict  { tmdb_id → url | None }.
    Only makes network requests for IDs not already in cache.
    """
    results: dict = {}
    for tid in tmdb_ids:
        results[tid] = get_poster_url(tid, api_key)
    return results


def clear_cache() -> None:
    """Clear the in-memory poster cache (useful for testing)."""
    _cache.clear()
    _search_cache.clear()


def prefetch_posters_for_movies(movies_df: pd.DataFrame, tmdb_map: dict, api_key: str, out_path: str | Path) -> pd.DataFrame:
    """Prefetch poster URLs for every movie in `movies_df` and save to CSV.

    CSV columns: movieId, tmdbId, poster_url
    Returns the dataframe that was written.
    """
    rows = []
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    for _, r in movies_df.iterrows():
        mid = int(r.get("movieId"))
        title = str(r.get("title", ""))
        tmdb_id = tmdb_map.get(mid)
        url = None
        if api_key and api_key.strip():
            if tmdb_id:
                url = get_poster_url(tmdb_id, api_key)
            if not url:
                url = get_poster_url_by_title(title, api_key)
        if not url:
            url = get_poster_url_from_wikipedia(title)
        rows.append({"movieId": mid, "tmdbId": int(tmdb_id) if tmdb_id else None, "poster_url": url or ""})

    df = pd.DataFrame(rows)
    df.to_csv(outp, index=False)
    return df


def load_prefetched_posters(path: str | Path) -> dict:
    """Load prefetched posters CSV and return { movieId -> poster_url | None }."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        result = {}
        for _, r in df.iterrows():
            try:
                mid = int(r.get("movieId"))
            except Exception:
                continue
            url = r.get("poster_url")
            result[mid] = url if pd.notna(url) and str(url).strip() else None
        return result
    except Exception:
        return {}
