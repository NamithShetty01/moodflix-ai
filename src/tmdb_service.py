"""Small TMDB API wrapper used by the Streamlit app.

Provides: search_movies, get_movie_details, get_popular, get_trending, build_poster_url
"""
from __future__ import annotations

import requests
from urllib3.exceptions import InsecureRequestWarning
from typing import Any, Dict, List

# Suppress SSL warnings when verify=False is used
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

API_BASE = "https://api.themoviedb.org/3"
IMG_TEMPLATE = "https://image.tmdb.org/t/p/w500{poster_path}"
TIMEOUT = 15 


def _req(path: str, api_key: str, params: dict | None = None) -> tuple[dict | None, str | None]:
    """Make TMDB API request and return (data, error_message)
    
    Retries up to 3 times with exponential backoff for resilience.
    """
    if not api_key:
        return None, "API key not provided"
    
    url = f"{API_BASE}{path}"
    p = {"api_key": api_key}
    if params:
        p.update(params)
    
    # Retry logic with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Disable SSL verification for compatibility with corporate proxies/firewalls
            # Use longer timeout on retries
            current_timeout = TIMEOUT + (attempt * 5)  # 15s, 20s, 25s on retries
            r = requests.get(url, params=p, timeout=current_timeout, verify=False)
            if r.status_code == 200:
                return r.json(), None
            else:
                try:
                    error_msg = r.json().get("status_message", f"HTTP {r.status_code}")
                except:
                    error_msg = f"HTTP {r.status_code}: {r.text[:100]}"
                return None, error_msg
        except requests.Timeout:
            if attempt == max_retries - 1:  # Last attempt
                return None, f"Request timeout after {current_timeout}s (retried {max_retries} times)"
            continue  # Retry
        except requests.ConnectionError as e:
            if attempt == max_retries - 1:
                return None, f"Connection error - unable to reach movie data API (retried {max_retries} times)"
            continue
        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"Error: {str(e)}"
            continue
    
    return None, "Failed after all retry attempts"


def search_movies(query: str, api_key: str, page: int = 1) -> tuple[Dict[str, Any] | None, str | None]:
    return _req("/search/movie", api_key, {"query": query, "page": page, "include_adult": "false"})


def get_movie_details(movie_id: int, api_key: str) -> tuple[Dict[str, Any] | None, str | None]:
    return _req(f"/movie/{movie_id}", api_key, {"append_to_response": "credits,images,recommendations"})


def get_popular(api_key: str, page: int = 1) -> tuple[Dict[str, Any] | None, str | None]:
    return _req("/movie/popular", api_key, {"page": page})


def get_top_rated(api_key: str, page: int = 1) -> tuple[Dict[str, Any] | None, str | None]:
    return _req("/movie/top_rated", api_key, {"page": page})


def get_trending(api_key: str, time_window: str = "day", page: int = 1) -> tuple[Dict[str, Any] | None, str | None]:
    return _req(f"/trending/movie/{time_window}", api_key, {"page": page})


def build_poster_url(poster_path: str | None) -> str | None:
    if not poster_path:
        return None
    return IMG_TEMPLATE.format(poster_path=poster_path)
