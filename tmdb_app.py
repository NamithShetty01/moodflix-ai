"""TMDB-only MoodFlix AI app implementation.

This module contains the live TMDB browsing experience. The legacy app.py
file imports and runs main() from here, then exits before executing the old
MovieLens-specific code that remains below in that file.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.poster_service import clear_cache, get_poster_url_by_title, get_poster_url_from_wikipedia
from src.tmdb_service import build_poster_url, get_movie_details, get_popular, get_top_rated, get_trending, search_movies

ROOT = Path(__file__).resolve().parent

# Mood-Based Hybrid Recommendation System
MOOD_GENRES = {
    "Action": ["Action", "Adventure"],
    "Happy": ["Comedy", "Animation", "Family"],
    "Romantic": ["Romance", "Drama"],
    "Thriller": ["Thriller", "Crime"],
    "Emotional": ["Drama"]
}

TMDB_GENRE_MAP = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}


@st.cache_data(show_spinner=False, ttl=1800)
def cached_search_movies(query: str, api_key: str, page: int = 1):
    return search_movies(query, api_key, page)


@st.cache_data(show_spinner=False, ttl=1800)
def cached_movie_details(movie_id: int, api_key: str):
    return get_movie_details(movie_id, api_key)


@st.cache_data(show_spinner=False, ttl=1800)
def cached_popular(api_key: str, page: int = 1):
    return get_popular(api_key, page)


@st.cache_data(show_spinner=False, ttl=1800)
def cached_top_rated(api_key: str, page: int = 1):
    return get_top_rated(api_key, page)


@st.cache_data(show_spinner=False, ttl=1800)
def cached_trending(api_key: str, time_window: str = "week", page: int = 1):
    return get_trending(api_key, time_window, page)


@st.cache_data(show_spinner=False, ttl=1800)
def cached_image_lookup(title: str, api_key: str):
    poster = get_poster_url_by_title(title, api_key)
    if not poster:
        poster = get_poster_url_from_wikipedia(title)
    return poster


def _payload(result):
    data, error = result
    return data or {}, error


def _youtube_trailer_url(title: str) -> str:
    query = quote_plus(f"{title} official trailer")
    return f"https://www.youtube.com/results?search_query={query}"


def _short(text: str, n: int = 24) -> str:
    return text[:n] + "…" if len(text) > n else text


def _overview_snippet(text: str, n: int = 180) -> str:
    value = (text or "").strip()
    if not value:
        return "No overview available."
    return value[:n] + ("…" if len(value) > n else "")


def _movie_link(movie_id: int) -> str:
    return f"https://www.themoviedb.org/movie/{movie_id}"


def _details_link(movie_id: int) -> str:
    return f"?movie_id={movie_id}"


def _poster_url(movie: dict, api_key: str) -> str | None:
    poster = build_poster_url(movie.get("poster_path"))
    if poster:
        return poster
    title = str(movie.get("title") or movie.get("name") or "")
    if title and api_key:
        return cached_image_lookup(title, api_key)
    return None

def _normalize_movie(item: dict, api_key: str) -> dict:
    title = str(item.get("title") or item.get("name") or "Untitled")
    release_date = str(item.get("release_date") or item.get("first_air_date") or "")
    year = release_date[:4] if len(release_date) >= 4 else "TBA"
    genres = item.get("genres") or []
    genre_names = [g.get("name", "") for g in genres if isinstance(g, dict) and g.get("name")]
    if not genre_names:
        genre_ids = item.get("genre_ids") or []
        genre_names = [TMDB_GENRE_MAP.get(int(genre_id), "") for genre_id in genre_ids if TMDB_GENRE_MAP.get(int(genre_id), "")]
    genre_text = ", ".join(genre_names)
    return {
        "id": int(item.get("id") or 0),
        "title": title,
        "year": year,
        "release_date": release_date,
        "genres": genre_text,
        "overview": item.get("overview") or "",
        "vote_average": float(item.get("vote_average") or 0.0),
        "vote_count": int(item.get("vote_count") or 0),
        "popularity": float(item.get("popularity") or 0.0),
        "poster_path": item.get("poster_path"),
        "poster_url": _poster_url(item, api_key),
        "runtime": item.get("runtime"),
        "status": item.get("status") or "",
    }


def _normalize_items(items: list[dict], api_key: str) -> list[dict]:
    return [_normalize_movie(item, api_key) for item in items if item.get("id")]


def _tmdb_results(result, api_key: str) -> list[dict]:
    data, error = _payload(result)
    if error and not data:
        st.session_state.setdefault("tmdb_data_unavailable", True)
        return []
    return _normalize_items(data.get("results", []), api_key)


def _unique_movies(*lists: list[dict]) -> list[dict]:
    seen: set[int] = set()
    combined: list[dict] = []
    for items in lists:
        for movie in items:
            movie_id = int(movie.get("id") or 0)
            if movie_id and movie_id not in seen:
                seen.add(movie_id)
                combined.append(movie)
    return combined


def hybrid_recommendation(movies: list[dict], mood: str, rating_w: float = 0.6, popularity_w: float = 0.4) -> list[dict]:
    """Adaptive Mood-Based Hybrid OTT Recommendation System.
    
    Filters and ranks movies based on mood, genres, popularity, and rating.
    Formula: Score = (vote_average * 0.6) + (popularity * 0.4)
    """
    if not movies or mood not in MOOD_GENRES:
        return movies
    
    genres = [g.lower() for g in MOOD_GENRES[mood]]
    scored_movies: list[dict] = []

    # Prepare normalization ranges
    ratings = [float(m.get("vote_average", 0.0)) for m in movies]
    pops = [float(m.get("popularity", 0.0)) for m in movies]
    votes = [float(m.get("vote_count", 0)) for m in movies]
    r_min, r_max = (min(ratings), max(ratings)) if ratings else (0.0, 1.0)
    p_min, p_max = (min(pops), max(pops)) if pops else (0.0, 1.0)
    v_min, v_max = (min(votes), max(votes)) if votes else (0.0, 1.0)

    def norm(val, vmin, vmax):
        try:
            return 0.0 if vmax - vmin <= 1e-9 else (float(val) - vmin) / (vmax - vmin)
        except Exception:
            return 0.0

    for movie in movies:
        movie_genres = str(movie.get("genres", "")).lower()
        matches = sum(1 for g in genres if g in movie_genres)

        nr = norm(float(movie.get("vote_average", 0.0)), r_min, r_max)
        npop = norm(float(movie.get("popularity", 0.0)), p_min, p_max)
        nvc = norm(float(movie.get("vote_count", 0)), v_min, v_max)

        base_score = rating_w * nr + popularity_w * npop

        # Small contribution from vote counts to prefer well-known titles
        base_score += 0.08 * nvc

        # Mood boosting: prioritize matching genres strongly to make mood selection visible
        mood_boost = 1.5 if matches > 0 else 0.0
        mood_bonus = matches * 0.25

        movie_score = base_score + mood_boost + mood_bonus
        movie["hybrid_score"] = movie_score
        scored_movies.append(movie)

    scored_movies.sort(key=lambda x: float(x.get("hybrid_score", 0.0)), reverse=True)
    return scored_movies[:20]


def render_details_overlay(movie_id: int, api_key: str) -> None:
    payload, error = _payload(cached_movie_details(movie_id, api_key))
    if error and not payload:
        st.error(error)
        return

    movie = _normalize_movie(payload, api_key)

    with st.expander(f"🎬 {movie.get('title')} — Details", expanded=True):
        # Header row with poster, title, and close
        top_cols = st.columns([1.2, 5, 0.8])
        with top_cols[0]:
            if movie.get('poster_url'):
                st.image(movie['poster_url'], width="stretch")
            else:
                st.markdown(f"**{movie.get('title')}**")
        
        with top_cols[1]:
            st.markdown(f"### {movie.get('title')}")
            st.markdown(f"**{movie.get('year')}** · {movie.get('genres')}")
            st.markdown(f"⭐ {movie.get('vote_average'):.1f}/10 ({movie.get('vote_count'):,} votes)")
            st.markdown(f"{_overview_snippet(movie.get('overview') or '', 300)}")
            
            # Action buttons
            btn_cols = st.columns([1, 1])
            with btn_cols[0]:
                st.link_button("▶ Trailer", _youtube_trailer_url(movie.get('title')), width="stretch")
            with btn_cols[1]:
                if st.button('🏠 Home', key=f'close_overlay_{movie_id}', width="stretch"):
                    st.session_state.show_overlay = False
                    st.session_state.selected_movie_id = None
                    try:
                        if hasattr(st, 'query_params'):
                            st.query_params.clear()
                        else:
                            st.experimental_set_query_params()
                    except Exception:
                        pass
                    st.rerun()
        
        with top_cols[2]:
            pass  # Spacer
        
        st.markdown('---')
        
        # Similar titles section
        recs = _normalize_items(payload.get('recommendations', {}).get('results', []), api_key)
        if recs:
            st.markdown('**Similar Titles**')
            render_movie_grid(recs[: st.session_state.cards_to_show], cols=st.session_state.cards_to_show, key_prefix=f"overlay_recs_{movie_id}", show_more_info=False)
        else:
            st.info('No similar titles available.')


def _render_card(movie: dict, key: str, show_more_info: bool = True) -> None:
    poster = movie.get("poster_url")
    title = str(movie.get("title") or "Untitled")
    year = str(movie.get("year") or "TBA")
    rating = float(movie.get("vote_average") or 0.0)
    votes = int(movie.get("vote_count") or 0)
    overview = _overview_snippet(str(movie.get("overview") or ""), 110)
    details_link = _details_link(int(movie.get("id") or 0))
    poster_nav_id = f"movie-nav-{int(movie.get('id') or 0)}"
    st.markdown(
        f'<a id="{poster_nav_id}" href="{details_link}" '
        'style="position:absolute;left:-9999px;top:auto;width:1px;height:1px;overflow:hidden;">'
        'open details'
        '</a>',
        unsafe_allow_html=True,
    )
    if poster:
        # Use a direct link to the details query parameter so Streamlit detects the change
        st.html(
            f'<a href="{details_link}" '
            'style="display:block;text-decoration:none;cursor:pointer;">'
            f'<img src="{poster}" alt="{title}" '
            'style="width:100%;aspect-ratio:2/3;object-fit:cover;border-radius:18px;display:block;box-shadow:0 12px 32px rgba(0,0,0,0.28);border:1px solid rgba(255,255,255,0.08);">'
            f'</a>'
        )
    else:
        st.html(
            f'<a href="{details_link}" style="display:flex;align-items:center;justify-content:center;min-height:240px;padding:14px;text-align:center;text-decoration:none;cursor:pointer;border-radius:18px;background:linear-gradient(135deg,#1a237e,#6a1b9a);color:rgba(255,255,255,0.95);font-size:13px;font-weight:700;line-height:1.3;box-shadow:0 12px 32px rgba(0,0,0,0.28);border:1px solid rgba(255,255,255,0.08);">'
            f'{_short(title, 28)}'
            '</a>'
        )

    st.markdown(
        f'<div class="nf-card">'
        f'<div class="nf-card-meta">'
        f'<div class="nf-card-title">{_short(title, 28)}</div>'
        f'<div class="nf-meta-line"><span>{year}</span></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Posters are clickable and open the details overlay; explicit "More Info" button removed.


def render_movie_grid(movies: list[dict] | pd.DataFrame, cols: int = 8, key_prefix: str = "card", show_more_info: bool = True) -> None:
    if isinstance(movies, pd.DataFrame):
        rows = movies.to_dict("records")
    else:
        rows = list(movies)

    if not rows:
        st.info("No movies to display.")
        return

    total = len(rows)
    start_key = f"{key_prefix}_start"
    st.session_state.setdefault(start_key, 0)

    max_start = max(0, total - cols)
    cur_start = min(max(0, int(st.session_state[start_key])), max_start)
    st.session_state[start_key] = cur_start

    st.markdown('<div class="nf-rail-nav">', unsafe_allow_html=True)
    n1, n2, n3 = st.columns([1.0, 1.0, 8.0])
    with n3:
        end_idx = min(cur_start + cols, total)
        st.markdown(f'<div class="nf-rail-count">{cur_start + 1}-{end_idx} of {total}</div>', unsafe_allow_html=True)
    with n1:
        if st.button("❮", key=f"{key_prefix}_prev", disabled=cur_start == 0):
            st.session_state[start_key] = max(0, cur_start - cols)
            st.rerun()
    with n2:
        if st.button("❯", key=f"{key_prefix}_next", disabled=cur_start >= max_start):
            st.session_state[start_key] = min(max_start, cur_start + cols)
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    chunk = rows[cur_start:cur_start + cols]
    columns = st.columns(max(len(chunk), 1))
    for col, movie in zip(columns, chunk):
        with col:
            _render_card(movie, key=f"{key_prefix}_{int(movie.get('id') or 0)}", show_more_info=show_more_info)


def render_hero(movie: dict, subtitle: str) -> None:
    poster = movie.get("poster_url")
    title = str(movie.get("title") or "Untitled")
    year = str(movie.get("year") or "TBA")
    rating = float(movie.get("vote_average") or 0.0)
    votes = int(movie.get("vote_count") or 0)
    overview = _overview_snippet(str(movie.get("overview") or ""), 200)
    bg = "linear-gradient(135deg,#08111d 0%,#0a1724 52%,#08111d 100%)"
    genre_text = str(movie.get("genres") or "")
    poster_html = (
        f'<div class="nf-hero-poster"><img src="{poster}" alt="{title}"></div>'
        if poster
        else '<div class="nf-hero-poster nf-hero-placeholder"><span>No poster available</span></div>'
    )

    st.markdown(
        f'<div class="nf-hero" style="background:{bg};">'
        f'<div class="nf-hero-body">'
        f'<div class="nf-hero-copy">'
        f'<div class="hero-tag">{subtitle}</div>'
        f'<div class="nf-hero-title">{title}</div>'
        f'<div class="nf-hero-meta">'
        f'<span class="nf-match">⭐ {rating:.1f}</span>'
        f'<span class="nf-age">{year}</span>'
        f'<span>{genre_text or "Animation"}</span>'
        f'<span>{votes:,}</span>'
        f'</div>'
        f'<div class="nf-hero-desc">{overview}</div>'
        f'</div>'
        f'{poster_html}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="MoodFlix AI",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Sora:wght@400;600;700&display=swap');

        :root {
            --bg: #070d16;
            --panel: #0f1724;
            --panel-2: #132033;
            --border: rgba(255,255,255,0.1);
            --text: #f5f8fc;
            --muted: #9cb0c4;
            --accent: #22b8e6;
            --accent-2: #2ec4b6;
            --shadow: 0 16px 40px rgba(2, 8, 18, 0.42);
            --shadow-lg: 0 28px 72px rgba(2, 8, 18, 0.55);
            --radius: 14px;
        }

        *, *::before, *::after { box-sizing: border-box; }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main > div {
            background:
                radial-gradient(circle at 15% 0%, rgba(34,184,230,0.12), transparent 33%),
                radial-gradient(circle at 100% 0%, rgba(46,196,182,0.08), transparent 34%),
                linear-gradient(180deg, #080f19 0%, #091320 52%, #080f19 100%) !important;
            color: var(--text);
            font-family: 'Plus Jakarta Sans', 'Sora', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        }

        [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], #MainMenu, header {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        [data-testid="stMain"] > div:first-child { padding-top: 0 !important; }
        .block-container {
            padding: 0 4% 56px 4% !important;
            max-width: 1400px !important;
        }

        [data-testid="stSidebar"] {
            display: none !important;
        }
        [data-testid="stMainBlockContainer"] { margin-left: 0 !important; }

        .stButton > button {
            background: linear-gradient(135deg, var(--accent), #1296c5) !important;
            color: #fff !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 11px !important;
            font-weight: 700 !important;
            font-size: 14px !important;
            padding: 9px 18px !important;
            box-shadow: 0 8px 20px rgba(8, 119, 162, 0.28) !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease !important;
        }
        .stButton > button:hover { 
            transform: translateY(-1px) !important;
            filter: brightness(1.04) !important;
            box-shadow: 0 12px 28px rgba(8, 119, 162, 0.34) !important;
        }
        .stButton > button:active { transform: translateY(0) !important; }

        [data-testid="stSidebar"] .stButton > button {
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid transparent !important;
            text-align: left !important;
            box-shadow: none !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(14, 165, 233, 0.12) !important;
            border-color: rgba(14, 165, 233, 0.25) !important;
        }

        .stTextInput > div > div > input,
        .stSelectbox > div > div,
        .stSlider > div {
            background: rgba(255,255,255,0.035) !important;
            color: #fff !important;
            border: 1px solid var(--border) !important;
            border-radius: 11px !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div:focus,
        .stSlider > div:focus {
            border-color: rgba(14, 165, 233, 0.5) !important;
            box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1) !important;
        }

        .stAlert {
            background: rgba(14, 165, 233, 0.05) !important;
            border: 1px solid rgba(14, 165, 233, 0.15) !important;
            border-radius: 10px !important;
            padding: 14px 16px !important;
        }
        .stAlert p { margin: 0 !important; font-size: 14px !important; }

        .nf-hero {
            position: relative;
            width: calc(100% + 8%);
            margin-left: -4%;
            margin-right: -4%;
            min-height: 62vh;
            max-height: 640px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            margin-bottom: 4px;
            box-shadow: var(--shadow-lg);
        }
        .nf-hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(to right, rgba(5,10,18,0.94) 0%, rgba(5,10,18,0.72) 44%, rgba(5,10,18,0.22) 78%),
                linear-gradient(to top, rgba(5,10,18,1.0) 0%, rgba(5,10,18,0.0) 30%);
        }
        .nf-hero-body {
            position: absolute;
            inset: 0;
            padding: 4.5% 4.5% 4.5% 4.5%;
            z-index: 2;
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 32px;
        }
        .nf-hero-copy {
            max-width: 56%;
            position: relative;
            z-index: 3;
        }
        .nf-hero-poster {
            position: relative;
            z-index: 3;
            flex: 0 0 min(24vw, 280px);
            align-self: center;
        }
        .nf-hero-poster img {
            width: 100%;
            display: block;
            border-radius: 16px;
            object-fit: cover;
            aspect-ratio: 2 / 3;
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.48);
            border: 1px solid rgba(255,255,255,0.12);
        }
        .nf-hero-placeholder {
            min-height: 360px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(1,180,228,0.18), rgba(10,14,24,0.92));
            border: 1px solid rgba(255,255,255,0.08);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #dbeafe;
            font-weight: 700;
            text-align: center;
            padding: 18px;
        }
        .hero-tag {
            font-size: 12px;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: var(--accent);
            font-weight: 700;
            margin-bottom: 14px;
            opacity: 0.95;
        }
        .nf-hero-title {
            font-size: clamp(2rem, 4.5vw, 3.8rem);
            font-weight: 800;
            line-height: 1.08;
            color: #fff;
            margin-bottom: 14px;
            text-shadow: 2px 4px 12px rgba(0,0,0,0.9);
            letter-spacing: -0.5px;
        }
        .nf-hero-desc {
            font-size: 1.05rem;
            color: #cbd5e1;
            line-height: 1.7;
            margin-bottom: 24px;
            text-shadow: 1px 2px 6px rgba(0,0,0,0.95);
            max-width: 95%;
        }
        .nf-hero-meta {
            font-size: 14px;
            color: var(--muted);
            margin-bottom: 24px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }
        .nf-match { color: #fbbf24; font-weight: 700; font-size: 15px; }
        .nf-age {
            border: 1.5px solid rgba(16, 185, 129, 0.5);
            padding: 2px 8px;
            font-size: 13px;
            color: var(--accent-2);
            border-radius: 4px;
            background: rgba(16, 185, 129, 0.08);
        }

        .section-header {
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--text);
            margin: 32px 0 10px 0;
            letter-spacing: -0.3px;
        }
        .section-caption { margin: 0 0 14px 0; color: var(--muted); font-size: 13px; font-weight: 500; }

        .tmdb-shell {
            width: calc(100% + 8%);
            margin-left: -4%;
            margin-right: -4%;
            padding: 28px 4% 22px 4%;
            background: linear-gradient(105deg, #153b5d 0%, #16527a 52%, #12334d 100%);
            border-radius: 0 0 16px 16px;
            border: 1px solid rgba(255,255,255,0.09);
            border-top: none;
            margin-bottom: 20px;
            box-shadow: 0 14px 36px rgba(6, 16, 30, 0.36);
        }
        .tmdb-shell-title {
            color: #ffffff;
            font-size: clamp(2.2rem, 3.5vw, 3.2rem);
            font-weight: 900;
            line-height: 1.05;
            letter-spacing: -0.7px;
            margin-bottom: 4px;
        }
        .tmdb-shell-subtitle {
            color: rgba(255,255,255,0.96);
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 20px;
            letter-spacing: -0.3px;
        }
        .tmdb-chip-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 10px 0 8px 0;
        }
        .tmdb-chip-title {
            color: #f8fbff;
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.4px;
        }
        .tmdb-chip-active {
            display: inline-flex;
            align-items: center;
            height: 30px;
            border-radius: 999px;
            background: linear-gradient(90deg, #90cea1 0%, #01b4e4 100%);
            color: #05243a;
            font-size: 13px;
            font-weight: 800;
            padding: 0 14px;
            margin-left: 6px;
        }

        .tmdb-topnav {
            width: calc(100% + 10%);
            margin-left: -5%;
            margin-right: -5%;
            padding: 8px 5% 8px 5%;
            background: linear-gradient(180deg, rgba(7,17,38,0.98), rgba(5,12,25,0.96));
            border-bottom: 1px solid var(--border);
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        .tmdb-topnav-item {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            padding: 7px 12px;
            border-radius: 5px;
            background: rgba(255,255,255,0.05);
            color: var(--text);
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.25s ease;
            border: 1px solid rgba(255,255,255,0.08);
            text-decoration: none;
            min-width: 48px;
        }
        .tmdb-topnav-item:hover {
            background: rgba(1,180,228,0.14);
            border-color: rgba(1,180,228,0.3);
            transform: translateY(-1px);
        }
        .tmdb-topnav-item.active {
            background: linear-gradient(135deg, #01b4e4, #0a90d9);
            color: #ffffff;
            border-color: rgba(1,180,228,0.5);
            box-shadow: 0 4px 12px rgba(1,180,228,0.25);
        }

        .metric-card,
        .score-box,
        .nf-card {
            background: linear-gradient(180deg, rgba(16, 22, 36, 0.95), rgba(14, 20, 32, 0.8));
            border: 1px solid var(--border);
            box-shadow: 0 8px 24px rgba(0,0,0,0.24);
            backdrop-filter: blur(8px);
        }
        .metric-card {
            border-radius: 14px;
            padding: 24px;
            text-align: center;
        }
        .metric-val { font-size: 2.2rem; font-weight: 800; color: var(--accent); }
        .metric-lbl { font-size: 13px; color: var(--muted); margin-top: 6px; font-weight: 500; }
        .score-box {
            border-radius: 14px;
            padding: 16px 20px;
            text-align: center;
        }
        .score-val { font-size: 1.9rem; font-weight: 800; color: var(--accent); }
        .score-lbl { font-size: 12px; color: var(--muted); margin-top: 4px; }

        .nf-card {
            position: relative;
            border-radius: 12px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 1;
        }
        .nf-card:hover {
            transform: translateY(-6px) scale(1.015);
            box-shadow: 0 18px 42px rgba(0,0,0,0.35);
            z-index: 10;
            border-color: rgba(1, 180, 228, 0.5);
        }
        .nf-thumb,
        .nf-thumb-placeholder {
            width: 100%;
            aspect-ratio: 2 / 3;
            display: block;
            object-fit: cover;
        }
        .nf-thumb {
            border-radius: 12px 12px 6px 6px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .nf-card:hover .nf-thumb {
            transform: scale(1.08);
        }
        .nf-thumb-placeholder {
            border-radius: 10px 10px 4px 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 14px;
            text-align: center;
            background: linear-gradient(135deg, #1e3a8a, #7c3aed);
            font-weight: 600;
        }
        .nf-card-meta {
            padding: 10px 10px 11px;
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0));
            border-top: 1px solid rgba(255,255,255,0.05);
        }
        .nf-card-title {
            font-size: 13px;
            font-weight: 700;
            color: #fff;
            margin-bottom: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .nf-meta-line {
            display: flex;
            justify-content: space-between;
            gap: 8px;
            font-size: 11px;
            color: var(--muted);
        }
        .nf-rating { color: #fbbf24; font-weight: 800; }
        .nf-card-info {
            display: none;
            background: linear-gradient(180deg, rgba(10,14,24,0.98), rgba(7,10,20,0.98));
            padding: 10px 10px 12px;
        }
        .nf-card:hover .nf-card-info { display: block; }
        .nf-card-row {
            display: flex;
            gap: 6px;
            align-items: center;
            flex-wrap: wrap;
            margin-top: 4px;
        }
        .nf-green { color: var(--accent-2); font-size: 12px; font-weight: 700; }
        .nf-tag {
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 3px;
            font-size: 10px;
            padding: 2px 5px;
            color: #cfe9f3;
            background: rgba(14, 165, 233, 0.08);
        }
        .nf-genre, .detail-meta { color: var(--muted); font-size: 13px; }

        .nf-rail-nav {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 8px;
            margin: 4px 0 12px;
        }
        .nf-rail-count {
            font-size: 12px;
            color: var(--muted);
            letter-spacing: 0.5px;
            margin-right: 8px;
            font-weight: 500;
        }
        .nf-rail-nav .stButton > button {
            background: rgba(10, 14, 24, 0.8) !important;
            color: #fff !important;
            border: 1px solid var(--border) !important;
            border-radius: 999px !important;
            min-width: 36px !important;
            height: 32px !important;
            padding: 0 !important;
            font-size: 14px !important;
            line-height: 1 !important;
            opacity: 0.3 !important;
            transition: all 0.2s ease !important;
        }
        .nf-rail-nav .stButton > button:hover { 
            opacity: 0.9 !important;
            background: rgba(14, 165, 233, 0.2) !important;
            border-color: rgba(14, 165, 233, 0.3) !important;
        }
        .nf-rail-nav .stButton > button:disabled { opacity: 0.1 !important; }

        .detail-hero {
            position: relative;
            min-height: 340px;
            overflow: hidden;
            margin-bottom: 32px;
            background-size: cover;
            background-position: center 25%;
            width: calc(100% + 10%);
            margin-left: -5%;
            margin-right: -5%;
            border-radius: 16px;
        }
        .detail-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(to right, rgba(10,14,24,0.96) 0%, rgba(10,14,24,0.72) 45%, rgba(10,14,24,0.25) 100%),
                linear-gradient(to top, rgba(10,14,24,1.0) 0%, transparent 38%);
        }
        .detail-hero-body {
            position: relative;
            z-index: 1;
            padding: 56px 5%;
            max-width: 65%;
        }
        .detail-hero-body h2 { 
            font-size: 3rem; 
            margin: 0 0 8px 0; 
            line-height: 1.08; 
            color: #fff;
            font-weight: 900;
            letter-spacing: -0.8px;
        }
        .detail-hero-body .detail-meta { color: #cbd5e1; margin-bottom: 14px; font-size: 13px; line-height: 1.6; }
        .detail-hero-chips { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
        .detail-chip {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 999px;
            padding: 6px 14px;
            font-size: 13px;
            color: #e0e7ff;
            font-weight: 500;
        }
        .detail-chip.red {
            background: rgba(14, 165, 233, 0.14);
            border-color: rgba(14, 165, 233, 0.4);
            color: #7dd3fc;
        }
        
        h1, h2, h3, h4 { color: #fff !important; font-weight: 800 !important; }
        h1 { font-size: 2rem !important; margin-bottom: 8px !important; }
        h3 { font-size: 1.2rem !important; }
        hr { border: none !important; height: 1px !important; background: var(--border) !important; }

        .footer {
            margin-top: 80px;
            padding-top: 40px;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--muted);
            font-size: 13px;
        }
        .footer p { margin: 4px 0; }
        .footer a { color: var(--accent); text-decoration: none; transition: color 0.2s ease; }
        .footer a:hover { color: #0284c7; text-decoration: underline; }

        @media (max-width: 980px) {
            .block-container { padding: 0 4% 44px 4% !important; }
            .nf-hero { min-height: 76vh; border-radius: 14px; }
            .nf-hero-body {
                align-items: flex-end;
                justify-content: flex-end;
                flex-direction: column-reverse;
                gap: 14px;
                padding: 7% 5%;
            }
            .nf-hero-copy { max-width: 100%; }
            .nf-hero-poster {
                align-self: flex-start;
                flex: 0 0 min(42vw, 220px);
            }
            .tmdb-shell {
                width: calc(100% + 8%);
                margin-left: -4%;
                margin-right: -4%;
                border-radius: 0 0 12px 12px;
            }
            .tmdb-chip-title { font-size: 1.35rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    defaults = {
        "page": "Home",
        "selected_movie_id": None,
        "search_query": "",
        "shell_query": "",
        "home_trending_window": "day",
        "poster_query": "",
        "seed_query": "",
        "cards_to_show": 8,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("show_overlay", False)

    if hasattr(st, "query_params"):
        query_params = dict(st.query_params)
    else:
        query_params = st.experimental_get_query_params()

    movie_id_param = query_params.get("movie_id")
    if movie_id_param:
        if isinstance(movie_id_param, list):
            movie_id_param = movie_id_param[0]
        try:
            st.session_state.selected_movie_id = int(movie_id_param)
            # Show inline overlay (Netflix-like) instead of navigating away
            st.session_state.show_overlay = True
        except (TypeError, ValueError):
            pass

    # Auto-load TMDB API key from environment variables or use default
    if "tmdb_api_key" not in st.session_state:
        st.session_state.tmdb_api_key = (
            os.getenv("TMDB_API_KEY", "") 
            or os.getenv("TMDB_V3_API_KEY", "")
            or "808345be3f2afb71295058b92d17c1e5"
        ).strip()

    api_key = st.session_state.get("tmdb_api_key", "").strip()
    if not api_key:
        st.markdown("# 🎬 MoodFlix AI")
        st.markdown(
            """
            ### Welcome to Your Movie Discovery Platform

            **MoodFlix AI** is a modern movie discovery platform with an extensive movie database.
            """
        )
        st.info("**Setup:** API key loaded from environment. Start exploring!")
        return

    popular_movies = _tmdb_results(cached_popular(api_key, 1), api_key)
    trending_window = st.session_state.get("home_trending_window", "day")
    if trending_window not in {"day", "week"}:
        trending_window = "day"
    trending_movies = _tmdb_results(cached_trending(api_key, trending_window, 1), api_key)
    top_rated_movies = _tmdb_results(cached_top_rated(api_key, 1), api_key)
    featured_pool = _unique_movies(trending_movies, popular_movies, top_rated_movies)
    featured_movie = featured_pool[0] if featured_pool else {"id": 0, "title": "Featured Movie", "poster_url": None, "vote_average": 0.0, "vote_count": 0, "overview": "", "year": "TBA", "genres": "Not specified"}

    selected_movie_id = st.session_state.get("selected_movie_id")

    if st.session_state.pop("tmdb_data_unavailable", False):
        st.info("Live movie data is temporarily unavailable. Showing the page shell while the API recovers.")

    # If the overlay flag is set, render ONLY the inline details overlay (modal behavior)
    if st.session_state.get("show_overlay") and selected_movie_id:
        render_details_overlay(int(selected_movie_id), api_key)

    elif st.session_state.page == "Home":
        st.markdown('<div class="tmdb-shell">', unsafe_allow_html=True)
        st.markdown('<div class="tmdb-shell-title">Welcome.</div>', unsafe_allow_html=True)
        st.markdown('<div class="tmdb-shell-subtitle">Millions of movies, TV shows and people to discover. Explore now.</div>', unsafe_allow_html=True)

        shell_cols = st.columns([6, 2, 2])
        with shell_cols[0]:
            shell_query = st.text_input(
                "Search for a movie, tv show, person...",
                value=st.session_state.get("shell_query", ""),
                key="home_shell_query_input",
                placeholder="Search for a movie, tv show, person...",
                label_visibility="collapsed",
            )
        with shell_cols[1]:
            mood = st.selectbox(
                "Choose Your Mood",
                ["Action", "Happy", "Romantic", "Thriller", "Emotional"],
                key="home_mood_selector",
                label_visibility="collapsed",
            )
            st.session_state.selected_mood = mood
        with shell_cols[2]:
            do_shell_search = st.button("Search", key="home_shell_search_btn", width="stretch")

        st.session_state.shell_query = shell_query
        if do_shell_search and shell_query.strip():
            st.session_state.search_query = shell_query.strip()
            st.session_state.page = "Search"
            st.rerun()

        trend_label = "Today" if st.session_state.get("home_trending_window", "day") == "day" else "This Week"
        st.markdown(f'<div class="tmdb-chip-row"><span class="tmdb-chip-title">Trending</span><span class="tmdb-chip-active">{trend_label}</span></div>', unsafe_allow_html=True)
        chip_cols = st.columns([1.2, 1.2, 8])
        with chip_cols[0]:
            if st.button("Today", key="chip_today", width="stretch"):
                st.session_state.home_trending_window = "day"
                st.rerun()
        with chip_cols[1]:
            if st.button("This Week", key="chip_week", width="stretch"):
                st.session_state.home_trending_window = "week"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Get selected mood with default fallback
        selected_mood = st.session_state.get("selected_mood", "Action")

        # Ranking is fixed: prefer popularity (views) first, then rating.
        # No user controls required per request.
        rating_w = 0.25
        popularity_w = 0.75

        trending_hybrid = hybrid_recommendation(trending_movies, selected_mood, rating_w, popularity_w)
        popular_hybrid = hybrid_recommendation(popular_movies, selected_mood, rating_w, popularity_w)
        top_rated_hybrid = hybrid_recommendation(top_rated_movies, selected_mood, rating_w, popularity_w)

        # Make hero mood-aware as well
        featured_movie = trending_hybrid[0] if trending_hybrid else featured_movie
        hero_subtitle = f"Trending {trend_label} • {selected_mood} Mood"
        render_hero(featured_movie, hero_subtitle)

        play_row = st.columns([1.3, 1.5, 8])
        # Hero 'More Info' button removed — poster/hero area is clickable to open details overlay.
        
        trending_caption = "Trending today" if st.session_state.get("home_trending_window", "day") == "day" else "Trending this week"
        st.markdown(f'<div class="section-header">🔥 Trending ({selected_mood} Mood)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-caption">{trending_caption}</div>', unsafe_allow_html=True)
        render_movie_grid(trending_hybrid[: st.session_state.cards_to_show], cols=st.session_state.cards_to_show, key_prefix="home_trending")

        st.markdown(f'<div class="section-header">👥 Popular ({selected_mood} Mood)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Popular movies in your mood.</div>', unsafe_allow_html=True)
        render_movie_grid(popular_hybrid[: st.session_state.cards_to_show], cols=st.session_state.cards_to_show, key_prefix="home_popular")

        st.markdown(f'<div class="section-header">⭐ Top Rated ({selected_mood} Mood)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Highest rated movies in your mood.</div>', unsafe_allow_html=True)
        render_movie_grid(top_rated_hybrid[: st.session_state.cards_to_show], cols=st.session_state.cards_to_show, key_prefix="home_top")

        if featured_movie.get("id"):
            detail, _ = _payload(cached_movie_details(int(featured_movie["id"]), api_key))
            recs = _normalize_items(detail.get("recommendations", {}).get("results", []), api_key) if detail else []
            recs_hybrid = hybrid_recommendation(recs, selected_mood)
            st.markdown(f'<div class="section-header">🎯 More Like This ({selected_mood} Mood)</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-caption">Recommended similar titles in your mood.</div>', unsafe_allow_html=True)
            if recs_hybrid:
                render_movie_grid(recs_hybrid[: st.session_state.cards_to_show], cols=st.session_state.cards_to_show, key_prefix="home_recs")

    elif st.session_state.page == "Search":
        st.markdown("# 🔍 Find Your Next Favorite")
        st.markdown("Search our comprehensive movie database and instantly discover related titles.")

        col_search, col_btn = st.columns([4, 1])
        with col_search:
            query = st.text_input(
                "Movie title",
                value=st.session_state.search_query,
                placeholder="Search by title, actor, director…",
                label_visibility="collapsed",
            )
        with col_btn:
            do_search = st.button("🔍 Search", width="stretch")

        if query:
            st.session_state.search_query = query
            payload, error = _payload(cached_search_movies(query, api_key, 1))
            results = _normalize_items(payload.get("results", []), api_key)
            if not results:
                st.info(f"📭 No results for **{query}**. Try a different title or actor name.")
            else:
                st.markdown(f'<div class="section-caption">{len(results)} result(s) found</div>', unsafe_allow_html=True)
                render_movie_grid(results[:20], cols=st.session_state.cards_to_show, key_prefix="search_results")
                first = results[0]
                first_detail, _ = _payload(cached_movie_details(int(first["id"]), api_key))
                recommendations = _normalize_items(first_detail.get("recommendations", {}).get("results", []), api_key) if first_detail else []
                st.markdown('<div class="section-header">🎯 Recommended From Your Search</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-caption">Related titles for your search.</div>', unsafe_allow_html=True)
                if recommendations:
                    render_movie_grid(recommendations[: st.session_state.cards_to_show], cols=st.session_state.cards_to_show, key_prefix="search_recs")
        else:
            st.markdown('<div class="section-header">📈 Popular Right Now</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-caption">Curated by real-time popularity rankings.</div>', unsafe_allow_html=True)
            render_movie_grid(popular_movies[:20], cols=st.session_state.cards_to_show, key_prefix="browse_popular")

    elif st.session_state.page == "Recommendations":
        st.markdown("# 🌟 Personalized Recommendations")
        st.markdown("Discover similar titles based on your selection.")

        if not selected_movie_id:
            st.info("Select a seed movie from Home, Search, or the sidebar to unlock recommendations.")
        else:
            detail_payload, error = _payload(cached_movie_details(int(selected_movie_id), api_key))
            if not detail_payload:
                st.error(error or "Movie details could not be loaded at this time.")
            else:
                movie = _normalize_movie(detail_payload, api_key)
                render_hero(movie, "Your Selected Title")
                if st.button("← Back to Details", key="recs_back_to_details"):
                    st.session_state.page = "Movie Details"
                    st.rerun()
                recommendations = _normalize_items(detail_payload.get("recommendations", {}).get("results", []), api_key)
                st.markdown('<div class="section-header">🎬 Recommended For You</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-caption">Ordered by similarity and audience relevance.</div>', unsafe_allow_html=True)
                if recommendations:
                    render_movie_grid(recommendations[: st.session_state.cards_to_show], cols=st.session_state.cards_to_show, key_prefix=f"recs_{selected_movie_id}")
                else:
                    st.info("No recommendations are currently available for this title.")

    elif st.session_state.page == "Movie Details":
        st.markdown("# 🎬 Title Details")
        if not selected_movie_id:
            st.info("Select a movie from Home or Search to view complete details.")
        else:
            detail_payload, error = _payload(cached_movie_details(int(selected_movie_id), api_key))
            if not detail_payload:
                st.error(error or "Movie details not found.")
            else:
                movie = _normalize_movie(detail_payload, api_key)
                render_hero(movie, "Now Showing")

                btn_row = st.columns([1.2, 1.2, 8])
                with btn_row[0]:
                    st.link_button("▶ Trailer", _youtube_trailer_url(movie["title"]), width="stretch")
                with btn_row[1]:
                    st.link_button("View Full Details", _movie_link(movie["id"]), width="stretch")

                c1, c2 = st.columns([1, 2])
                with c1:
                    if movie.get("poster_url"):
                        st.image(movie["poster_url"], width="stretch")
                    else:
                        st.markdown(
                            '<div style="width:100%;aspect-ratio:2/3;border-radius:16px;background:linear-gradient(135deg,#1a237e,#6a1b9a);display:flex;align-items:center;justify-content:center;padding:18px;text-align:center;color:white;font-weight:700;">Poster unavailable</div>',
                            unsafe_allow_html=True,
                        )
                with c2:
                    st.markdown(f'**Title:** {movie["title"]}')
                    st.markdown(f'**Release date:** {movie.get("release_date") or "Unknown"}')
                    st.markdown(f'**Status:** {movie.get("status") or "Unknown"}')
                    st.markdown(f'**Genres:** {movie.get("genres") or "Not specified"}')
                    st.markdown(f'**Runtime:** {movie.get("runtime") or "N/A"} min')
                    st.markdown(f'**Overview:** {_overview_snippet(movie.get("overview", ""), 1000)}')

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(
                            f'<div class="score-box"><div class="score-val">{movie["vote_average"]:.1f}</div><div class="score-lbl">Vote Average</div></div>',
                            unsafe_allow_html=True,
                        )
                    with m2:
                        st.markdown(
                            f'<div class="score-box"><div class="score-val">{movie["vote_count"]:,}</div><div class="score-lbl">Vote Count</div></div>',
                            unsafe_allow_html=True,
                        )
                    with m3:
                        st.markdown(
                            f'<div class="score-box"><div class="score-val">{movie["popularity"]:.0f}</div><div class="score-lbl">Popularity</div></div>',
                            unsafe_allow_html=True,
                        )
                    with m4:
                        st.markdown(
                            f'<div class="score-box"><div class="score-val">{movie["year"]}</div><div class="score-lbl">Release Year</div></div>',
                            unsafe_allow_html=True,
                        )

                    credits = detail_payload.get("credits", {})
                    cast = credits.get("cast", [])[:6]
                    if cast:
                        st.markdown("**Top Cast:**")
                        st.markdown(", ".join(person.get("name", "") for person in cast if person.get("name")))

                recommendations = _normalize_items(detail_payload.get("recommendations", {}).get("results", []), api_key)
                st.markdown('<div class="section-header">🎯 Similar Titles</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-caption">Similar titles in style and genre.</div>', unsafe_allow_html=True)
                if recommendations:
                    render_movie_grid(recommendations[: st.session_state.cards_to_show], cols=st.session_state.cards_to_show, key_prefix=f"detail_recs_{selected_movie_id}", show_more_info=False)
                else:
                    st.info("No similar titles found for this movie.")

    elif st.session_state.page == "Posters":
        st.markdown("# 🖼️ Posters")
        st.markdown("Enter one or more titles or numeric IDs separated by commas.")

        query = st.text_input(
            "Movie IDs or titles",
            value=st.session_state.poster_query,
            placeholder="e.g. 27205, Inception, Parasite",
            help="This page uses live search. No local movie catalog is used.",
        )
        st.session_state.poster_query = query

        parts = [part.strip() for part in query.split(",") if part.strip()]
        if parts:
            cards: list[dict] = []
            for token in parts:
                if token.isdigit():
                    payload, _ = _payload(cached_movie_details(int(token), api_key))
                    if payload:
                        cards.append(_normalize_movie(payload, api_key))
                        continue
                payload, error = _payload(cached_search_movies(token, api_key, 1))
                movies = _normalize_items(payload.get("results", []), api_key)
                if movies:
                    cards.append(movies[0])
                else:
                    cards.append({
                        "id": 0,
                        "title": token,
                        "year": "TBA",
                        "poster_url": None,
                        "overview": error or "No match found.",
                        "vote_average": 0.0,
                        "vote_count": 0,
                    })
            render_movie_grid(cards, cols=min(5, len(cards)) or 1, key_prefix="poster_lookup")
        else:
            st.info("Type a title or movie ID to preview posters.")
        st.markdown("# 📊 Analytics")
        st.markdown("A lightweight analytics view built from live data samples fetched during this session.")

        sample = _unique_movies(popular_movies, trending_movies, top_rated_movies)[:30]
        detailed_sample: list[dict] = []
        for movie in sample:
            payload, _ = _payload(cached_movie_details(int(movie["id"]), api_key))
            detailed_sample.append(_normalize_movie(payload, api_key) if payload else movie)

        sample_count = len(detailed_sample)
        avg_vote = sum(float(m.get("vote_average") or 0.0) for m in detailed_sample) / sample_count if sample_count else 0.0
        avg_popularity = sum(float(m.get("popularity") or 0.0) for m in detailed_sample) / sample_count if sample_count else 0.0
        unique_genres = sorted({genre for movie in detailed_sample for genre in str(movie.get("genres") or "").split(", ") if genre})

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{sample_count}</div><div class="metric-lbl">Sampled Movies</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_vote:.2f}★</div><div class="metric-lbl">Average Vote</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_popularity:.0f}</div><div class="metric-lbl">Average Popularity</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{len(unique_genres)}</div><div class="metric-lbl">Unique Genres</div></div>', unsafe_allow_html=True)

        genre_counter: Counter[str] = Counter()
        year_counter: Counter[str] = Counter()
        for movie in detailed_sample:
            genre_counter.update([genre.strip() for genre in str(movie.get("genres") or "").split(",") if genre.strip()])
            year = str(movie.get("year") or "")
            if year.isdigit():
                year_counter[year] += 1

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Top Genres")
            if genre_counter:
                top = genre_counter.most_common(10)
                fig, ax = plt.subplots(figsize=(7, 4))
                fig.patch.set_facecolor("#111827")
                ax.set_facecolor("#111827")
                labels = [item[0] for item in top]
                values = [item[1] for item in top]
                ax.barh(labels, values, color="#01b4e4")
                ax.invert_yaxis()
                ax.tick_params(colors="#fff")
                ax.set_title("Genre frequency in sampled titles", color="#fff")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
            else:
                st.info("Not enough genre data to plot yet.")

        with col_b:
            st.markdown("#### Release Years")
            if year_counter:
                top_years = sorted(year_counter.items())
                fig, ax = plt.subplots(figsize=(7, 4))
                fig.patch.set_facecolor("#111827")
                ax.set_facecolor("#111827")
                ax.bar([year for year, _ in top_years], [count for _, count in top_years], color="#90cea1")
                ax.tick_params(axis="x", rotation=45, colors="#fff")
                ax.tick_params(axis="y", colors="#fff")
                ax.set_title("Release year distribution in sampled titles", color="#fff")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
            else:
                st.info("Not enough year data to plot yet.")

        st.markdown("---")
        st.markdown("#### Sampled Titles")
        render_movie_grid(detailed_sample[: st.session_state.cards_to_show], cols=st.session_state.cards_to_show, key_prefix="analytics_sample")


if __name__ == "__main__":
    main()
