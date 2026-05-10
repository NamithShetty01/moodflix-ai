# MRS - Movie Recommendation System

A movie discovery web app built with Streamlit.

- Live search and browsing from an external movie catalog
- Movie detail overlay with cast, overview, ratings, trailer, and similar titles
- Single-poster hero layout for the featured movie
- Popular, trending, top-rated, and analytics views
- Mood-aware recommendation filtering on the Home page

## Features

- Adaptive mood-based hybrid recommendations
- Mood selector for Action, Happy, Romantic, Thriller, and Emotional
- Genre matching with popularity and rating ranking
- Trending, top-rated, and popular movie views
- Search by movie title
- Interactive analytics and dataset visualizations
- Poster lookup for movie titles and catalog IDs

## Methodology

**Adaptive Mood-Based Hybrid OTT Recommendation System**

The app uses movie catalog data to generate personalized recommendations through a mood-based hybrid flow:

Movie Catalog -> Mood Filter -> Genre Matching -> Popularity Analysis -> Rating Ranking -> Personalized Recommendations

The ranking stage uses a hybrid score that blends rating and popularity so that higher-quality, more relevant titles are shown first.

## Tech Stack

- Python 3.10+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Requests

## Project Structure

```text
movie-recommendation/
	app.py
	tmdb_app.py
	README.md
	requirements.txt
	src/
		__init__.py
		tmdb_service.py
		poster_service.py
	visualization/
		__init__.py
		plots.py
```

## Setup and Run

### 1. Clone the repository

```bash
git clone https://github.com/NamithShetty01/MRS.git
cd MRS
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Mac or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

http://localhost:8501

## Data Source Setup Notes

- Add the required API key in the sidebar before starting the app.
- If no key is provided, the app will prompt you to enter one and will not load catalog data.

## Evaluation Support

The project includes evaluation utilities for offline testing:

- RMSE
- MAE
- Precision at K
- Recall at K
- F1 at K

See `src/evaluation.py` for metric implementations.

## Troubleshooting

- If Streamlit command is missing, install dependencies again in the active virtual environment.
- If movie data does not load, confirm your API key is valid and the network can reach the movie data service.

## Author

Namith Naveen Shetty
