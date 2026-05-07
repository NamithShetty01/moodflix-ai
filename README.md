# MRS - Movie Recommendation System

A  movie recommendation web app built with Streamlit and a hybrid recommender that combines:

- Collaborative Filtering (SVD-based)
- Content-Based Filtering (genre similarity)

The app supports personalized recommendations, similar-movie discovery, popularity views, and analytics plots.

## Features

- Hybrid recommendation engine with weighted blending
- User-based recommendations by userId
- Similar movie suggestions for any selected title
- Trending, top-rated, and most-popular movie views
- Search by movie title
- Interactive analytics and dataset visualizations
- Automatic MovieLens download fallback if data is missing

## How the Hybrid Model Works

The final recommendation score is computed as:

Final Score = alpha * CF Score + (1 - alpha) * CB Score

Where:

- CF Score comes from matrix factorization (SVD)
- CB Score comes from genre-based similarity
- alpha controls the blend between collaborative and content signals

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
	.gitignore
	app.py
	requirements.txt
	README.md
	data/
		links.csv
		movies.csv
		ratings.csv
	src/
		__init__.py
		collaborative_filtering.py
		content_filtering.py
		data_preprocessing.py
		evaluation.py
		hybrid_recommender.py
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

## Dataset Notes

- The full MovieLens ratings dataset can be very large.
- The project currently excludes very large dataset files from Git tracking.
- If required files are missing, the preprocessing pipeline attempts to:
	1) download MovieLens data, or
	2) generate a realistic sample dataset for offline use.

## Evaluation Support

The project includes evaluation utilities for offline testing:

- RMSE
- MAE
- Precision at K
- Recall at K
- F1 at K

See src/evaluation.py for metric implementations.

## Troubleshooting

- If the app is slow with large ratings data, reduce collaborative filtering matrix size using these environment variables:
	- CF_MAX_USERS
	- CF_MAX_MOVIES
	- CF_MIN_USER_RATINGS
	- CF_MIN_MOVIE_RATINGS
- If Streamlit command is missing, install dependencies again in the active virtual environment.

## Author

Namith Naveen Shetty
