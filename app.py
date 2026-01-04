import os
import numpy as np
import pandas as pd
import joblib
import requests
import atexit
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- LOAD & MERGE DATASETS ----------
netflix_disney = pd.read_csv(os.path.join(BASE_DIR, "models", "data.csv"))
prime = pd.read_csv(os.path.join(BASE_DIR, "models", "prime_movies.csv"))

movies = pd.concat([netflix_disney, prime], ignore_index=True)

movies["title"] = movies["title"].str.lower().str.strip()
movies.drop_duplicates(subset="title", inplace=True)
movies = movies.reset_index(drop=True)

movies = movies.fillna("")   




# Build TF-IDF and cosine similarity at startup
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=20000   # IMPORTANT: caps memory
)
tfidf_matrix = tfidf.fit_transform(movies["content"])


# Normalize titles
movies["title"] = movies["title"].str.lower().str.strip()
indices = pd.Series(movies.index, index=movies["title"])

# ---------- GOOGLE IMAGE SEARCH ----------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CX = os.environ.get("GOOGLE_CX")

if not GOOGLE_API_KEY or not GOOGLE_CX:
    raise RuntimeError("GOOGLE_API_KEY or GOOGLE_CX not set")


# Image cache (disk-backed)
CACHE_FILE = os.path.join(BASE_DIR, "image_cache.pkl")

if os.path.exists(CACHE_FILE):
    image_cache = joblib.load(CACHE_FILE)
else:
    image_cache = {}

@atexit.register
def save_cache():
    joblib.dump(image_cache, CACHE_FILE)

def clean_title_for_search(title: str) -> str:
    title = title.split(":")[0]          # remove subtitles
    title = title.replace("&", "and")
    return title.strip()

def get_poster(title):
    if title in image_cache:
        return image_cache[title]

    try:
        query = f"{clean_title_for_search(title)} movie poster"

        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CX,
                "q": query,
                "searchType": "image",
                "num": 1
            },
            timeout=5
        ).json()

        items = r.get("items")
        if items:
            link = items[0].get("link")
            image_cache[title] = link
            return link

    except Exception:
        pass

    image_cache[title] = "/static/placeholder.jpg"
    return image_cache[title]

# ---------- RECOMMENDER ----------
def resolve_index(query):
    query = query.lower().strip()

    if query in indices:
        idx = indices[query]
        return int(idx.iloc[0]) if hasattr(idx, "iloc") else int(idx)

    matches = movies[movies["title"].str.contains(query, na=False)]
    if matches.empty:
        return None

    starts = matches[matches["title"].str.startswith(query)]
    if not starts.empty:
        return int(starts.index[0])

    return int(matches.index[0])

from sklearn.metrics.pairwise import linear_kernel  # place once, at top-level


def recommend_like_this(query, n=10):
    idx = resolve_index(query)
    if idx is None:
        return []

    # Compute similarity ON DEMAND (memory safe)
    query_vec = tfidf_matrix[idx]
    scores = linear_kernel(query_vec, tfidf_matrix).flatten()
    ranked = scores.argsort()[::-1]

    rec_idx = [i for i in ranked if i != idx][:n]

    results = movies.iloc[rec_idx][
        ["title", "type", "director", "cast", "language", "genres"]
    ].copy()

    results["poster"] = results["title"].apply(get_poster)
    return results.to_dict(orient="records")


# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(force=True)
    movie = data.get("movie", "")
    results = recommend_like_this(movie, n=10)
    return jsonify(results)

# ---------- RUN ----------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)







