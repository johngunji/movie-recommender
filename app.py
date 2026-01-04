import os
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# ---------- PATH ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- LOAD & MERGE DATASETS ----------
netflix_disney = pd.read_csv(os.path.join(BASE_DIR, "models", "data.csv"))
prime = pd.read_csv(os.path.join(BASE_DIR, "models", "prime_movies.csv"))

movies = pd.concat([netflix_disney, prime], ignore_index=True)

# Clean data
movies = movies.fillna("")
movies["title"] = movies["title"].str.lower().str.strip()
movies.drop_duplicates(subset="title", inplace=True)
movies.reset_index(drop=True, inplace=True)

# Ensure content exists
if "content" not in movies.columns:
    movies["content"] = (
        movies["genres"] + " " +
        movies["genres"] + " " +
        movies["cast"] + " " +
        movies["director"] + " " +
        movies["description"]
    )

# ---------- TF-IDF ----------
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=20000
)
tfidf_matrix = tfidf.fit_transform(movies["content"])
indices = pd.Series(movies.index, index=movies["title"])

# ---------- POSTER (SAFE) ----------
OMDB_API_KEY = os.environ.get("OMDB_API_KEY")

def get_poster(title):
    if not OMDB_API_KEY:
        return "/static/placeholder.jpg"
    try:
        r = requests.get(
            "http://www.omdbapi.com/",
            params={"apikey": OMDB_API_KEY, "t": title},
            timeout=5
        ).json()
        poster = r.get("Poster")
        if poster and poster != "N/A":
            return poster
    except Exception:
        pass
    return "/static/placeholder.jpg"

# ---------- RECOMMENDER ----------
def resolve_index(query):
    query = query.lower().strip()
    if query in indices:
        return int(indices[query])
    matches = movies[movies["title"].str.contains(query)]
    if matches.empty:
        return None
    return int(matches.index[0])

def recommend_like_this(query, n=10):
    idx = resolve_index(query)
    if idx is None:
        return []

    query_vec = tfidf_matrix[idx]
    scores = linear_kernel(query_vec, tfidf_matrix).flatten()
    ranked = scores.argsort()[::-1]

    rec_idx = [i for i in ranked if i != idx][:n]

    results = movies.iloc[rec_idx][
        ["title", "type", "language", "genres", "director", "cast"]
    ].copy()

    results["poster"] = results["title"].apply(get_poster)
    return results.to_dict(orient="records")

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json(force=True)
        movie = data.get("movie", "")
        results = recommend_like_this(movie, n=10)
        return jsonify(results)
    except Exception as e:
        print("ERROR:", e)
        return jsonify([])

# ---------- RUN ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
