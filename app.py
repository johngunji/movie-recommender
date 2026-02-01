import os
import difflib
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- LOAD DATA ----------
netflix_disney = pd.read_csv(os.path.join(BASE_DIR, "models", "data.csv"))
prime = pd.read_csv(os.path.join(BASE_DIR, "models", "prime_movies.csv"))

netflix_disney["platform"] = "Netflix / Disney"
prime["platform"] = "Prime Video"

movies = pd.concat([netflix_disney, prime], ignore_index=True)

# ---------- CLEAN ----------
movies = movies.fillna("")
movies["title"] = movies["title"].str.lower().str.strip()
movies.drop_duplicates(subset="title", inplace=True)
movies.reset_index(drop=True, inplace=True)

# ---------- CONTENT ----------
movies["content"] = (
    movies["genres"] + " " +
    movies["cast"] + " " +
    movies["director"] + " " +
    movies["description"]
)

# ---------- TF-IDF ----------
tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
tfidf_matrix = tfidf.fit_transform(movies["content"])

# ---------- POSTER ----------
poster_cache = {}

def get_poster(title):
    return "/static/placeholder.jpg"

# ---------- FUZZY ----------
def resolve_title_fuzzy(query, titles):
    match = difflib.get_close_matches(query, titles, n=1, cutoff=0.6)
    return match[0] if match else None

# ---------- RECOMMENDER ----------
def recommend_like_this(query, start, limit, content_type, platform):
    query = query.lower().strip()
    data = movies.copy()

    if content_type:
        data = data[data["type"].str.lower() == content_type.lower()]

    if platform:
        data = data[data["platform"] == platform]

    if data.empty:
        return []

    titles = data["title"].tolist()
    if query not in titles:
        query = resolve_title_fuzzy(query, titles)
        if not query:
            return []

    idx = data[data["title"] == query].index[0]
    scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    ranked = scores.argsort()[::-1]

    all_idx = [i for i in ranked if i != idx]
    paged_idx = all_idx[start:start + limit]

    results = movies.loc[paged_idx][
        ["title", "type", "genres", "platform"]
    ].copy()

    results["poster"] = results["title"].apply(get_poster)
    return results.to_dict(orient="records")

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    return jsonify(
        recommend_like_this(
            data.get("movie", ""),
            int(data.get("start", 0)),
            int(data.get("limit", 10)),
            data.get("type", ""),
            data.get("platform", "")
        )
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


