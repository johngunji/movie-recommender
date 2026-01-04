import os
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# ---------- PATH ----------
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

# ---------- ENSURE CONTENT ----------
if "content" not in movies.columns:
    movies["content"] = (
        movies["genres"] + " " +
        movies["genres"] + " " +
        movies["cast"] + " " +
        movies["director"] + " " +
        movies["description"]
    )

# ---------- POPULARITY SAFE ----------
if "popularity" not in movies.columns:
    movies["popularity"] = 0

# ---------- TF-IDF ----------
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=20000
)
tfidf_matrix = tfidf.fit_transform(movies["content"])
indices = pd.Series(movies.index, index=movies["title"])

# ---------- POSTER ----------
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

def recommend_like_this(query, n=10, content_type="", language="", platform=""):
    idx = resolve_index(query)
    if idx is None:
        return []

    data = movies.copy()

    if content_type:
        data = data[data["type"].str.lower() == content_type.lower()]

    if language:
        data = data[data["language"].str.contains(language, case=False)]

    if platform:
        data = data[data["platform"] == platform]

    query_vec = tfidf_matrix[idx]
    similarity = linear_kernel(query_vec, tfidf_matrix).flatten()

    pop = movies["popularity"].astype(float).values
    if pop.max() > 0:
        pop = pop / pop.max()

    final_score = 0.85 * similarity + 0.15 * pop
    ranked = final_score.argsort()[::-1]

    rec_idx = [i for i in ranked if i != idx][:n]

    results = data.iloc[rec_idx][
        ["title", "type", "language", "genres", "director", "cast", "platform"]
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
    content_type = data.get("type", "")
    language = data.get("language", "")
    platform = data.get("platform", "")

    results = recommend_like_this(
        movie, 10,
        content_type=content_type,
        language=language,
        platform=platform
    )
    return jsonify(results)

# ---------- RUN ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
