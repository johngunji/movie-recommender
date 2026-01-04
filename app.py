import os
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
poster_cache = {}


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
def clean_title(title):
    return title.split(":")[0].strip()

OMDB_API_KEY = os.environ.get("OMDB_API_KEY")
def fetch_omdb(title):
    OMDB_API_KEY = os.environ.get("OMDB_API_KEY")
    if not OMDB_API_KEY:
        return None

    try:
        r = requests.get(
            "http://www.omdbapi.com/",
            params={"apikey": OMDB_API_KEY, "t": clean_title(title)},
            timeout=5
        ).json()

        poster = r.get("Poster")
        if poster and poster != "N/A":
            return poster
    except Exception:
        pass

    return None


def fetch_google_image(title):
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    GOOGLE_CX = os.environ.get("GOOGLE_CX")

    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return None

    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CX,
                "q": f"{clean_title(title)} movie poster",
                "searchType": "image",
                "num": 1
            },
            timeout=5
        ).json()

        items = r.get("items")
        if items:
            return items[0].get("link")
    except Exception:
        pass

    return None
def get_poster(title):
    if title in poster_cache:
        return poster_cache[title]

    # 1️⃣ Try OMDb
    poster = fetch_omdb(title)
    if poster:
        poster_cache[title] = poster
        return poster

    # 2️⃣ Fallback to Google
    poster = fetch_google_image(title)
    if poster:
        poster_cache[title] = poster
        return poster

    # 3️⃣ Placeholder
    poster_cache[title] = "/static/placeholder.jpg"
    return poster_cache[title]

def recommend_like_this(query, n=10, content_type="", language="", platform=""):
    query = query.lower().strip()

    # Filter first
    data = movies.copy()

    if content_type:
        data = data[data["type"].str.lower() == content_type.lower()]

    if language:
        data = data[data["language"].str.contains(language, case=False)]

    if platform:
        data = data[data["platform"] == platform]

    if data.empty:
        return []

    # Build local indices
    local_indices = pd.Series(data.index, index=data["title"])

    if query not in local_indices:
        return []

    idx = local_indices[query]

    # Compute similarity ONLY on filtered data
    filtered_matrix = tfidf_matrix[data.index]

    query_vec = tfidf_matrix[idx]
    similarity = linear_kernel(query_vec, filtered_matrix).flatten()

    ranked = similarity.argsort()[::-1]
    rec_idx = [data.index[i] for i in ranked if data.index[i] != idx][:n]

    results = movies.loc[rec_idx][
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


