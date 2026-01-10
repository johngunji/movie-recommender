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

# ---------- ENSURE CONTENT ----------
if "content" not in movies.columns:
    movies["content"] = (
        movies["genres"] + " " +
        movies["cast"] + " " +
        movies["director"] + " " +
        movies["description"]
    )

# ---------- TF-IDF ----------
tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
tfidf_matrix = tfidf.fit_transform(movies["content"])

# ---------- POSTER CACHE ----------
poster_cache = {}

def clean_title(title):
    return title.split(":")[0].strip()

def fetch_omdb(title):
    key = os.environ.get("OMDB_API_KEY")
    if not key:
        return None
    try:
        r = requests.get(
            "http://www.omdbapi.com/",
            params={"apikey": key, "t": clean_title(title)},
            timeout=5
        ).json()
        poster = r.get("Poster")
        if poster and poster != "N/A":
            return poster
    except Exception:
        pass
    return None

def fetch_google(title):
    key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CX")
    if not key or not cx:
        return None
    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": key,
                "cx": cx,
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

    poster = fetch_omdb(title)
    if poster:
        poster_cache[title] = poster
        return poster

    poster = fetch_google(title)
    if poster:
        poster_cache[title] = poster
        return poster

    poster_cache[title] = "/static/placeholder.jpg"
    return poster_cache[title]

# ---------- FUZZY MATCH ----------
def resolve_title_fuzzy(query, titles):
    match = difflib.get_close_matches(query, titles, n=1, cutoff=0.6)
    return match[0] if match else None

# ---------- RECOMMENDER WITH PAGINATION ----------
def recommend_like_this(query, start, limit, content_type, language, platform):
    query = query.lower().strip()
    data = movies.copy()

    if content_type:
        data = data[data["type"].str.lower() == content_type.lower()]
    if language:
        data = data[data["language"].str.contains(language, case=False)]
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
    filtered_matrix = tfidf_matrix[data.index]

    scores = linear_kernel(tfidf_matrix[idx], filtered_matrix).flatten()
    ranked = scores.argsort()[::-1]

    all_idx = [data.index[i] for i in ranked if data.index[i] != idx]
    paged_idx = all_idx[start:start + limit]

    results = movies.loc[paged_idx][
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
    return jsonify(
        recommend_like_this(
            data.get("movie", ""),
            int(data.get("start", 0)),
            int(data.get("limit", 5)),
            data.get("type", ""),
            data.get("language", ""),
            data.get("platform", "")
        )
    )

# ---------- RUN ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
