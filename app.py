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
movies["genres"] = movies["genres"].str.replace("|", " ")


# ---------- CONTENT ----------
movies["content"] = (
    movies["genres"] *10+ " " +
    movies["cast"]*8 + " " +
    movies["director"]*5 + " " +
    movies["description"]*6
)

# ---------- TF-IDF ----------
tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
tfidf_matrix = tfidf.fit_transform(movies["content"])

# ---------- POSTERS ----------
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
        p = r.get("Poster")
        if p and p != "N/A":
            return p
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
            return items[0]["link"]
    except Exception:
        pass
    return None

def get_poster(title):
    if title in poster_cache:
        return poster_cache[title]

    poster = fetch_omdb(title) or fetch_google(title)
    if not poster:
        poster = "/static/placeholder.jpg"

    poster_cache[title] = poster
    return poster

# ---------- FUZZY MATCH ----------
def fuzzy_match(query, titles):
    match = difflib.get_close_matches(query, titles, n=1, cutoff=0.6)
    return match[0] if match else None

# ---------- RECOMMENDER ----------
def recommend(query, start, limit, content_type, platform):
    data = movies.copy()
    query = query.lower().strip()

    if content_type:
        data = data[data["type"].str.lower() == content_type.lower()]
    if platform:
        data = data[data["platform"] == platform]

    if data.empty:
        return []

    titles = data["title"].tolist()
    if query not in titles:
        query = fuzzy_match(query, titles)
        if not query:
            return []

    idx = data[data["title"] == query].index[0]
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix[data.index]).flatten()
    ranked = sim_scores.argsort()[::-1]

    rec_ids = [data.index[i] for i in ranked if data.index[i] != idx]
    rec_ids = rec_ids[start:start+limit]

    results = movies.loc[rec_ids, ["title", "type", "genres", "platform"]].copy()
    results["poster"] = results["title"].apply(get_poster)

    return results.to_dict(orient="records")

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend_api():
    d = request.get_json(force=True)
    return jsonify(
        recommend(
            d.get("movie", ""),
            int(d.get("start", 0)),
            int(d.get("limit", 10)),
            d.get("type", ""),
            d.get("platform", "")
        )
    )

# ---------- RUN ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

