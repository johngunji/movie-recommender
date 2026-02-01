import os
import difflib
import pandas as pd
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

# ---------- POSTER (INTENTIONAL FALLBACK) ----------
def get_poster(title: str) -> str:
    return "/static/placeholder.jpg"

# ---------- FUZZY MATCH ----------
def resolve_title_fuzzy(query, titles):
    match = difflib.get_close_matches(query, titles, n=1, cutoff=0.6)
    return match[0] if match else None

# ---------- RECOMMENDER (CORRECT LOGIC) ----------
def recommend_like_this(query, start, limit, content_type, platform):
    query = query.lower().strip()
    data = movies.copy()

    if content_type:
        data = data[data["type"].str.lower() == content_type.lower()]

    if platform:
        data = data[data["platform"] == platform]

    if data.empty:
        return []

    titles = data["title"]()
