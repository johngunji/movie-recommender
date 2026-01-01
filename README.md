🎬 Movie Recommendation System (ML + Web)

A content-based movie recommender system built using Machine Learning (TF-IDF) and deployed as a live web application.

🔗 Live Demo:
https://movie-recommender-nx2b.onrender.com

🚀 Features

Content-based recommendations using TF-IDF + cosine similarity

Fast, memory-safe similarity computation (no precomputed NxN matrix)

Dynamic movie poster fetching via Google Custom Search API

Clean web interface built with Flask + HTML/CSS

Fully deployed on Render (Free Tier)

🧠 How It Works

Movie metadata is combined into a single content field

TF-IDF Vectorization converts text into numerical features

On each request:

The selected movie’s vector is compared against all movies

Top-N similar movies are returned

Posters are fetched dynamically and cached for performance

🛠 Tech Stack
Machine Learning

TF-IDF Vectorizer

Cosine similarity (computed on demand using linear_kernel)

NumPy, Pandas, Scikit-learn

Backend

Python

Flask

Joblib

Frontend

HTML

CSS

JavaScript (Fetch API)

Deployment

Render (Free Tier)

Environment variables for API security

📂 Project Structure
├── app.py
├── requirements.txt
├── models/
│   └── movies.pkl
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── placeholder.jpg

🔐 Environment Variables

The following environment variables are required:

GOOGLE_API_KEY = your_api_key
GOOGLE_CX      = your_search_engine_id(06ee71665e2d143d5)-mine


(They are not hard-coded for security.)

⚡ Performance Notes

Optimized for 512 MB RAM environments

Uses sparse matrices to avoid memory overflow

First request may be slower due to free-tier cold start

Subsequent requests are fast

🧪 Future Improvements

User-based collaborative filtering

Movie search autocomplete

Persistent poster caching (Redis)

UI enhancements (skeleton loaders, animations)

API rate limiting

📌 Learning Outcomes

Practical ML system design

Memory-efficient similarity computation

Real-world cloud deployment

Debugging Linux vs Windows issues

Secure API handling

🧑‍💻 Author

John Gunji
B.Tech CSE | IIT Dhanbad
Inter interests: Machine Learning · Data Science · Web Development
