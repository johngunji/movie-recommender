# 🎬 Movie & TV Show Recommendation System (ML-Based)

A machine learning–based content recommendation web application built with Python, Flask, and scikit-learn.  
The system recommends similar movies or TV shows based on semantic similarity using TF-IDF vectorization and cosine similarity, with support for fuzzy search, multi-platform filtering, and a responsive UI.

This project focuses on practical ML integration, not just model building.

---

## 🚀 Key Features

### 🔍 Content-Based Recommendations
- Uses TF-IDF Vectorization on movie metadata
- Similarity computed via Cosine Similarity
- No user accounts required (cold-start safe)

### ⚖️ Explicit Metadata Weighting
To improve recommendation relevance, structured metadata is intentionally up-weighted:

| Field        | Weight |
|-------------|--------|
| Genres      | ×10 |
| Cast        | ×8 |
| Description | ×6 |
| Director    | ×5 |

This biases similarity toward genre and cast alignment, while still preserving plot context.

---

### ✏️ Fuzzy Search (Misspelling-Tolerant)
- Handles approximate or misspelled input
- Implemented using difflib.get_close_matches
- Prevents empty results caused by minor typos

---

### 📺 Multi-Platform Support
Combined datasets from:
- Netflix
- Disney+
- Prime Video

Users can filter recommendations by:
- Content type (Movie / TV Show)
- Streaming platform

---

### ❤️ Local Favourites (No Login)
- Users can like/save movies
- Stored using browser localStorage
- Persists across reloads without authentication

---

### 🖼️ Smart Poster Fetching
Hybrid poster retrieval system:
1. OMDb API (primary source)
2. Google Custom Search API (fallback)
3. Local placeholder image if both fail

This design prevents broken UI due to API limits or failures.

---

### 📄 Pagination (“Show More”)
- Initial recommendations are limited
- Additional results loaded incrementally
- Improves performance and UI clarity

---

### 🎨 Modern Responsive UI
- Dark-mode themed interface
- Card-based layout
- Built using Bootstrap + custom CSS
- Mobile-friendly design

---

## 🧠 How Recommendations Work (Pipeline)

1. User enters a movie/show name
2. Input is normalized (lowercased, trimmed)
3. Fuzzy matching resolves approximate titles
4. Movie metadata is combined with explicit weighting
5. TF-IDF vectors are generated
6. Cosine similarity is computed against the dataset
7. Results are filtered by platform and content type
8. Top-N recommendations are returned with posters

---

## 🧠 Tech Stack

### Frontend
- HTML5
- CSS3
- Bootstrap
- JavaScript (Fetch API, localStorage)

### Backend
- Python
- Flask
- Pandas

### Machine Learning
- scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

### APIs
- OMDb API
- Google Custom Search API

---

## 📂 Project Structure

├── app.py  
├── models/  
│   ├── data.csv              # Netflix + Disney dataset  
│   └── prime_movies.csv      # Prime Video dataset  
├── templates/  
│   └── index.html  
├── static/  
│   ├── style.css  
│   └── placeholder.jpg  
├── requirements.txt  
└── README.md  

---

## ⚙️ Environment Variables

Set the following environment variables before running:

export OMDB_API_KEY=your_omdb_api_key
export GOOGLE_API_KEY=your_google_api_key
export GOOGLE_CX=your_custom_search_cx




▶️ Run Locally
pip install -r requirements.txt
python app.py


Open in browser:

http://127.0.0.1:5000

🌐 Deployment

Deployed on Render

Uses dynamic port binding via PORT environment variable

Automatically sleeps on inactivity (free tier behavior)

Secrets managed using platform environment variables

🧩 Future Improvements

Recommendation evaluation metrics

Trending movies based on interaction frequency

Backend-stored favourites with user accounts

Weighted similarity tuning via validation

Performance optimizations for large datasets

👨‍💻 Author

Gunji John
B.Tech Computer Science & Engineering
IIT (ISM) Dhanbad

GitHub: https://github.com/johngunji

📜 License

This project is intended for learning and portfolio purposes.
