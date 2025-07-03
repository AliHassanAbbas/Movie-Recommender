import streamlit as st
import torch
import os
import requests
from utils.dataloader import load_movielens_1m, preprocess_ratings, build_movie_dict
from models.ncf import NCF

# TMDb API key (replace with your own if needed)
TMDB_API_KEY = "f88387f379069be7aee429272d5c460a"

# === 1) Load MovieLens 1M data ===
ratings_df, movies_df = load_movielens_1m()
ratings_df, num_users, num_items, user2idx, item2idx = preprocess_ratings(ratings_df)

# === 2) Build movie metadata dictionary ===
movie_id_to_meta = build_movie_dict(movies_df)

# === 3) Build mappings: title ↔ original movieId ===
title_to_original_id = {meta["title"]: movie_id for movie_id, meta in movie_id_to_meta.items()}

# === 4) Map original movieId → encoded item ID ===
original_to_encoded_item = {orig: item2idx[orig] for orig in item2idx}

# === 5) Load trained model ===
device = torch.device("cpu")
model = NCF(num_users, num_items, embedding_dim=64)
model_path = os.path.join(os.path.dirname(__file__), "ncf_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

st.title("🎬 Movie Recommender System (1M Dataset)")

def clean_title(title):
    return title.split(" (")[0] if "(" in title else title

def fetch_movie_info(title):
    """Fetch poster, overview, genres from TMDb."""
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    resp = requests.get(url, params=params)
    if resp.status_code != 200 or not resp.json().get("results"):
        return None, "No summary available.", []

    movie = resp.json()["results"][0]
    poster = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None
    overview = movie.get("overview", "No summary available.")
    genre_ids = movie.get("genre_ids", [])

    # Fetch genre names
    genre_resp = requests.get("https://api.themoviedb.org/3/genre/movie/list", params={"api_key": TMDB_API_KEY})
    genre_map = {g["id"]: g["name"] for g in genre_resp.json().get("genres", [])}
    genres = [genre_map.get(gid, "Other") for gid in genre_ids]

    return poster, overview, genres

# Show first 40 titles for performance reasons
all_titles = list(title_to_original_id.keys())[:40]

if "selected_movie" not in st.session_state:
    # === Grid view with movie posters ===
    cols = st.columns(4)
    for idx, title in enumerate(all_titles):
        cleaned = clean_title(title)
        with cols[idx % 4]:
            poster, overview, genres = fetch_movie_info(cleaned)
            st.image(
                poster if poster else "https://via.placeholder.com/200x300?text=No+Image",
                caption=cleaned,
                use_container_width=True,
            )
            st.write(f"**Genres:** {', '.join(genres) if genres else 'N/A'}")
            st.caption(overview[:150] + "..." if len(overview) > 150 else overview)
            if st.button("🎥 Get Recommendations", key=f"rec_{idx}"):
                st.session_state["selected_movie"] = title
else:
    # === Recommendations view ===
    movie_title = st.session_state["selected_movie"]
    cleaned = clean_title(movie_title)
    st.subheader(f"🔎 Recommendations for **{cleaned}**")
    original_item_id = title_to_original_id[movie_title]
    encoded_item_id = original_to_encoded_item.get(original_item_id, None)

    if encoded_item_id is None:
        st.warning("❌ Sorry, no data available for this movie.")
    else:
        with torch.no_grad():
            all_user_ids = torch.arange(num_users, device=device)
            repeated_items = torch.tensor([encoded_item_id] * num_users, device=device)
            predictions = model(all_user_ids, repeated_items)
            top_users = predictions.argsort(descending=True)[:6]

        recommendations = []
        for user_idx in top_users.cpu().numpy():
            user_ratings = ratings_df[ratings_df['user'] == user_idx]
            top_movie_row = user_ratings.sort_values('rating', ascending=False).head(1)
            for _, row in top_movie_row.iterrows():
                rec_item_id = int(row['movieId'])
                rec_title = movie_id_to_meta.get(rec_item_id, {}).get("title", "Unknown Movie")
                recommendations.append(rec_title)

        if recommendations:
            rec_cols = st.columns(3)
            for i, rec in enumerate(recommendations):
                with rec_cols[i % 3]:
                    cleaned_rec = clean_title(rec)
                    poster, overview, genres = fetch_movie_info(cleaned_rec)
                    st.image(
                        poster if poster else "https://via.placeholder.com/200x300?text=No+Image",
                        caption=cleaned_rec,
                        use_container_width=True,
                    )
                    st.write(f"**Genres:** {', '.join(genres) if genres else 'N/A'}")
                    st.caption(overview[:150] + "..." if len(overview) > 150 else overview)
        else:
            st.info("🤔 No recommendations found.")

    if st.button("🔙 Go Back"):
        del st.session_state["selected_movie"] 
