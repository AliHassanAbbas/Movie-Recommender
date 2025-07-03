# 🎬 Neural Collaborative Filtering Movie Recommender

A neural network–based movie recommender system built using the MovieLens 1M dataset and TMDb API for rich movie metadata.

This project trains a Neural Collaborative Filtering (NCF) model to predict user ratings and generates personalized movie recommendations. It also provides an interactive frontend via Streamlit where users can browse movies, see posters, summaries, genres, and get recommendations.

---

## 🚀 Features

✅ Trains an NCF model on MovieLens 1M  

✅ Fetches movie posters, genres, and summaries via TMDb API  

✅ Streamlit app with interactive movie grid & recommendation viewer  

✅ Easily customizable & extendable for your own datasets

---

## 📂 Project Structure

recommender-system/

├── data/

│ └── ml-1m/ # Place MovieLens 1M files here after downloading
├── models/

│ └── ncf.py # Neural Collaborative Filtering model definition
├── utils/

│ ├── dataloader.py # Functions to load & preprocess MovieLens data

│ ├── dataset.py # PyTorch Dataset class

│ └── evaluate.py # Evaluation helper

├── app.py # Streamlit app entry point

├── main.py # Training script

├── requirements.txt # Python dependencies

├── README.md # This readme

└── .gitignore


---

## 📥 Dataset

This project uses the **MovieLens 1M dataset**. Download it from [GroupLens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/).

After downloading and extracting, place the folder like this:

recommender-system/

└── data/

└── ml-1m/

├── ratings.dat

└── movies.dat

---

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AliHassanAbbas/recommender-system.git
   cd recommender-system
Create a virtual environment (recommended) and install dependencies:
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

🔑 TMDb API Key

To fetch movie posters, summaries, and genres, this project uses the TMDb API. Sign up at TMDb to get a free API key. Replace the TMDB_API_KEY variable in app.py with your own key:
TMDB_API_KEY = "your_api_key_here"


🏋️ Training the Model

Train the recommender with:
python main.py

🎨 Running the Streamlit App

Launch the interactive recommender app with:

streamlit run streamlit_app.py

This will open your browser. You can:

✅ Browse movies with posters and descriptions

✅ Click Get Recommendations for personalized suggestions

✅ See recommended movies with posters, genres, and summaries


📜 License

MIT License © 2025 AliHassan

🙌 Acknowledgements

MovieLens dataset by GroupLens Research

TMDb API for movie metadata


🔗 Connect

Feel free to open issues or pull requests if you’d like to contribute!
