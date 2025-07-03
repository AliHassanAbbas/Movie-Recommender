import pandas as pd

def load_movielens_1m(data_path_ratings="data/ml-1m/ratings.dat",
                      data_path_movies="data/ml-1m/movies.dat"):
    """
    Loads MovieLens 1M ratings and movies files into Pandas DataFrames.
    """
    ratings_df = pd.read_csv(
        data_path_ratings,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"]
    )
    movies_df = pd.read_csv(
        data_path_movies,
        sep="::",
        engine="python",
        encoding="latin-1",  # <-- Fix UnicodeDecodeError here!
        names=["movieId", "title", "genres"]
    )
    return ratings_df, movies_df



def preprocess_ratings(ratings_df):
    """
    Encodes user and item IDs to contiguous integers starting at 0.
    Returns updated DataFrame and total numbers of users and items.
    """
    unique_users = ratings_df['userId'].unique()
    unique_items = ratings_df['movieId'].unique()
    
    # Create mappings: original ID -> new ID
    user2idx = {original: idx for idx, original in enumerate(unique_users)}
    item2idx = {original: idx for idx, original in enumerate(unique_items)}
    
    # Apply mappings
    ratings_df['user'] = ratings_df['userId'].map(user2idx)
    ratings_df['item'] = ratings_df['movieId'].map(item2idx)
    
    num_users = len(unique_users)
    num_items = len(unique_items)
    
    print(f"Number of users: {num_users}, Number of items: {num_items}")
    
    return ratings_df[['user', 'userId', 'item', 'movieId', 'rating']], num_users, num_items, user2idx, item2idx


def build_movie_dict(movies_df):
    """
    Creates a mapping from original movieId -> movie metadata (title, genres).
    Returns a dictionary: {movieId: {"title": str, "genres": str}}
    """
    movie_dict = {}
    for _, row in movies_df.iterrows():
        movie_dict[int(row['movieId'])] = {
            "title": row['title'],
            "genres": row['genres']
        }
    return movie_dict
