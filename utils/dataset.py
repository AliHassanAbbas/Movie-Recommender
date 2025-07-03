import torch
from torch.utils.data import Dataset

class RatingsDataset(Dataset):
    """
    PyTorch Dataset for user-item-rating triplets.
    """
    def __init__(self, ratings_df):
        self.users = torch.tensor(ratings_df['user'].values, dtype=torch.long)
        self.items = torch.tensor(ratings_df['item'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

