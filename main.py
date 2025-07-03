import torch
from utils.dataloader import load_movielens_1m, preprocess_ratings
from utils.dataset import RatingsDataset
from torch.utils.data import DataLoader
from models.ncf import NCF
from utils.evaluate import evaluate_model
import os

if __name__ == "__main__":
    # === 1) Load ratings and movies from MovieLens 1M ===
    ratings_df, movies_df = load_movielens_1m()
    print("Ratings shape:", ratings_df.shape)
    print(ratings_df.head())

    # === 2) Preprocess ratings ===
    ratings_df, num_users, num_items, user2idx, item2idx = preprocess_ratings(ratings_df)
    print("\nEncoded ratings:")
    print(ratings_df.head())

    # === 3) Dataset and DataLoader ===
    dataset = RatingsDataset(ratings_df)
    print(f"Total samples in dataset: {len(dataset)}")

    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)

    # Show example batch
    for users, items, ratings in train_loader:
        print(f"Batch - users: {users.shape}, items: {items.shape}, ratings: {ratings.shape}")
        break

    # === 4) Initialize training ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = NCF(num_users, num_items, embedding_dim=64).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50  # Adjust as needed

    # === 5) Training Loop ===
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for users, items, ratings in train_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)

            optimizer.zero_grad()
            outputs = model(users, items)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * users.size(0)  # Accumulate weighted batch loss

        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Evaluate model after each epoch
        evaluate_model(model, train_loader, device)

    # === 6) Test prediction on a sample user/item ===
    model.eval()
    user_id = 0  # example encoded user index
    item_id = 1  # example encoded item index
    with torch.no_grad():
        pred_rating = model(
            torch.tensor([user_id], device=device),
            torch.tensor([item_id], device=device)
        )
    print(f"Predicted rating by user {user_id} for item {item_id}: {pred_rating.item():.2f}")

    # === 7) Save model ===
    save_path = os.path.join(os.path.dirname(__file__), "ncf_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at: {save_path}")
