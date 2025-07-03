import torch

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for users, items, ratings in data_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            outputs = model(users, items)
            loss = criterion(outputs, ratings)
            total_loss += loss.item() * users.size(0)

    avg_loss = total_loss / len(data_loader.dataset)
    rmse = avg_loss ** 0.5
    print(f"Evaluation RMSE: {rmse:.4f}")
    model.train()
