#Neural Collaborative Filtering model.
import torch
import torch.nn as nn

class NCF(nn.Module):
    """
    Neural Collaborative Filtering model.
    """
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[128, 64, 32]):
        super(NCF, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        mlp_layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Final prediction layer
        self.output_layer = nn.Linear(hidden_layers[-1], 1)

    def forward(self, user_indices, item_indices):
        user_embeds = self.user_embedding(user_indices)
        item_embeds = self.item_embedding(item_indices)
        
        # Concatenate embeddings
        x = torch.cat([user_embeds, item_embeds], dim=-1)

        x = self.mlp(x)
        out = self.output_layer(x)
        return out.squeeze()  # predicted rating as 1D tensor
