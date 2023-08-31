import torch
import torch.nn as nn

from typing import *


class MLP(nn.Module):
    def __init__(self, num_users: int, num_items: int, layers: List[int], latent_dim: int=64):
        super(MLP, self).__init__()

        self.embedding_user = nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim)

        self.fc_layers = nn.ModuleList()

        for idx, _ in enumerate(layers[:-1]):
            self.fc_layers.append(nn.Linear(layers[idx], layers[idx+1]))

        self.fc_out = nn.Linear(in_features=layers[-1], out_features=1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        for layer in self.fc_layers:
            x = layer(x)

        logits = self.fc_out(x)
        rating = self.logistic(logits)
        return rating