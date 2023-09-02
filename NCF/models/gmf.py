import torch
import torch.nn as nn


class GMF(nn.Module):

    def __init__(
        self, 
        num_users: int, 
        num_items: int, 
        latent_dim: int=16,
    ):
        super(GMF, self).__init__()

        self.embedding_user = nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim)

        self.fc = nn.Linear(in_features=latent_dim, out_features=1)
        self.logistic = nn.Sigmoid()

        self._init_weights_()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.fc(element_product)
        rating = self.logistic(logits)
        return rating
    
    def _init_weights_(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal(module.weight, mean=0.0, std=0.01)