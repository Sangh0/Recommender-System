import torch
import torch.nn as nn

from typing import *

from mlp import MLP
from gmf import GMF


class NeuMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        latent_dim_mf: int,
        latent_dim_mlp: int,
        layers: List[int],
    ):
        super(NeuMF, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim_mf = latent_dim_mf
        self.latent_dim_mlp = latent_dim_mlp
        self.layers = layers

        self.embedding_user_mlp = nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dim_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim_mlp)
        
        self.embedding_user_mf = nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dim_mf)
        self.embedding_item_mf = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim_mf)

        self.fc_layers = nn.ModuleList()
        for idx, _ in enumerate(layers[:-1]):
            self.fc_layers.append(nn.Linear(layers[idx], layers[idx+1]))
            self.fc_layers.append(nn.ReLU())

        self.relu = nn.ReLU()

        self.fc_out = nn.Linear(layers[-1] + latent_dim_mf, 1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        user_embeddimg_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mf_vector = torch.cat([user_embeddimg_mf, item_embedding_mf], dim=-1)

        for layer in self.fc_layers:
            mlp_vector = layer(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.fc_out(vector)
        rating = self.logistic(logits)
        return rating
    
    def load_pretrained_weight(self, mlp_weight, gmf_weight):
        """ Load pre-trained weights for MLP model """
        mlp_model = MLP(
            num_users=self.num_users, 
            num_items=self.num_items, 
            layers=self.layers, 
            latent_dim=self.latent_dim_mlp,
        )
        mlp_model.load_state_dict(torch.load(mlp_weight))

        self.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
        self.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data
        
        for idx, layer in enumerate(self.layers):
            layer.weight.data = mlp_model.fc_layers[idx].weight.data

        """ Load pre-trained weights for GMF model """
        gmf_model = GMF(
            num_users=self.num_users, 
            num_items=self.num_items, 
            latent_dim=self.latent_dim_mf,
        )
        gmf_model.load_state_dict(torch.load(gmf_weight))

        self.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data

        self.fc_out.weight.data = 0.5 * torch.cat([mlp_model.fc_out.weight.data, gmf_model.fc.weight.data], dim=-1)
        self.fc_out.bias.data = 0.5 * (mlp_model.fc_out.bias.data + gmf_model.fc.bias.data)