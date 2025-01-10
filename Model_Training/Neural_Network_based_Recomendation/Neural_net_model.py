import torch
import torch.nn as nn
from itertools import zip_longest

def get_list(n):
    if isinstance(n, (int, float)):
        return [n]
    elif hasattr(n, '__iter__'):
        return list(n)
    raise TypeError('Layers configuration should be a single number or a list of numbers')

class Neural_Net(nn.Module):
    """
    A deep learning model for collaborative filtering to recommend the next product in a sequence.

    Args:
        n_users: Number of unique users in the dataset.
        n_products: Number of unique products in the dataset.
        n_factors: Number of columns in the embeddings matrix (default=50).
        embedding_dropout: Dropout rate to apply right after embeddings layer.
        hidden: Number of units in hidden layers (either a single int or a list).
        dropouts: Dropout rates applied after each hidden layer (single int or list).
    """
    def __init__(self, n_users, n_products, n_factors=50, embedding_dropout=0.02,
                 hidden=10, dropouts=0.2):
        super().__init__()
        hidden = get_list(hidden)
        dropouts = get_list(dropouts)

        def gen_layers(n_in):
            """
            Generator that yields a sequence of hidden layers and their activations/dropouts.
            """
            nonlocal hidden, dropouts
            assert len(dropouts) <= len(hidden)
            for n_out, rate in zip_longest(hidden, dropouts):
                yield nn.Linear(n_in, n_out)
                yield nn.ReLU()
                if rate is not None and rate > 0.:
                    yield nn.Dropout(rate)
                n_in = n_out

        # User and product embeddings
        self.u = nn.Embedding(n_users, n_factors)
        self.p = nn.Embedding(n_products, n_factors)

        # Dropout after embeddings
        self.drop = nn.Dropout(embedding_dropout)

        # Hidden layers
        self.hidden = nn.Sequential(*list(gen_layers(n_factors * 3)))  # Updated for two products

        # Output layer: Predict next product (as a multi-class classification task)
        self.fc = nn.Linear(hidden[-1], n_products)

        # Initialize weights
        self._init()

    def forward(self, users, product_1, product_2):
        """
        Forward pass through the network.

        Args:
            users: Tensor of user indices.
            product_1: Tensor of first product indices.
            product_2: Tensor of second product indices.

        Returns:
            Predicted scores for the next product (multi-class probabilities).
        """
        user_embeds = self.u(users)
        product_1_embeds = self.p(product_1)
        product_2_embeds = self.p(product_2)

        combined_embeds = torch.cat([user_embeds, product_1_embeds, product_2_embeds], dim=1)
        x = self.drop(combined_embeds)
        x = self.hidden(x)
        out = self.fc(x)  # Raw scores for each product class
        return out

    def _init(self):
        """
        Initialize weights of the model with Xavier uniform distribution for layers.
        """
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.u.weight.data.uniform_(-0.05, 0.05)
        self.p.weight.data.uniform_(-0.05, 0.05)
        self.hidden.apply(init)
        init(self.fc)
