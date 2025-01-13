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
                 hidden=[100, 50], dropouts=[0.3, 0.2]):
        super().__init__()
        hidden = get_list(hidden)
        dropouts = get_list(dropouts)

        def gen_layers(n_in):
            """
            Generator that yields a sequence of hidden layers and their activations/dropouts.
            """
            nonlocal hidden, dropouts
            for n_out, rate in zip_longest(hidden, dropouts + [None] * (len(hidden) - len(dropouts))):
                yield nn.Linear(n_in, n_out)
                yield nn.BatchNorm1d(n_out)  
                yield nn.ReLU()
                if rate is not None and rate > 0.:
                    yield nn.Dropout(rate)
                n_in = n_out

        # User and product embeddings
        self.u = nn.Embedding(n_users, n_factors)
        self.p1 = nn.Embedding(n_products, n_factors)
        self.p2 = nn.Embedding(n_products, n_factors)

        # Dropout layer for embedding space
        self.drop = nn.Dropout(embedding_dropout)
        
        # Sequential layers for hidden layers
        self.hidden_layers = nn.Sequential(*list(gen_layers(3 * n_factors)))
        
        # Output layer (size of products)
        self.output = nn.Linear(hidden[-1], n_products)

    def forward(self, u, p1, p2):
        """
        Forward pass to predict the next product for a given user and product pair.

        Args:
            u: User indices.
            p1: First product indices.
            p2: Second product indices.
        """
        u = self.u(u)
        p1 = self.p1(p1)
        p2 = self.p2(p2)
        x = torch.cat([u, p1, p2], 1)  
        x = self.drop(x) 
        x = self.hidden_layers(x)  
        return self.output(x)  

