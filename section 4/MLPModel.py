import torch
from torch import nn


class MLPModel(nn.Module):
    max_len = 1000

    def __init__(self, embedding_matrix, hidden_dims=[512, 256, 128], output_dim=1):
        super(MLPModel, self).__init__()

        # Embedding Layer with frozen weights
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=True,
            padding_idx=0
        )

        # Calculate input dimension
        input_dim = embedding_matrix.shape[1] * MLPModel.max_len

        # Create list to hold all layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # Combine all layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Get embeddings and flatten
        embedded = self.embedding(x)
        flattened = embedded.view(embedded.size(0), -1)

        # Forward pass through all layers
        return self.model(flattened)
