import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, emb_dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(variance + self.epsilon)
        return self.scale * x_norm + self.shift
