import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feedforward import FeedForward
from model.layernorm import LayerNorm


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=config.d_model,
            d_out=config.d_model,
            context_length=config.context_length,
            dropout=config.dropout,
            num_heads=config.n_heads,
            qkv_bias=config.qkv_bias,
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)
        self.drop_shortcut = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.drop_shortcut(self.attn(self.norm1(x)))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x
