import torch
import torch.nn as nn

from model.block import TransformerBlock
from model.layernorm import LayerNorm


class LLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)
        self.drop_emb = nn.Dropout(config.dropout)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok_emb + pos_emb)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.output_head(x)
