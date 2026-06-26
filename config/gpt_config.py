from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    context_length: int = 256
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    dropout: float = 0.1
    qkv_bias: bool = False
