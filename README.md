# LLM From Scratch

A GPT-style language model built from scratch, following Sebastian Raschka's *Build a Large Language Model (From Scratch)*.

## Architecture

| Param | Value |
|---|---|
| Parameters | ~70M |
| d_model | 768 |
| n_heads | 8 |
| n_layers | 8 |
| context_length | 512 |
| vocab_size | 8,192 |
| dropout | 0.1 |

- Pre-norm transformer blocks (LayerNorm before attention/FFN)
- GELU activation (tanh approximation)
- Causal self-attention with learned positional embeddings
- Custom BPE tokenizer trained from scratch

## Project Structure

```
config/         GPTConfig dataclass
tokenizer/      Custom BPE tokenizer
model/          Transformer architecture (attention, FFN, blocks, LLM)
data/           Dataset and dataloader
training/       Trainer, LR schedule, checkpointing, plotting
Notebooks/      Experiment notebooks (tokenizer → dataset → attention → LLM → pretraining → eval)
train.py        Training script
```

## Training

**Optimizer:** AdamW (weight_decay=0.1)  
**LR schedule:** Cosine decay with linear warmup  
**Gradient clipping:** max_norm=1.0  
**Dataset:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

### Run 1 — Baseline
| | Value |
|---|---|
| Dataset | 40k stories (~7.9M tokens) |
| Epochs | 7 |
| Batch size | 16 |
| Max LR | 3e-4 |
| Hardware | T4 GPU (Lightning AI) |
| Train loss | 2.03 |
| Val loss | 2.38 |

### Run 2 — Scaling up
| | Value |
|---|---|
| Dataset | 100k stories (~20M tokens) |
| Epochs | 20 |
| Batch size | 16 |
| Max LR | 3e-4 |
| Hardware | A100 GPU (Lightning AI) |
| Status | In progress |

## Sample Output (Run 1, Epoch 7)

> *Once upon a time there was a little girl called Jane. Jane was a very cheerful girl and loved playing in her garden. One day, Jane was walking in the garden when she saw a white bird. It was yellow and pink and she was very happy. She picked it up and opened it.*

## Setup

```bash
git clone https://github.com/naamShahreyar/LLM-From-Scratch.git
cd LLM-From-Scratch
pip install -r requirements.txt
wandb login
python train.py
```

## Generation

```python
from model.llm import LLM
from model.generate import text_to_token_idx, token_idx_to_text
from training.trainer import load_checkpoint
from config.gpt_config import GPTConfig
from tokenizer.bpe import BPETokenizer
import torch

tokenizer = BPETokenizer()
tokenizer.load("data/vocab.json", "data/merges.json")

model = LLM(GPTConfig())
optimizer = torch.optim.AdamW(model.parameters())
load_checkpoint("checkpoints/latest.pt", model, optimizer, torch.device("cpu"))
model.eval()

def generate(model, prompt, max_new_tokens=100, temperature=0.8, top_k=40):
    idx = text_to_token_idx(prompt, tokenizer)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -512:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :] / temperature
        if top_k:
            top_vals, _ = torch.topk(logits, top_k)
            logits[logits < top_vals[:, -1:]] = float("-inf")
        idx_next = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        idx = torch.cat((idx, idx_next), dim=1)
    return token_idx_to_text(idx, tokenizer)

print(generate(model, "Once upon a time"))
```
