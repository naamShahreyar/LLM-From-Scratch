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
**Mixed precision:** BF16 (A100)  
**Dataset:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

### Run 1 — Baseline (T4)
| | Value |
|---|---|
| Dataset | 40k stories (~7.9M tokens) |
| Epochs | 7 |
| Batch size | 16 |
| Max LR | 3e-4 |
| Hardware | T4 GPU (Lightning AI) |
| Train loss | 2.03 |
| Val loss | 2.38 |

### Run 2 — Full dataset (A100)
| | Value |
|---|---|
| Dataset | 2.1M stories (~417M tokens) |
| Epochs | 3 |
| Batch size | 128 |
| Max LR | 3e-4 |
| Hardware | A100 80GB (Lightning AI) |
| Train loss | ~1.51 |
| Val loss | ~1.56 |

## Sample Output (Run 2, Epoch 3)

> *Once upon a time there was a brave little boy named John. He wanted to play in the garden, so he got out his little yellow rake and started to rake the leaves.*
>
> *John's mom saw him struggling and came to help him. She told him to not be so careless. "John, always try your best and don't be careless."*
>
> *John smiled and said "I will!" He kept raking the leaves and he was very happy. After a while, John felt tired. He knew it was time to go inside and put his rake away.*
>
> *John's mom was so proud of him and hugged him tightly. She was very happy that he had listened to her and followed her instructions. The end.*

## Fine-Tuning (LoRA)

Fine-tuned on [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) (12k dialogue-summary pairs) using LoRA via the HuggingFace PEFT library.

**LoRA config:** r=16, lora_alpha=64, target modules: W_query, W_key, W_value, out_proj  
**Training:** 10 epochs, batch size 16, lr=3e-4, BF16 (A100)  
**Trainable params:** ~786k / 70M (1.12%)

### ROUGE Results (200 test samples)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Base model (no fine-tuning) | 0.098 | 0.007 | 0.072 |
| Base model + LoRA | **0.314** | **0.064** | **0.245** |
| Flan-T5 zero-shot (reference) | 0.234 | 0.072 | 0.202 |
| Flan-T5 + LoRA (reference) | 0.422 | 0.174 | 0.337 |

LoRA fine-tuning improved ROUGE-1 by **+0.22** absolute. The fine-tuned model outperforms Flan-T5 zero-shot on ROUGE-1 and ROUGE-L despite being pretrained only on TinyStories.

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

def generate(model, prompt, max_new_tokens=200, temperature=0.8, top_k=40):
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
