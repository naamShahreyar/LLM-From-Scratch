import os
import torch
import wandb
from datasets import load_dataset

from config.gpt_config import GPTConfig
from tokenizer.bpe import BPETokenizer
from model.llm import LLM
from data.dataset import create_dataloader_from_ids
from training.trainer import train_model, load_checkpoint

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")

MAX_LR = 3e-4
MIN_LR = MAX_LR * 0.1
NUM_EPOCHS = 20
WARMUP_STEPS = 200
BATCH_SIZE = 128
CONTEXT_LENGTH = GPTConfig.context_length
STRIDE = CONTEXT_LENGTH
EVAL_FREQ = 200
EVAL_ITER = 20
CHECKPOINT_FREQ = 1
START_CONTEXT = "Once upon a time"

# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────

tokenizer = BPETokenizer()
tokenizer.load("data/vocab.json", "data/merges.json")
print(f"Tokenizer loaded — vocab size: {len(tokenizer.vocab)}")

# ── Dataset ───────────────────────────────────────────────────────────────────

TRAIN_CACHE = "data/train_tokens.pt"
VAL_CACHE = "data/val_tokens.pt"

if os.path.exists(TRAIN_CACHE) and os.path.exists(VAL_CACHE):
    print("Loading cached token IDs...")
    train_ids = torch.load(TRAIN_CACHE)
    val_ids = torch.load(VAL_CACHE)
else:
    print("Tokenizing TinyStories (one-time, will be cached)...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    train_text = "\n".join(ds["text"])
    train_ids = tokenizer.encode(train_text)
    torch.save(train_ids, TRAIN_CACHE)
    print(f"Train tokens: {len(train_ids):,} — saved to {TRAIN_CACHE}")

    ds_val = load_dataset("roneneldan/TinyStories", split="validation")
    val_text = "\n".join(ds_val["text"])
    val_ids = tokenizer.encode(val_text)
    torch.save(val_ids, VAL_CACHE)
    print(f"Val tokens: {len(val_ids):,} — saved to {VAL_CACHE}")

print(f"Train tokens: {len(train_ids):,} | Val tokens: {len(val_ids):,}")

train_loader = create_dataloader_from_ids(
    train_ids,
    max_length=CONTEXT_LENGTH, stride=STRIDE,
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0,
)
val_loader = create_dataloader_from_ids(
    val_ids,
    max_length=CONTEXT_LENGTH, stride=STRIDE,
    batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0,
)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ── Model & Optimizer ─────────────────────────────────────────────────────────

config = GPTConfig()
model = LLM(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

max_steps = NUM_EPOCHS * len(train_loader)

# ── Wandb + Resume ────────────────────────────────────────────────────────────

if os.path.exists(CHECKPOINT_PATH):
    start_epoch, global_step, train_losses, val_losses, wandb_run_id = load_checkpoint(
        CHECKPOINT_PATH, model, optimizer, device
    )
    start_epoch += 1
    wandb.init(
        project="gpt-from-scratch",
        id=wandb_run_id,
        resume="allow",
    )
else:
    torch.manual_seed(123)
    start_epoch, global_step = 0, 0
    train_losses, val_losses = [], []
    run = wandb.init(
        project="gpt-from-scratch",
        config={
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "max_lr": MAX_LR,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
        }
    )
    wandb_run_id = run.id
    print("Starting fresh training")

# ── Train ─────────────────────────────────────────────────────────────────────

train_losses, val_losses, tokens_seen = train_model(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=NUM_EPOCHS,
    max_lr=MAX_LR, min_lr=MIN_LR,
    warmup_steps=WARMUP_STEPS, max_steps=max_steps,
    eval_freq=EVAL_FREQ, eval_iter=EVAL_ITER,
    start_context=START_CONTEXT, tokenizer=tokenizer,
    checkpoint_dir=CHECKPOINT_DIR, checkpoint_freq=CHECKPOINT_FREQ,
    start_epoch=start_epoch, initial_step=global_step,
    wandb_run_id=wandb_run_id,
)

wandb.finish()
print("Training complete.")
