import os
import math
import torch
import wandb

from model.generate import generate_text_simple, text_to_token_idx, token_idx_to_text


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(input_batch)
        return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    num_batches = min(num_batches, len(data_loader)) if num_batches else len(data_loader)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_idx(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size,
        )
    print(token_idx_to_text(token_ids, tokenizer).replace("\n", " "))
    model.train()


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def save_checkpoint(model, optimizer, epoch, global_step, train_losses, val_losses, path, wandb_run_id=None, mid_epoch=False):
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "wandb_run_id": wandb_run_id,
        "mid_epoch": mid_epoch,
    }, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    mid_epoch = checkpoint.get("mid_epoch", False)
    epoch = checkpoint["epoch"]
    print(f"Resumed from {path} (epoch {epoch+1}, step {checkpoint['global_step']}, mid_epoch={mid_epoch})")
    return (
        epoch,
        checkpoint["global_step"],
        checkpoint["train_losses"],
        checkpoint["val_losses"],
        checkpoint.get("wandb_run_id"),
        mid_epoch,
    )


def train_model(model, train_loader, val_loader, optimizer, device,
                num_epochs, warmup_steps, max_steps, max_lr, min_lr,
                eval_freq, eval_iter, start_context, tokenizer,
                checkpoint_dir="checkpoints", checkpoint_freq=1,
                start_epoch=0, initial_step=0, wandb_run_id=None,
                save_every=1000):

    os.makedirs(checkpoint_dir, exist_ok=True)
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, initial_step

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            global_step += 1

            lr = get_lr(global_step, warmup_steps, max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tokens_seen += input_batch.numel()

            if global_step % save_every == 0:
                save_checkpoint(model, optimizer, epoch, global_step,
                                train_losses, val_losses,
                                os.path.join(checkpoint_dir, "latest.pt"),
                                wandb_run_id=wandb_run_id, mid_epoch=True)

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f} | "
                      f"lr={lr:.2e}")
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                    "tokens_seen": tokens_seen,
                }, step=global_step)

        generate_and_print_sample(model, tokenizer, device, start_context)

        save_checkpoint(model, optimizer, epoch, global_step,
                        train_losses, val_losses,
                        os.path.join(checkpoint_dir, "latest.pt"),
                        wandb_run_id=wandb_run_id, mid_epoch=False)

        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(model, optimizer, epoch, global_step,
                            train_losses, val_losses,
                            os.path.join(checkpoint_dir, f"ckpt_epoch{epoch+1}.pt"),
                            wandb_run_id=wandb_run_id, mid_epoch=False)

    return train_losses, val_losses, track_tokens_seen
