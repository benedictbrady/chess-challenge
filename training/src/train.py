"""Training loop with cosine LR, checkpointing, and multiple loss functions."""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .dataset import ChessEvalDataset, ChessOutcomeDataset, ChessBlendedDataset, ChessMultiOutcomeDataset, ChessMultiDataset
from .model import build_model


# ── Loss functions ────────────────────────────────────────────────────────────


def sigmoid_mse_loss(pred_cp, target_cp, eval_scale=400.0, **_):
    """MSE in sigmoid space. Weights close positions more than blowouts."""
    return nn.functional.mse_loss(
        torch.sigmoid(pred_cp / eval_scale),
        torch.sigmoid(target_cp / eval_scale),
    )


def direct_mse_loss(pred_cp, target_cp, clamp_cp=2000.0, **_):
    """Direct MSE on clamped centipawn values."""
    return nn.functional.mse_loss(
        torch.clamp(pred_cp, -clamp_cp, clamp_cp),
        torch.clamp(target_cp, -clamp_cp, clamp_cp),
    )


def outcome_bce_loss(pred_cp, target_outcome, eval_scale=400.0, **_):
    """BCE for game-outcome prediction. Maps NN output through sigmoid."""
    pred_prob = torch.sigmoid(pred_cp / eval_scale)
    return nn.functional.binary_cross_entropy(pred_prob, target_outcome)


def blended_wdl_mse_loss(pred_cp, target_wdl, eval_scale=400.0, **_):
    """MSE in WDL space — the Stockfish NNUE loss."""
    pred_wdl = torch.sigmoid(pred_cp / eval_scale)
    return nn.functional.mse_loss(pred_wdl, target_wdl)


LOSS_FUNCTIONS = {
    "sigmoid_mse": sigmoid_mse_loss,
    "direct_mse": direct_mse_loss,
    "outcome_bce": outcome_bce_loss,
    "blended_wdl_mse": blended_wdl_mse_loss,
}


# ── Training loop ─────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, loss_fn, eval_scale, grad_clip, device):
    model.train()
    total_loss = 0.0
    n = 0
    for positions, labels in loader:
        positions, labels = positions.to(device), labels.to(device)
        pred = model(positions).squeeze(-1)
        loss = loss_fn(pred, labels, eval_scale=eval_scale)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, eval_scale, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for positions, labels in loader:
        positions, labels = positions.to(device), labels.to(device)
        pred = model(positions).squeeze(-1)
        loss = loss_fn(pred, labels, eval_scale=eval_scale)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def train(config: dict):
    """Full training loop from config dict. Returns (model, best_val_loss)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset — supports comma-separated paths for multi-file loading
    data_path = config["data_path"]
    loss_type = config.get("loss_type", "sigmoid_mse")

    # Check if multiple paths (comma-separated)
    data_paths = [p.strip() for p in data_path.split(",")]
    is_multi = len(data_paths) > 1

    if loss_type == "blended_wdl_mse":
        assert not is_multi, "Multi-file loading not yet supported for blended_wdl_mse"
        dataset = ChessBlendedDataset(
            data_path,
            eval_scale=config.get("eval_scale", 400.0),
            blend_lambda=config.get("blend_lambda", 0.75),
        )
    elif loss_type == "outcome_bce":
        if is_multi:
            dataset = ChessMultiOutcomeDataset(
                data_paths,
                decisive_only=config.get("decisive_only", False),
            )
        else:
            dataset = ChessOutcomeDataset(
                data_path,
                decisive_only=config.get("decisive_only", False),
            )
    else:
        if is_multi:
            dataset = ChessMultiDataset(data_paths, label_key="evals",
                                        max_abs_eval=config.get("max_abs_eval", 10000.0))
        else:
            dataset = ChessEvalDataset(
                data_path,
                max_abs_eval=config.get("max_abs_eval", 10000.0),
            )

    print(f"Dataset: {len(dataset):,} samples")

    # Train/val split
    val_frac = config.get("val_fraction", 0.05)
    val_size = int(len(dataset) * val_frac)
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(config.get("seed", 42))
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)
    print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    batch_size = config.get("batch_size", 16384)
    workers = config.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)

    # Model
    arch = config.get("architecture", "nnue_256x2_32_32")
    model = build_model(arch).to(device)
    assert model.num_params <= 10_000_000, f"Too large: {model.num_params:,}"

    # Optimizer
    lr = config.get("learning_rate", 1e-3)
    wd = config.get("weight_decay", 1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Cosine schedule with warmup
    epochs = config.get("epochs", 30)
    warmup = config.get("warmup_epochs", 2)
    min_lr = config.get("min_lr", 1e-6)

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return max(min_lr / lr, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss
    loss_fn = LOSS_FUNCTIONS[loss_type]
    eval_scale = config.get("eval_scale", 400.0)
    grad_clip = config.get("grad_clip", 1.0)

    # Resume
    start_epoch = 0
    ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints/default"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if config.get("resume_from"):
        print(f"Resuming from {config['resume_from']}")
        state = torch.load(config["resume_from"], map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model = model.to(device)

    # Print config
    print(f"\nConfig: arch={arch}, params={model.num_params:,}, epochs={epochs}")
    print(f"  lr={lr}, wd={wd}, batch={batch_size}, loss={loss_type}, scale={eval_scale}")
    print()

    best_val_loss = float("inf")
    best_path = ckpt_dir / "best.pt"
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, eval_scale, grad_clip, device)
        val_loss = evaluate(model, val_loader, loss_fn, eval_scale, device)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }, best_path)

        marker = "*BEST" if is_best else ""
        print(f"Epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={cur_lr:.2e}  {marker}  ({dt:.1f}s)")

    # Save final
    final_path = ckpt_dir / "final.pt"
    torch.save({
        "epoch": epochs - 1,
        "model_state_dict": model.state_dict(),
        "val_loss": best_val_loss,
        "config": config,
    }, final_path)

    total = time.time() - start_time
    print(f"\nDone in {total:.1f}s. Best val loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {best_path}")
    return model, best_val_loss
