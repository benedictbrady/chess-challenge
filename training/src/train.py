"""Training loop with checkpointing, logging, and cosine LR schedule."""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

from .checkpoint import CheckpointManager
from .dataset import ChessDataset, ChessOutcomeDataset, ChessBlendedDataset, ChessMovePairDataset
from .model import build_model


def sigmoid_mse_loss(
    pred_cp: torch.Tensor,
    target_cp: torch.Tensor,
    eval_scale: float = 400.0,
) -> torch.Tensor:
    """MSE loss in sigmoid space.

    Maps centipawn values through sigmoid(x / eval_scale) before computing MSE.
    This weights positions with close evaluations more heavily than blowouts.
    """
    pred_sig = torch.sigmoid(pred_cp / eval_scale)
    target_sig = torch.sigmoid(target_cp / eval_scale)
    return nn.functional.mse_loss(pred_sig, target_sig)


def direct_mse_loss(
    pred_cp: torch.Tensor,
    target_cp: torch.Tensor,
    clamp_cp: float = 2000.0,
    **_kwargs,
) -> torch.Tensor:
    """Direct MSE loss on clamped centipawn values.

    No sigmoid compression — equal weight to all eval differences.
    Values are clamped to [-clamp_cp, clamp_cp] to avoid extreme outliers
    dominating the loss.
    """
    target_clamped = torch.clamp(target_cp, -clamp_cp, clamp_cp)
    pred_clamped = torch.clamp(pred_cp, -clamp_cp, clamp_cp)
    return nn.functional.mse_loss(pred_clamped, target_clamped)


def blended_wdl_mse_loss(
    pred_cp: torch.Tensor,
    target_wdl: torch.Tensor,
    eval_scale: float = 400.0,
    **_kwargs,
) -> torch.Tensor:
    """MSE loss in WDL space — the Stockfish NNUE loss function.

    The target is a pre-computed blend of sigmoid(eval/scale) and game outcome.
    The model output is mapped through sigmoid to WDL space before computing MSE.
    """
    pred_wdl = torch.sigmoid(pred_cp / eval_scale)
    return nn.functional.mse_loss(pred_wdl, target_wdl)


def outcome_bce_loss(
    pred_cp: torch.Tensor,
    target_outcome: torch.Tensor,
    eval_scale: float = 400.0,
    **_kwargs,
) -> torch.Tensor:
    """BCE loss for game-outcome prediction.

    Maps NN output through sigmoid to get win probability, then BCE against
    actual game outcome (1.0 = win, 0.0 = loss, 0.5 = draw).
    """
    pred_prob = torch.sigmoid(pred_cp / eval_scale)
    return nn.functional.binary_cross_entropy(pred_prob, target_outcome)


def ranking_margin_loss(
    pred_good: torch.Tensor,
    pred_bad: torch.Tensor,
    margin_cp: torch.Tensor,
    eval_scale: float = 400.0,
) -> torch.Tensor:
    """Margin ranking loss: good position should score higher than bad.

    Operates in sigmoid space to match the main loss.
    """
    sig_good = torch.sigmoid(pred_good / eval_scale)
    sig_bad = torch.sigmoid(pred_bad / eval_scale)
    target_margin = torch.sigmoid(margin_cp / eval_scale) - 0.5
    return nn.functional.relu(target_margin - (sig_good - sig_bad)).mean()


LOSS_FUNCTIONS = {
    "sigmoid_mse": sigmoid_mse_loss,
    "direct_mse": direct_mse_loss,
    "outcome_bce": outcome_bce_loss,
    "blended_wdl_mse": blended_wdl_mse_loss,
}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    eval_scale: float,
    grad_clip: float,
    device: torch.device,
    loss_type: str = "sigmoid_mse",
    pair_loader: DataLoader | None = None,
    ranking_lambda: float = 0.0,
) -> float:
    """Train for one epoch. Returns average loss.

    If pair_loader and ranking_lambda > 0, adds ranking loss from move pairs.
    """
    loss_fn = LOSS_FUNCTIONS[loss_type]
    model.train()
    total_loss = 0.0
    num_batches = 0

    pair_iter = iter(pair_loader) if pair_loader is not None and ranking_lambda > 0 else None

    for positions, evals_cp in loader:
        positions = positions.to(device)
        evals_cp = evals_cp.to(device)

        pred = model(positions).squeeze(-1)  # [N]
        loss = loss_fn(pred, evals_cp, eval_scale=eval_scale)

        # Add ranking loss if available
        if pair_iter is not None:
            try:
                pg, pb, margins = next(pair_iter)
            except StopIteration:
                pair_iter = iter(pair_loader)
                pg, pb, margins = next(pair_iter)

            pg = pg.to(device)
            pb = pb.to(device)
            margins = margins.to(device)

            pred_good = model(pg).squeeze(-1)
            pred_bad = model(pb).squeeze(-1)
            rloss = ranking_margin_loss(pred_good, pred_bad, margins, eval_scale)
            loss = loss + ranking_lambda * rloss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    eval_scale: float,
    device: torch.device,
    loss_type: str = "sigmoid_mse",
) -> float:
    """Evaluate model on a dataset. Returns average loss."""
    loss_fn = LOSS_FUNCTIONS[loss_type]
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for positions, evals_cp in loader:
        positions = positions.to(device)
        evals_cp = evals_cp.to(device)

        pred = model(positions).squeeze(-1)
        loss = loss_fn(pred, evals_cp, eval_scale=eval_scale)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train(config: dict):
    """Run the full training loop from a config dict."""
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    data_path = config["data_path"]
    print(f"Loading data from {data_path}")
    loss_type = config.get("loss_type", "sigmoid_mse")
    if loss_type == "blended_wdl_mse":
        dataset = ChessBlendedDataset(
            data_path,
            eval_scale=config.get("eval_scale", 400.0),
            blend_lambda=config.get("blend_lambda", 0.75),
            augment_hflip=config.get("augment_hflip", False),
        )
    elif loss_type == "outcome_bce":
        dataset = ChessOutcomeDataset(
            data_path,
            augment_hflip=config.get("augment_hflip", False),
            decisive_only=config.get("decisive_only", True),
        )
    else:
        dataset = ChessDataset(
            data_path,
            eval_scale=config.get("eval_scale", 400.0),
            augment_hflip=config.get("augment_hflip", False),
            blend_alpha=config.get("blend_alpha", 1.0),
        )
    print(f"Dataset size: {len(dataset):,}")

    # Train/val split
    val_frac = config.get("val_fraction", 0.05)
    val_size = int(len(dataset) * val_frac)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(config.get("seed", 42))
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    print(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")

    batch_size = config.get("batch_size", 8192)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    # Model
    arch = config.get("architecture", "mlp_512_256_128")
    model = build_model(arch)
    assert model.num_params <= 10_000_000, f"Model too large: {model.num_params:,} params"
    model = model.to(device)

    # Optimizer
    lr = config.get("learning_rate", 1e-3)
    weight_decay = config.get("weight_decay", 1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler: cosine annealing with warmup
    epochs = config.get("epochs", 30)
    warmup_epochs = config.get("warmup_epochs", 2)
    min_lr = config.get("min_lr", 1e-6)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return max(min_lr / lr, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Checkpoint manager
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints/default")
    ckpt_mgr = CheckpointManager(checkpoint_dir, keep=config.get("keep_checkpoints", 3))

    # Resume or initialize from pretrained weights
    start_epoch = 0
    if config.get("resume", False):
        start_epoch = ckpt_mgr.resume(model, optimizer, scheduler)
    elif config.get("resume_from"):
        # Load model weights only (for fine-tuning). Fresh optimizer/scheduler.
        resume_path = config["resume_from"]
        print(f"Initializing weights from {resume_path}")
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model = model.to(device)
        print(f"  Loaded weights (epoch {state.get('epoch', '?')}, val_loss={state.get('val_loss', '?')})")

    # Training config
    eval_scale = config.get("eval_scale", 400.0)
    grad_clip = config.get("grad_clip", 1.0)
    loss_type = config.get("loss_type", "sigmoid_mse")
    if loss_type not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss_type: {loss_type}. Available: {list(LOSS_FUNCTIONS.keys())}")

    # Optional ranking loss from move pairs
    ranking_lambda = config.get("ranking_lambda", 0.0)
    pair_loader = None
    if ranking_lambda > 0 and config.get("pair_data_path"):
        pair_dataset = ChessMovePairDataset(
            config["pair_data_path"],
            augment_hflip=config.get("augment_hflip", False),
        )
        pair_loader = DataLoader(
            pair_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )
        print(f"Ranking loss: λ={ranking_lambda}, {len(pair_dataset):,} pairs")

    print(f"\nTraining config:")
    print(f"  Architecture: {arch}")
    print(f"  Parameters: {model.num_params:,}")
    print(f"  Epochs: {start_epoch} → {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR: {lr}, min_lr: {min_lr}")
    print(f"  Eval scale: {eval_scale}")
    print(f"  Loss type: {loss_type}")
    print(f"  Grad clip: {grad_clip}")
    print(f"  Ranking lambda: {ranking_lambda}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print()

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, eval_scale, grad_clip, device, loss_type,
            pair_loader=pair_loader, ranking_lambda=ranking_lambda,
        )
        val_loss = evaluate(model, val_loader, eval_scale, device, loss_type)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"lr={current_lr:.2e}  "
            f"{'*BEST' if is_best else '     '}  "
            f"({epoch_time:.1f}s)"
        )

        ckpt_mgr.save(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=train_loss,
            val_loss=val_loss,
            config=config,
        )

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best val loss: {best_val_loss:.6f}")

    return model, ckpt_mgr


def main():
    parser = argparse.ArgumentParser(description="Train chess evaluation network")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.resume:
        config["resume"] = True

    train(config)


if __name__ == "__main__":
    main()
