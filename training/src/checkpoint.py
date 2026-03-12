"""Checkpoint save/load/resume manager for resilient training."""

import os
import re
from pathlib import Path

import torch


class CheckpointManager:
    """Manages training checkpoints with auto-resume support.

    Saves full training state every epoch. Keeps last `keep` checkpoints
    plus a `best.pt` based on validation loss.
    """

    def __init__(self, checkpoint_dir: str, keep: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.best_val_loss = float("inf")

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loss: float,
        val_loss: float | None = None,
        config: dict | None = None,
        extra: dict | None = None,
    ):
        """Save a checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": config,
            "extra": extra or {},
        }

        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(state, path)

        # Update best if val_loss improved
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)

        # Clean up old checkpoints (keep last N + best)
        self._cleanup()

    def _cleanup(self):
        """Remove old checkpoints, keeping only the last `keep`."""
        pattern = re.compile(r"checkpoint_epoch_(\d+)\.pt")
        checkpoints = []
        for f in self.checkpoint_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                checkpoints.append((int(m.group(1)), f))

        checkpoints.sort(key=lambda x: x[0])

        # Remove all but the last `keep`
        to_remove = checkpoints[: -self.keep] if len(checkpoints) > self.keep else []
        for _, path in to_remove:
            path.unlink(missing_ok=True)

    def latest_checkpoint(self) -> Path | None:
        """Find the latest epoch checkpoint."""
        pattern = re.compile(r"checkpoint_epoch_(\d+)\.pt")
        checkpoints = []
        for f in self.checkpoint_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                checkpoints.append((int(m.group(1)), f))

        if not checkpoints:
            return None

        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1][1]

    def best_checkpoint(self) -> Path | None:
        """Find the best checkpoint."""
        best = self.checkpoint_dir / "best.pt"
        return best if best.exists() else None

    def load(self, path: Path) -> dict:
        """Load a checkpoint."""
        return torch.load(path, map_location="cpu", weights_only=False)

    def resume(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
    ) -> int:
        """Resume from latest checkpoint. Returns the next epoch to train.

        Returns 0 if no checkpoint found.
        """
        latest = self.latest_checkpoint()
        if latest is None:
            print("No checkpoint found, starting from scratch.")
            return 0

        print(f"Resuming from {latest}")
        state = self.load(latest)

        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler and state.get("scheduler_state_dict"):
            scheduler.load_state_dict(state["scheduler_state_dict"])

        if state.get("val_loss") is not None:
            self.best_val_loss = state["val_loss"]

        next_epoch = state["epoch"] + 1
        print(f"  Resumed at epoch {state['epoch']}, train_loss={state['train_loss']:.6f}")
        return next_epoch
