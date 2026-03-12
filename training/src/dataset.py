"""PyTorch Datasets for chess training data."""

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


def _build_hflip_indices(num_channels: int) -> torch.Tensor:
    """Build horizontal flip indices for a tensor with num_channels × 64 layout.

    Works for both 768 (12 channels) and 1536 (24 channels) encodings.
    """
    flip_indices = []
    for channel in range(num_channels):
        for rank in range(8):
            for file in range(8):
                flip_indices.append(channel * 64 + rank * 8 + (7 - file))
    return torch.tensor(flip_indices, dtype=torch.long)


class ChessDataset(Dataset):
    """Dataset of chess positions with evaluations.

    Supports single-label (evals) and dual-label (baseline_evals + sf_evals)
    datasets. For dual-label, blends at configurable ratio.
    """

    def __init__(
        self,
        data_path: str,
        eval_scale: float = 400.0,
        augment_hflip: bool = False,
        blend_alpha: float = 1.0,
    ):
        """Load dataset from .npz file.

        Args:
            data_path: Path to data.npz
            eval_scale: Scale factor for sigmoid transformation.
            augment_hflip: If True, random horizontal flip (chess is symmetric).
            blend_alpha: For dual-label data, target = alpha*baseline + (1-alpha)*sf.
                         For single-label data, this is ignored.
        """
        data = np.load(data_path)
        self.positions = torch.from_numpy(data["positions"])  # [N, 768] or [N, 1536]

        # Dual-label or single-label?
        if "baseline_evals" in data and "sf_evals" in data:
            bl = torch.from_numpy(data["baseline_evals"].astype(np.float32))
            sf = torch.from_numpy(data["sf_evals"].astype(np.float32))
            self.evals_cp = blend_alpha * bl + (1.0 - blend_alpha) * sf
            print(f"  Dual-label dataset: blend_alpha={blend_alpha}")
        else:
            self.evals_cp = torch.from_numpy(data["evals"].astype(np.float32))

        self.eval_scale = eval_scale
        self.augment_hflip = augment_hflip

        if augment_hflip:
            num_channels = self.positions.shape[1] // 64  # 12 or 24
            self._flip_indices = _build_hflip_indices(num_channels)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.positions[idx]
        ev = self.evals_cp[idx]
        if self.augment_hflip and torch.rand(1).item() < 0.5:
            pos = pos[self._flip_indices]
        return pos, ev


class ChessOutcomeDataset(Dataset):
    """Dataset of positions labeled with game outcomes (win probability).

    Each sample is (position_tensor, win_probability) where win_probability
    is from the side-to-move's perspective: 1.0 = win, 0.0 = loss, 0.5 = draw.
    """

    def __init__(self, data_path: str, augment_hflip: bool = False, decisive_only: bool = False):
        data = np.load(data_path)
        positions = data["positions"]
        outcomes = data["outcomes"].astype(np.float32)

        if decisive_only:
            # Filter to wins/losses only (remove draws at 0.5)
            mask = (outcomes > 0.7) | (outcomes < 0.3)
            positions = positions[mask]
            outcomes = outcomes[mask]
            print(f"  Decisive only: {mask.sum():,} / {len(mask):,} positions ({100*mask.mean():.1f}%)")

        self.positions = torch.from_numpy(positions)  # [N, 768] or [N, 1536]
        self.outcomes = torch.from_numpy(outcomes)  # [N]

        self.augment_hflip = augment_hflip
        if augment_hflip:
            num_channels = self.positions.shape[1] // 64
            self._flip_indices = _build_hflip_indices(num_channels)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.positions[idx]
        out = self.outcomes[idx]
        if self.augment_hflip and torch.rand(1).item() < 0.5:
            pos = pos[self._flip_indices]
        return pos, out


class ChessBlendedDataset(Dataset):
    """Dataset with both eval labels and game outcomes for blended training.

    Each sample is (position_tensor, blended_target) where:
        blended_target = λ * sigmoid(eval/scale) + (1-λ) * game_outcome

    This is how Stockfish NNUE trains — the eval anchors the model while
    game outcomes push it beyond the teacher.
    """

    def __init__(
        self,
        data_path: str,
        eval_scale: float = 400.0,
        blend_lambda: float = 0.75,
        augment_hflip: bool = False,
    ):
        data = np.load(data_path)
        self.positions = torch.from_numpy(data["positions"])  # [N, 768] or [N, 1536]
        evals = torch.from_numpy(data["evals"].astype(np.float32))  # [N]
        outcomes = torch.from_numpy(data["outcomes"].astype(np.float32))  # [N]

        # Pre-compute blended targets
        eval_wdl = torch.sigmoid(evals / eval_scale)
        self.targets = blend_lambda * eval_wdl + (1.0 - blend_lambda) * outcomes

        print(f"  Blended dataset: λ={blend_lambda}, scale={eval_scale}")
        print(f"  Eval WDL: mean={eval_wdl.mean():.3f}, std={eval_wdl.std():.3f}")
        print(f"  Outcomes: mean={outcomes.mean():.3f}")
        print(f"  Blended targets: mean={self.targets.mean():.3f}, std={self.targets.std():.3f}")

        self.augment_hflip = augment_hflip
        if augment_hflip:
            num_channels = self.positions.shape[1] // 64
            self._flip_indices = _build_hflip_indices(num_channels)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.positions[idx]
        target = self.targets[idx]
        if self.augment_hflip and torch.rand(1).item() < 0.5:
            pos = pos[self._flip_indices]
        return pos, target


class ChessMovePairDataset(Dataset):
    """Dataset of (good_position, bad_position, margin) triples for ranking loss.

    Each sample is a pair of positions resulting from legal moves in the same
    parent position. The "good" position has a higher baseline eval than "bad".
    The margin is the eval difference in centipawns.
    """

    def __init__(self, data_path: str, augment_hflip: bool = False):
        data = np.load(data_path)
        self.pos_good = torch.from_numpy(data["pos_good"])  # [N, 768] or [N, 1536]
        self.pos_bad = torch.from_numpy(data["pos_bad"])  # [N, 768] or [N, 1536]
        self.margins = torch.from_numpy(data["margins"].astype(np.float32))  # [N]

        self.augment_hflip = augment_hflip
        if augment_hflip:
            num_channels = self.pos_good.shape[1] // 64
            self._flip_indices = _build_hflip_indices(num_channels)

    def __len__(self) -> int:
        return len(self.pos_good)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pg = self.pos_good[idx]
        pb = self.pos_bad[idx]
        m = self.margins[idx]
        if self.augment_hflip and torch.rand(1).item() < 0.5:
            pg = pg[self._flip_indices]
            pb = pb[self._flip_indices]
        return pg, pb, m
