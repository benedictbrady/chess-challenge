"""PyTorch Datasets for chess training data — 1540 dual-perspective encoding."""

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class ChessEvalDataset(Dataset):
    """Positions labeled with centipawn evaluations.

    Each sample: (position [1540], eval_cp [scalar])
    """

    def __init__(self, data_path: str, max_abs_eval: float = 10000.0):
        data = np.load(data_path)
        self.positions = torch.from_numpy(data["positions"])  # [N, 1540]
        self.evals_cp = torch.from_numpy(data["evals"].astype(np.float32))  # [N]

        # Filter extreme evals
        mask = self.evals_cp.abs() <= max_abs_eval
        self.positions = self.positions[mask]
        self.evals_cp = self.evals_cp[mask]

        assert self.positions.shape[1] == 1540, f"Expected 1540 features, got {self.positions.shape[1]}"

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.positions[idx], self.evals_cp[idx]


class ChessOutcomeDataset(Dataset):
    """Positions labeled with game outcomes.

    Each sample: (position [1540], outcome [scalar])
    outcome: 1.0 = win for STM, 0.0 = loss, 0.5 = draw
    """

    def __init__(self, data_path: str, decisive_only: bool = False):
        data = np.load(data_path)
        positions = data["positions"]
        outcomes = data["outcomes"].astype(np.float32)

        assert positions.shape[1] == 1540, f"Expected 1540 features, got {positions.shape[1]}"

        if decisive_only:
            mask = (outcomes > 0.7) | (outcomes < 0.3)
            positions = positions[mask]
            outcomes = outcomes[mask]
            print(f"  Decisive only: {mask.sum():,} / {len(mask):,} ({100 * mask.mean():.1f}%)")

        self.positions = torch.from_numpy(positions)
        self.outcomes = torch.from_numpy(outcomes)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.positions[idx], self.outcomes[idx]


class ChessBlendedDataset(Dataset):
    """Positions with eval + game outcome for blended WDL training.

    target = lambda * sigmoid(eval/scale) + (1-lambda) * outcome
    This is the Stockfish NNUE loss approach.
    """

    def __init__(
        self,
        data_path: str,
        eval_scale: float = 400.0,
        blend_lambda: float = 0.75,
    ):
        data = np.load(data_path)
        self.positions = torch.from_numpy(data["positions"])  # [N, 1540]
        evals = torch.from_numpy(data["evals"].astype(np.float32))
        outcomes = torch.from_numpy(data["outcomes"].astype(np.float32))

        assert self.positions.shape[1] == 1540

        eval_wdl = torch.sigmoid(evals / eval_scale)
        self.targets = blend_lambda * eval_wdl + (1.0 - blend_lambda) * outcomes

        print(f"  Blended: λ={blend_lambda}, eval_mean_wdl={eval_wdl.mean():.3f}, "
              f"outcome_mean={outcomes.mean():.3f}, target_mean={self.targets.mean():.3f}")

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.positions[idx], self.targets[idx]


class ChessMultiDataset(Dataset):
    """Concatenation of multiple .npz files into one dataset.

    Supports eval-only, outcome-only, or both (for blended training).
    """

    def __init__(self, data_paths: list[str], label_key: str = "evals", max_abs_eval: float = 10000.0):
        all_positions = []
        all_labels = []

        for path in data_paths:
            data = np.load(path)
            all_positions.append(data["positions"])
            all_labels.append(data[label_key].astype(np.float32))

        positions = np.concatenate(all_positions, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        assert positions.shape[1] == 1540, f"Expected 1540 features, got {positions.shape[1]}"

        # Filter extreme evals
        mask = np.abs(labels) <= max_abs_eval
        if mask.sum() < len(mask):
            print(f"  Filtered to |eval| <= {max_abs_eval}: {mask.sum():,} / {len(mask):,}")
            positions = positions[mask]
            labels = labels[mask]

        self.positions = torch.from_numpy(positions)
        self.labels = torch.from_numpy(labels)
        print(f"  Multi-dataset: {len(self.positions):,} samples from {len(data_paths)} files")

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.positions[idx], self.labels[idx]


class ChessMultiOutcomeDataset(Dataset):
    """Concatenation of multiple .npz files for outcome-based training.

    Loads positions and outcomes from each file, concatenates them.
    This avoids merging large datasets into a single file on disk.
    """

    def __init__(self, data_paths: list[str], decisive_only: bool = False):
        all_positions = []
        all_outcomes = []

        for path in data_paths:
            print(f"  Loading {path}...")
            data = np.load(path)
            all_positions.append(data["positions"])
            all_outcomes.append(data["outcomes"].astype(np.float32))
            print(f"    {data['positions'].shape[0]:,} positions")

        positions = np.concatenate(all_positions, axis=0)
        outcomes = np.concatenate(all_outcomes, axis=0)

        assert positions.shape[1] == 1540, f"Expected 1540 features, got {positions.shape[1]}"

        if decisive_only:
            mask = (outcomes > 0.7) | (outcomes < 0.3)
            positions = positions[mask]
            outcomes = outcomes[mask]
            print(f"  Decisive only: {mask.sum():,} / {len(mask):,} ({100 * mask.mean():.1f}%)")

        self.positions = torch.from_numpy(positions)
        self.outcomes = torch.from_numpy(outcomes)
        print(f"  Multi-outcome-dataset: {len(self.positions):,} total samples from {len(data_paths)} files")

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.positions[idx], self.outcomes[idx]
