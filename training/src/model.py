"""Neural network architectures for chess evaluation — 1540 dual-perspective input.

All models take [N, 1540] float32 input and produce [N, 1] scalar output.
The input is two 770-element halves: STM perspective and NSTM perspective.
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class ClippedReLU(nn.Module):
    """Clamped activation: clamp(x, 0, 1). Used by Stockfish NNUE."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0)


class SquaredClippedReLU(nn.Module):
    """SCReLU: clamp(x, 0, 1)^2. Used by newer NNUE variants for richer gradients."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0).square()


ACTIVATIONS = {
    "relu": nn.ReLU,
    "clipped_relu": ClippedReLU,
    "screlu": SquaredClippedReLU,
}


class NNUEModel(nn.Module):
    """NNUE-style perspective network for 1540 dual-perspective input.

    Architecture:
      1. Split input into STM half (770) and NSTM half (770)
      2. Shared feature transformer processes each half: 770 → ft_size
      3. Concatenate [STM_features, NSTM_features] → 2*ft_size
      4. Output layers: 2*ft_size → ... → 1

    This is how Stockfish NNUE and all competitive small engines work.
    The shared transformer learns piece-square features once, and the output
    layers learn to compare the two sides' features.
    """

    def __init__(
        self,
        ft_size: int = 256,
        output_sizes: list[int] | None = None,
        activation: str = "clipped_relu",
    ):
        super().__init__()

        if output_sizes is None:
            output_sizes = [32, 32]

        act_cls = ACTIVATIONS[activation]

        # Shared feature transformer: 770 (one side) → ft_size
        self.ft = nn.Linear(770, ft_size)
        self.ft_act = act_cls()

        # Output layers: ft_size*2 → ... → 1
        layers = []
        in_size = ft_size * 2
        for h in output_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(act_cls())
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.output = nn.Sequential(*layers)

        self.num_params = sum(p.numel() for p in self.parameters())

        # Initialize weights for centipawn-scale output.
        # Default PyTorch init produces outputs near 0 with tiny std, which
        # makes sigmoid(x/400) ≈ 0.5 with vanishing gradients.
        # We scale the final layer so initial outputs span ~[-300, 300] cp.
        self._init_weights()

    def _init_weights(self):
        """Initialize weights so the network produces centipawn-scale outputs
        from the start, preventing sigmoid(x/400) gradient vanishing.

        Strategy:
        - Feature transformer: small positive bias so ClippedReLU passes signal
        - Hidden layers: Kaiming uniform (good for ReLU-family)
        - Final linear layer: large gain so outputs span [-300, 300] initially
          With ClippedReLU activations in [0,1] and 32 inputs to the final layer,
          the expected active inputs have mean ~0.25, so we need weights ~±20
          to get outputs in the ±300 range.
        """
        # Feature transformer
        init.kaiming_uniform_(self.ft.weight, nonlinearity='relu')
        if self.ft.bias is not None:
            init.uniform_(self.ft.bias, 0.0, 0.1)

        # Hidden layers
        for module in self.output:
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    init.zeros_(module.bias)

        # Final layer: scale up to produce centipawn-range outputs.
        # Find the last Linear in self.output
        final_linear = None
        for module in self.output:
            if isinstance(module, nn.Linear):
                final_linear = module
        if final_linear is not None:
            fan_in = final_linear.in_features
            # With ClippedReLU inputs averaging ~0.3, we want
            # sum(w_i * x_i) to have std ≈ 200 cp.
            # std = sqrt(fan_in) * w_std * x_std ≈ sqrt(32) * w_std * 0.3
            # For std=200: w_std ≈ 200 / (sqrt(32) * 0.3) ≈ 118
            # Use ±150 uniform range (std ≈ 87) for a conservative start
            scale = 150.0 / (fan_in ** 0.5 * 0.3)
            init.uniform_(final_linear.weight, -scale, scale)
            init.zeros_(final_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 1540] = [STM_half (770), NSTM_half (770)]
        stm = x[:, :770]
        nstm = x[:, 770:]

        # Shared feature transformer
        stm_ft = self.ft_act(self.ft(stm))
        nstm_ft = self.ft_act(self.ft(nstm))

        # Concatenate: STM first, NSTM second
        combined = torch.cat([stm_ft, nstm_ft], dim=1)
        return self.output(combined)


class NNUEBucketModel(nn.Module):
    """NNUE with output buckets based on piece count (game phase proxy).

    Different output heads for different game phases:
    - Bucket 0: endgame (few pieces)
    - Bucket 1: middlegame
    - Bucket 2: opening (many pieces)

    The feature transformer is shared; only the output heads differ.
    This allows the model to specialize its evaluation for each phase.
    """

    def __init__(
        self,
        ft_size: int = 256,
        output_sizes: list[int] | None = None,
        num_buckets: int = 8,
        activation: str = "clipped_relu",
    ):
        super().__init__()

        if output_sizes is None:
            output_sizes = [32, 32]

        act_cls = ACTIVATIONS[activation]
        self.num_buckets = num_buckets

        # Shared feature transformer
        self.ft = nn.Linear(770, ft_size)
        self.ft_act = act_cls()

        # Per-bucket output heads
        self.bucket_heads = nn.ModuleList()
        for _ in range(num_buckets):
            layers = []
            in_size = ft_size * 2
            for h in output_sizes:
                layers.append(nn.Linear(in_size, h))
                layers.append(act_cls())
                in_size = h
            layers.append(nn.Linear(in_size, 1))
            self.bucket_heads.append(nn.Sequential(*layers))

        self.num_params = sum(p.numel() for p in self.parameters())

    def _piece_count(self, x: torch.Tensor) -> torch.Tensor:
        """Count pieces from the STM half (channels 0-11, 64 squares each)."""
        stm = x[:, :768]  # piece planes only (exclude castling)
        return stm.sum(dim=1)  # [N]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stm = x[:, :770]
        nstm = x[:, 770:]

        stm_ft = self.ft_act(self.ft(stm))
        nstm_ft = self.ft_act(self.ft(nstm))
        combined = torch.cat([stm_ft, nstm_ft], dim=1)

        # Determine bucket from piece count
        # Piece count ranges from ~2 (K vs K) to 32 (opening)
        # Map to bucket: bucket = clamp(piece_count * num_buckets / 33, 0, num_buckets-1)
        pc = self._piece_count(x)
        bucket_idx = torch.clamp(
            (pc * self.num_buckets / 33).long(), 0, self.num_buckets - 1
        )

        # Evaluate all buckets and select (for ONNX exportability)
        all_outputs = torch.stack(
            [head(combined) for head in self.bucket_heads], dim=1
        )  # [N, num_buckets, 1]

        # Gather the right bucket for each sample
        idx = bucket_idx.view(-1, 1, 1).expand(-1, 1, 1)
        return all_outputs.gather(1, idx).squeeze(1)  # [N, 1]


class SimpleMLP(nn.Module):
    """Simple MLP baseline for comparison. Takes full 1540 input."""

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        activation: str = "relu",
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        act_cls = ACTIVATIONS[activation]

        layers = []
        in_size = 1540
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(act_cls())
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Architecture registry
ARCHITECTURES = {
    # NNUE perspective models (recommended)
    "nnue_256x2_32_32": {"cls": NNUEModel, "ft_size": 256, "output_sizes": [32, 32]},
    "nnue_512x2_32_32": {"cls": NNUEModel, "ft_size": 512, "output_sizes": [32, 32]},
    "nnue_1024x2_32_32": {"cls": NNUEModel, "ft_size": 1024, "output_sizes": [32, 32]},
    "nnue_2048x2_64_32": {"cls": NNUEModel, "ft_size": 2048, "output_sizes": [64, 32]},
    "nnue_4096x2_128_32": {"cls": NNUEModel, "ft_size": 4096, "output_sizes": [128, 32]},
    "nnue_256x2_16": {"cls": NNUEModel, "ft_size": 256, "output_sizes": [16]},
    "nnue_512x2_64": {"cls": NNUEModel, "ft_size": 512, "output_sizes": [64]},
    # SCReLU variants
    "nnue_256x2_32_32_screlu": {"cls": NNUEModel, "ft_size": 256, "output_sizes": [32, 32], "activation": "screlu"},
    "nnue_512x2_32_32_screlu": {"cls": NNUEModel, "ft_size": 512, "output_sizes": [32, 32], "activation": "screlu"},
    "nnue_1024x2_32_32_screlu": {"cls": NNUEModel, "ft_size": 1024, "output_sizes": [32, 32], "activation": "screlu"},
    "nnue_2048x2_64_32_screlu": {"cls": NNUEModel, "ft_size": 2048, "output_sizes": [64, 32], "activation": "screlu"},
    # Bucket models (game-phase-aware)
    "nnue_bucket_256x2_32": {"cls": NNUEBucketModel, "ft_size": 256, "output_sizes": [32], "num_buckets": 8},
    "nnue_bucket_512x2_32": {"cls": NNUEBucketModel, "ft_size": 512, "output_sizes": [32], "num_buckets": 8},
    # Simple MLP baselines
    "mlp_512_256_128": {"cls": SimpleMLP, "hidden_sizes": [512, 256, 128]},
    "mlp_1024_256_64": {"cls": SimpleMLP, "hidden_sizes": [1024, 256, 64]},
}


def build_model(arch_name: str) -> nn.Module:
    """Build a model from the architecture registry."""
    if arch_name not in ARCHITECTURES:
        raise ValueError(f"Unknown arch: {arch_name}. Available: {list(ARCHITECTURES.keys())}")

    config = dict(ARCHITECTURES[arch_name])
    cls = config.pop("cls")
    model = cls(**config)
    print(f"Model: {arch_name}, params: {model.num_params:,}")
    return model


def list_architectures():
    """Print all available architectures with param counts."""
    for name in ARCHITECTURES:
        config = dict(ARCHITECTURES[name])
        cls = config.pop("cls")
        model = cls(**config)
        print(f"  {name:40s}  {model.num_params:>10,} params")


if __name__ == "__main__":
    list_architectures()
    print()

    # Verify all models work with 1540 input
    for name in ARCHITECTURES:
        model = build_model(name)
        x = torch.randn(4, 1540)
        y = model(x)
        assert y.shape == (4, 1), f"Bad output shape for {name}: {y.shape}"
        print(f"  {name}: output shape OK, range [{y.min():.3f}, {y.max():.3f}]")
