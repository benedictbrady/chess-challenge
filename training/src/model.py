"""Neural network architectures for chess evaluation.

Models take binary float32 input and produce [N, 1] scalar output.
- MLP/CNN/Perspective models: [N, 768] input
- NNUE models: [N, 1536] input (dual perspective)
The output represents evaluation from the side-to-move's perspective.
"""

import torch
import torch.nn as nn


class ClippedReLU(nn.Module):
    """Clamped activation: clamp(x, 0, 1). Used by Stockfish NNUE."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0)


class ChessEvalMLP(nn.Module):
    """Simple MLP for chess evaluation.

    Default architecture (558K params):
        768 → 512 → 256 → 128 → 1
    """

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        input_size: int = 768,
        activation: str = "clipped_relu",
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        act_fn = {
            "clipped_relu": ClippedReLU,
            "relu": nn.ReLU,
        }
        if activation not in act_fn:
            raise ValueError(f"Unknown activation: {activation}. Available: {list(act_fn.keys())}")

        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(act_fn[activation]())
            in_size = h
        layers.append(nn.Linear(in_size, 1))

        self.net = nn.Sequential(*layers)
        self._count_params()

    def _count_params(self):
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChessEvalCNN(nn.Module):
    """CNN for chess evaluation — reshapes 768 flat inputs to 12×8×8 board image.

    Convolutions naturally capture spatial patterns like:
    - Sliding piece mobility (diagonals, files, ranks)
    - King zone attacks (local neighborhoods)
    - Pawn structure (file-based patterns)
    """

    def __init__(
        self,
        channels: list[int] | None = None,
        fc_sizes: list[int] | None = None,
    ):
        super().__init__()

        if channels is None:
            channels = [64, 64, 32]
        if fc_sizes is None:
            fc_sizes = [256, 128]

        # Convolutional layers: process 12-channel 8x8 board
        conv_layers = []
        in_ch = 12  # 6 own piece types + 6 opponent piece types
        for out_ch in channels:
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)

        # Flatten: channels[-1] * 8 * 8
        flat_size = channels[-1] * 8 * 8

        # Fully connected layers
        fc_layers = []
        in_size = flat_size
        for h in fc_sizes:
            fc_layers.append(nn.Linear(in_size, h))
            fc_layers.append(nn.ReLU())
            in_size = h
        fc_layers.append(nn.Linear(in_size, 1))
        self.fc = nn.Sequential(*fc_layers)

        self._count_params()

    def _count_params(self):
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape flat 768 → 12 channels × 8 × 8
        x = x.view(-1, 12, 8, 8)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)  # flatten
        return self.fc(x)


# Pre-defined architecture configs
ARCHITECTURES = {
    # ClippedReLU variants (original)
    # 558K params — safe under 1M limit
    "mlp_512_256_128": {"hidden_sizes": [512, 256, 128]},
    # 796K params — wider first layer, still under 1M
    "mlp_768_256_32": {"hidden_sizes": [768, 256, 32]},
    # 329K params — compact
    "mlp_384_128_32": {"hidden_sizes": [384, 128, 32]},
    # 428K params — two-layer
    "mlp_512_64": {"hidden_sizes": [512, 64]},
    # ReLU variants — more expressive (no clamping at 1.0)
    "mlp_512_256_128_relu": {"hidden_sizes": [512, 256, 128], "activation": "relu"},
    "mlp_768_256_32_relu": {"hidden_sizes": [768, 256, 32], "activation": "relu"},
    "mlp_384_128_32_relu": {"hidden_sizes": [384, 128, 32], "activation": "relu"},
    # Wide first layer — 960 feature detectors, 798K params
    "mlp_960_64_relu": {"hidden_sizes": [960, 64], "activation": "relu"},
    # Very wide + deeper, 936K params
    "mlp_1024_128_relu": {"hidden_sizes": [1024, 128], "activation": "relu"},
    # Ensemble member architectures (~320K each, 3 fit under 1M)
    "ens_a": {"hidden_sizes": [384, 64, 32], "activation": "relu"},  # 322K, ReLU (ClippedReLU too restrictive for small models)
    "ens_b_relu": {"hidden_sizes": [320, 128, 32], "activation": "relu"},  # 291K, ReLU
    "ens_c_relu": {"hidden_sizes": [384, 64, 32], "activation": "relu"},  # 322K, ReLU
    # 10M-budget architectures (wide first layer = rich piece-square features)
    "mlp_4096_256_128_relu": {"hidden_sizes": [4096, 256, 128], "activation": "relu"},  # 4.2M
    "mlp_2048_512_256_relu": {"hidden_sizes": [2048, 512, 256], "activation": "relu"},  # 2.8M
    "mlp_2048_256_128_relu": {"hidden_sizes": [2048, 256, 128], "activation": "relu"},  # 2.1M
}

# CNN architectures — process board as 12×8×8 image
CNN_ARCHITECTURES = {
    # 3 conv layers + 2 FC layers, ~657K params
    "cnn_64_64_32": {"channels": [64, 64, 32], "fc_sizes": [256, 128]},
    # Deeper: 4 conv layers, ~694K params
    "cnn_64_64_64_32": {"channels": [64, 64, 64, 32], "fc_sizes": [256, 128]},
    # Wider: more channels, ~870K params
    "cnn_96_64_32": {"channels": [96, 64, 32], "fc_sizes": [256, 128]},
    # Smaller FC for bigger conv, ~550K
    "cnn_64_64_32_small": {"channels": [64, 64, 32], "fc_sizes": [128]},
}


class SCReLU(nn.Module):
    """Squared Clipped ReLU: clamp(x, 0, 1)^2.

    More expressive than ClippedReLU — proven ~50% parameter efficiency
    gain in modern NNUE engines (Stockfish, Lc0).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0).square()


class ChessEvalNNUE(nn.Module):
    """NNUE-style dual perspective network.

    Takes [N, 1536] input: two 768-element halves (STM and NSTM perspectives).
    Both halves contain all 12 piece channels from their respective viewpoint.

    A SHARED feature transformer processes each perspective identically,
    then the output layers compare the concatenated perspectives.

    This differs from ChessEvalPerspective which splits a SINGLE 768 input
    into own/opponent halves (384 each). Here, each perspective is a FULL
    768-element encoding, preserving cross-side interactions.
    """

    def __init__(
        self,
        ft_size: int = 2048,
        output_sizes: list[int] | None = None,
    ):
        super().__init__()

        if output_sizes is None:
            output_sizes = [16, 32]

        # Shared feature transformer: 768 (one perspective) → ft_size
        self.ft = nn.Linear(768, ft_size)
        self.ft_act = SCReLU()

        # Output layers: ft_size*2 (both perspectives concatenated) → ... → 1
        layers: list[nn.Module] = []
        in_size = ft_size * 2
        for h in output_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(SCReLU())
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.output = nn.Sequential(*layers)

        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stm = x[:, :768]       # STM full perspective (all 12 channels)
        nstm = x[:, 768:]      # NSTM full perspective (all 12 channels)

        # Shared feature transformer (same weights for both perspectives)
        stm_ft = self.ft_act(self.ft(stm))
        nstm_ft = self.ft_act(self.ft(nstm))

        # Concatenate: STM first, NSTM second
        combined = torch.cat([stm_ft, nstm_ft], dim=1)  # [N, ft_size*2]

        return self.output(combined)


# NNUE architecture configs (dual perspective, 1536 input)
NNUE_ARCHITECTURES = {
    # ft_size=2048, output=[16, 32] → ~1.64M params
    "nnue_2048_16_32": {"ft_size": 2048, "output_sizes": [16, 32]},
    # ft_size=1024, output=[16, 32] → ~820K params
    "nnue_1024_16_32": {"ft_size": 1024, "output_sizes": [16, 32]},
    # ft_size=1536, output=[16, 32] → ~1.22M params
    "nnue_1536_16_32": {"ft_size": 1536, "output_sizes": [16, 32]},
    # ft_size=2048, output=[32] → ~1.64M params (simpler output)
    "nnue_2048_32": {"ft_size": 2048, "output_sizes": [32]},
    # ft_size=4096, output=[16, 32] → ~3.21M params
    "nnue_4096_16_32": {"ft_size": 4096, "output_sizes": [16, 32]},
}


class ChessEvalPerspective(nn.Module):
    """Perspective network — processes own and opponent pieces separately.

    The 768 inputs are split into own pieces (384) and opponent pieces (384).
    Both halves are processed by a SHARED feature transformer, then
    concatenated (STM first, NSTM second) before the output layers.

    This is how Stockfish NNUE, Leorik, and all competitive small engines work.
    The shared transformer learns piece-square features once, and the output
    layers learn how to COMPARE the two sides' features.
    """

    def __init__(
        self,
        ft_size: int = 256,
        output_sizes: list[int] | None = None,
        activation: str = "relu",
    ):
        super().__init__()

        if output_sizes is None:
            output_sizes = [32]

        act_fn = {"relu": nn.ReLU, "clipped_relu": ClippedReLU}
        if activation not in act_fn:
            raise ValueError(f"Unknown activation: {activation}")
        act_cls = act_fn[activation]

        # Shared feature transformer: 384 (one side) → ft_size
        self.ft = nn.Linear(384, ft_size)
        self.ft_act = act_cls()

        # Output layers: ft_size*2 (both sides concatenated) → ... → 1
        layers = []
        in_size = ft_size * 2
        for h in output_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(act_cls())
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.output = nn.Sequential(*layers)

        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 768] = [own_P..own_K (6×64), opp_P..opp_K (6×64)]
        own = x[:, :384]   # own pieces
        opp = x[:, 384:]   # opponent pieces

        # Shared feature transformer
        own_ft = self.ft_act(self.ft(own))
        opp_ft = self.ft_act(self.ft(opp))

        # Concatenate: STM first, NSTM second
        combined = torch.cat([own_ft, opp_ft], dim=1)

        return self.output(combined)


# Perspective architecture configs
PERSPECTIVE_ARCHITECTURES = {
    # ft_size=256, output=[32] → 115K params (tiny but competitive)
    "perspective_256_32": {"ft_size": 256, "output_sizes": [32]},
    # ft_size=512, output=[32] → 395K params
    "perspective_512_32": {"ft_size": 512, "output_sizes": [32]},
    # ft_size=1024, output=[64, 32] → 1.5M params
    "perspective_1024_64_32": {"ft_size": 1024, "output_sizes": [64, 32]},
    # ft_size=2048, output=[64, 32] → 2.9M params
    "perspective_2048_64_32": {"ft_size": 2048, "output_sizes": [64, 32]},
    # ft_size=4096, output=[64, 32] → 5.7M params
    "perspective_4096_64_32": {"ft_size": 4096, "output_sizes": [64, 32]},
}


class ChessEvalEnsemble(nn.Module):
    """Ensemble of multiple models — averages their outputs.

    Used to combine diverse small models into a single ONNX file.
    Total params must stay under 1M.
    """

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([m(x) for m in self.models], dim=0)  # [K, N, 1]
        return outputs.mean(dim=0)  # [N, 1]


def build_ensemble(arch_names: list[str]) -> ChessEvalEnsemble:
    """Build an ensemble from a list of architecture names."""
    models = [build_model(name) for name in arch_names]
    ensemble = ChessEvalEnsemble(models)
    print(f"Ensemble: {len(models)} models, total params: {ensemble.num_params:,}")
    return ensemble


def build_model(arch_name: str = "mlp_512_256_128") -> ChessEvalMLP | ChessEvalCNN | ChessEvalPerspective | ChessEvalNNUE:
    """Build a model from a named architecture."""
    if arch_name in NNUE_ARCHITECTURES:
        config = NNUE_ARCHITECTURES[arch_name]
        model = ChessEvalNNUE(**config)
        print(f"Model: {arch_name}, params: {model.num_params:,}")
        return model
    if arch_name in PERSPECTIVE_ARCHITECTURES:
        config = PERSPECTIVE_ARCHITECTURES[arch_name]
        model = ChessEvalPerspective(**config)
        print(f"Model: {arch_name}, params: {model.num_params:,}")
        return model
    if arch_name in CNN_ARCHITECTURES:
        config = CNN_ARCHITECTURES[arch_name]
        model = ChessEvalCNN(**config)
        print(f"Model: {arch_name}, params: {model.num_params:,}")
        return model
    if arch_name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch_name}. Available: {list(NNUE_ARCHITECTURES.keys()) + list(ARCHITECTURES.keys()) + list(CNN_ARCHITECTURES.keys()) + list(PERSPECTIVE_ARCHITECTURES.keys())}")
    config = ARCHITECTURES[arch_name]
    model = ChessEvalMLP(**config)
    print(f"Model: {arch_name}, params: {model.num_params:,}")
    return model


if __name__ == "__main__":
    # Print param counts for all architectures
    for name in ARCHITECTURES:
        model = build_model(name)
        # Verify with a forward pass
        x = torch.randn(4, 768)
        y = model(x)
        assert y.shape == (4, 1), f"Bad output shape: {y.shape}"
        print(f"  output shape: {y.shape}, range: [{y.min():.3f}, {y.max():.3f}]")
        print()
