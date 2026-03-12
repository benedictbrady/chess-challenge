"""Export PyTorch model to ONNX with strict requirements matching the Rust harness.

Requirements (from engine/src/nn.rs):
- Input name: "board"
- Input shape: [N, 1536] (dynamic batch) for NNUE models, [N, 768] for legacy
- Output shape: [N, 1]
- ir_version = 8, opset 17
- Parameters stored as initializers (for counting)
- Total params <= 10,000,000
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .checkpoint import CheckpointManager
from .model import build_model


def count_onnx_params(model_path: str) -> int:
    """Count parameters the same way the Rust harness does.

    Counts all initializer scalars + Constant node tensor scalars.
    Must match engine/src/nn.rs count_parameters().
    """
    model = onnx.load(model_path)
    total = 0

    # Count initializers
    for tensor in model.graph.initializer:
        if tensor.dims:
            count = 1
            for d in tensor.dims:
                count *= max(d, 1)
            total += count

    # Count Constant nodes
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.t and attr.t.dims:
                    count = 1
                    for d in attr.t.dims:
                        count *= max(d, 1)
                    total += count

    return total


def export_onnx(
    model: torch.nn.Module,
    output_path: str,
    opset_version: int = 17,
    ir_version: int = 8,
    input_size: int = 1536,
) -> str:
    """Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model
        output_path: Where to save the .onnx file
        opset_version: ONNX opset version (default 17)
        ir_version: ONNX IR version (must be 8 for the harness)
        input_size: Input tensor width (1536 for NNUE, 768 for legacy)

    Returns:
        Path to the exported ONNX file
    """
    model.eval()
    model.cpu()

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Dummy input for tracing
    dummy_input = torch.randn(1, input_size)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["board"],
        output_names=["eval"],
        dynamic_axes={
            "board": {0: "batch_size"},
            "eval": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Patch ir_version post-export
    onnx_model = onnx.load(output_path)
    onnx_model.ir_version = ir_version
    onnx.save(onnx_model, output_path)

    return output_path


def validate_onnx(model_path: str, max_params: int = 10_000_000, input_size: int = 1536) -> bool:
    """Validate an exported ONNX model against harness requirements."""
    print(f"Validating {model_path}")
    success = True

    # 1. Load and check structure
    model = onnx.load(model_path)
    print(f"  IR version: {model.ir_version}")
    if model.ir_version != 8:
        print(f"  WARNING: ir_version should be 8, got {model.ir_version}")
        success = False

    # 2. Check input name
    inputs = [i.name for i in model.graph.input]
    print(f"  Inputs: {inputs}")
    if "board" not in inputs:
        print(f"  ERROR: Input must be named 'board', got {inputs}")
        success = False

    # 3. Check parameter count
    param_count = count_onnx_params(model_path)
    print(f"  Parameters: {param_count:,}")
    if param_count > max_params:
        print(f"  ERROR: Exceeds {max_params:,} parameter limit")
        success = False

    # 4. Test inference with ORT
    session = ort.InferenceSession(model_path)

    # Batch size = 1
    input_1 = np.random.randn(1, input_size).astype(np.float32)
    output_1 = session.run(None, {"board": input_1})
    print(f"  Batch=1 output shape: {output_1[0].shape}")
    if output_1[0].shape != (1, 1):
        print(f"  ERROR: Expected (1, 1), got {output_1[0].shape}")
        success = False

    # Batch size = 35 (typical: ~35 legal moves)
    input_35 = np.random.randn(35, input_size).astype(np.float32)
    output_35 = session.run(None, {"board": input_35})
    print(f"  Batch=35 output shape: {output_35[0].shape}")
    if output_35[0].shape != (35, 1):
        print(f"  ERROR: Expected (35, 1), got {output_35[0].shape}")
        success = False

    # 5. Check output range with binary inputs (like real board tensors)
    binary_input = np.zeros((10, input_size), dtype=np.float32)
    # Set some random bits to 1 (simulating piece positions)
    for i in range(10):
        indices = np.random.choice(input_size, size=16, replace=False)
        binary_input[i, indices] = 1.0
    output_binary = session.run(None, {"board": binary_input})
    print(f"  Binary input eval range: [{output_binary[0].min():.3f}, {output_binary[0].max():.3f}]")

    if success:
        print("  PASSED all validation checks")
    else:
        print("  FAILED validation")

    return success


def export_ensemble_onnx(
    checkpoint_paths: list[str],
    arch_names: list[str],
    output_path: str,
    max_params: int = 1_000_000,
) -> str:
    """Export an ensemble of models as a single ONNX file.

    Loads each checkpoint, builds the ensemble, exports, and validates.
    """
    from .model import build_ensemble

    models = []
    for ckpt_path, arch_name in zip(checkpoint_paths, arch_names):
        model = build_model(arch_name)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        models.append(model)
        print(f"  Loaded {arch_name} from {ckpt_path}")

    from .model import ChessEvalEnsemble
    ensemble = ChessEvalEnsemble(models)
    print(f"Ensemble: {len(models)} models, {ensemble.num_params:,} params")

    if ensemble.num_params > max_params:
        raise ValueError(f"Ensemble too large: {ensemble.num_params:,} > {max_params:,}")

    export_onnx(ensemble, output_path)
    valid = validate_onnx(output_path, max_params=max_params)
    if not valid:
        raise RuntimeError("Ensemble ONNX validation failed!")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory to find best checkpoint")
    parser.add_argument("--architecture", type=str, default="mlp_512_256_128")
    parser.add_argument("--output", type=str, default="model.onnx")
    parser.add_argument("--validate", action="store_true", default=True)
    args = parser.parse_args()

    # Load model
    model = build_model(args.architecture)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.checkpoint_dir:
        mgr = CheckpointManager(args.checkpoint_dir)
        ckpt_path = mgr.best_checkpoint() or mgr.latest_checkpoint()
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoints in {args.checkpoint_dir}")
    else:
        raise ValueError("Must provide --checkpoint or --checkpoint-dir")

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    # Export
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, args.output)
    print(f"Exported to {args.output}")

    # Validate
    if args.validate:
        validate_onnx(args.output)


if __name__ == "__main__":
    main()
