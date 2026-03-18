"""Export PyTorch model to ONNX — 1540 dual-perspective input.

Requirements (from engine/src/nn.rs):
- Input name: "board", shape [N, 1540]
- Output shape: [N, 1]
- ir_version = 8, opset 17
- Max 10,000,000 parameters
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .model import build_model

INPUT_SIZE = 1540


def count_onnx_params(model_path: str) -> int:
    """Count parameters matching engine/src/nn.rs count_parameters()."""
    model = onnx.load(model_path)
    total = 0
    for tensor in model.graph.initializer:
        if tensor.dims:
            count = 1
            for d in tensor.dims:
                count *= max(d, 1)
            total += count
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
) -> str:
    """Export PyTorch model to ONNX."""
    model.eval()
    model.cpu()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, INPUT_SIZE)

    torch.onnx.export(
        model, dummy, output_path,
        input_names=["board"],
        output_names=["eval"],
        dynamic_axes={"board": {0: "batch"}, "eval": {0: "batch"}},
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Patch ir_version
    onnx_model = onnx.load(output_path)
    onnx_model.ir_version = ir_version
    onnx.save(onnx_model, output_path)

    return output_path


def validate_onnx(model_path: str, max_params: int = 10_000_000) -> bool:
    """Validate ONNX model against harness requirements."""
    print(f"Validating {model_path}")
    ok = True

    model = onnx.load(model_path)
    print(f"  IR version: {model.ir_version}")
    if model.ir_version != 8:
        print(f"  WARN: expected ir_version=8")
        ok = False

    inputs = [i.name for i in model.graph.input]
    if "board" not in inputs:
        print(f"  ERROR: input must be named 'board', got {inputs}")
        ok = False

    params = count_onnx_params(model_path)
    print(f"  Parameters: {params:,}")
    if params > max_params:
        print(f"  ERROR: exceeds {max_params:,} limit")
        ok = False

    # Check batch dimensions are named (not anonymous)
    for tensor_desc in list(model.graph.input) + list(model.graph.output):
        dim0 = tensor_desc.type.tensor_type.shape.dim[0]
        if not dim0.dim_param:
            print(f"  ERROR: '{tensor_desc.name}' has unnamed batch dimension (dim 0).")
            print(f"         The Rust harness requires a named dynamic batch dim (e.g. 'batch').")
            print(f"         Fix: use dynamic_axes={{\"board\": {{0: \"batch\"}}, \"eval\": {{0: \"batch\"}}}} in torch.onnx.export,")
            print(f"         or use nn.Linear (Gemm) for the final layer instead of MatMul+Add.")
            ok = False

    # Test inference
    sess = ort.InferenceSession(model_path)

    for batch in [1, 35]:
        inp = np.random.randn(batch, INPUT_SIZE).astype(np.float32)
        out = sess.run(None, {"board": inp})
        expected = (batch, 1)
        if out[0].shape != expected:
            print(f"  ERROR: batch={batch} expected {expected}, got {out[0].shape}")
            ok = False
        else:
            print(f"  Batch={batch}: shape OK, range [{out[0].min():.3f}, {out[0].max():.3f}]")

    # Binary input test (realistic board positions)
    binary = np.zeros((10, INPUT_SIZE), dtype=np.float32)
    for i in range(10):
        idx = np.random.choice(INPUT_SIZE, size=32, replace=False)
        binary[i, idx] = 1.0
    out = sess.run(None, {"board": binary})
    print(f"  Binary input range: [{out[0].min():.3f}, {out[0].max():.3f}]")

    print(f"  {'PASSED' if ok else 'FAILED'}")
    return ok


def export_from_checkpoint(
    checkpoint_path: str,
    arch_name: str,
    output_path: str,
) -> str:
    """Load checkpoint and export to ONNX."""
    model = build_model(arch_name)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    export_onnx(model, output_path)
    validate_onnx(output_path)
    return output_path
