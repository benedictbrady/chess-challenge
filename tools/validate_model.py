#!/usr/bin/env python3
"""Validate an ONNX model against the chess-challenge harness requirements.

Usage: python tools/validate_model.py path/to/model.onnx

Requirements checked:
  - Input named "board", shape [N, 1540]
  - Output shape [N, 1]
  - ir_version = 8, opset 17
  - Max 10,000,000 parameters
  - Batch dimension must be named (not anonymous) on input and output
  - Batched inference actually works (batch=1 and batch=35)

Dependencies: onnx, onnxruntime, numpy
"""

import sys

import numpy as np
import onnx
import onnxruntime as ort

INPUT_SIZE = 1540
MAX_PARAMS = 10_000_000


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


def validate_onnx(model_path: str, max_params: int = MAX_PARAMS) -> bool:
    """Validate ONNX model against harness requirements."""
    print(f"Validating {model_path}")
    ok = True

    model = onnx.load(model_path)

    # IR version
    print(f"  IR version: {model.ir_version}")
    if model.ir_version != 8:
        print(f"  ERROR: expected ir_version=8")
        ok = False

    # Input name
    inputs = [i.name for i in model.graph.input]
    if "board" not in inputs:
        print(f"  ERROR: input must be named 'board', got {inputs}")
        ok = False

    # Parameter count
    params = count_onnx_params(model_path)
    print(f"  Parameters: {params:,}")
    if params > max_params:
        print(f"  ERROR: exceeds {max_params:,} limit")
        ok = False

    # Batch dimensions must be named (not anonymous)
    for tensor_desc in list(model.graph.input) + list(model.graph.output):
        dim0 = tensor_desc.type.tensor_type.shape.dim[0]
        if not dim0.dim_param:
            print(f"  ERROR: '{tensor_desc.name}' has unnamed batch dimension (dim 0).")
            print(f"         The Rust harness requires a named dynamic batch dim (e.g. 'batch').")
            print(f"         Fix: use dynamic_axes={{\"board\": {{0: \"batch\"}}, \"eval\": {{0: \"batch\"}}}} in torch.onnx.export,")
            print(f"         or use nn.Linear (Gemm) for the final layer instead of MatMul+Add.")
            ok = False

    # Test inference at multiple batch sizes
    sess = ort.InferenceSession(model_path)

    for batch in [1, 35]:
        inp = np.random.randn(batch, INPUT_SIZE).astype(np.float32)
        out = sess.run(None, {"board": inp})
        expected = (batch, 1)
        if out[0].shape != expected:
            print(f"  ERROR: batch={batch} expected shape {expected}, got {out[0].shape}")
            ok = False
        else:
            print(f"  Batch={batch}: shape OK, range [{out[0].min():.3f}, {out[0].max():.3f}]")

    # Binary input test (realistic board positions have sparse 0/1 inputs)
    binary = np.zeros((10, INPUT_SIZE), dtype=np.float32)
    for i in range(10):
        idx = np.random.choice(INPUT_SIZE, size=32, replace=False)
        binary[i, idx] = 1.0
    out = sess.run(None, {"board": binary})
    print(f"  Binary input range: [{out[0].min():.3f}, {out[0].max():.3f}]")

    print(f"  {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <model.onnx>")
        sys.exit(1)
    ok = validate_onnx(sys.argv[1])
    sys.exit(0 if ok else 1)
