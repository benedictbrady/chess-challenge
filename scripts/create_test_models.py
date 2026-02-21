#!/usr/bin/env python3
"""
Create test ONNX models for the chess competition pipeline.

Models produced:
  models/random_policy.onnx  — tiny MLP (768→64→4096) with random weights
                               Plays essentially random chess (~315K params)
  models/capture_policy.onnx — same architecture, hand-crafted weights that
                               strongly prefer captures of valuable pieces

Requirements:
  pip install onnx numpy

The models conform to the competition spec:
  Input:  "board"  shape [1, 768]  float32
  Output: "policy" shape [1, 4096] float32 (raw logits)
"""

import os
import struct
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    from onnx.checker import check_model
except ImportError:
    print("ERROR: onnx not installed.  Run:  pip install onnx numpy")
    raise

os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: build a 2-layer MLP graph
#   board [1, 768] → Linear(768, hidden) → ReLU → Linear(hidden, 4096) → policy
# ---------------------------------------------------------------------------

def make_mlp_model(name: str, W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray) -> onnx.ModelProto:
    """
    W1: [768, hidden]  b1: [hidden]
    W2: [hidden, 4096] b2: [4096]
    """
    hidden = W1.shape[1]
    assert W1.shape == (768, hidden)
    assert b1.shape == (hidden,)
    assert W2.shape == (hidden, 4096)
    assert b2.shape == (4096,)

    def init(arr, iname):
        return numpy_helper.from_array(arr.astype(np.float32), name=iname)

    initializers = [
        init(W1, "W1"), init(b1, "b1"),
        init(W2, "W2"), init(b2, "b2"),
    ]

    nodes = [
        helper.make_node("MatMul",  ["board", "W1"],        ["mm1"]),
        helper.make_node("Add",     ["mm1", "b1"],           ["hidden"]),
        helper.make_node("Relu",    ["hidden"],              ["relu"]),
        helper.make_node("MatMul",  ["relu", "W2"],          ["mm2"]),
        helper.make_node("Add",     ["mm2", "b2"],           ["policy"]),
    ]

    graph = helper.make_graph(
        nodes, name,
        inputs  = [helper.make_tensor_value_info("board",  TensorProto.FLOAT, [1, 768])],
        outputs = [helper.make_tensor_value_info("policy", TensorProto.FLOAT, [1, 4096])],
        initializer = initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    check_model(model)
    return model


def param_count(model: onnx.ModelProto) -> int:
    total = 0
    for init in model.graph.initializer:
        n = 1
        for d in init.dims:
            n *= d
        total += n
    return total


# ---------------------------------------------------------------------------
# Model 1: random_policy — near-uniform play (validates the pipeline)
# ---------------------------------------------------------------------------

HIDDEN = 64

rng = np.random.default_rng(42)
W1_rand = rng.normal(0, 0.01, (768, HIDDEN)).astype(np.float32)
b1_rand = np.zeros(HIDDEN, dtype=np.float32)
W2_rand = rng.normal(0, 0.01, (HIDDEN, 4096)).astype(np.float32)
b2_rand = np.zeros(4096, dtype=np.float32)

rand_model = make_mlp_model("random_policy", W1_rand, b1_rand, W2_rand, b2_rand)
path_rand = "models/random_policy.onnx"
onnx.save(rand_model, path_rand)
print(f"Saved {path_rand}  ({param_count(rand_model):,} parameters)")


# ---------------------------------------------------------------------------
# Model 2: capture_policy — hand-crafted weights that prefer captures
#
# Board tensor layout (input, shape [1, 768]):
#   ch 0..5  : current player's pieces (P,N,B,R,Q,K) at squares 0..63
#   ch 6..11 : opponent's pieces       (P,N,B,R,Q,K) at squares 0..63
#   index = channel * 64 + square
#
# Policy tensor layout (output, shape [1, 4096]):
#   index = from_square * 64 + to_square
#
# Strategy: compute, for each square, the value of opponent piece there.
#   opp_value[sq] = sum_p piece_value[p] * input[(6+p)*64 + sq]
# Then set logit[f*64+t] = opp_value[t] + small noise so the model
# prefers moving TO squares occupied by valuable opponent pieces.
#
# We implement this through W1/W2:
#   W1: [768, 64] — maps input → opp_value[sq] for sq in 0..63
#   W2: [64, 4096] — maps opp_value[to] → logit[f*64+to] for all f
# ---------------------------------------------------------------------------

PIECE_VALUES = np.array([1, 3, 3, 5, 9, 0], dtype=np.float32)  # P N B R Q K

# W1: shape [768, 64]
# W1[(6+p)*64 + sq, sq] = piece_value[p]  for p in 0..6, sq in 0..64
W1_cap = np.zeros((768, 64), dtype=np.float32)
for p, val in enumerate(PIECE_VALUES):
    for sq in range(64):
        W1_cap[(6 + p) * 64 + sq, sq] = val

b1_cap = np.zeros(64, dtype=np.float32)

# W2: shape [64, 4096]
# W2[to_sq, f*64+to_sq] = 1.0  for all f in 0..64
# This "tiles" the opp_value vector across all from-squares
W2_cap = np.zeros((64, 4096), dtype=np.float32)
for to_sq in range(64):
    for f in range(64):
        W2_cap[to_sq, f * 64 + to_sq] = 1.0

b2_cap = np.zeros(4096, dtype=np.float32)
# Small position bonus: prefer e4/e5/d4/d5 as destination squares (central control)
center_squares = [27, 28, 35, 36]  # d4, e4, d5, e5
for f in range(64):
    for sq in center_squares:
        b2_cap[f * 64 + sq] += 0.2

cap_model = make_mlp_model("capture_policy", W1_cap, b1_cap, W2_cap, b2_cap)
path_cap = "models/capture_policy.onnx"
onnx.save(cap_model, path_cap)
print(f"Saved {path_cap}  ({param_count(cap_model):,} parameters)")

print()
print("Done. Run competition with:")
print("  cargo run -p cli --bin compete -- models/random_policy.onnx")
print("  cargo run -p cli --bin compete -- models/capture_policy.onnx")
