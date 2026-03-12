#!/usr/bin/env python3
"""Diagnose an ONNX chess eval model on known positions.

Checks whether the model produces sensible evaluations for positions
where we know the expected direction (not exact value).

Usage:
    python -m scripts.diagnose_model model.onnx
    python -m scripts.diagnose_model /data/models/level1_v2/model.onnx
"""

import argparse
import sys

import chess
import numpy as np
import onnxruntime as ort

# Add parent dir so we can import src.encoding
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src.encoding import board_to_tensor_dual as board_to_tensor


def evaluate_fen(session: ort.InferenceSession, fen: str) -> float:
    """Evaluate a single FEN string. Returns raw model output (centipawns)."""
    board = chess.Board(fen)
    tensor = board_to_tensor(board).reshape(1, 768)
    result = session.run(None, {"board": tensor})
    return float(result[0][0, 0])


def main():
    parser = argparse.ArgumentParser(description="Diagnose ONNX chess model")
    parser.add_argument("model_path", help="Path to .onnx model file")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    session = ort.InferenceSession(args.model_path)
    print(f"  Input: {session.get_inputs()[0].name} {session.get_inputs()[0].shape}")
    print(f"  Output: {session.get_outputs()[0].name} {session.get_outputs()[0].shape}")
    print()

    # Test cases: (description, FEN, expected_direction)
    # expected_direction: "~0" (near zero), "+" (positive/good for side to move),
    #                     "-" (negative/bad for side to move), "> X" (greater than X)
    tests = [
        # 1. Starting position — should be near 0 (equal)
        (
            "Starting position (white to move)",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "~0",
        ),
        # 2. White missing queen — white to move, should be very negative
        (
            "White missing queen (white to move)",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
            "-",
        ),
        # 3. Black missing queen — white to move, should be very positive
        (
            "Black missing queen (white to move)",
            "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "+",
        ),
        # 4. Black missing queen — black to move, should be very negative
        (
            "Black missing queen (black to move)",
            "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "-",
        ),
        # 5. White missing queen — black to move, should be very positive
        (
            "White missing queen (black to move)",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1",
            "+",
        ),
        # 6. After 1.e4 — should score slightly positive for white
        (
            "After 1.e4 (black to move)",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "~0",
        ),
        # 7. White up a rook
        (
            "White up a rook (white to move)",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1",
            "-",  # white is MISSING a rook
        ),
        (
            "Black up a rook (white to move) [white missing rook]",
            "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1",
            "-",  # white is missing a1 rook
        ),
        # 8. Completely won position — white has queen vs lone king
        (
            "Q+K vs K (white to move)",
            "8/8/8/4k3/8/8/8/4K2Q w - - 0 1",
            "+",
        ),
        # 9. Material symmetry check
        (
            "Symmetric — each side has K+R+P",
            "4k3/4r3/4p3/8/8/4P3/4R3/4K3 w - - 0 1",
            "~0",
        ),
    ]

    print("=" * 70)
    print(f"{'Test':<50} {'Eval':>8} {'Expected':>10} {'OK?':>5}")
    print("=" * 70)

    passes = 0
    fails = 0

    for desc, fen, expected in tests:
        val = evaluate_fen(session, fen)

        if expected == "~0":
            ok = abs(val) < 500
        elif expected == "+":
            ok = val > 50
        elif expected == "-":
            ok = val < -50
        else:
            ok = False

        status = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1
        else:
            passes += 1

        print(f"{desc:<50} {val:>8.1f} {expected:>10} {status:>5}")

    print("=" * 70)
    print(f"Results: {passes} passed, {fails} failed out of {len(tests)} tests")
    print()

    # Additional: check raw output range on random binary inputs
    print("Output range check on 100 random board-like inputs:")
    rng = np.random.default_rng(42)
    inputs = np.zeros((100, 768), dtype=np.float32)
    for i in range(100):
        # Simulate ~16 pieces on random squares
        indices = rng.choice(768, size=16, replace=False)
        inputs[i, indices] = 1.0
    outputs = session.run(None, {"board": inputs})[0].flatten()
    print(f"  min={outputs.min():.1f}, max={outputs.max():.1f}, "
          f"mean={outputs.mean():.1f}, std={outputs.std():.1f}")

    # Check if outputs are all the same (collapsed model)
    if outputs.std() < 1.0:
        print("  WARNING: Very low output variance — model may have collapsed!")
    elif outputs.std() < 10.0:
        print("  NOTE: Low output variance — model may not be expressive enough")
    else:
        print("  Output variance looks reasonable")

    # Relative comparison: 1.e4 vs 1.a3 from white's perspective
    print()
    print("Relative move quality (from white's perspective):")
    after_e4 = evaluate_fen(session, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    after_a3 = evaluate_fen(session, "rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq - 0 1")
    # Note: these are from black's perspective (black to move), so negate for white's view
    print(f"  After 1.e4 (black's view): {after_e4:.1f}")
    print(f"  After 1.a3 (black's view): {after_a3:.1f}")
    print(f"  e4 better for white? (black's eval lower): {after_e4 < after_a3} "
          f"(diff: {after_a3 - after_e4:.1f})")


if __name__ == "__main__":
    main()
