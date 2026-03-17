"""Quick pipeline test: generate data with heuristic eval, train tiny model, export ONNX.

This validates the full pipeline locally without needing Modal or Stockfish.
Run from the training/ directory:
  python3 quick_test.py
"""

import sys
import time
import random
import numpy as np
import chess
import torch

sys.path.insert(0, ".")
from src.encoding import board_to_tensor, TENSOR_SIZE
from src.model import build_model
from src.export_onnx import export_onnx, validate_onnx


# ── Simple heuristic eval (material + basic PST) ─────────────────────────────

PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}

# Simple center bonus for pieces
CENTER_BONUS = {}
for sq in chess.SQUARES:
    file, rank = chess.square_file(sq), chess.square_rank(sq)
    # Distance from center (3.5, 3.5)
    d = abs(file - 3.5) + abs(rank - 3.5)
    CENTER_BONUS[sq] = int(max(0, 7 - d) * 3)


def heuristic_eval(board: chess.Board) -> float:
    """Simple material + center control eval (centipawns, STM perspective)."""
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        val = PIECE_VALUES[piece.piece_type] + CENTER_BONUS[sq]
        if piece.color == board.turn:
            score += val
        else:
            score -= val

    # Mobility bonus
    our_moves = len(list(board.legal_moves))
    board.push(chess.Move.null())
    their_moves = len(list(board.legal_moves))
    board.pop()
    score += (our_moves - their_moves) * 5

    return float(score)


# ── Generate positions ────────────────────────────────────────────────────────

def generate_positions(n: int = 50_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Play random games, label with heuristic eval."""
    rng = random.Random(seed)
    positions = []
    evals = []

    while len(positions) < n:
        board = chess.Board()
        for _ in range(rng.randint(5, 60)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))

        if board.is_game_over() or len(board.piece_map()) < 4:
            continue

        ev = heuristic_eval(board)
        if abs(ev) > 5000:  # skip extreme
            continue

        positions.append(board_to_tensor(board))
        evals.append(ev)

        if len(positions) % 10000 == 0:
            print(f"  Generated {len(positions)}/{n}...")

    return np.array(positions, dtype=np.float32), np.array(evals, dtype=np.float32)


# ── Main pipeline test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Quick Pipeline Test ===\n")

    # 1. Generate data
    print("Step 1: Generating 50K positions with heuristic eval...")
    t0 = time.time()
    positions, evals = generate_positions(50_000)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Positions shape: {positions.shape}")
    print(f"  Eval range: [{evals.min():.0f}, {evals.max():.0f}] cp")
    print(f"  Eval mean: {evals.mean():.1f}, std: {evals.std():.1f}")

    # Save
    np.savez_compressed("/tmp/quick_test_data.npz", positions=positions, evals=evals)

    # 2. Train tiny model
    print("\nStep 2: Training nnue_256x2_16 for 10 epochs...")
    from src.train import train

    config = {
        "data_path": "/tmp/quick_test_data.npz",
        "architecture": "nnue_256x2_16",
        "epochs": 10,
        "batch_size": 4096,
        "learning_rate": 1e-3,
        "loss_type": "sigmoid_mse",
        "eval_scale": 400.0,
        "checkpoint_dir": "/tmp/quick_test_ckpts",
        "num_workers": 0,
        "warmup_epochs": 1,
    }

    model, val_loss = train(config)
    print(f"  Final val loss: {val_loss:.6f}")

    # 3. Export ONNX
    print("\nStep 3: Exporting to ONNX...")
    onnx_path = "/tmp/quick_test_model.onnx"

    # Load best checkpoint
    state = torch.load("/tmp/quick_test_ckpts/best.pt", map_location="cpu", weights_only=False)
    model = build_model("nnue_256x2_16")
    model.load_state_dict(state["model_state_dict"])

    export_onnx(model, onnx_path)
    ok = validate_onnx(onnx_path)

    if ok:
        print(f"\n=== Pipeline test PASSED ===")
        print(f"Model at: {onnx_path}")
        print(f"Params: {model.num_params:,}")
        print(f"Run: cargo run -p cli --bin compete -- {onnx_path} --level 1")
    else:
        print(f"\n=== Pipeline test FAILED ===")
