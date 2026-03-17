"""Python mirror of engine/src/eval.rs — the handcrafted baseline evaluator.

Used for two purposes:
1. Generate training data labeled with the exact eval the NN must beat
2. Quick local evaluation without needing the Rust binary

The eval is tapered: blends middlegame and endgame PSTs based on material.
"""

import chess
import numpy as np

# Material values (centipawns)
PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}

# Game phase weights
PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
TOTAL_PHASE = 2 * (2 * 1 + 2 * 1 + 2 * 2 + 4)  # 24

# PSTs indexed rank 8 at index 0 (white perspective)
PAWN_MG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
PAWN_EG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10,
     0,  0,  0,  0,  0,  0,  0,  0,
]
KNIGHT_MG = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
KNIGHT_EG = KNIGHT_MG  # same in eval.rs
BISHOP_MG = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
BISHOP_EG = BISHOP_MG
ROOK_MG = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]
ROOK_EG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    10, 15, 15, 15, 15, 15, 15, 10,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]
QUEEN_MG = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]
QUEEN_EG = QUEEN_MG
KING_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]
KING_EG = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

PST_MG = {
    chess.PAWN: PAWN_MG, chess.KNIGHT: KNIGHT_MG, chess.BISHOP: BISHOP_MG,
    chess.ROOK: ROOK_MG, chess.QUEEN: QUEEN_MG, chess.KING: KING_MG,
}
PST_EG = {
    chess.PAWN: PAWN_EG, chess.KNIGHT: KNIGHT_EG, chess.BISHOP: BISHOP_EG,
    chess.ROOK: ROOK_EG, chess.QUEEN: QUEEN_EG, chess.KING: KING_EG,
}


def pst_index(sq: int, color: bool) -> int:
    """PST index: rank 8 at index 0, from white's perspective."""
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    rank_from_top = (7 - rank) if color == chess.WHITE else rank
    return rank_from_top * 8 + file


def game_phase(board: chess.Board) -> int:
    """Game phase: 256 = middlegame, 0 = endgame."""
    phase = 0
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type, weight in PHASE_WEIGHTS.items():
            phase += len(board.pieces(piece_type, color)) * weight
    return (phase * 256 + TOTAL_PHASE // 2) // TOTAL_PHASE


def evaluate(board: chess.Board) -> float:
    """Evaluate position from STM perspective. Mirrors eval.rs evaluate()."""
    side = board.turn
    phase = game_phase(board)

    mg_score = 0
    eg_score = 0

    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == side else -1

        for piece_type in chess.PIECE_TYPES:
            if piece_type == chess.KING:
                pt = chess.KING
            else:
                pt = piece_type

            for sq in board.pieces(pt, color):
                mat = PIECE_VALUES.get(pt, 0)
                idx = pst_index(sq, color)
                mg_pst = PST_MG.get(pt, [0]*64)[idx]
                eg_pst = PST_EG.get(pt, [0]*64)[idx]
                mg_score += sign * (mat + mg_pst)
                eg_score += sign * (mat + eg_pst)

    # Tapered eval
    return (mg_score * phase + eg_score * (256 - phase)) / 256


def generate_baseline_data(
    num_positions: int = 500_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate positions from random games, label with baseline eval."""
    from .encoding import board_to_tensor

    import random
    rng = random.Random(seed)

    positions = []
    evals = []

    while len(positions) < num_positions:
        board = chess.Board()
        # Play random game
        for _ in range(rng.randint(5, 80)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))

        if board.is_game_over() or len(board.piece_map()) < 4:
            continue

        ev = evaluate(board)
        if abs(ev) > 5000:
            continue

        positions.append(board_to_tensor(board))
        evals.append(ev)

        if len(positions) % 50000 == 0:
            print(f"  Generated {len(positions)}/{num_positions}...")

    return (
        np.array(positions[:num_positions], dtype=np.float32),
        np.array(evals[:num_positions], dtype=np.float32),
    )


if __name__ == "__main__":
    # Quick test
    board = chess.Board()
    ev = evaluate(board)
    print(f"Starting position eval: {ev:.0f} cp (should be ~0)")

    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    ev = evaluate(board)
    print(f"After 1.e4 (black to move): {ev:.0f} cp")
