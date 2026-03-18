"""Reference board encoding matching engine/src/nn.rs board_to_tensor() — 1540 dual perspective.

This is the Python reference implementation of the board encoding used by
the Rust engine. Use it to verify your training pipeline encodes positions
identically to the harness.

Layout: two 770-element halves = 1540 floats total.

Each half:
  - 768 floats: 12 piece planes x 64 squares
  - 2 floats: castling rights (kingside, queenside)

First 770 (STM perspective):
  - Channels 0-5:  STM's Pawn, Knight, Bishop, Rook, Queen, King
  - Channels 6-11: Opponent's pieces
  - [768]: STM can castle kingside
  - [769]: STM can castle queenside

Last 770 (NSTM perspective):
  - Channels 0-5:  NSTM's Pawn, Knight, Bishop, Rook, Queen, King
  - Channels 6-11: STM's pieces
  - [1538]: NSTM can castle kingside
  - [1539]: NSTM can castle queenside

Square index: file + rank * 8  (a1=0, b1=1, ..., h8=63)
Ranks flipped when perspective color is Black.

Dependencies: python-chess, numpy
"""

import chess
import numpy as np

HALF_SIZE = 770
TENSOR_SIZE = 1540

PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def square_idx(sq: int, flip: bool) -> int:
    """Convert python-chess square to tensor index, optionally flipping ranks."""
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    if flip:
        rank = 7 - rank
    return file + rank * 8


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Encode a chess.Board as a flat [1540] float32 array.

    Exactly replicates engine/src/nn.rs board_to_tensor().
    """
    us = board.turn  # True = White, False = Black
    them = not us

    tensor = np.zeros(TENSOR_SIZE, dtype=np.float32)

    # First half: STM perspective (our pieces ch 0-5, their pieces ch 6-11)
    stm_flip = us == chess.BLACK
    for ch, piece_type in enumerate(PIECE_ORDER):
        for sq in board.pieces(piece_type, us):
            tensor[ch * 64 + square_idx(sq, stm_flip)] = 1.0
        for sq in board.pieces(piece_type, them):
            tensor[(ch + 6) * 64 + square_idx(sq, stm_flip)] = 1.0
    # STM castling rights
    if board.has_kingside_castling_rights(us):
        tensor[768] = 1.0
    if board.has_queenside_castling_rights(us):
        tensor[769] = 1.0

    # Second half: NSTM perspective (their pieces ch 0-5, our pieces ch 6-11)
    nstm_flip = them == chess.BLACK
    for ch, piece_type in enumerate(PIECE_ORDER):
        for sq in board.pieces(piece_type, them):
            tensor[HALF_SIZE + ch * 64 + square_idx(sq, nstm_flip)] = 1.0
        for sq in board.pieces(piece_type, us):
            tensor[HALF_SIZE + (ch + 6) * 64 + square_idx(sq, nstm_flip)] = 1.0
    # NSTM castling rights
    if board.has_kingside_castling_rights(them):
        tensor[HALF_SIZE + 768] = 1.0
    if board.has_queenside_castling_rights(them):
        tensor[HALF_SIZE + 769] = 1.0

    return tensor


def board_to_tensor_batch(boards: list[chess.Board]) -> np.ndarray:
    """Encode a batch of boards as [N, 1540] float32 array."""
    return np.stack([board_to_tensor(b) for b in boards])


if __name__ == "__main__":
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",  # endgame, no castling
    ]

    for fen in test_fens:
        board = chess.Board(fen)
        t = board_to_tensor(board)
        piece_count = len(board.piece_map())
        # Each piece appears twice (once in each half)
        expected_ones = piece_count * 2
        # Add castling rights
        for color in [chess.WHITE, chess.BLACK]:
            if board.has_kingside_castling_rights(color):
                expected_ones += 1
            if board.has_queenside_castling_rights(color):
                expected_ones += 1
        actual_ones = int(t.sum())
        print(f"FEN: {fen}")
        print(f"  pieces: {piece_count}, expected ones: {expected_ones}, actual: {actual_ones}")
        assert actual_ones == expected_ones, f"Mismatch! {actual_ones} != {expected_ones}"
        print(f"  OK")
        print()

    print("All encoding tests passed.")
