"""Board encoding that exactly matches engine/src/nn.rs board_to_tensor().

Layout: 12 planes x 64 squares = 768 floats.
- Channels 0-5:  current player's Pawn, Knight, Bishop, Rook, Queen, King
- Channels 6-11: opponent's Pawn, Knight, Bishop, Rook, Queen, King

Square index: file + rank * 8  (a1=0, b1=1, ..., h8=63)
When Black to move, ranks are flipped (7 - rank) so the current player
always sees their pieces advancing "up" from rank 0.
"""

import chess
import numpy as np

# Piece channel order — must match nn.rs PIECE_TYPES
PIECE_ORDER = [
    chess.PAWN,    # channel 0 / 6
    chess.KNIGHT,  # channel 1 / 7
    chess.BISHOP,  # channel 2 / 8
    chess.ROOK,    # channel 3 / 9
    chess.QUEEN,   # channel 4 / 10
    chess.KING,    # channel 5 / 11
]


def square_idx(sq: int, flip: bool) -> int:
    """Convert a python-chess square to our tensor index.

    python-chess: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
    This matches cozy_chess exactly, so no re-mapping needed.
    """
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    if flip:
        rank = 7 - rank
    return file + rank * 8


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Encode a chess.Board as a flat [768] float32 array.

    Exactly replicates engine/src/nn.rs board_to_tensor().
    """
    us = board.turn  # True = White, False = Black
    them = not us
    flip = us == chess.BLACK

    tensor = np.zeros(768, dtype=np.float32)

    for ch, piece_type in enumerate(PIECE_ORDER):
        # Current player's pieces → channels 0-5
        for sq in board.pieces(piece_type, us):
            tensor[ch * 64 + square_idx(sq, flip)] = 1.0
        # Opponent's pieces → channels 6-11
        for sq in board.pieces(piece_type, them):
            tensor[(ch + 6) * 64 + square_idx(sq, flip)] = 1.0

    return tensor


def board_to_tensor_batch(boards: list[chess.Board]) -> np.ndarray:
    """Encode a batch of boards as [N, 768] float32 array."""
    return np.stack([board_to_tensor(b) for b in boards])


def board_to_tensor_dual(board: chess.Board) -> np.ndarray:
    """Encode a board as a flat [1536] float32 array (dual perspective).

    Exactly replicates engine/src/nn.rs board_to_tensor() dual-perspective layout.

    First 768: STM perspective (STM pieces ch 0-5, NSTM pieces ch 6-11)
    Last 768:  NSTM perspective (NSTM pieces ch 0-5, STM pieces ch 6-11)

    Both halves see all 12 piece channels from their respective viewpoint.
    Ranks are flipped when the perspective's side is Black.
    """
    us = board.turn  # True = White, False = Black
    them = not us

    tensor = np.zeros(1536, dtype=np.float32)

    # First 768: STM perspective
    stm_flip = us == chess.BLACK
    for ch, piece_type in enumerate(PIECE_ORDER):
        for sq in board.pieces(piece_type, us):
            tensor[ch * 64 + square_idx(sq, stm_flip)] = 1.0
        for sq in board.pieces(piece_type, them):
            tensor[(ch + 6) * 64 + square_idx(sq, stm_flip)] = 1.0

    # Last 768: NSTM perspective
    nstm_flip = them == chess.BLACK
    for ch, piece_type in enumerate(PIECE_ORDER):
        for sq in board.pieces(piece_type, them):
            tensor[768 + ch * 64 + square_idx(sq, nstm_flip)] = 1.0
        for sq in board.pieces(piece_type, us):
            tensor[768 + (ch + 6) * 64 + square_idx(sq, nstm_flip)] = 1.0

    return tensor


def board_to_tensor_dual_batch(boards: list[chess.Board]) -> np.ndarray:
    """Encode a batch of boards as [N, 1536] float32 array (dual perspective)."""
    return np.stack([board_to_tensor_dual(b) for b in boards])


if __name__ == "__main__":
    # Quick self-test: encode starting position and a known FEN
    import sys

    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
        "8/4p3/4k3/8/8/4K3/8/8 b - - 0 1",
    ]

    for fen in test_fens:
        board = chess.Board(fen)
        t = board_to_tensor(board)
        ones = int(t.sum())
        print(f"FEN: {fen}")
        print(f"  pieces on board: {len(board.piece_map())}, ones in tensor: {ones}")
        assert ones == len(board.piece_map()), "Mismatch!"
        print(f"  non-zero indices: {np.nonzero(t)[0].tolist()}")
        print()

    print("All encoding tests passed.")
