"""Simple Python chess evaluation matching the engine's baseline eval.

Material values + piece-square tables (middlegame only for simplicity).
Used for pipeline validation: if the NN can learn this eval, the pipeline works.

Returns centipawns from the side-to-move's perspective.
"""

import chess

# Material values — must match engine/src/eval.rs
MATERIAL = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-square tables from engine/src/eval.rs (middlegame, white's perspective)
# Indexed with rank 8 at index 0 (same as Rust code)
# fmt: off
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
# fmt: on

PST = {
    chess.PAWN: PAWN_MG,
    chess.KNIGHT: KNIGHT_MG,
    chess.BISHOP: BISHOP_MG,
    chess.ROOK: ROOK_MG,
    chess.QUEEN: QUEEN_MG,
    chess.KING: KING_MG,
}


def _pst_index(square: int, color: bool) -> int:
    """Convert a python-chess square to PST index.

    PSTs are stored with rank 8 at index 0 (from white's perspective).
    """
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    # rank_from_top for white: 7 - rank; for black: rank
    rank_from_top = (7 - rank) if color == chess.WHITE else rank
    return rank_from_top * 8 + file


def evaluate(board: chess.Board) -> int:
    """Evaluate a position. Returns centipawns from side-to-move's perspective.

    Simple material + middlegame PST evaluation.
    No mobility, king safety, pawn structure, etc.
    """
    side = board.turn
    score = 0

    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == side else -1
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                           chess.ROOK, chess.QUEEN, chess.KING]:
            for sq in board.pieces(piece_type, color):
                mat = MATERIAL[piece_type]
                pst = PST[piece_type][_pst_index(sq, color)]
                score += sign * (mat + pst)

    return score


if __name__ == "__main__":
    # Self-test with known positions
    tests = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "startpos"),
        ("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "black missing queen"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "after 1.e4"),
        ("rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq - 0 1", "after 1.a3"),
        ("8/8/8/4k3/8/8/8/4K2Q w - - 0 1", "Q+K vs K"),
    ]

    for fen, desc in tests:
        board = chess.Board(fen)
        val = evaluate(board)
        print(f"{desc:30s}  eval = {val:+6d} cp  (side: {'white' if board.turn else 'black'})")
