"""Minimax search using baseline_eval for search distillation labels.

Generates labels that represent "what the baseline eval says at depth N",
so the NN can internalize deeper search knowledge while being evaluated at depth 1.
"""

import chess
from .baseline_eval import evaluate as static_eval

MATE_SCORE = 30000


def quiescence(board: chess.Board, alpha: int, beta: int, max_depth: int = 8) -> int:
    """Quiescence search — follow captures until quiet."""
    stand_pat = static_eval(board)

    if max_depth <= 0:
        return stand_pat

    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    # Generate and sort captures by MVV-LVA
    captures = []
    for move in board.legal_moves:
        if board.is_capture(move):
            # MVV-LVA: prioritize capturing valuable pieces with cheap pieces
            victim = board.piece_type_at(move.to_square) or chess.PAWN
            attacker = board.piece_type_at(move.from_square) or chess.PAWN
            captures.append((victim * 10 - attacker, move))
    captures.sort(key=lambda x: -x[0])  # best MVV-LVA first

    for _, move in captures:
        board.push(move)
        score = -quiescence(board, -beta, -alpha, max_depth - 1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    """Negamax with alpha-beta pruning + quiescence at leaves."""
    if board.is_game_over():
        result = board.result()
        if result == "1/2-1/2":
            return 0
        # Current side lost (opponent checkmated us)
        return -MATE_SCORE

    if depth <= 0:
        return quiescence(board, alpha, beta)

    best = -MATE_SCORE

    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break  # beta cutoff

    return best


def search_evaluate(board: chess.Board, depth: int = 2) -> int:
    """Evaluate a position using minimax search at given depth.

    Returns centipawns from side-to-move perspective.
    """
    return negamax(board, depth, -MATE_SCORE, MATE_SCORE)


if __name__ == "__main__":
    import time

    tests = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "startpos"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "after 1.e4"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "1.e4 e5 2.Nf3 Nc6"),
    ]

    for depth in [1, 2, 3]:
        print(f"\n=== Depth {depth} ===")
        for fen, desc in tests:
            board = chess.Board(fen)
            start = time.time()
            score = search_evaluate(board, depth)
            elapsed = time.time() - start
            static = static_eval(board)
            print(f"  {desc:35s}  search={score:+6d}  static={static:+6d}  diff={score-static:+5d}  ({elapsed:.2f}s)")

    # Benchmark
    import random
    print("\n=== Benchmark: depth 2 ===")
    boards = []
    for _ in range(50):
        b = chess.Board()
        for _ in range(random.randint(8, 30)):
            legal = list(b.legal_moves)
            if not legal or b.is_game_over():
                break
            b.push(random.choice(legal))
        if not b.is_game_over():
            boards.append(b)

    start = time.time()
    for b in boards:
        search_evaluate(b, 2)
    elapsed = time.time() - start
    print(f"  {len(boards)} positions in {elapsed:.1f}s = {len(boards)/elapsed:.1f} pos/s")
