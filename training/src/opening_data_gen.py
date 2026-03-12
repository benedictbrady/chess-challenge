"""Generate training positions by playing random continuations from real openings.

Instead of random positions from startpos (which don't resemble real games),
this generates positions by:
1. Starting from curated opening positions (the same ones used in competition)
2. Playing 1-30 random moves from each opening
3. Evaluating with Stockfish

This creates training data that matches the distribution of test positions.
"""

import random
import chess
import chess.engine
import numpy as np
from pathlib import Path
from .encoding import board_to_tensor_dual as board_to_tensor


def load_openings(openings_path: str) -> list[str]:
    """Load FEN strings from openings file."""
    fens = []
    with open(openings_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                fens.append(line)
    return fens


def generate_from_openings(
    num_positions: int,
    openings_path: str,
    stockfish_path: str = "stockfish",
    depth: int = 10,
    min_extra_ply: int = 1,
    max_extra_ply: int = 30,
    max_eval: int = 10000,
    min_pieces: int = 4,
    quiet_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate positions by playing random moves from opening positions.

    Args:
        num_positions: Number of positions to generate
        openings_path: Path to openings.txt file
        depth: Stockfish analysis depth
        min_extra_ply: Minimum random moves to play from opening
        max_extra_ply: Maximum random moves to play from opening
        quiet_only: If True, only keep positions where best move is not a capture

    Returns:
        positions: [N, 768] float32 array
        evals: [N] float32 array (centipawns, side-to-move)
        fens: list of FEN strings
    """
    openings = load_openings(openings_path)
    print(f"Loaded {len(openings)} opening positions")

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 16})

    positions = []
    evals_list = []
    fens_list = []
    skipped_tactical = 0

    while len(positions) < num_positions:
        # Pick a random opening
        fen = random.choice(openings)
        board = chess.Board(fen)

        # Play random moves
        num_extra = random.randint(min_extra_ply, max_extra_ply)
        for _ in range(num_extra):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            board.push(random.choice(legal))

        if board.is_game_over():
            continue
        if len(board.piece_map()) < min_pieces:
            continue

        # Evaluate
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].pov(board.turn)
            cp = score.score(mate_score=15000)
            if cp is None:
                continue
            cp = float(cp)

            if quiet_only:
                best_move = info.get("pv", [None])[0]
                if best_move is None:
                    continue
                if board.is_capture(best_move):
                    skipped_tactical += 1
                    continue
        except Exception:
            continue

        if abs(cp) > max_eval:
            continue

        tensor = board_to_tensor(board)
        positions.append(tensor)
        evals_list.append(cp)
        fens_list.append(board.fen())

        if len(positions) % 5000 == 0:
            print(f"  Generated {len(positions)}/{num_positions} positions...")

    engine.quit()

    if quiet_only:
        print(f"Kept {len(positions)}, skipped {skipped_tactical} tactical positions")

    return (
        np.array(positions, dtype=np.float32),
        np.array(evals_list, dtype=np.float32),
        fens_list,
    )
