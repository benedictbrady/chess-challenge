"""Generate training positions with Stockfish evaluations.

Strategy:
1. Play random legal moves from startpos for 8-80 half-moves → diverse positions
2. Evaluate each with Stockfish at configurable depth
3. Filter out trivial positions (|eval| > 10000cp, < 4 pieces)
4. Optionally filter for "quiet" positions (best move is not a capture)
5. Save as compressed .npz: positions [N, 1536] + evals [N] (centipawns, side-to-move)
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
from tqdm import tqdm

from .encoding import board_to_tensor_dual as board_to_tensor


def random_position(min_ply: int = 8, max_ply: int = 80) -> chess.Board | None:
    """Generate a random position by playing random moves from startpos."""
    board = chess.Board()
    num_plies = random.randint(min_ply, max_ply)

    for _ in range(num_plies):
        legal = list(board.legal_moves)
        if not legal:
            return None  # game ended
        board.push(random.choice(legal))
        if board.is_game_over():
            return None

    return board


def evaluate_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int = 10,
) -> float | None:
    """Evaluate a position with Stockfish. Returns centipawns from side-to-move perspective."""
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].pov(board.turn)
        cp = score.score(mate_score=15000)
        if cp is None:
            return None
        return float(cp)
    except Exception:
        return None


def is_quiet_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int = 10,
) -> bool:
    """Check if a position is quiet (best move is not a capture).

    Quiet positions have more stable evals and are better for training
    positional understanding. Filtering out tactical positions (where the
    best move is a capture) removes noisy labels where the eval depends
    on seeing a specific tactical sequence.
    """
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        best_move = info.get("pv", [None])[0]
        if best_move is None:
            return False
        return not board.is_capture(best_move)
    except Exception:
        return False


def _worker_generate(args: tuple) -> tuple[list, list, list]:
    """Worker function for parallel data generation."""
    worker_id, num_positions, stockfish_path, depth, min_ply, max_ply, max_eval, min_pieces, quiet_only, eval_engine = args

    use_simple = eval_engine == "simple"
    use_baseline = eval_engine == "baseline"
    use_search = eval_engine.startswith("search_d") if eval_engine else False
    if use_simple:
        from .simple_eval import evaluate as simple_evaluate
        engine = None
    elif use_baseline:
        from .baseline_eval import evaluate as baseline_evaluate
        engine = None
    elif use_search:
        from .search_eval import search_evaluate
        search_depth = int(eval_engine.split("_d")[1])  # "search_d2" -> 2
        engine = None
    else:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": 1, "Hash": 16})

    positions = []
    evals_list = []
    fens_list = []
    skipped_tactical = 0

    while len(positions) < num_positions:
        board = random_position(min_ply, max_ply)
        if board is None:
            continue
        if len(board.piece_map()) < min_pieces:
            continue

        if use_simple:
            cp = float(simple_evaluate(board))
        elif use_baseline:
            cp = float(baseline_evaluate(board))
        elif use_search:
            cp = float(search_evaluate(board, search_depth))
        elif quiet_only:
            # Combined: get eval + check if quiet in one analysis call
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
                score = info["score"].pov(board.turn)
                cp = score.score(mate_score=15000)
                if cp is None:
                    continue
                cp = float(cp)
                best_move = info.get("pv", [None])[0]
                if best_move is None:
                    continue
                if board.is_capture(best_move):
                    skipped_tactical += 1
                    continue
            except Exception:
                continue
        else:
            cp = evaluate_position(engine, board, depth=depth)
            if cp is None:
                continue

        if abs(cp) > max_eval:
            continue

        tensor = board_to_tensor(board)
        positions.append(tensor)
        evals_list.append(cp)
        fens_list.append(board.fen())

    if engine is not None:
        engine.quit()
    if quiet_only and not use_simple and not use_baseline and not use_search:
        print(f"Worker {worker_id}: kept {len(positions)}, skipped {skipped_tactical} tactical positions")
    return positions, evals_list, fens_list


def generate_positions(
    num_positions: int,
    stockfish_path: str = "stockfish",
    depth: int = 10,
    min_ply: int = 8,
    max_ply: int = 80,
    max_eval: int = 10000,
    min_pieces: int = 4,
    num_threads: int = 1,
    hash_mb: int = 64,
    quiet_only: bool = False,
    eval_engine: str = "stockfish",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate positions and evaluate them.

    Uses multiple parallel processes for speed.

    Args:
        quiet_only: If True, only keep positions where the best move is not a capture.
            This filters out tactical positions for cleaner training labels.
        eval_engine: "stockfish", "simple", or "baseline".
            simple: material+PST only (no Stockfish needed, pipeline validation).
            baseline: full port of engine's eval.rs (no Stockfish needed, distillation).

    Returns:
        positions: [N, 1536] float32 array of encoded boards
        evals: [N] float32 array of centipawn evaluations (side-to-move perspective)
        fens: list of FEN strings for debugging
    """
    import multiprocessing as mp

    num_workers = max(num_threads, 1)
    per_worker = (num_positions + num_workers - 1) // num_workers

    engine_str = f" (eval: {eval_engine})"
    quiet_str = " (quiet only)" if quiet_only else ""
    print(f"Generating {num_positions} positions{quiet_str}{engine_str} with {num_workers} workers...")

    worker_args = [
        (i, per_worker, stockfish_path, depth, min_ply, max_ply, max_eval, min_pieces, quiet_only, eval_engine)
        for i in range(num_workers)
    ]

    if num_workers == 1:
        # Single-threaded: run directly (simpler for debugging)
        all_results = [_worker_generate(worker_args[0])]
    else:
        with mp.Pool(num_workers) as pool:
            all_results = pool.map(_worker_generate, worker_args)

    # Merge results
    positions = []
    evals_list = []
    fens_list = []
    for pos, ev, fn in all_results:
        positions.extend(pos)
        evals_list.extend(ev)
        fens_list.extend(fn)

    # Trim to exact count
    positions = positions[:num_positions]
    evals_list = evals_list[:num_positions]
    fens_list = fens_list[:num_positions]

    print(f"Generated {len(positions)} positions")

    return (
        np.array(positions, dtype=np.float32),
        np.array(evals_list, dtype=np.float32),
        fens_list,
    )


def save_dataset(
    output_dir: str,
    positions: np.ndarray,
    evals: np.ndarray,
    fens: list[str],
    metadata: dict,
):
    """Save generated dataset to disk."""
    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(output_dir, "data.npz"),
        positions=positions,
        evals=evals,
    )

    # Save FENs separately (useful for debugging but can be large)
    with open(os.path.join(output_dir, "fens.txt"), "w") as f:
        for fen in fens:
            f.write(fen + "\n")

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved dataset to {output_dir}")
    print(f"  positions: {positions.shape}")
    print(f"  evals: {evals.shape}")
    print(f"  eval range: [{evals.min():.0f}, {evals.max():.0f}] cp")
    print(f"  eval mean: {evals.mean():.1f} cp, std: {evals.std():.1f} cp")


def _worker_generate_dual(args: tuple) -> tuple[list, list, list, list]:
    """Worker that computes BOTH baseline and SF evals per position.

    Returns: (positions, baseline_evals, sf_evals, fens)
    """
    worker_id, num_positions, stockfish_path, depth, min_ply, max_ply, max_eval, min_pieces = args

    from .baseline_eval import evaluate as baseline_evaluate

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 16})

    positions = []
    baseline_evals = []
    sf_evals = []
    fens_list = []

    while len(positions) < num_positions:
        board = random_position(min_ply, max_ply)
        if board is None:
            continue
        if len(board.piece_map()) < min_pieces:
            continue

        # Baseline eval (fast, pure Python)
        bl_cp = float(baseline_evaluate(board))
        if abs(bl_cp) > max_eval:
            continue

        # SF eval
        sf_cp = evaluate_position(engine, board, depth=depth)
        if sf_cp is None:
            continue
        if abs(sf_cp) > max_eval:
            continue

        tensor = board_to_tensor(board)
        positions.append(tensor)
        baseline_evals.append(bl_cp)
        sf_evals.append(sf_cp)
        fens_list.append(board.fen())

    engine.quit()
    print(f"Worker {worker_id}: generated {len(positions)} dual-labeled positions")
    return positions, baseline_evals, sf_evals, fens_list


def generate_dual_positions(
    num_positions: int,
    stockfish_path: str = "stockfish",
    depth: int = 8,
    min_ply: int = 8,
    max_ply: int = 80,
    max_eval: int = 10000,
    min_pieces: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate positions with both baseline and SF labels.

    Returns: (positions [N,1536], baseline_evals [N], sf_evals [N])
    """
    args = (0, num_positions, stockfish_path, depth, min_ply, max_ply, max_eval, min_pieces)
    positions, bl_evals, sf_evals, _fens = _worker_generate_dual(args)

    return (
        np.array(positions, dtype=np.float32),
        np.array(bl_evals, dtype=np.float32),
        np.array(sf_evals, dtype=np.float32),
    )


def generate_move_pairs(
    num_pairs: int,
    min_ply: int = 8,
    max_ply: int = 80,
    max_eval: int = 10000,
    min_pieces: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate move-pair training data for ranking loss.

    For each random position, enumerate all legal moves, evaluate the
    resulting positions with baseline eval, then store the best and a
    random non-best child as a pair with the eval margin.

    Returns: (pos_good [N,1536], pos_bad [N,1536], margins [N])
    """
    from .baseline_eval import evaluate as baseline_evaluate

    pos_good = []
    pos_bad = []
    margins = []

    while len(pos_good) < num_pairs:
        board = random_position(min_ply, max_ply)
        if board is None:
            continue
        if len(board.piece_map()) < min_pieces:
            continue

        moves = list(board.legal_moves)
        if len(moves) < 2:
            continue

        # Evaluate all children with baseline eval
        children = []
        for move in moves:
            board.push(move)
            # Negate because baseline returns from side-to-move perspective,
            # and after pushing, side-to-move has changed
            ev = -float(baseline_evaluate(board))
            if abs(ev) <= max_eval:
                children.append((board_to_tensor(board), ev))
            board.pop()

        if len(children) < 2:
            continue

        # Find best and pick a random non-best
        children.sort(key=lambda x: x[1], reverse=True)
        best_tensor, best_eval = children[0]
        # Pick a random non-best child
        other_idx = random.randint(1, len(children) - 1)
        other_tensor, other_eval = children[other_idx]

        margin = best_eval - other_eval
        if margin <= 0:
            continue  # tie or measurement issue

        pos_good.append(best_tensor)
        pos_bad.append(other_tensor)
        margins.append(margin)

    return (
        np.array(pos_good[:num_pairs], dtype=np.float32),
        np.array(pos_bad[:num_pairs], dtype=np.float32),
        np.array(margins[:num_pairs], dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate chess training data")
    parser.add_argument("--num-positions", type=int, default=100_000)
    parser.add_argument("--output-dir", type=str, default="data/dataset_v1")
    parser.add_argument("--stockfish-path", type=str, default="stockfish")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--max-ply", type=int, default=80)
    parser.add_argument("--max-eval", type=int, default=10000)
    parser.add_argument("--min-pieces", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--quiet-only", action="store_true",
                        help="Only keep positions where best move is not a capture")
    parser.add_argument("--eval-engine", type=str, default="stockfish",
                        choices=["stockfish", "simple", "baseline"],
                        help="Eval engine: stockfish (default), simple (material+PST), or baseline (full eval.rs port)")
    args = parser.parse_args()

    start = time.time()

    positions, evals, fens = generate_positions(
        num_positions=args.num_positions,
        stockfish_path=args.stockfish_path,
        depth=args.depth,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        max_eval=args.max_eval,
        min_pieces=args.min_pieces,
        num_threads=args.threads,
        hash_mb=args.hash_mb,
        quiet_only=args.quiet_only,
        eval_engine=args.eval_engine,
    )

    metadata = {
        "num_positions": len(positions),
        "depth": args.depth,
        "min_ply": args.min_ply,
        "max_ply": args.max_ply,
        "max_eval": args.max_eval,
        "min_pieces": args.min_pieces,
        "generation_time_seconds": time.time() - start,
    }

    save_dataset(args.output_dir, positions, evals, fens, metadata)


if __name__ == "__main__":
    main()
