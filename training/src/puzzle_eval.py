"""Lichess puzzle evaluation — measure how well a model can solve tactical puzzles.

Downloads Lichess puzzle database CSV, tests model's ability to find the best move
by comparing eval of correct move vs other legal moves.

Lichess puzzle CSV format: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
- FEN: position before the first move (opponent's last move)
- Moves: space-separated UCI moves. First move is the opponent's move that sets up the puzzle,
  remaining moves are the solution.

Usage:
  python3 -m src.puzzle_eval --model model.onnx --puzzles puzzles.csv --max-puzzles 1000
"""

import csv
import time
from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort

from .encoding import board_to_tensor, TENSOR_SIZE


def load_puzzles(
    csv_path: str,
    max_puzzles: int = 5000,
    min_rating: int = 600,
    max_rating: int = 2500,
    themes_filter: set[str] | None = None,
) -> list[dict]:
    """Load Lichess puzzles from CSV.

    Returns list of dicts with:
      - fen: position after opponent's setup move
      - solution_moves: list of UCI solution moves
      - rating: puzzle rating
      - themes: set of theme strings
    """
    puzzles = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            if len(row) < 8:
                continue

            puzzle_id, fen, moves_str, rating_str = row[0], row[1], row[2], row[3]
            themes = set(row[7].split()) if len(row) > 7 else set()

            try:
                rating = int(rating_str)
            except ValueError:
                continue

            if rating < min_rating or rating > max_rating:
                continue

            if themes_filter and not themes_filter.intersection(themes):
                continue

            moves = moves_str.split()
            if len(moves) < 2:
                continue

            # Apply the setup move to get the puzzle position
            board = chess.Board(fen)
            setup_move = chess.Move.from_uci(moves[0])
            board.push(setup_move)

            puzzles.append({
                "fen": board.fen(),
                "solution_moves": moves[1:],
                "rating": rating,
                "themes": themes,
            })

            if len(puzzles) >= max_puzzles:
                break

    return puzzles


def eval_position(session: ort.InferenceSession, board: chess.Board) -> float:
    """Evaluate a single position with the ONNX model."""
    tensor = board_to_tensor(board).reshape(1, TENSOR_SIZE)
    output = session.run(None, {"board": tensor})
    return float(output[0][0, 0])


def solve_puzzle(session: ort.InferenceSession, puzzle: dict) -> dict:
    """Try to solve a puzzle by picking the move with the best eval.

    Returns dict with:
      - correct: bool
      - best_move: UCI string (model's pick)
      - solution_move: UCI string (correct answer)
      - eval_gap: eval of correct move minus eval of model's pick
    """
    board = chess.Board(puzzle["fen"])
    solution_uci = puzzle["solution_moves"][0]
    solution_move = chess.Move.from_uci(solution_uci)

    # Evaluate all legal moves (depth 1: apply move, evaluate child)
    best_move = None
    best_eval = float("-inf")
    solution_eval = None

    for move in board.legal_moves:
        child = board.copy()
        child.push(move)

        if child.is_checkmate():
            ev = 100000.0
        elif child.is_stalemate() or child.is_insufficient_material():
            ev = 0.0
        else:
            ev = -eval_position(session, child)  # negate: child is opponent's perspective

        if move == solution_move:
            solution_eval = ev

        if ev > best_eval:
            best_eval = ev
            best_move = move

    correct = best_move == solution_move

    return {
        "correct": correct,
        "best_move": best_move.uci() if best_move else "",
        "solution_move": solution_uci,
        "eval_gap": (solution_eval - best_eval) if solution_eval is not None else 0,
        "rating": puzzle["rating"],
        "themes": puzzle["themes"],
    }


def evaluate_on_puzzles(
    model_path: str,
    puzzles: list[dict],
    verbose: bool = True,
) -> dict:
    """Run puzzle evaluation. Returns summary statistics."""
    session = ort.InferenceSession(model_path)

    results = []
    t0 = time.time()

    for i, puzzle in enumerate(puzzles):
        result = solve_puzzle(session, puzzle)
        results.append(result)

        if verbose and (i + 1) % 100 == 0:
            correct_so_far = sum(r["correct"] for r in results)
            print(f"  {i+1}/{len(puzzles)}: {correct_so_far}/{i+1} correct "
                  f"({100*correct_so_far/(i+1):.1f}%)")

    elapsed = time.time() - t0

    # Aggregate
    total = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = correct / total if total > 0 else 0

    # By rating bucket
    buckets = {}
    for r in results:
        rating = r["rating"]
        bucket = (rating // 200) * 200
        if bucket not in buckets:
            buckets[bucket] = {"correct": 0, "total": 0}
        buckets[bucket]["total"] += 1
        if r["correct"]:
            buckets[bucket]["correct"] += 1

    # By theme
    theme_stats = {}
    for r in results:
        for theme in r["themes"]:
            if theme not in theme_stats:
                theme_stats[theme] = {"correct": 0, "total": 0}
            theme_stats[theme]["total"] += 1
            if r["correct"]:
                theme_stats[theme]["correct"] += 1

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed": elapsed,
        "buckets": buckets,
        "theme_stats": theme_stats,
    }

    if verbose:
        print(f"\n=== Puzzle Evaluation ===")
        print(f"Total: {correct}/{total} ({100*accuracy:.1f}%)")
        print(f"Time: {elapsed:.1f}s ({elapsed/total*1000:.1f}ms/puzzle)")
        print(f"\nBy rating:")
        for bucket in sorted(buckets.keys()):
            b = buckets[bucket]
            pct = 100 * b["correct"] / b["total"] if b["total"] > 0 else 0
            print(f"  {bucket}-{bucket+199}: {b['correct']}/{b['total']} ({pct:.1f}%)")
        print(f"\nTop themes:")
        sorted_themes = sorted(theme_stats.items(), key=lambda x: x[1]["total"], reverse=True)
        for theme, stats in sorted_themes[:10]:
            pct = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {theme}: {stats['correct']}/{stats['total']} ({pct:.1f}%)")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--puzzles", required=True, help="Lichess puzzles CSV")
    parser.add_argument("--max-puzzles", type=int, default=1000)
    parser.add_argument("--min-rating", type=int, default=600)
    parser.add_argument("--max-rating", type=int, default=2500)
    args = parser.parse_args()

    puzzles = load_puzzles(args.puzzles, args.max_puzzles, args.min_rating, args.max_rating)
    print(f"Loaded {len(puzzles)} puzzles")

    evaluate_on_puzzles(args.model, puzzles)
