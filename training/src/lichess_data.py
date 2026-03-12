"""Extract training positions from Lichess game PGNs.

Parses Lichess PGN files (optionally zstd-compressed), extracts positions
from real games, and labels them with Stockfish evaluations.

Output format: same .npz as data_gen.py (positions [N, 768] + evals [N]).
"""

import argparse
import io
import os
import random
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from .encoding import board_to_tensor_dual as board_to_tensor


def extract_positions_from_pgn(
    pgn_path: str,
    max_positions: int = 1_000_000,
    min_elo: int = 1500,
    sample_moves: list[int] | None = None,
    min_pieces: int = 6,
    max_abs_eval: int = 2000,
    seed: int = 42,
) -> list[chess.Board]:
    """Extract positions from a Lichess PGN file.

    Args:
        pgn_path: Path to PGN file (or .pgn.zst for zstd-compressed)
        max_positions: Maximum positions to extract
        min_elo: Minimum average Elo of the two players
        sample_moves: Which move numbers to sample (default: [10, 15, 20, 25, 30])
        min_pieces: Skip positions with fewer pieces
        max_abs_eval: Skip if Lichess eval annotation |eval| > this (if available)
        seed: Random seed

    Returns:
        List of chess.Board objects (positions to label)
    """
    if sample_moves is None:
        sample_moves = [10, 15, 20, 25, 30]

    rng = random.Random(seed)
    positions = []
    games_parsed = 0
    games_used = 0

    # Handle zstd compression
    if pgn_path.endswith(".zst"):
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        raw_fh = open(pgn_path, "rb")
        reader = dctx.stream_reader(raw_fh)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
    else:
        text_stream = open(pgn_path, "r")
        raw_fh = None

    try:
        while len(positions) < max_positions:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break

            games_parsed += 1
            if games_parsed % 10000 == 0:
                print(f"  Parsed {games_parsed} games, extracted {len(positions)} positions...")

            # Filter by Elo
            try:
                white_elo = int(game.headers.get("WhiteElo", "0"))
                black_elo = int(game.headers.get("BlackElo", "0"))
            except ValueError:
                continue
            if (white_elo + black_elo) / 2 < min_elo:
                continue

            # Skip non-standard games
            if game.headers.get("Variant", "Standard") != "Standard":
                continue

            # Walk through moves
            board = game.board()
            move_num = 0
            for node in game.mainline():
                board.push(node.move)
                move_num += 1

                if move_num not in sample_moves:
                    continue

                # Filter: enough pieces
                if len(board.piece_map()) < min_pieces:
                    continue

                # Filter: not game over
                if board.is_game_over():
                    continue

                # Optional: add some randomness — also sample move_num +/- 1
                # to avoid always getting the same ply across games
                jitter = rng.randint(-1, 1)
                effective_move = move_num + jitter
                if effective_move != move_num and effective_move > 0:
                    continue  # skip this sample point, jittered out

                positions.append(board.copy())

            games_used += 1
    finally:
        text_stream.close()
        if raw_fh is not None:
            raw_fh.close()

    print(f"Parsed {games_parsed} games, used {games_used}, extracted {len(positions)} positions")
    return positions[:max_positions]


def label_positions_with_stockfish(
    positions: list[chess.Board],
    stockfish_path: str = "stockfish",
    depth: int = 12,
    threads: int = 1,
    hash_mb: int = 16,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Label positions with Stockfish evaluations.

    Returns:
        positions_array: [N, 768] float32
        evals_array: [N] float32 (centipawns, side-to-move perspective)
        fens: list of FEN strings
    """
    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": threads, "Hash": hash_mb})

    encoded = []
    evals = []
    fens = []

    for i, board in enumerate(positions):
        if i % 1000 == 0 and i > 0:
            print(f"  Labeled {i}/{len(positions)} positions...")

        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].pov(board.turn)
            cp = score.score(mate_score=15000)
            if cp is None:
                continue
        except Exception:
            continue

        encoded.append(board_to_tensor(board))
        evals.append(float(cp))
        fens.append(board.fen())

    engine.quit()

    print(f"Labeled {len(encoded)} / {len(positions)} positions successfully")

    return (
        np.array(encoded, dtype=np.float32),
        np.array(evals, dtype=np.float32),
        fens,
    )


def main():
    parser = argparse.ArgumentParser(description="Extract positions from Lichess PGN")
    parser.add_argument("pgn_path", help="Path to PGN file (or .pgn.zst)")
    parser.add_argument("--output-dir", type=str, default="data/lichess_v1")
    parser.add_argument("--max-positions", type=int, default=1_000_000)
    parser.add_argument("--min-elo", type=int, default=1500)
    parser.add_argument("--sample-moves", type=str, default="10,15,20,25,30",
                        help="Comma-separated move numbers to sample")
    parser.add_argument("--stockfish-path", type=str, default="stockfish")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample_moves = [int(x) for x in args.sample_moves.split(",")]

    start = time.time()

    # Step 1: Extract positions
    print(f"Extracting positions from {args.pgn_path}...")
    positions = extract_positions_from_pgn(
        pgn_path=args.pgn_path,
        max_positions=args.max_positions,
        min_elo=args.min_elo,
        sample_moves=sample_moves,
        seed=args.seed,
    )

    if not positions:
        print("No positions extracted!")
        return

    # Step 2: Label with Stockfish
    print(f"\nLabeling {len(positions)} positions with Stockfish depth {args.depth}...")
    positions_arr, evals_arr, fens = label_positions_with_stockfish(
        positions,
        stockfish_path=args.stockfish_path,
        depth=args.depth,
        threads=args.threads,
    )

    # Step 3: Save
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.output_dir, "data.npz"),
        positions=positions_arr,
        evals=evals_arr,
    )
    with open(os.path.join(args.output_dir, "fens.txt"), "w") as f:
        for fen in fens:
            f.write(fen + "\n")

    elapsed = time.time() - start
    print(f"\nSaved {len(positions_arr)} positions to {args.output_dir}")
    print(f"Eval range: [{evals_arr.min():.0f}, {evals_arr.max():.0f}] cp")
    print(f"Eval mean: {evals_arr.mean():.1f}, std: {evals_arr.std():.1f}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
