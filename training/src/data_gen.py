"""Data generation: extract positions from Lichess PGNs and label them.

Generates [N, 1540] dual-perspective encoded positions with either:
- Stockfish centipawn evaluations
- Game outcome labels (win/draw/loss)
- Both (for blended training)
"""

import io
import random
import time

import chess
import chess.pgn
import numpy as np

from .encoding import board_to_tensor, TENSOR_SIZE


def is_quiet_position(board: chess.Board) -> bool:
    """Check if position is 'quiet' — no captures available on obvious hanging pieces.

    Based on research: training on quiet positions is critical (+100 Elo).
    Skip positions where the best move is a capture.
    """
    if board.is_check():
        return False

    # Check if there are obvious capturing advantages
    # Simple heuristic: if there's a capture of a higher-value piece by a lower-value one
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    for move in board.legal_moves:
        if board.is_capture(move):
            # Check if it's a very unbalanced capture
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                v_val = piece_values.get(victim.piece_type, 0)
                a_val = piece_values.get(attacker.piece_type, 0)
                if v_val >= a_val + 2:  # free piece capture
                    return False
    return True


def extract_positions_from_pgn(
    pgn_path: str,
    max_positions: int = 1_000_000,
    min_elo: int = 1800,
    sample_every_n: int = 4,
    min_ply: int = 10,
    max_ply: int = 200,
    min_pieces: int = 6,
    seed: int = 42,
    include_outcomes: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract positions from Lichess PGN file.

    Returns:
        positions: [N, 1540] float32
        outcomes: [N] float32 (1.0=STM wins, 0.0=STM loses, 0.5=draw) or None
    """
    rng = random.Random(seed)
    positions_list = []
    outcomes_list = []
    games_parsed = 0

    # Handle zstd compression
    if pgn_path.endswith(".zst"):
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        raw_fh = open(pgn_path, "rb")
        reader = dctx.stream_reader(raw_fh)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    else:
        text_stream = open(pgn_path, "r")
        raw_fh = None

    t0 = time.time()
    try:
        while len(positions_list) < max_positions:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break

            games_parsed += 1
            if games_parsed % 10000 == 0:
                elapsed = time.time() - t0
                print(f"  {games_parsed} games, {len(positions_list)} positions, {elapsed:.0f}s")

            # Filter by Elo
            try:
                w_elo = int(game.headers.get("WhiteElo", "0"))
                b_elo = int(game.headers.get("BlackElo", "0"))
            except ValueError:
                continue
            if (w_elo + b_elo) / 2 < min_elo:
                continue

            # Skip variants
            if game.headers.get("Variant", "Standard") != "Standard":
                continue

            # Determine game outcome
            result = game.headers.get("Result", "*")
            if result == "1-0":
                white_outcome = 1.0
            elif result == "0-1":
                white_outcome = 0.0
            elif result == "1/2-1/2":
                white_outcome = 0.5
            else:
                continue  # skip unfinished games

            # Walk through moves, sample positions
            board = game.board()
            ply = 0
            for node in game.mainline():
                board.push(node.move)
                ply += 1

                if ply < min_ply or ply > max_ply:
                    continue
                if ply % sample_every_n != 0:
                    continue
                if len(board.piece_map()) < min_pieces:
                    continue
                if board.is_game_over():
                    continue

                # Research-informed filtering: skip non-quiet positions
                if not is_quiet_position(board):
                    continue

                # Small random skip for variety
                if rng.random() < 0.2:
                    continue

                tensor = board_to_tensor(board)
                positions_list.append(tensor)

                if include_outcomes:
                    # Outcome from STM perspective
                    if board.turn == chess.WHITE:
                        outcomes_list.append(white_outcome)
                    else:
                        outcomes_list.append(1.0 - white_outcome)

                if len(positions_list) >= max_positions:
                    break

    finally:
        text_stream.close()
        if raw_fh is not None:
            raw_fh.close()

    elapsed = time.time() - t0
    print(f"Extracted {len(positions_list)} positions from {games_parsed} games in {elapsed:.0f}s")

    positions = np.array(positions_list, dtype=np.float32)
    outcomes = np.array(outcomes_list, dtype=np.float32) if include_outcomes else None

    return positions, outcomes


def label_with_stockfish(
    positions: np.ndarray,
    stockfish_path: str = "stockfish",
    depth: int = 12,
    threads: int = 1,
    hash_mb: int = 16,
) -> np.ndarray:
    """Label 1540-encoded positions with Stockfish centipawn evals.

    Takes encoded positions and reconstructs boards from them for analysis.
    This is slow — use sparingly or in parallel.

    Actually, we should pass boards directly. Let's accept FENs instead.
    """
    raise NotImplementedError("Use label_boards_with_stockfish instead")


def label_boards_with_stockfish(
    boards: list[chess.Board],
    stockfish_path: str = "stockfish",
    depth: int = 12,
    threads: int = 1,
    hash_mb: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Label boards with Stockfish evals and encode them.

    Returns:
        positions: [N, 1540] float32
        evals: [N] float32 (centipawns, STM perspective)
    """
    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": threads, "Hash": hash_mb})

    encoded = []
    evals = []

    for i, board in enumerate(boards):
        if i % 1000 == 0 and i > 0:
            print(f"  Labeled {i}/{len(boards)}...")

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

    engine.quit()

    return (
        np.array(encoded, dtype=np.float32),
        np.array(evals, dtype=np.float32),
    )


def generate_random_positions(
    num_positions: int = 100_000,
    stockfish_path: str = "stockfish",
    depth: int = 8,
    min_ply: int = 8,
    max_ply: int = 80,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate positions by playing random games and labeling with Stockfish.

    Quick way to get data without Lichess PGNs.
    """
    import chess.engine

    rng = random.Random(seed)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 16})

    positions = []
    evals = []

    while len(positions) < num_positions:
        board = chess.Board()
        ply = 0

        while not board.is_game_over() and ply < max_ply:
            moves = list(board.legal_moves)
            if not moves:
                break

            # Mix of random and engine moves for variety
            if rng.random() < 0.3:
                # Engine move (weak)
                result = engine.play(board, chess.engine.Limit(depth=2))
                board.push(result.move)
            else:
                board.push(rng.choice(moves))
            ply += 1

            if ply < min_ply:
                continue
            if board.is_game_over():
                break
            if len(board.piece_map()) < 6:
                continue
            if rng.random() > 0.3:  # sample ~30% of positions
                continue

            try:
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
                score = info["score"].pov(board.turn)
                cp = score.score(mate_score=15000)
                if cp is None:
                    continue
            except Exception:
                continue

            positions.append(board_to_tensor(board))
            evals.append(float(cp))

            if len(positions) % 1000 == 0:
                print(f"  Generated {len(positions)}/{num_positions} positions...")

    engine.quit()

    return (
        np.array(positions[:num_positions], dtype=np.float32),
        np.array(evals[:num_positions], dtype=np.float32),
    )
