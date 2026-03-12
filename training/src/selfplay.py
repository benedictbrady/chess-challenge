"""Self-play game engine for generating game-outcome training data.

Plays games using depth-1 + quiescence search (matching the competition format).
Supports baseline eval and NN eval (via ONNX Runtime) as eval functions.
Records all positions + game outcomes for training.
"""

import chess
import numpy as np
import onnxruntime as ort

from .baseline_eval import evaluate as baseline_eval
from .encoding import board_to_tensor_dual as board_to_tensor


MATE_SCORE = 30000
MAX_PLIES = 500


def quiescence(board: chess.Board, alpha: int, beta: int, eval_fn, max_depth: int = 8) -> int:
    """Quiescence search with pluggable eval function."""
    stand_pat = eval_fn(board)

    if max_depth <= 0:
        return stand_pat
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    captures = []
    for move in board.legal_moves:
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square) or chess.PAWN
            attacker = board.piece_type_at(move.from_square) or chess.PAWN
            captures.append((victim * 10 - attacker, move))
    captures.sort(key=lambda x: -x[0])

    for _, move in captures:
        board.push(move)
        score = -quiescence(board, -beta, -alpha, eval_fn, max_depth - 1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def depth1_search(board: chess.Board, eval_fn) -> tuple[chess.Move | None, int]:
    """Depth-1 + quiescence search. Returns (best_move, score)."""
    if board.is_game_over():
        return None, 0

    best_move = None
    best_score = -MATE_SCORE

    for move in board.legal_moves:
        board.push(move)
        score = -quiescence(board, -MATE_SCORE, MATE_SCORE, eval_fn)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move, best_score


def make_nn_eval(onnx_path: str):
    """Create an eval function from an ONNX model."""
    session = ort.InferenceSession(onnx_path)

    def nn_eval(board: chess.Board) -> int:
        tensor = board_to_tensor(board).reshape(1, 768)
        result = session.run(None, {"board": tensor.astype(np.float32)})
        return int(result[0][0][0])

    return nn_eval


def play_game(
    white_eval_fn,
    black_eval_fn,
    opening_fen: str = chess.STARTING_FEN,
) -> tuple[list[np.ndarray], list[float], float]:
    """Play a complete game with depth-1 + quiescence search.

    Returns:
        positions: list of board tensors (768-dim) for each position visited
        evals: list of baseline eval (centipawns, side-to-move perspective) per position
        outcome: 1.0 = white wins, 0.0 = black wins, 0.5 = draw
    """
    board = chess.Board(opening_fen)
    positions = []
    evals = []

    for _ in range(MAX_PLIES):
        if board.is_game_over():
            break

        # Skip positions in check (noisy for training)
        if not board.is_check():
            positions.append(board_to_tensor(board))
            evals.append(float(baseline_eval(board)))

        # Pick move
        eval_fn = white_eval_fn if board.turn == chess.WHITE else black_eval_fn
        move, _ = depth1_search(board, eval_fn)

        if move is None:
            break
        board.push(move)

    # Determine outcome
    result = board.result(claim_draw=True)
    if result == "1-0":
        outcome = 1.0
    elif result == "0-1":
        outcome = 0.0
    else:
        outcome = 0.5

    return positions, evals, outcome


def play_games_batch(
    nn_onnx_path: str,
    openings: list[str],
    games_per_opening: int = 100,
    nn_plays_white_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Play many games and collect (position, eval, outcome) triples.

    Each position gets BOTH the baseline eval AND the game outcome.
    This enables blended training: target = λ*sigmoid(eval/s) + (1-λ)*outcome.

    Returns:
        positions: [N, 768] float32 array
        evals: [N] float32 array (baseline eval in centipawns, side-to-move perspective)
        outcomes: [N] float32 array (1.0 = side-to-move wins, 0.0 = loses, 0.5 = draw)
    """
    nn_eval = make_nn_eval(nn_onnx_path)
    all_positions = []
    all_evals = []
    all_outcomes = []

    total_games = len(openings) * games_per_opening
    nn_wins = 0
    nn_losses = 0
    draws = 0

    for oi, opening in enumerate(openings):
        for gi in range(games_per_opening):
            # Alternate who plays white
            nn_is_white = gi < int(games_per_opening * nn_plays_white_frac)

            if nn_is_white:
                white_fn, black_fn = nn_eval, baseline_eval
            else:
                white_fn, black_fn = baseline_eval, nn_eval

            positions, evals, outcome = play_game(white_fn, black_fn, opening)

            # Convert outcome to side-to-move perspective for each position
            # Evals are already in side-to-move perspective (from baseline_eval)
            board = chess.Board(opening)
            opening_stm_is_white = board.turn == chess.WHITE
            for i, (pos_tensor, ev) in enumerate(zip(positions, evals)):
                stm_is_white = (i % 2 == 0) if opening_stm_is_white else (i % 2 == 1)
                if stm_is_white:
                    stm_outcome = outcome
                else:
                    stm_outcome = 1.0 - outcome

                all_positions.append(pos_tensor)
                all_evals.append(ev)
                all_outcomes.append(stm_outcome)

            # Track stats
            nn_outcome = outcome if nn_is_white else 1.0 - outcome
            if nn_outcome > 0.7:
                nn_wins += 1
            elif nn_outcome < 0.3:
                nn_losses += 1
            else:
                draws += 1

            game_num = oi * games_per_opening + gi + 1
            if game_num % 100 == 0:
                print(f"  Game {game_num}/{total_games}: NN {nn_wins}W/{draws}D/{nn_losses}L")

    print(f"Final: NN {nn_wins}W/{draws}D/{nn_losses}L out of {total_games} games")
    print(f"Collected {len(all_positions)} positions")

    return (
        np.array(all_positions, dtype=np.float32),
        np.array(all_evals, dtype=np.float32),
        np.array(all_outcomes, dtype=np.float32),
    )
