use cozy_chess::{Board, GameStatus, Move};

use crate::eval::evaluate;

const MATE_SCORE: i32 = 100_000;
const DRAW_SCORE: i32 = 0;

/// Returns moves with captures first for better alpha-beta cutoffs.
fn ordered_moves(board: &Board) -> Vec<Move> {
    let mut captures = Vec::new();
    let mut quiets = Vec::new();

    board.generate_moves(|piece_moves| {
        for mv in piece_moves {
            if board.piece_on(mv.to).is_some() {
                captures.push(mv);
            } else {
                quiets.push(mv);
            }
        }
        false
    });

    captures.extend(quiets);
    captures
}

/// Returns only capture moves, ordered by MVV-LVA (most valuable victim first).
fn capture_moves(board: &Board) -> Vec<Move> {
    let mut captures = Vec::new();

    board.generate_moves(|piece_moves| {
        for mv in piece_moves {
            if board.piece_on(mv.to).is_some() {
                captures.push(mv);
            }
        }
        false
    });

    // MVV-LVA: sort by victim value descending, then attacker value ascending
    captures.sort_by(|a, b| {
        let val = |mv: &Move| {
            let victim = piece_val(board.piece_on(mv.to).unwrap());
            let attacker = piece_val(board.piece_on(mv.from).unwrap());
            (victim, std::cmp::Reverse(attacker))
        };
        val(b).cmp(&val(a))
    });

    captures
}

fn piece_val(p: cozy_chess::Piece) -> i32 {
    use cozy_chess::Piece::*;
    match p {
        Pawn => 1,
        Knight => 3,
        Bishop => 3,
        Rook => 5,
        Queen => 9,
        King => 100,
    }
}

/// Quiescence search: only explore captures until the position is quiet.
fn quiescence(board: &Board, mut alpha: i32, beta: i32) -> i32 {
    match board.status() {
        GameStatus::Won => return -MATE_SCORE,
        GameStatus::Drawn => return DRAW_SCORE,
        GameStatus::Ongoing => {}
    }

    // Stand-pat: the side to move can choose not to capture
    let stand_pat = evaluate(board);
    if stand_pat >= beta {
        return beta;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    for mv in capture_moves(board) {
        let mut child = board.clone();
        child.play_unchecked(mv);
        let score = -quiescence(&child, -beta, -alpha);
        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

/// Negamax with alpha-beta pruning. Returns score from the perspective of the side to move.
pub fn negamax(board: &Board, depth: u32, mut alpha: i32, beta: i32) -> i32 {
    match board.status() {
        GameStatus::Won => return -MATE_SCORE,
        GameStatus::Drawn => return DRAW_SCORE,
        GameStatus::Ongoing => {}
    }

    if depth == 0 {
        return quiescence(board, alpha, beta);
    }

    let moves = ordered_moves(board);

    for mv in moves {
        let mut child = board.clone();
        child.play_unchecked(mv);
        let score = -negamax(&child, depth - 1, -beta, -alpha);
        if score >= beta {
            return beta; // beta cutoff
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

/// Find the best move and its score at the given depth.
pub fn best_move_with_scores(board: &Board, depth: u32) -> Vec<(Move, i32)> {
    let moves = ordered_moves(board);
    let mut results = Vec::new();

    for mv in moves {
        let mut child = board.clone();
        child.play_unchecked(mv);
        let score = -negamax(&child, depth - 1, -MATE_SCORE, MATE_SCORE);
        results.push((mv, score));
    }

    results
}
