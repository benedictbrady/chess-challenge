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

/// Negamax with alpha-beta pruning. Returns score from the perspective of the side to move.
pub fn negamax(board: &Board, depth: u32, mut alpha: i32, beta: i32) -> i32 {
    match board.status() {
        GameStatus::Won => return -MATE_SCORE,
        GameStatus::Drawn => return DRAW_SCORE,
        GameStatus::Ongoing => {}
    }

    if depth == 0 {
        return evaluate(board);
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
