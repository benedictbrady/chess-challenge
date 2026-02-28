use cozy_chess::{Board, GameStatus, Move, Piece};

use crate::eval::evaluate;

const MATE_SCORE: i32 = 100_000;
const DRAW_SCORE: i32 = 0;

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum TTFlag {
    Exact,
    LowerBound,
    UpperBound,
}

#[derive(Clone, Copy)]
struct TTEntry {
    hash: u64,
    depth: u32,
    score: i32,
    flag: TTFlag,
    best_move: Option<Move>,
}

pub struct TTable {
    entries: Vec<Option<TTEntry>>,
    mask: usize,
}

impl TTable {
    pub fn new(power: u32) -> Self {
        let size = 1 << power;
        TTable {
            entries: vec![None; size],
            mask: size - 1,
        }
    }

    fn probe(&self, hash: u64) -> Option<&TTEntry> {
        let idx = hash as usize & self.mask;
        self.entries[idx].as_ref().filter(|e| e.hash == hash)
    }

    fn store(&mut self, hash: u64, depth: u32, score: i32, flag: TTFlag, best_move: Option<Move>) {
        let idx = hash as usize & self.mask;
        self.entries[idx] = Some(TTEntry {
            hash,
            depth,
            score,
            flag,
            best_move,
        });
    }
}

// ---------------------------------------------------------------------------
// Move ordering
// ---------------------------------------------------------------------------

fn piece_val(p: Piece) -> i32 {
    match p {
        Piece::Pawn => 1,
        Piece::Knight => 3,
        Piece::Bishop => 3,
        Piece::Rook => 5,
        Piece::Queen => 9,
        Piece::King => 100,
    }
}

fn move_score(
    board: &Board,
    mv: Move,
    tt_move: Option<Move>,
    killers: &[Option<Move>; 2],
    history: &[[i32; 64]; 64],
) -> i32 {
    if tt_move == Some(mv) {
        return 1_000_000;
    }
    if let Some(victim) = board.piece_on(mv.to) {
        let attacker = board.piece_on(mv.from).unwrap_or(Piece::Pawn);
        return 100_000 + piece_val(victim) * 100 - piece_val(attacker);
    }
    if killers[0] == Some(mv) {
        return 90_000;
    }
    if killers[1] == Some(mv) {
        return 80_000;
    }
    history[mv.from as usize][mv.to as usize]
}

fn sorted_moves(
    board: &Board,
    tt_move: Option<Move>,
    killers: &[Option<Move>; 2],
    history: &[[i32; 64]; 64],
) -> Vec<Move> {
    let mut moves = Vec::with_capacity(40);
    board.generate_moves(|piece_moves| {
        moves.extend(piece_moves);
        false
    });
    moves.sort_unstable_by(|a, b| {
        let sa = move_score(board, *a, tt_move, killers, history);
        let sb = move_score(board, *b, tt_move, killers, history);
        sb.cmp(&sa)
    });
    moves
}

/// Captures with MVV-LVA ordering (used by both classic and enhanced quiescence).
pub fn capture_moves(board: &Board) -> Vec<Move> {
    let mut captures = Vec::with_capacity(16);
    board.generate_moves(|piece_moves| {
        for mv in piece_moves {
            if board.piece_on(mv.to).is_some() {
                captures.push(mv);
            }
        }
        false
    });
    captures.sort_unstable_by(|a, b| {
        let val = |mv: &Move| {
            let victim = piece_val(board.piece_on(mv.to).unwrap());
            let attacker = piece_val(board.piece_on(mv.from).unwrap());
            (victim, std::cmp::Reverse(attacker))
        };
        val(b).cmp(&val(a))
    });
    captures
}

/// Simple move ordering: captures first (unsorted), then quiets.
fn ordered_moves_classic(board: &Board) -> Vec<Move> {
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

// ===========================================================================
// CLASSIC SEARCH — original algorithm, no enhancements
// ===========================================================================

fn quiescence_classic(board: &Board, mut alpha: i32, beta: i32) -> i32 {
    match board.status() {
        GameStatus::Won => return -MATE_SCORE,
        GameStatus::Drawn => return DRAW_SCORE,
        GameStatus::Ongoing => {}
    }

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
        let score = -quiescence_classic(&child, -beta, -alpha);
        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

pub fn negamax_classic(board: &Board, depth: u32, mut alpha: i32, beta: i32) -> i32 {
    match board.status() {
        GameStatus::Won => return -MATE_SCORE,
        GameStatus::Drawn => return DRAW_SCORE,
        GameStatus::Ongoing => {}
    }

    if depth == 0 {
        return quiescence_classic(board, alpha, beta);
    }

    let moves = ordered_moves_classic(board);

    for mv in moves {
        let mut child = board.clone();
        child.play_unchecked(mv);
        let score = -negamax_classic(&child, depth - 1, -beta, -alpha);
        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

pub fn best_move_with_scores_classic(board: &Board, depth: u32) -> Vec<(Move, i32)> {
    let moves = ordered_moves_classic(board);
    let mut results = Vec::new();

    for mv in moves {
        let mut child = board.clone();
        child.play_unchecked(mv);
        let score = -negamax_classic(&child, depth - 1, -MATE_SCORE, MATE_SCORE);
        results.push((mv, score));
    }

    results
}

// ===========================================================================
// ENHANCED SEARCH — TT, PVS, null move pruning, delta pruning, killers, history
// ===========================================================================

const DELTA_MARGIN: i32 = 1100;

fn quiescence_enhanced(board: &Board, mut alpha: i32, beta: i32) -> i32 {
    match board.status() {
        GameStatus::Won => return -MATE_SCORE,
        GameStatus::Drawn => return DRAW_SCORE,
        GameStatus::Ongoing => {}
    }

    let stand_pat = evaluate(board);
    if stand_pat >= beta {
        return beta;
    }
    if stand_pat + DELTA_MARGIN < alpha {
        return alpha;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    for mv in capture_moves(board) {
        if let Some(victim) = board.piece_on(mv.to) {
            let gain = match victim {
                Piece::Pawn => 100,
                Piece::Knight => 320,
                Piece::Bishop => 330,
                Piece::Rook => 500,
                Piece::Queen => 900,
                Piece::King => 0,
            };
            if stand_pat + gain + 200 < alpha {
                continue;
            }
        }

        let mut child = board.clone();
        child.play_unchecked(mv);
        let score = -quiescence_enhanced(&child, -beta, -alpha);
        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

pub struct SearchContext {
    tt: TTable,
    killers: Vec<[Option<Move>; 2]>,
    history: Box<[[i32; 64]; 64]>,
}

impl SearchContext {
    pub fn new() -> Self {
        SearchContext {
            tt: TTable::new(20),
            killers: vec![[None; 2]; 64],
            history: Box::new([[0i32; 64]; 64]),
        }
    }

    fn add_killer(&mut self, ply: usize, mv: Move) {
        if self.killers[ply][0] != Some(mv) {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = Some(mv);
        }
    }

    fn add_history(&mut self, mv: Move, depth: u32) {
        self.history[mv.from as usize][mv.to as usize] += (depth * depth) as i32;
    }
}

impl Default for SearchContext {
    fn default() -> Self {
        Self::new()
    }
}

fn can_null_move(board: &Board) -> bool {
    let us = board.side_to_move();
    let non_pawn = board.colored_pieces(us, Piece::Knight)
        | board.colored_pieces(us, Piece::Bishop)
        | board.colored_pieces(us, Piece::Rook)
        | board.colored_pieces(us, Piece::Queen);
    !non_pawn.is_empty()
}

fn negamax_enhanced(
    ctx: &mut SearchContext,
    board: &Board,
    depth: u32,
    mut alpha: i32,
    beta: i32,
    ply: usize,
    allow_null: bool,
) -> i32 {
    match board.status() {
        GameStatus::Won => return -MATE_SCORE,
        GameStatus::Drawn => return DRAW_SCORE,
        GameStatus::Ongoing => {}
    }

    let orig_alpha = alpha;
    let hash = board.hash();
    let mut tt_move = None;

    if let Some(entry) = ctx.tt.probe(hash) {
        tt_move = entry.best_move;
        if entry.depth >= depth {
            match entry.flag {
                TTFlag::Exact => return entry.score,
                TTFlag::LowerBound => {
                    if entry.score >= beta {
                        return entry.score;
                    }
                }
                TTFlag::UpperBound => {
                    if entry.score <= alpha {
                        return entry.score;
                    }
                }
            }
        }
    }

    if depth == 0 {
        return quiescence_enhanced(board, alpha, beta);
    }

    // Null move pruning (R=2)
    if allow_null && depth >= 3 && can_null_move(board) {
        if let Some(null_board) = board.null_move() {
            let score =
                -negamax_enhanced(ctx, &null_board, depth - 3, -beta, -beta + 1, ply + 1, false);
            if score >= beta {
                return beta;
            }
        }
    }

    let killers = if ply < ctx.killers.len() {
        ctx.killers[ply]
    } else {
        [None; 2]
    };
    let moves = sorted_moves(board, tt_move, &killers, &ctx.history);

    if moves.is_empty() {
        return evaluate(board);
    }

    let mut best_score = i32::MIN;
    let mut best_move = moves[0];

    for (i, &mv) in moves.iter().enumerate() {
        let mut child = board.clone();
        child.play_unchecked(mv);

        let score;
        if i == 0 {
            score = -negamax_enhanced(ctx, &child, depth - 1, -beta, -alpha, ply + 1, true);
        } else {
            let zw = -negamax_enhanced(ctx, &child, depth - 1, -alpha - 1, -alpha, ply + 1, true);
            if zw > alpha && zw < beta {
                score = -negamax_enhanced(ctx, &child, depth - 1, -beta, -alpha, ply + 1, true);
            } else {
                score = zw;
            }
        }

        if score > best_score {
            best_score = score;
            best_move = mv;
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            if board.piece_on(mv.to).is_none() {
                ctx.add_killer(ply, mv);
                ctx.add_history(mv, depth);
            }
            break;
        }
    }

    let flag = if best_score >= beta {
        TTFlag::LowerBound
    } else if best_score <= orig_alpha {
        TTFlag::UpperBound
    } else {
        TTFlag::Exact
    };
    ctx.tt.store(hash, depth, best_score, flag, Some(best_move));

    best_score
}

pub fn best_move_with_scores_enhanced(
    ctx: &mut SearchContext,
    board: &Board,
    depth: u32,
) -> Vec<(Move, i32)> {
    let tt_move = ctx.tt.probe(board.hash()).and_then(|e| e.best_move);
    let killers = [None; 2];
    let moves = sorted_moves(board, tt_move, &killers, &ctx.history);
    let mut results = Vec::with_capacity(moves.len());

    // At root we need exact scores for every move so the caller can compare them.
    // Full window for each move — PVS is used inside negamax for subtrees.
    for &mv in &moves {
        let mut child = board.clone();
        child.play_unchecked(mv);
        let score = -negamax_enhanced(ctx, &child, depth - 1, -MATE_SCORE, MATE_SCORE, 1, true);
        results.push((mv, score));
    }

    results
}
