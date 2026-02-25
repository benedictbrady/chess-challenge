use cozy_chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves, Board,
    Color, File, Piece, Rank, Square,
};

// Material values in centipawns
pub const PAWN_VALUE: i32 = 100;
pub const KNIGHT_VALUE: i32 = 320;
pub const BISHOP_VALUE: i32 = 330;
pub const ROOK_VALUE: i32 = 500;
pub const QUEEN_VALUE: i32 = 900;

fn piece_value(piece: Piece) -> i32 {
    match piece {
        Piece::Pawn => PAWN_VALUE,
        Piece::Knight => KNIGHT_VALUE,
        Piece::Bishop => BISHOP_VALUE,
        Piece::Rook => ROOK_VALUE,
        Piece::Queen => QUEEN_VALUE,
        Piece::King => 0,
    }
}

// ---------------------------------------------------------------------------
// Game phase
// ---------------------------------------------------------------------------

// Phase weights per piece type (knights, bishops, rooks, queens only)
const KNIGHT_PHASE: i32 = 1;
const BISHOP_PHASE: i32 = 1;
const ROOK_PHASE: i32 = 2;
const QUEEN_PHASE: i32 = 4;
const TOTAL_PHASE: i32 = 2 * (2 * KNIGHT_PHASE + 2 * BISHOP_PHASE + 2 * ROOK_PHASE + QUEEN_PHASE); // 24

/// Returns game phase: 256 = pure middlegame, 0 = pure endgame.
fn game_phase(board: &Board) -> i32 {
    let mut phase = 0;
    for color in [Color::White, Color::Black] {
        phase += board.colored_pieces(color, Piece::Knight).len() as i32 * KNIGHT_PHASE;
        phase += board.colored_pieces(color, Piece::Bishop).len() as i32 * BISHOP_PHASE;
        phase += board.colored_pieces(color, Piece::Rook).len() as i32 * ROOK_PHASE;
        phase += board.colored_pieces(color, Piece::Queen).len() as i32 * QUEEN_PHASE;
    }
    // Scale to 0..256
    (phase * 256 + TOTAL_PHASE / 2) / TOTAL_PHASE
}

// ---------------------------------------------------------------------------
// Middlegame piece-square tables (from white's perspective, rank 8 at index 0)
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const PAWN_MG: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const KNIGHT_MG: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

#[rustfmt::skip]
const BISHOP_MG: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

#[rustfmt::skip]
const ROOK_MG: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
];

#[rustfmt::skip]
const QUEEN_MG: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];

#[rustfmt::skip]
const KING_MG: [i32; 64] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
];

// ---------------------------------------------------------------------------
// Endgame piece-square tables
// ---------------------------------------------------------------------------

// Pawns: advanced pawns are worth much more in the endgame (close to promotion)
#[rustfmt::skip]
const PAWN_EG: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10,
     0,  0,  0,  0,  0,  0,  0,  0,
];

// Knights: still centralized, but slightly less valuable on the rim
#[rustfmt::skip]
const KNIGHT_EG: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

// Bishops: same as middlegame (good throughout)
#[rustfmt::skip]
const BISHOP_EG: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

// Rooks: 7th rank bonus more pronounced in endgame
#[rustfmt::skip]
const ROOK_EG: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    10, 15, 15, 15, 15, 15, 15, 10,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
];

// Queens: same as middlegame
#[rustfmt::skip]
const QUEEN_EG: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];

// King: centralize! The opposite of middlegame.
#[rustfmt::skip]
const KING_EG: [i32; 64] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
];

// ---------------------------------------------------------------------------
// PST lookup
// ---------------------------------------------------------------------------

fn pst_index(sq: Square, color: Color) -> usize {
    let file = sq.file() as usize;
    let rank = sq.rank() as usize;
    // PSTs are indexed with rank 8 at index 0 (from white's perspective)
    let rank_from_top = match color {
        Color::White => 7 - rank,
        Color::Black => rank,
    };
    rank_from_top * 8 + file
}

/// Returns (middlegame_bonus, endgame_bonus) for a piece on a square.
fn pst_bonus(piece: Piece, sq: Square, color: Color) -> (i32, i32) {
    let idx = pst_index(sq, color);
    match piece {
        Piece::Pawn => (PAWN_MG[idx], PAWN_EG[idx]),
        Piece::Knight => (KNIGHT_MG[idx], KNIGHT_EG[idx]),
        Piece::Bishop => (BISHOP_MG[idx], BISHOP_EG[idx]),
        Piece::Rook => (ROOK_MG[idx], ROOK_EG[idx]),
        Piece::Queen => (QUEEN_MG[idx], QUEEN_EG[idx]),
        Piece::King => (KING_MG[idx], KING_EG[idx]),
    }
}

// ---------------------------------------------------------------------------
// King safety
// ---------------------------------------------------------------------------

/// +15cp per friendly pawn shielding the king (max +45cp).
/// Only computed when king is on its back two ranks.
fn pawn_shield_bonus(board: &Board, color: Color) -> i32 {
    let king_sq = board.king(color);
    let king_rank = king_sq.rank() as usize;

    let home_ranks = match color {
        Color::White => king_rank <= 1,
        Color::Black => king_rank >= 6,
    };
    if !home_ranks {
        return 0;
    }

    let shield_rank = match color {
        Color::White => king_rank + 1,
        Color::Black => king_rank - 1,
    };

    let king_file = king_sq.file() as usize;
    let friendly_pawns = board.colored_pieces(color, Piece::Pawn);
    let mut bonus = 0;

    let lo = if king_file > 0 { king_file - 1 } else { 0 };
    let hi = if king_file < 7 { king_file + 1 } else { 7 };
    for f in lo..=hi {
        let sq = Square::new(File::index(f), Rank::index(shield_rank));
        if friendly_pawns.has(sq) {
            bonus += 15;
        }
    }

    bonus
}

/// Penalty for open/semi-open files near the king.
fn open_file_penalty(board: &Board, color: Color) -> i32 {
    let king_file = board.king(color).file() as usize;
    let friendly_pawns = board.colored_pieces(color, Piece::Pawn);
    let enemy_pawns = board.colored_pieces(!color, Piece::Pawn);
    let mut penalty = 0;

    let lo = if king_file > 0 { king_file - 1 } else { 0 };
    let hi = if king_file < 7 { king_file + 1 } else { 7 };
    for f in lo..=hi {
        let file_bb = File::index(f).bitboard();
        let has_friendly = !(friendly_pawns & file_bb).is_empty();
        let has_enemy = !(enemy_pawns & file_bb).is_empty();
        if !has_friendly && !has_enemy {
            penalty -= 20;
        } else if !has_friendly {
            penalty -= 10;
        }
    }

    penalty
}

/// Non-linear penalty based on how many enemy pieces attack the king zone.
fn attacker_penalty(board: &Board, color: Color) -> i32 {
    const PENALTIES: [i32; 7] = [0, -5, -20, -45, -80, -120, -160];

    let king_sq = board.king(color);
    let king_zone = get_king_moves(king_sq) | king_sq.bitboard();
    let them = !color;
    let occupied = board.occupied();
    let mut attackers = 0u32;

    for sq in board.colored_pieces(them, Piece::Knight) {
        if !(get_knight_moves(sq) & king_zone).is_empty() {
            attackers += 1;
        }
    }
    for sq in board.colored_pieces(them, Piece::Bishop) {
        if !(get_bishop_moves(sq, occupied) & king_zone).is_empty() {
            attackers += 1;
        }
    }
    for sq in board.colored_pieces(them, Piece::Rook) {
        if !(get_rook_moves(sq, occupied) & king_zone).is_empty() {
            attackers += 1;
        }
    }
    for sq in board.colored_pieces(them, Piece::Queen) {
        let attacks = get_rook_moves(sq, occupied) | get_bishop_moves(sq, occupied);
        if !(attacks & king_zone).is_empty() {
            attackers += 1;
        }
    }
    for sq in board.colored_pieces(them, Piece::Pawn) {
        if !(get_pawn_attacks(sq, them) & king_zone).is_empty() {
            attackers += 1;
        }
    }

    let idx = (attackers as usize).min(PENALTIES.len() - 1);
    PENALTIES[idx]
}

// ---------------------------------------------------------------------------
// Main evaluation — tapered
// ---------------------------------------------------------------------------

/// Evaluate the board from the perspective of the side to move.
/// Positive = good for side to move.
///
/// Uses tapered evaluation: blends middlegame and endgame scores based
/// on how much material remains on the board.
pub fn evaluate(board: &Board) -> i32 {
    let side = board.side_to_move();
    let phase = game_phase(board); // 256 = middlegame, 0 = endgame

    let mut mg_score = 0i32;
    let mut eg_score = 0i32;

    for color in [Color::White, Color::Black] {
        let sign = if color == side { 1 } else { -1 };
        for piece in [
            Piece::Pawn,
            Piece::Knight,
            Piece::Bishop,
            Piece::Rook,
            Piece::Queen,
            Piece::King,
        ] {
            let bb = board.pieces(piece) & board.colors(color);
            for sq in bb {
                let mat = piece_value(piece);
                let (mg_pst, eg_pst) = pst_bonus(piece, sq, color);
                mg_score += sign * (mat + mg_pst);
                eg_score += sign * (mat + eg_pst);
            }
        }

        // King safety (tapered: full weight in middlegame, zero in endgame)
        let safety =
            pawn_shield_bonus(board, color) + open_file_penalty(board, color) + attacker_penalty(board, color);
        mg_score += sign * safety;
        // eg_score gets no king safety — in endgames it's irrelevant
    }

    // Blend: phase=256 → pure middlegame, phase=0 → pure endgame
    (mg_score * phase + eg_score * (256 - phase)) / 256
}
