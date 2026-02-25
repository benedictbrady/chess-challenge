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

// Piece-square tables (from white's perspective, rank 1 = index 0)
// Values are bonuses in centipawns

#[rustfmt::skip]
const PAWN_PST: [i32; 64] = [
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
const KNIGHT_PST: [i32; 64] = [
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
const BISHOP_PST: [i32; 64] = [
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
const ROOK_PST: [i32; 64] = [
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
const QUEEN_PST: [i32; 64] = [
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
const KING_MIDDLE_PST: [i32; 64] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
];

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

fn pst_bonus(piece: Piece, sq: Square, color: Color) -> i32 {
    let idx = pst_index(sq, color);
    match piece {
        Piece::Pawn => PAWN_PST[idx],
        Piece::Knight => KNIGHT_PST[idx],
        Piece::Bishop => BISHOP_PST[idx],
        Piece::Rook => ROOK_PST[idx],
        Piece::Queen => QUEEN_PST[idx],
        Piece::King => KING_MIDDLE_PST[idx],
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

    // Only compute if king is on back two ranks
    let home_ranks = match color {
        Color::White => king_rank <= 1, // Rank 1-2
        Color::Black => king_rank >= 6, // Rank 7-8
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
/// Fully open (no pawns): -20cp. Semi-open (no friendly pawns): -10cp.
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
            penalty -= 20; // fully open
        } else if !has_friendly {
            penalty -= 10; // semi-open
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

    // Knights
    for sq in board.colored_pieces(them, Piece::Knight) {
        if !(get_knight_moves(sq) & king_zone).is_empty() {
            attackers += 1;
        }
    }
    // Bishops
    for sq in board.colored_pieces(them, Piece::Bishop) {
        if !(get_bishop_moves(sq, occupied) & king_zone).is_empty() {
            attackers += 1;
        }
    }
    // Rooks
    for sq in board.colored_pieces(them, Piece::Rook) {
        if !(get_rook_moves(sq, occupied) & king_zone).is_empty() {
            attackers += 1;
        }
    }
    // Queens (rook + bishop moves)
    for sq in board.colored_pieces(them, Piece::Queen) {
        let attacks = get_rook_moves(sq, occupied) | get_bishop_moves(sq, occupied);
        if !(attacks & king_zone).is_empty() {
            attackers += 1;
        }
    }
    // Pawns
    for sq in board.colored_pieces(them, Piece::Pawn) {
        if !(get_pawn_attacks(sq, them) & king_zone).is_empty() {
            attackers += 1;
        }
    }

    let idx = (attackers as usize).min(PENALTIES.len() - 1);
    PENALTIES[idx]
}

/// Evaluate the board from the perspective of the side to move.
/// Positive = good for side to move.
pub fn evaluate(board: &Board) -> i32 {
    let side = board.side_to_move();
    let mut score = 0i32;

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
                score += sign * (piece_value(piece) + pst_bonus(piece, sq, color));
            }
        }

        // King safety
        let safety =
            pawn_shield_bonus(board, color) + open_file_penalty(board, color) + attacker_penalty(board, color);
        score += sign * safety;
    }

    score
}
