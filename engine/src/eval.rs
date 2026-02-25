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
// Passed pawns
// ---------------------------------------------------------------------------

/// Check if a pawn is passed (no enemy pawns on same or adjacent files ahead).
fn is_passed_pawn(board: &Board, sq: Square, color: Color) -> bool {
    let file = sq.file() as usize;
    let rank = sq.rank() as usize;
    let enemy_pawns = board.colored_pieces(!color, Piece::Pawn);

    let lo = if file > 0 { file - 1 } else { 0 };
    let hi = if file < 7 { file + 1 } else { 7 };

    for f in lo..=hi {
        // Check all ranks ahead of the pawn
        let start_rank = match color {
            Color::White => rank + 1,
            Color::Black => 0,
        };
        let end_rank = match color {
            Color::White => 8,
            Color::Black => rank,
        };
        for r in start_rank..end_rank {
            let check_sq = Square::new(File::index(f), Rank::index(r));
            if enemy_pawns.has(check_sq) {
                return false;
            }
        }
    }
    true
}

/// Evaluate passed pawns for one side. Returns (mg_bonus, eg_bonus).
/// Uses quadratic scaling by rank (inspired by Stockfish classical eval).
fn passed_pawn_eval(board: &Board, color: Color) -> (i32, i32) {
    let friendly_pawns = board.colored_pieces(color, Piece::Pawn);
    let mut mg = 0i32;
    let mut eg = 0i32;

    for sq in friendly_pawns {
        if !is_passed_pawn(board, sq, color) {
            continue;
        }

        // Ranks advanced from starting position (0-based: rank 2→0, rank 7→5)
        let r = match color {
            Color::White => sq.rank() as i32 - 1, // rank 2 = 1 → r=0, rank 7 = 6 → r=5
            Color::Black => 6 - sq.rank() as i32,  // rank 7 = 6 → r=0, rank 2 = 1 → r=5
        };
        let r = r.max(0);
        let rr = r * (r - 1).max(0);

        // Base bonuses (quadratic scaling)
        mg += 15 * rr;
        eg += 10 * (rr + r + 1);

        // King distance bonus in endgame: friendly king near passer is good,
        // enemy king far from passer is good
        let friendly_king = board.king(color);
        let enemy_king = board.king(!color);

        let promo_sq = match color {
            Color::White => Square::new(sq.file(), Rank::Eighth),
            Color::Black => Square::new(sq.file(), Rank::First),
        };

        let friendly_dist = chebyshev_distance(friendly_king, sq);
        let enemy_dist = chebyshev_distance(enemy_king, promo_sq);

        // Bonus for enemy king being far from the promotion square
        eg += (enemy_dist as i32) * 5 * r;
        // Bonus for friendly king being close to the pawn
        eg -= (friendly_dist as i32) * 2 * r;
    }

    (mg, eg)
}

fn chebyshev_distance(a: Square, b: Square) -> u32 {
    let df = (a.file() as i32 - b.file() as i32).unsigned_abs();
    let dr = (a.rank() as i32 - b.rank() as i32).unsigned_abs();
    df.max(dr)
}

// ---------------------------------------------------------------------------
// Piece mobility
// ---------------------------------------------------------------------------

/// Count pseudo-legal moves for pieces (excluding pawns and king).
/// Returns (mg_bonus, eg_bonus).
fn mobility_eval(board: &Board, color: Color) -> (i32, i32) {
    let occupied = board.occupied();
    // Squares controlled by enemy pawns are "unsafe" for minors
    let enemy_pawn_attacks = {
        let mut attacks = cozy_chess::BitBoard::EMPTY;
        for sq in board.colored_pieces(!color, Piece::Pawn) {
            attacks = attacks | get_pawn_attacks(sq, !color);
        }
        attacks
    };

    let mut mg = 0i32;
    let mut eg = 0i32;

    // Knights: ~4cp MG, ~4cp EG per move (excluding pawn-controlled squares)
    for sq in board.colored_pieces(color, Piece::Knight) {
        let moves = get_knight_moves(sq) & !enemy_pawn_attacks;
        let count = moves.len() as i32;
        mg += (count - 4) * 4; // baseline 4 moves = 0 bonus
        eg += (count - 4) * 4;
    }

    // Bishops: ~5cp MG, ~5cp EG per move
    for sq in board.colored_pieces(color, Piece::Bishop) {
        let moves = get_bishop_moves(sq, occupied) & !enemy_pawn_attacks;
        let count = moves.len() as i32;
        mg += (count - 6) * 5; // baseline 6 moves = 0 bonus
        eg += (count - 6) * 5;
    }

    // Rooks: ~3cp MG, ~7cp EG per move (much more important in endgames)
    for sq in board.colored_pieces(color, Piece::Rook) {
        let moves = get_rook_moves(sq, occupied);
        let count = moves.len() as i32;
        mg += (count - 7) * 3; // baseline 7 moves = 0 bonus
        eg += (count - 7) * 7;
    }

    // Queens: ~1cp MG, ~2cp EG per move (queens are usually mobile)
    for sq in board.colored_pieces(color, Piece::Queen) {
        let moves = get_rook_moves(sq, occupied) | get_bishop_moves(sq, occupied);
        let count = moves.len() as i32;
        mg += (count - 14) * 1;
        eg += (count - 14) * 2;
    }

    (mg, eg)
}

// ---------------------------------------------------------------------------
// Pawn structure
// ---------------------------------------------------------------------------

/// Penalties for doubled and isolated pawns. Returns (mg_penalty, eg_penalty).
fn pawn_structure_eval(board: &Board, color: Color) -> (i32, i32) {
    let friendly_pawns = board.colored_pieces(color, Piece::Pawn);
    let mut mg = 0i32;
    let mut eg = 0i32;

    for f in 0..8u8 {
        let file_bb = File::index(f as usize).bitboard();
        let pawns_on_file = (friendly_pawns & file_bb).len() as i32;

        // Doubled pawns: penalty for each extra pawn on same file
        if pawns_on_file > 1 {
            mg -= (pawns_on_file - 1) * 10;
            eg -= (pawns_on_file - 1) * 20;
        }

        // Isolated pawns: no friendly pawns on adjacent files
        if pawns_on_file > 0 {
            let has_adjacent = {
                let left = if f > 0 { !(friendly_pawns & File::index((f - 1) as usize).bitboard()).is_empty() } else { false };
                let right = if f < 7 { !(friendly_pawns & File::index((f + 1) as usize).bitboard()).is_empty() } else { false };
                left || right
            };
            if !has_adjacent {
                mg -= 10;
                eg -= 15;
            }
        }
    }

    (mg, eg)
}

// ---------------------------------------------------------------------------
// Bishop pair
// ---------------------------------------------------------------------------

fn bishop_pair_bonus(board: &Board, color: Color) -> (i32, i32) {
    if board.colored_pieces(color, Piece::Bishop).len() >= 2 {
        (30, 50) // bishop pair is very strong, especially in endgame
    } else {
        (0, 0)
    }
}

// ---------------------------------------------------------------------------
// Rook on open/semi-open file
// ---------------------------------------------------------------------------

fn rook_file_bonus(board: &Board, color: Color) -> (i32, i32) {
    let friendly_pawns = board.colored_pieces(color, Piece::Pawn);
    let enemy_pawns = board.colored_pieces(!color, Piece::Pawn);
    let mut mg = 0i32;
    let mut eg = 0i32;

    for sq in board.colored_pieces(color, Piece::Rook) {
        let file_bb = sq.file().bitboard();
        let has_friendly_pawn = !(friendly_pawns & file_bb).is_empty();
        let has_enemy_pawn = !(enemy_pawns & file_bb).is_empty();

        if !has_friendly_pawn && !has_enemy_pawn {
            mg += 20; // open file
            eg += 25;
        } else if !has_friendly_pawn {
            mg += 10; // semi-open
            eg += 15;
        }
    }

    // Rook on 7th rank bonus
    let seventh_rank = match color {
        Color::White => Rank::Seventh.bitboard(),
        Color::Black => Rank::Second.bitboard(),
    };
    for _sq in board.colored_pieces(color, Piece::Rook) & seventh_rank {
        mg += 20;
        eg += 40;
    }

    (mg, eg)
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

        // Passed pawns
        let (pp_mg, pp_eg) = passed_pawn_eval(board, color);
        mg_score += sign * pp_mg;
        eg_score += sign * pp_eg;

        // Piece mobility
        let (mob_mg, mob_eg) = mobility_eval(board, color);
        mg_score += sign * mob_mg;
        eg_score += sign * mob_eg;

        // Pawn structure
        let (ps_mg, ps_eg) = pawn_structure_eval(board, color);
        mg_score += sign * ps_mg;
        eg_score += sign * ps_eg;

        // Bishop pair
        let (bp_mg, bp_eg) = bishop_pair_bonus(board, color);
        mg_score += sign * bp_mg;
        eg_score += sign * bp_eg;

        // Rook on open files + 7th rank
        let (rf_mg, rf_eg) = rook_file_bonus(board, color);
        mg_score += sign * rf_mg;
        eg_score += sign * rf_eg;
    }

    // Blend: phase=256 → pure middlegame, phase=0 → pure endgame
    (mg_score * phase + eg_score * (256 - phase)) / 256
}
