"""Full Python port of engine/src/eval.rs — all 20 evaluation terms.

This is a faithful reimplementation of the baseline engine's static evaluation
so we can generate training labels that exactly match what the baseline sees.

All constants, PSTs, and logic are copied directly from eval.rs.

Returns centipawns from the side-to-move's perspective.
"""

import chess

# ---------------------------------------------------------------------------
# Material values (must match eval.rs)
# ---------------------------------------------------------------------------

MATERIAL = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# ---------------------------------------------------------------------------
# Game phase
# ---------------------------------------------------------------------------

KNIGHT_PHASE = 1
BISHOP_PHASE = 1
ROOK_PHASE = 2
QUEEN_PHASE = 4
TOTAL_PHASE = 2 * (2 * KNIGHT_PHASE + 2 * BISHOP_PHASE + 2 * ROOK_PHASE + QUEEN_PHASE)  # 24

PHASE_WEIGHT = {
    chess.KNIGHT: KNIGHT_PHASE,
    chess.BISHOP: BISHOP_PHASE,
    chess.ROOK: ROOK_PHASE,
    chess.QUEEN: QUEEN_PHASE,
}


def game_phase(board: chess.Board) -> int:
    """Returns game phase: 256 = pure middlegame, 0 = pure endgame."""
    phase = 0
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type, weight in PHASE_WEIGHT.items():
            phase += len(board.pieces(piece_type, color)) * weight
    return (phase * 256 + TOTAL_PHASE // 2) // TOTAL_PHASE


# ---------------------------------------------------------------------------
# Piece-square tables (from white's perspective, rank 8 at index 0)
# ---------------------------------------------------------------------------
# fmt: off

PAWN_MG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_MG = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_MG = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_MG = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_MG = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

KING_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

# Endgame PSTs

PAWN_EG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_EG = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_EG = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_EG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    10, 15, 15, 15, 15, 15, 15, 10,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_EG = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

KING_EG = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

# fmt: on

PST_MG = {
    chess.PAWN: PAWN_MG,
    chess.KNIGHT: KNIGHT_MG,
    chess.BISHOP: BISHOP_MG,
    chess.ROOK: ROOK_MG,
    chess.QUEEN: QUEEN_MG,
    chess.KING: KING_MG,
}

PST_EG = {
    chess.PAWN: PAWN_EG,
    chess.KNIGHT: KNIGHT_EG,
    chess.BISHOP: BISHOP_EG,
    chess.ROOK: ROOK_EG,
    chess.QUEEN: QUEEN_EG,
    chess.KING: KING_EG,
}


# ---------------------------------------------------------------------------
# PST lookup
# ---------------------------------------------------------------------------

def _pst_index(square: int, color: bool) -> int:
    """Convert python-chess square to PST index.

    PSTs are stored with rank 8 at index 0 (from white's perspective).
    """
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    rank_from_top = (7 - rank) if color == chess.WHITE else rank
    return rank_from_top * 8 + file


def _pst_bonus(piece_type: int, square: int, color: bool) -> tuple[int, int]:
    """Returns (mg_bonus, eg_bonus) for a piece on a square."""
    idx = _pst_index(square, color)
    return PST_MG[piece_type][idx], PST_EG[piece_type][idx]


# ---------------------------------------------------------------------------
# King safety
# ---------------------------------------------------------------------------

def _pawn_shield_bonus(board: chess.Board, color: bool) -> int:
    """+15cp per friendly pawn shielding the king (max +45cp).
    Only when king is on back two ranks."""
    king_sq = board.king(color)
    king_rank = chess.square_rank(king_sq)

    if color == chess.WHITE:
        home_ranks = king_rank <= 1
    else:
        home_ranks = king_rank >= 6

    if not home_ranks:
        return 0

    if color == chess.WHITE:
        shield_rank = king_rank + 1
    else:
        shield_rank = king_rank - 1

    king_file = chess.square_file(king_sq)
    friendly_pawns = board.pieces(chess.PAWN, color)
    bonus = 0

    lo = max(king_file - 1, 0)
    hi = min(king_file + 1, 7)
    for f in range(lo, hi + 1):
        sq = chess.square(f, shield_rank)
        if sq in friendly_pawns:
            bonus += 15

    return bonus


def _open_file_penalty(board: chess.Board, color: bool) -> int:
    """Penalty for open/semi-open files near the king."""
    king_file = chess.square_file(board.king(color))
    friendly_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    penalty = 0

    lo = max(king_file - 1, 0)
    hi = min(king_file + 1, 7)
    for f in range(lo, hi + 1):
        file_mask = chess.BB_FILES[f]
        has_friendly = bool(friendly_pawns & file_mask)
        has_enemy = bool(enemy_pawns & file_mask)
        if not has_friendly and not has_enemy:
            penalty -= 20  # open file
        elif not has_friendly:
            penalty -= 10  # semi-open

    return penalty


# ---------------------------------------------------------------------------
# Piece mobility + king attackers (combined, like eval.rs)
# ---------------------------------------------------------------------------

# Precompute pawn attack masks for each side
def _pawn_attack_bb(board: chess.Board, color: bool) -> int:
    """Compute the union of all pawn attacks for a color."""
    pawns = board.pieces(chess.PAWN, color)
    attacks = 0
    for sq in pawns:
        attacks |= chess.BB_PAWN_ATTACKS[color][sq]
    return attacks


def _count_bits(bb: int) -> int:
    """Count set bits in a bitboard."""
    return bin(bb).count('1')


def _king_zone(king_sq: int) -> int:
    """King zone = king square + adjacent squares."""
    return chess.BB_KING_ATTACKS[king_sq] | chess.BB_SQUARES[king_sq]


def _slider_attacks(board: chess.Board, sq: int, piece_type: int) -> int:
    """Get slider attack bitboard using python-chess's attack generation."""
    # python-chess board.attacks() gives the attack set for a piece on sq
    # But we need to compute it manually for a hypothetical piece
    occupied = board.occupied
    if piece_type == chess.BISHOP:
        return chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occupied]
    elif piece_type == chess.ROOK:
        return (chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occupied] |
                chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occupied])
    else:
        return 0


def _piece_eval(board: chess.Board, color: bool) -> tuple[int, int, int]:
    """Combined king safety (attacker count) + piece mobility.

    Returns (mg_king_safety, mg_mobility, eg_mobility) for one color.
    """
    ATTACK_PENALTIES = [0, -5, -20, -45, -80, -120, -160]

    them = not color
    king_sq = board.king(color)
    kzone = _king_zone(king_sq)

    # Enemy pawn attacks (for mobility exclusion on minors)
    enemy_pawn_attacks = _pawn_attack_bb(board, them)

    attackers = 0
    mob_mg = 0
    mob_eg = 0

    # --- Our pieces: mobility ---
    for sq in board.pieces(chess.KNIGHT, color):
        moves = chess.BB_KNIGHT_ATTACKS[sq] & ~enemy_pawn_attacks
        count = _count_bits(moves)
        mob_mg += (count - 4) * 4
        mob_eg += (count - 4) * 4

    occupied = board.occupied
    for sq in board.pieces(chess.BISHOP, color):
        # Use board.attacks() for slider moves (handles blockers correctly)
        moves = board.attacks_mask(sq) & ~enemy_pawn_attacks
        count = _count_bits(moves)
        mob_mg += (count - 6) * 5
        mob_eg += (count - 6) * 5

    for sq in board.pieces(chess.ROOK, color):
        moves = board.attacks_mask(sq)
        count = _count_bits(moves)
        mob_mg += (count - 7) * 3
        mob_eg += (count - 7) * 7

    for sq in board.pieces(chess.QUEEN, color):
        moves = board.attacks_mask(sq)
        count = _count_bits(moves)
        mob_mg += (count - 14) * 1
        mob_eg += (count - 14) * 2

    # --- Enemy pieces: king attackers ---
    for sq in board.pieces(chess.KNIGHT, them):
        if chess.BB_KNIGHT_ATTACKS[sq] & kzone:
            attackers += 1

    for sq in board.pieces(chess.BISHOP, them):
        if board.attacks_mask(sq) & kzone:
            attackers += 1

    for sq in board.pieces(chess.ROOK, them):
        if board.attacks_mask(sq) & kzone:
            attackers += 1

    for sq in board.pieces(chess.QUEEN, them):
        if board.attacks_mask(sq) & kzone:
            attackers += 1

    # Pawn attackers on king zone
    pawn_attacks_on_kzone = enemy_pawn_attacks & kzone
    if pawn_attacks_on_kzone:
        attackers += _count_bits(pawn_attacks_on_kzone)

    idx = min(attackers, len(ATTACK_PENALTIES) - 1)
    king_safety_mg = ATTACK_PENALTIES[idx]

    return king_safety_mg, mob_mg, mob_eg


# ---------------------------------------------------------------------------
# Passed pawns
# ---------------------------------------------------------------------------

# Precomputed forward masks for passed pawn detection.
# FORWARD_MASKS[color][sq] covers the file + adjacent files, all ranks ahead.
_FORWARD_MASKS: dict[bool, list[int]] = {chess.WHITE: [0] * 64, chess.BLACK: [0] * 64}

def _init_forward_masks():
    for file in range(8):
        lo = max(file - 1, 0)
        hi = min(file + 1, 7)
        for rank in range(8):
            # White: ranks above
            w = 0
            for f in range(lo, hi + 1):
                for r in range(rank + 1, 8):
                    w |= chess.BB_SQUARES[chess.square(f, r)]
            _FORWARD_MASKS[chess.WHITE][rank * 8 + file] = w

            # Black: ranks below
            b = 0
            for f in range(lo, hi + 1):
                for r in range(0, rank):
                    b |= chess.BB_SQUARES[chess.square(f, r)]
            _FORWARD_MASKS[chess.BLACK][rank * 8 + file] = b

_init_forward_masks()


def _chebyshev_distance(a: int, b: int) -> int:
    """Chebyshev (king) distance between two squares."""
    df = abs(chess.square_file(a) - chess.square_file(b))
    dr = abs(chess.square_rank(a) - chess.square_rank(b))
    return max(df, dr)


def _passed_pawn_eval(board: chess.Board, color: bool) -> tuple[int, int]:
    """Evaluate passed pawns for one side. Returns (mg, eg)."""
    friendly_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    friendly_king = board.king(color)
    enemy_king = board.king(not color)
    mg = 0
    eg = 0

    enemy_pawn_bb = int(enemy_pawns)

    for sq in friendly_pawns:
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)

        # Check if passed using precomputed mask
        idx = rank * 8 + file
        mask = _FORWARD_MASKS[color][idx]
        if enemy_pawn_bb & mask:
            continue  # not passed

        if color == chess.WHITE:
            r = rank - 1
        else:
            r = 6 - rank
        r = max(r, 0)
        rr = r * max(r - 1, 0)

        mg += 15 * rr
        eg += 10 * (rr + r + 1)

        # King distance bonuses (endgame only)
        if color == chess.WHITE:
            promo_sq = chess.square(file, 7)
        else:
            promo_sq = chess.square(file, 0)

        f_dist = _chebyshev_distance(friendly_king, sq)
        e_dist = _chebyshev_distance(enemy_king, promo_sq)
        eg += e_dist * 5 * r
        eg -= f_dist * 2 * r

    return mg, eg


# ---------------------------------------------------------------------------
# Pawn structure
# ---------------------------------------------------------------------------

def _pawn_structure_eval(board: chess.Board, color: bool) -> tuple[int, int]:
    """Evaluate doubled and isolated pawns. Returns (mg, eg)."""
    friendly_pawns = board.pieces(chess.PAWN, color)
    friendly_pawn_bb = int(friendly_pawns)
    mg = 0
    eg = 0

    for f in range(8):
        file_mask = chess.BB_FILES[f]
        pawns_on_file = _count_bits(friendly_pawn_bb & file_mask)

        # Doubled pawns
        if pawns_on_file > 1:
            mg -= (pawns_on_file - 1) * 10
            eg -= (pawns_on_file - 1) * 20

        # Isolated pawns
        if pawns_on_file > 0:
            # Adjacent files mask
            adj = 0
            if f > 0:
                adj |= chess.BB_FILES[f - 1]
            if f < 7:
                adj |= chess.BB_FILES[f + 1]
            if not (friendly_pawn_bb & adj):
                mg -= 10
                eg -= 15

    return mg, eg


# ---------------------------------------------------------------------------
# Bishop pair
# ---------------------------------------------------------------------------

def _bishop_pair_bonus(board: chess.Board, color: bool) -> tuple[int, int]:
    if len(board.pieces(chess.BISHOP, color)) >= 2:
        return 30, 50
    return 0, 0


# ---------------------------------------------------------------------------
# Rook on open/semi-open file + 7th rank
# ---------------------------------------------------------------------------

def _rook_file_bonus(board: chess.Board, color: bool) -> tuple[int, int]:
    friendly_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    friendly_pawn_bb = int(friendly_pawns)
    enemy_pawn_bb = int(enemy_pawns)
    mg = 0
    eg = 0

    for sq in board.pieces(chess.ROOK, color):
        file_mask = chess.BB_FILES[chess.square_file(sq)]
        has_friendly_pawn = bool(friendly_pawn_bb & file_mask)
        has_enemy_pawn = bool(enemy_pawn_bb & file_mask)

        if not has_friendly_pawn and not has_enemy_pawn:
            mg += 20  # open file
            eg += 25
        elif not has_friendly_pawn:
            mg += 10  # semi-open
            eg += 15

    # Rook on 7th rank
    if color == chess.WHITE:
        seventh_rank = chess.BB_RANKS[6]  # rank 7 (0-indexed: rank index 6)
    else:
        seventh_rank = chess.BB_RANKS[1]  # rank 2 (0-indexed: rank index 1)

    rook_bb = int(board.pieces(chess.ROOK, color))
    rooks_on_7th = _count_bits(rook_bb & seventh_rank)
    mg += rooks_on_7th * 20
    eg += rooks_on_7th * 40

    return mg, eg


# ---------------------------------------------------------------------------
# Main evaluation — tapered
# ---------------------------------------------------------------------------

def evaluate(board: chess.Board) -> int:
    """Evaluate the board from the perspective of the side to move.

    Positive = good for side to move.
    Uses tapered evaluation blending MG and EG scores.
    """
    side = board.turn
    phase = game_phase(board)  # 256 = middlegame, 0 = endgame

    mg_score = 0
    eg_score = 0

    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == side else -1

        # Material + PSTs
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                           chess.ROOK, chess.QUEEN, chess.KING]:
            for sq in board.pieces(piece_type, color):
                mat = MATERIAL[piece_type]
                mg_pst, eg_pst = _pst_bonus(piece_type, sq, color)
                mg_score += sign * (mat + mg_pst)
                eg_score += sign * (mat + eg_pst)

        # King safety + mobility
        king_safety_mg, mob_mg, mob_eg = _piece_eval(board, color)
        mg_score += sign * (king_safety_mg + _pawn_shield_bonus(board, color) + _open_file_penalty(board, color))
        mg_score += sign * mob_mg
        eg_score += sign * mob_eg

        # Passed pawns
        pp_mg, pp_eg = _passed_pawn_eval(board, color)
        mg_score += sign * pp_mg
        eg_score += sign * pp_eg

        # Pawn structure
        ps_mg, ps_eg = _pawn_structure_eval(board, color)
        mg_score += sign * ps_mg
        eg_score += sign * ps_eg

        # Bishop pair
        bp_mg, bp_eg = _bishop_pair_bonus(board, color)
        mg_score += sign * bp_mg
        eg_score += sign * bp_eg

        # Rook open files + 7th rank
        rf_mg, rf_eg = _rook_file_bonus(board, color)
        mg_score += sign * rf_mg
        eg_score += sign * rf_eg

    # Taper: phase=256 → pure MG, phase=0 → pure EG
    return (mg_score * phase + eg_score * (256 - phase)) // 256


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "startpos"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "after 1.e4"),
        ("rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq - 0 1", "after 1.a3"),
        ("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "black missing queen"),
        ("8/8/8/4k3/8/8/8/4K2Q w - - 0 1", "Q+K vs K"),
        ("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2", "1.e4 Nc6"),
        ("rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2", "1.e4 Nf6"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "1.e4 e5 2.Nf3 Nc6"),
    ]

    print("Baseline eval self-test:")
    print("-" * 70)
    for fen, desc in tests:
        board = chess.Board(fen)
        val = evaluate(board)
        phase = game_phase(board)
        print(f"{desc:35s}  eval={val:+6d}cp  phase={phase:3d}  side={'white' if board.turn else 'black'}")
