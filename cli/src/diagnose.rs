/// Evaluate a set of diagnostic positions to identify eval blindspots.
/// Prints the eval score for each position alongside what the score SHOULD be.

use engine::eval::evaluate;
use engine::game::GameState;
use engine::search::best_move_with_scores;
use engine::{Board, Color, Move, Piece};

struct DiagPosition {
    fen: &'static str,
    description: &'static str,
    expected: &'static str, // qualitative expectation
}

const POSITIONS: &[DiagPosition] = &[
    // === Material imbalances ===
    DiagPosition {
        fen: "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        description: "KP vs K (white pawn on e2)",
        expected: "+200 to +400 (winning endgame)",
    },
    DiagPosition {
        fen: "4k3/8/8/8/4P3/8/8/4K3 w - - 0 1",
        description: "KP vs K (white pawn on e4)",
        expected: "+250 to +500 (more advanced = better)",
    },
    DiagPosition {
        fen: "4k3/8/4P3/8/8/8/8/4K3 w - - 0 1",
        description: "KP vs K (white pawn on e6, almost promoting)",
        expected: "+500 to +800 (nearly won)",
    },
    DiagPosition {
        fen: "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1",
        description: "KP vs K (white pawn on e7, one step from queening)",
        expected: "+700 to +900 (basically won)",
    },

    // === Passed pawn recognition ===
    DiagPosition {
        fen: "4k3/pp6/8/3P4/8/8/PP6/4K3 w - - 0 1",
        description: "White has passed d-pawn on d5, black pawns on a/b can't stop it",
        expected: "+150 to +300 (passed pawn advantage)",
    },
    DiagPosition {
        fen: "4k3/pp6/3P4/8/8/8/PP6/4K3 w - - 0 1",
        description: "White has passed d-pawn on d6 (more advanced)",
        expected: "+300 to +500 (dangerous passer)",
    },
    DiagPosition {
        fen: "4k3/8/8/3PP3/8/8/8/4K3 w - - 0 1",
        description: "Connected passed pawns on d5+e5",
        expected: "+400 to +700 (connected passers are very strong)",
    },
    DiagPosition {
        fen: "4k3/3p4/8/3P4/8/8/8/4K3 w - - 0 1",
        description: "Blocked passed pawn (d5 vs d7) — NOT a real passer threat",
        expected: "+50 to +150 (blocked, less dangerous)",
    },

    // === Piece activity ===
    DiagPosition {
        fen: "4k3/8/8/8/4N3/8/8/4K3 w - - 0 1",
        description: "Knight on e4 (center) vs bare king",
        expected: "+300 (knight value + central placement)",
    },
    DiagPosition {
        fen: "4k3/8/8/8/8/8/8/N3K3 w - - 0 1",
        description: "Knight on a1 (corner, trapped) vs bare king",
        expected: "+250 (knight value but terrible placement)",
    },
    DiagPosition {
        fen: "r3k3/8/8/8/8/8/8/R3K3 w - - 0 1",
        description: "Rook on open file vs rook not on open file (equal material)",
        expected: "~0 (material equal, slight positional edge)",
    },

    // === King position in endgame ===
    DiagPosition {
        fen: "8/4k3/8/3pKp2/3P1P2/8/8/8 w - - 0 1",
        description: "White king centralized (e5), black king passive (e7)",
        expected: "+50 to +150 (active king advantage)",
    },
    DiagPosition {
        fen: "8/8/8/3pKp2/3P1P2/8/8/4k3 w - - 0 1",
        description: "White king centralized, black king on back rank",
        expected: "+100 to +200 (much better king)",
    },

    // === Middlegame positions ===
    DiagPosition {
        fen: "r1bqkbnr/pppppppp/2n5/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        description: "White space advantage (pawn on e5)",
        expected: "+30 to +80 (slight space edge)",
    },
    DiagPosition {
        fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        description: "Standard position after 1.e4 (black to move)",
        expected: "-20 to +20 (roughly equal, slight tempo edge for white)",
    },

    // === Pawn structure ===
    DiagPosition {
        fen: "4k3/pp3ppp/8/8/8/8/PPP2PPP/4K3 w - - 0 1",
        description: "Equal pawns, symmetric structure",
        expected: "~0 (dead equal)",
    },
    DiagPosition {
        fen: "4k3/pp3ppp/8/8/8/8/PP1P1PPP/4K3 w - - 0 1",
        description: "White has isolated d-pawn (weakness)",
        expected: "~0 (equal material, slight structural weakness)",
    },
    DiagPosition {
        fen: "4k3/pp3ppp/8/8/8/3P4/PP1P1PPP/4K3 w - - 0 1",
        description: "White has doubled d-pawns (structural weakness but extra pawn)",
        expected: "+50 to +100 (extra pawn but doubled)",
    },

    // === Rook endgames ===
    DiagPosition {
        fen: "4k3/8/8/4p3/4P3/8/R7/4K3 w - - 0 1",
        description: "Rook + pawn vs pawn (white should win)",
        expected: "+400 to +600 (rook advantage is decisive)",
    },
    DiagPosition {
        fen: "4k3/R7/8/4p3/4P3/8/8/4K3 w - - 0 1",
        description: "Rook on 7th rank + pawn vs pawn",
        expected: "+500 to +700 (rook on 7th is dominant)",
    },
];

fn main() {
    println!("=== Eval Diagnostics ===");
    println!("Evaluating {} positions to identify blindspots\n", POSITIONS.len());

    for (i, pos) in POSITIONS.iter().enumerate() {
        let board: Board = pos.fen.parse().expect("valid FEN");
        let stm = board.side_to_move();
        let eval = evaluate(&board);

        // Also get the best move at depth 5
        let scored = best_move_with_scores(&board, 5);
        let best = scored.iter().max_by_key(|(_, s)| *s);
        let (best_mv, search_score) = match best {
            Some((mv, s)) => (format!("{}{}", mv.from, mv.to), *s),
            None => ("none".to_string(), eval),
        };

        let eval_display = if stm == Color::White {
            eval // already from white's perspective since white is STM
        } else {
            eval // eval is from STM's perspective, so this is black's view
        };

        let status = if eval.abs() < 30 && pos.expected.contains("+1") || pos.expected.contains("+2") || pos.expected.contains("+3") || pos.expected.contains("+4") || pos.expected.contains("+5") || pos.expected.contains("+6") || pos.expected.contains("+7") || pos.expected.contains("+8") {
            "BLIND"
        } else {
            "ok"
        };

        println!("Position {}: {}", i + 1, pos.description);
        println!("  FEN:      {}", pos.fen);
        println!("  STM:      {:?}", stm);
        println!("  Static:   {:+}cp", eval_display);
        println!("  Search:   {:+}cp (best: {})", search_score, best_mv);
        println!("  Expected: {}", pos.expected);
        println!("  Status:   {}", status);
        println!();
    }

    // Summary
    println!("=== Summary ===");
    println!("Static eval = evaluate() only (no search)");
    println!("Search eval = depth-5 alpha-beta + quiescence");
    println!("'BLIND' = eval severely underestimates a known advantage");
}
