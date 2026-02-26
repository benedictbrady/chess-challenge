use cozy_chess::Move;
use rand::Rng;

use crate::game::GameState;
use crate::search::best_move_with_scores;

pub trait Bot {
    fn choose_move(&self, game: &GameState) -> Option<Move>;
}

/// Strong baseline bot (~1600-1800 Elo) using alpha-beta search with
/// material + PST + king safety evaluation.
pub struct BaselineBot {
    /// Search depth for negamax
    pub depth: u32,
    /// Centipawn window — pick randomly from moves within this many cp of best
    pub candidate_window: i32,
    /// Probability (0.0–1.0) of playing a completely random legal move
    pub blunder_rate: f64,
}

impl Default for BaselineBot {
    fn default() -> Self {
        BaselineBot {
            depth: 5,
            candidate_window: 0,
            blunder_rate: 0.0,
        }
    }
}

impl BaselineBot {
    pub fn description() -> &'static str {
        "Alpha-beta depth 5 + quiescence, tapered eval (material, PSTs, king safety, passed pawns, mobility, pawn structure, bishop pair, rook bonuses)"
    }
}

impl Bot for BaselineBot {
    fn choose_move(&self, game: &GameState) -> Option<Move> {
        let legal = game.legal_moves();
        if legal.is_empty() {
            return None;
        }

        let mut rng = rand::thread_rng();

        // Blunder: random legal move
        if self.blunder_rate > 0.0 && rng.gen::<f64>() < self.blunder_rate {
            let idx = rng.gen_range(0..legal.len());
            return Some(legal[idx]);
        }

        // Score all moves
        let mut scored = best_move_with_scores(&game.board, self.depth);
        if scored.is_empty() {
            let idx = rng.gen_range(0..legal.len());
            return Some(legal[idx]);
        }

        // Find best score
        let best_score = scored.iter().map(|(_, s)| *s).max().unwrap();

        // Collect candidates within window of best
        let threshold = best_score - self.candidate_window;
        scored.retain(|(_, s)| *s >= threshold);

        // Pick randomly among candidates
        let idx = rng.gen_range(0..scored.len());
        Some(scored[idx].0)
    }
}
