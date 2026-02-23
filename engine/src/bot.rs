use cozy_chess::Move;
use rand::Rng;

use crate::game::GameState;
use crate::search::best_move_with_scores;

pub trait Bot {
    fn choose_move(&self, game: &GameState) -> Option<Move>;
}

/// ~1000 ELO bot with some randomness and occasional blunders.
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
            depth: 3,
            candidate_window: 80,
            blunder_rate: 0.15,
        }
    }
}

// ---------------------------------------------------------------------------
// Challenger fleet
// ---------------------------------------------------------------------------

/// Configuration for one of the 5 challenger bots in the competition fleet.
pub struct ChallengerConfig {
    pub name: &'static str,
    pub description: &'static str,
    pub depth: u32,
    pub candidate_window: i32,
    pub blunder_rate: f64,
}

impl ChallengerConfig {
    pub fn to_bot(&self) -> BaselineBot {
        BaselineBot {
            depth: self.depth,
            candidate_window: self.candidate_window,
            blunder_rate: self.blunder_rate,
        }
    }
}

pub const CHALLENGERS: &[ChallengerConfig] = &[
    ChallengerConfig {
        name: "Grunt",
        description: "Shallow but principled (depth=2, window=50cp, blunder=10%)",
        depth: 2,
        candidate_window: 50,
        blunder_rate: 0.10,
    },
    ChallengerConfig {
        name: "Fortress",
        description: "Always plays the engine's best move when not blundering (depth=3, window=0cp, blunder=12%)",
        depth: 3,
        candidate_window: 0,
        blunder_rate: 0.12,
    },
    ChallengerConfig {
        name: "Scholar",
        description: "Deep calculator with high blunder rate (depth=4, window=40cp, blunder=20%)",
        depth: 4,
        candidate_window: 40,
        blunder_rate: 0.20,
    },
    ChallengerConfig {
        name: "Chaos",
        description: "Extremely wide candidate window, unpredictable play (depth=2, window=200cp, blunder=8%)",
        depth: 2,
        candidate_window: 200,
        blunder_rate: 0.08,
    },
    ChallengerConfig {
        name: "Wall",
        description: "Zero blunders, must outplay on pure merit (depth=3, window=60cp, blunder=0%)",
        depth: 3,
        candidate_window: 60,
        blunder_rate: 0.0,
    },
];

pub fn challenger_by_name(name: &str) -> Option<&'static ChallengerConfig> {
    CHALLENGERS.iter().find(|c| c.name.eq_ignore_ascii_case(name))
}

impl Bot for BaselineBot {
    fn choose_move(&self, game: &GameState) -> Option<Move> {
        let legal = game.legal_moves();
        if legal.is_empty() {
            return None;
        }

        let mut rng = rand::thread_rng();

        // 15% chance to blunder with a fully random move
        if rng.gen::<f64>() < self.blunder_rate {
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
