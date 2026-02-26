use cozy_chess::Move;
use rand::Rng;

use crate::game::GameState;
use crate::search::{
    best_move_with_scores_classic, best_move_with_scores_enhanced, SearchContext,
};

pub trait Bot {
    fn choose_move(&self, game: &GameState) -> Option<Move>;
}

/// Baseline bot with configurable search mode.
pub struct BaselineBot {
    pub depth: u32,
    pub candidate_window: i32,
    pub blunder_rate: f64,
    /// true = enhanced (TT, PVS, NMP, delta pruning), false = classic
    pub enhanced: bool,
    /// Shared search context for enhanced mode (persists across moves)
    ctx: std::cell::RefCell<SearchContext>,
}

impl Default for BaselineBot {
    fn default() -> Self {
        BaselineBot {
            depth: 4,
            candidate_window: 0,
            blunder_rate: 0.0,
            enhanced: true,
            ctx: std::cell::RefCell::new(SearchContext::new()),
        }
    }
}

impl BaselineBot {
    pub fn new(depth: u32, candidate_window: i32, blunder_rate: f64, enhanced: bool) -> Self {
        BaselineBot {
            depth,
            candidate_window,
            blunder_rate,
            enhanced,
            ctx: std::cell::RefCell::new(SearchContext::new()),
        }
    }

    /// Create a classic (original) bot with no search enhancements.
    pub fn classic(depth: u32) -> Self {
        Self::new(depth, 0, 0.0, false)
    }

    /// Reset search context (call between games to avoid TT pollution).
    pub fn reset(&self) {
        *self.ctx.borrow_mut() = SearchContext::new();
    }

    pub fn description(&self) -> String {
        if self.enhanced {
            format!(
                "Alpha-beta depth {} + TT + PVS + NMP + delta pruning, tapered eval",
                self.depth
            )
        } else {
            format!(
                "Alpha-beta depth {} + quiescence (classic), tapered eval",
                self.depth
            )
        }
    }
}

impl Bot for BaselineBot {
    fn choose_move(&self, game: &GameState) -> Option<Move> {
        let legal = game.legal_moves();
        if legal.is_empty() {
            return None;
        }

        let mut rng = rand::thread_rng();

        if self.blunder_rate > 0.0 && rng.gen::<f64>() < self.blunder_rate {
            let idx = rng.gen_range(0..legal.len());
            return Some(legal[idx]);
        }

        let mut scored = if self.enhanced {
            let mut ctx = self.ctx.borrow_mut();
            best_move_with_scores_enhanced(&mut ctx, &game.board, self.depth)
        } else {
            best_move_with_scores_classic(&game.board, self.depth)
        };

        if scored.is_empty() {
            let idx = rng.gen_range(0..legal.len());
            return Some(legal[idx]);
        }

        let best_score = scored.iter().map(|(_, s)| *s).max().unwrap();
        let threshold = best_score - self.candidate_window;
        scored.retain(|(_, s)| *s >= threshold);

        let idx = rng.gen_range(0..scored.len());
        Some(scored[idx].0)
    }
}
