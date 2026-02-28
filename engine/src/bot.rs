use cozy_chess::Move;
use rand::Rng;

use crate::game::GameState;
use crate::search::{
    best_move_with_scores_classic, best_move_with_scores_enhanced, SearchContext,
};

pub trait Bot {
    fn choose_move(&self, game: &GameState) -> Option<Move>;
}

// ---------------------------------------------------------------------------
// Level system
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct Level {
    value: u8,
}

pub const ALL_LEVELS: [Level; 5] = [
    Level { value: 1 },
    Level { value: 2 },
    Level { value: 3 },
    Level { value: 4 },
    Level { value: 5 },
];

impl Level {
    pub fn new(n: u8) -> Option<Level> {
        if (1..=5).contains(&n) {
            Some(Level { value: n })
        } else {
            None
        }
    }

    pub fn value(self) -> u8 {
        self.value
    }

    pub fn name(self) -> &'static str {
        match self.value {
            1 => "Beginner",
            2 => "Novice",
            3 => "Intermediate",
            4 => "Advanced",
            5 => "Expert",
            _ => unreachable!(),
        }
    }

    pub fn description(self) -> &'static str {
        match self.value {
            1 => "Depth 1 classic + quiescence — both sides follow captures",
            2 => "Depth 2 classic — 2-ply alpha-beta + quiescence",
            3 => "Depth 3 classic — 3-ply alpha-beta + quiescence",
            4 => "Depth 3 enhanced — TT/PVS/NMP/delta pruning",
            5 => "Depth 4 enhanced — full strength baseline",
            _ => unreachable!(),
        }
    }

    pub fn depth(self) -> u32 {
        match self.value {
            1 => 1,
            2 => 2,
            3 => 3,
            4 => 3,
            5 => 4,
            _ => unreachable!(),
        }
    }

    pub fn enhanced(self) -> bool {
        self.value >= 4
    }
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

    /// Create a baseline bot configured for the given level.
    pub fn from_level(level: Level) -> Self {
        Self::new(level.depth(), 0, 0.0, level.enhanced())
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
