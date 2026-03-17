use cozy_chess::Move;

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

pub const ALL_LEVELS: [Level; 4] = [
    Level { value: 1 },
    Level { value: 2 },
    Level { value: 3 },
    Level { value: 4 },
];

impl Level {
    pub fn new(n: u8) -> Option<Level> {
        if (1..=4).contains(&n) {
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
            3 => "Advanced",
            4 => "Expert",
            _ => unreachable!(),
        }
    }

    pub fn description(self) -> &'static str {
        match self.value {
            1 => "Depth 1, same search as your NN",
            2 => "Depth 2, sees your response to each move",
            3 => "Depth 3, with search optimizations",
            4 => "Depth 4, with search optimizations",
            _ => unreachable!(),
        }
    }

    pub fn depth(self) -> u32 {
        match self.value {
            1 => 1,
            2 => 2,
            3 => 3,
            4 => 4,
            _ => unreachable!(),
        }
    }

    pub fn enhanced(self) -> bool {
        self.value >= 3
    }
}

/// Baseline bot with configurable search depth.
pub struct BaselineBot {
    pub depth: u32,
    /// true = adds TT, PVS, NMP to the search
    pub enhanced: bool,
    /// Shared search context for enhanced mode (persists across moves)
    ctx: std::cell::RefCell<SearchContext>,
}

impl Default for BaselineBot {
    fn default() -> Self {
        BaselineBot {
            depth: 4,
            enhanced: true,
            ctx: std::cell::RefCell::new(SearchContext::new()),
        }
    }
}

impl BaselineBot {
    /// Create a baseline bot configured for the given level.
    pub fn from_level(level: Level) -> Self {
        BaselineBot {
            depth: level.depth(),
            enhanced: level.enhanced(),
            ctx: std::cell::RefCell::new(SearchContext::new()),
        }
    }

    /// Reset search context (call between games to avoid TT pollution).
    pub fn reset(&self) {
        *self.ctx.borrow_mut() = SearchContext::new();
    }
}

impl Bot for BaselineBot {
    fn choose_move(&self, game: &GameState) -> Option<Move> {
        let legal = game.legal_moves();
        if legal.is_empty() {
            return None;
        }

        let scored = if self.enhanced {
            let mut ctx = self.ctx.borrow_mut();
            best_move_with_scores_enhanced(&mut ctx, &game.board, self.depth)
        } else {
            best_move_with_scores_classic(&game.board, self.depth)
        };

        if scored.is_empty() {
            return legal.into_iter().next();
        }

        let best_score = scored.iter().map(|(_, s)| *s).max().unwrap();
        scored
            .into_iter()
            .find(|(_, s)| *s == best_score)
            .map(|(mv, _)| mv)
    }
}
