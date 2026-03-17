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

#[cfg(test)]
mod tests {
    use super::*;
    use cozy_chess::GameStatus;

    #[test]
    fn level_new_boundaries() {
        assert!(Level::new(0).is_none());
        for n in 1..=4 {
            assert!(Level::new(n).is_some(), "Level::new({n}) should be Some");
        }
        assert!(Level::new(5).is_none());
        assert!(Level::new(255).is_none());
    }

    #[test]
    fn from_level_depth_and_enhanced() {
        let expected = [(1, 1, false), (2, 2, false), (3, 3, true), (4, 4, true)];
        for (n, depth, enhanced) in expected {
            let bot = BaselineBot::from_level(Level::new(n).unwrap());
            assert_eq!(bot.depth, depth, "level {n} depth");
            assert_eq!(bot.enhanced, enhanced, "level {n} enhanced");
        }
    }

    #[test]
    fn choose_move_returns_legal_from_startpos() {
        let bot = BaselineBot::from_level(Level::new(1).unwrap());
        let game = GameState::new();
        let mv = bot.choose_move(&game).expect("should find a move from startpos");
        assert!(
            game.legal_moves().contains(&mv),
            "returned move should be legal"
        );
    }

    #[test]
    fn choose_move_finds_mate_in_one() {
        // Fool's mate position: after 1.f3 e5 2.g4, Black plays Qh4#
        let fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2";
        let game = GameState::from_fen(fen).unwrap();
        let bot = BaselineBot::from_level(Level::new(1).unwrap());
        let mv = bot.choose_move(&game).expect("should find a move");
        let mut after = game.board.clone();
        after.play(mv);
        assert_eq!(
            after.status(),
            GameStatus::Won,
            "depth-1 bot should find Qh4# checkmate"
        );
    }
}
