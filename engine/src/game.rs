use cozy_chess::{Board, Color, GameStatus, Move};
use std::collections::HashMap;

#[derive(Clone)]
pub struct GameState {
    pub board: Board,
    pub history: Vec<Move>,
    position_counts: HashMap<u64, u32>,
}

impl GameState {
    pub fn new() -> Self {
        let board = Board::default();
        let mut position_counts = HashMap::new();
        position_counts.insert(board.hash(), 1);
        GameState {
            board,
            history: Vec::new(),
            position_counts,
        }
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut moves = Vec::new();
        self.board.generate_moves(|piece_moves| {
            moves.extend(piece_moves);
            false
        });
        moves
    }

    pub fn make_move(&mut self, mv: Move) -> bool {
        let mut legal = false;
        self.board.generate_moves(|piece_moves| {
            if piece_moves.into_iter().any(|m| m == mv) {
                legal = true;
                true
            } else {
                false
            }
        });

        if !legal {
            return false;
        }

        self.board.play(mv);
        self.history.push(mv);
        let hash = self.board.hash();
        *self.position_counts.entry(hash).or_insert(0) += 1;
        true
    }

    pub fn is_game_over(&self) -> bool {
        self.board.status() != GameStatus::Ongoing || self.is_threefold_repetition()
    }

    pub fn is_threefold_repetition(&self) -> bool {
        self.position_counts
            .get(&self.board.hash())
            .copied()
            .unwrap_or(0)
            >= 3
    }

    pub fn outcome(&self) -> Option<Outcome> {
        if self.is_threefold_repetition() {
            return Some(Outcome::Draw);
        }
        match self.board.status() {
            GameStatus::Won => Some(Outcome::Checkmate {
                winner: !self.board.side_to_move(),
            }),
            GameStatus::Drawn => Some(Outcome::Draw),
            GameStatus::Ongoing => None,
        }
    }

    pub fn side_to_move(&self) -> Color {
        self.board.side_to_move()
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Checkmate { winner: Color },
    Draw,
}
