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

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let board: Board = fen.parse().map_err(|e| format!("Invalid FEN: {:?}", e))?;
        let mut position_counts = HashMap::new();
        position_counts.insert(board.hash(), 1);
        Ok(GameState {
            board,
            history: Vec::new(),
            position_counts,
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_fen_italian_game() {
        // Italian Game after 1.e4 e5 2.Nf3 Nc6 3.Bc4
        let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3";
        let game = GameState::from_fen(fen).expect("valid FEN should parse");
        assert_eq!(game.side_to_move(), Color::Black);
        assert!(game.history.is_empty());
        assert!(!game.is_game_over());
        assert!(game.legal_moves().len() > 0);
    }

    #[test]
    fn from_fen_invalid() {
        let result = GameState::from_fen("not a valid fen");
        assert!(result.is_err());
    }

    #[test]
    fn from_fen_startpos_matches_new() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let from_fen = GameState::from_fen(fen).unwrap();
        let from_new = GameState::new();
        assert_eq!(from_fen.side_to_move(), from_new.side_to_move());
        assert_eq!(from_fen.legal_moves().len(), from_new.legal_moves().len());
    }
}
