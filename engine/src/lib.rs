pub mod bot;
pub mod eval;
pub mod game;
pub mod nn;
pub mod openings;
pub mod search;

pub use bot::{ChallengerConfig, CHALLENGERS, challenger_by_name};
pub use cozy_chess::{Board, Color, File, Move, Piece, Rank, Square};
pub use nn::NnBot;
