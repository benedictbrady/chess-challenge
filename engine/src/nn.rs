use cozy_chess::{Color, Piece, Square};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;

use crate::bot::Bot;
use crate::game::GameState;
use crate::Move;

// Piece channel order (matches both current-player and opponent halves)
const PIECE_TYPES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

/// Convert a square to a 0..64 index.
/// Square ordering: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63.
/// When `flip` is true (black to move), ranks are mirrored so the
/// current player always sees their pieces at "rank 1".
fn square_idx(sq: Square, flip: bool) -> usize {
    let file = sq.file() as usize;
    let rank = sq.rank() as usize;
    let rank = if flip { 7 - rank } else { rank };
    file + rank * 8
}

/// Encode a board position as a flat [768] float32 tensor.
///
/// Layout: 12 planes × 64 squares.
/// - Channels 0–5:  current player's pieces (P, N, B, R, Q, K)
/// - Channels 6–11: opponent's pieces       (P, N, B, R, Q, K)
///
/// Board is always represented from the current player's perspective:
/// if it is Black's turn, ranks are flipped so both sides appear to
/// "move up the board" from their own rank-1.
pub fn board_to_tensor(game: &GameState) -> Vec<f32> {
    let board = &game.board;
    let us = board.side_to_move();
    let them = !us;
    let flip = us == Color::Black;

    let mut tensor = vec![0.0f32; 768];

    for (ch, &piece) in PIECE_TYPES.iter().enumerate() {
        for sq in board.colored_pieces(us, piece) {
            tensor[ch * 64 + square_idx(sq, flip)] = 1.0;
        }
        for sq in board.colored_pieces(them, piece) {
            tensor[(ch + 6) * 64 + square_idx(sq, flip)] = 1.0;
        }
    }

    tensor
}

/// Pick the best legal move given raw policy logits of length 4096.
///
/// Logit index = from_square_idx * 64 + to_square_idx, where square
/// indices use the same flipped coordinate system as `board_to_tensor`.
///
/// Under-promotions are ignored; pawn moves to the back rank are
/// treated as queen promotions.
pub fn decode_policy(logits: &[f32], game: &GameState) -> Option<Move> {
    let flip = game.board.side_to_move() == Color::Black;
    let legal = game.legal_moves();

    if legal.is_empty() {
        return None;
    }

    let mut best_mv = None;
    let mut best_score = f32::NEG_INFINITY;

    for &mv in &legal {
        // Skip under-promotions; queen promotion and non-promotion moves are scored
        if mv.promotion.map_or(false, |p| p != Piece::Queen) {
            continue;
        }

        let from_idx = square_idx(mv.from, flip);
        let to_idx = square_idx(mv.to, flip);
        let score = logits[from_idx * 64 + to_idx];

        if score > best_score {
            best_score = score;
            best_mv = Some(mv);
        }
    }

    // Edge case: position had only under-promotions (extremely rare)
    best_mv.or_else(|| legal.into_iter().next())
}

// ---------------------------------------------------------------------------
// Parameter counting via minimal ONNX protobuf parsing
// ---------------------------------------------------------------------------

mod onnx_proto {
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct TensorProto {
        #[prost(int64, repeated, tag = "1")]
        pub dims: Vec<i64>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct GraphProto {
        #[prost(message, repeated, tag = "5")]
        pub initializer: Vec<TensorProto>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct ModelProto {
        #[prost(message, optional, tag = "7")]
        pub graph: Option<GraphProto>,
    }
}

/// Count every scalar parameter (weight or bias) stored as ONNX initializers.
pub fn count_parameters(path: &Path) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    use prost::Message;
    let bytes = std::fs::read(path)?;
    let model = onnx_proto::ModelProto::decode(bytes.as_slice())?;

    let mut total = 0u64;
    if let Some(graph) = model.graph {
        for tensor in &graph.initializer {
            if !tensor.dims.is_empty() {
                let count: u64 = tensor.dims.iter().map(|&d| d.max(1) as u64).product();
                total += count;
            }
        }
    }

    Ok(total)
}

// ---------------------------------------------------------------------------
// NnBot
// ---------------------------------------------------------------------------

/// A chess bot that runs an ONNX policy network.
///
/// The model must accept input "board" [1, 768] float32 and output
/// "policy" [1, 4096] float32 raw logits. See the competition spec
/// for the board encoding and move indexing convention.
pub struct NnBot {
    // Session::run takes &mut self, so we wrap in Mutex for Bot's &self interface
    session: Mutex<Session>,
    pub param_count: u64,
}

impl NnBot {
    pub fn load(path: &Path) -> Result<NnBot, Box<dyn std::error::Error + Send + Sync>> {
        let param_count = count_parameters(path)?;
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(NnBot {
            session: Mutex::new(session),
            param_count,
        })
    }

    fn try_choose_move(&self, game: &GameState) -> Result<Option<Move>, Box<dyn std::error::Error>> {
        let tensor_data = board_to_tensor(game);

        // Build [1, 768] input tensor
        let input = Tensor::<f32>::from_array(([1usize, 768], tensor_data))?;

        // Run inference — session.run() takes &mut self
        let mut session = self
            .session
            .lock()
            .map_err(|_| "session mutex poisoned")?;

        let outputs = session.run(ort::inputs!["board" => input])?;

        // Extract policy logits [1, 4096] → flat &[f32]
        let (_, logits_slice) = outputs[0].try_extract_tensor::<f32>()?;
        let logits = logits_slice.to_vec();

        Ok(decode_policy(&logits, game))
    }
}

impl Bot for NnBot {
    fn choose_move(&self, game: &GameState) -> Option<Move> {
        match self.try_choose_move(game) {
            Ok(mv) => mv,
            Err(e) => {
                eprintln!("NnBot inference error: {e}");
                None
            }
        }
    }
}
