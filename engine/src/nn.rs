use cozy_chess::{Board, Color, GameStatus, Piece, Square};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;

use crate::bot::Bot;
use crate::game::GameState;
use crate::search::capture_moves;
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
    pub struct AttributeProto {
        #[prost(message, optional, tag = "5")]
        pub t: Option<TensorProto>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct NodeProto {
        #[prost(string, tag = "4")]
        pub op_type: String,
        #[prost(message, repeated, tag = "5")]
        pub attribute: Vec<AttributeProto>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct GraphProto {
        #[prost(message, repeated, tag = "1")]
        pub node: Vec<NodeProto>,
        #[prost(message, repeated, tag = "5")]
        pub initializer: Vec<TensorProto>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct ModelProto {
        #[prost(message, optional, tag = "7")]
        pub graph: Option<GraphProto>,
    }
}

/// Count every scalar parameter stored in ONNX initializers and Constant nodes.
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

        for node in &graph.node {
            if node.op_type == "Constant" {
                for attr in &node.attribute {
                    if let Some(tensor) = &attr.t {
                        if !tensor.dims.is_empty() {
                            let count: u64 =
                                tensor.dims.iter().map(|&d| d.max(1) as u64).product();
                            total += count;
                        }
                    }
                }
            }
        }
    }

    Ok(total)
}

// ---------------------------------------------------------------------------
// NnEvalBot — scalar eval network with depth-1 + quiescence search
// ---------------------------------------------------------------------------

const MATE_SCORE_F: f32 = 100_000.0;
const DRAW_SCORE_F: f32 = 0.0;

/// A chess bot that runs an ONNX scalar evaluation network with depth-1
/// search plus quiescence (follows captures to quiet positions).
///
/// The model must accept input "board" [N, 768] float32 and output a scalar
/// eval [N, 1] float32 (positive = good for side to move).
pub struct NnEvalBot {
    session: Mutex<Session>,
    pub param_count: u64,
}

impl NnEvalBot {
    pub fn load(path: &Path) -> Result<NnEvalBot, Box<dyn std::error::Error + Send + Sync>> {
        let param_count = count_parameters(path)?;
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(NnEvalBot {
            session: Mutex::new(session),
            param_count,
        })
    }

    /// Evaluate a batch of positions in a single ONNX call.
    /// Each tensor in `tensors` is a flat [768] encoding.
    /// Returns one scalar eval per position.
    /// Falls back to one-at-a-time inference if batching fails.
    fn nn_eval_batch(
        &self,
        tensors: &[Vec<f32>],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let n = tensors.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Try batched inference first
        if let Ok(results) = self.nn_eval_batch_inner(tensors) {
            return Ok(results);
        }

        // Fall back to one-at-a-time
        let mut results = Vec::with_capacity(n);
        for t in tensors {
            let r = self.nn_eval_batch_inner(&[t.clone()])?;
            results.push(r[0]);
        }
        Ok(results)
    }

    fn nn_eval_batch_inner(
        &self,
        tensors: &[Vec<f32>],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let n = tensors.len();

        let mut flat = Vec::with_capacity(n * 768);
        for t in tensors {
            flat.extend_from_slice(t);
        }

        let input = Tensor::<f32>::from_array(([n, 768], flat))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| "session mutex poisoned")?;

        let outputs = session.run(ort::inputs!["board" => input])?;

        let (_, raw) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(raw.to_vec())
    }

    /// Evaluate a single position. Returns eval from the perspective of the side to move.
    pub fn nn_eval(&self, game: &GameState) -> Result<f32, Box<dyn std::error::Error>> {
        let tensor = board_to_tensor(game);
        let results = self.nn_eval_batch(&[tensor])?;
        Ok(results[0])
    }

    /// Quiescence search using the NN eval. Follows captures until the
    /// position is quiet, then returns the NN evaluation.
    fn quiescence_nn(
        &self,
        board: &Board,
        mut alpha: f32,
        beta: f32,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        match board.status() {
            GameStatus::Won => return Ok(-MATE_SCORE_F),
            GameStatus::Drawn => return Ok(DRAW_SCORE_F),
            GameStatus::Ongoing => {}
        }

        let stand_pat = self.nn_eval(&GameState::from_board(board.clone()))?;
        if stand_pat >= beta {
            return Ok(beta);
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        for mv in capture_moves(board) {
            let mut child = board.clone();
            child.play_unchecked(mv);
            let score = -self.quiescence_nn(&child, -beta, -alpha)?;
            if score >= beta {
                return Ok(beta);
            }
            if score > alpha {
                alpha = score;
            }
        }

        Ok(alpha)
    }

    /// Depth-1 search with quiescence: for each legal move, run quiescence
    /// on the resulting position to follow captures to quiet positions.
    fn try_choose_move(
        &self,
        game: &GameState,
    ) -> Result<Option<Move>, Box<dyn std::error::Error>> {
        let legal = game.legal_moves();
        if legal.is_empty() {
            return Ok(None);
        }

        let mut best_mv: Option<Move> = None;
        let mut best_eval = f32::NEG_INFINITY;

        for &mv in &legal {
            let mut child_board = game.board.clone();
            child_board.play_unchecked(mv);

            let eval = match child_board.status() {
                GameStatus::Won => MATE_SCORE_F,
                GameStatus::Drawn => DRAW_SCORE_F,
                GameStatus::Ongoing => {
                    -self.quiescence_nn(&child_board, f32::NEG_INFINITY, f32::INFINITY)?
                }
            };

            if eval > best_eval {
                best_eval = eval;
                best_mv = Some(mv);
            }

            // Immediate checkmate — no need to keep searching
            if eval >= MATE_SCORE_F {
                break;
            }
        }

        if best_mv.is_none() {
            best_mv = legal.into_iter().next();
        }

        Ok(best_mv)
    }
}

impl Bot for NnEvalBot {
    fn choose_move(&self, game: &GameState) -> Option<Move> {
        match self.try_choose_move(game) {
            Ok(mv) => mv,
            Err(e) => {
                eprintln!("NnEvalBot inference error: {e}");
                None
            }
        }
    }
}
