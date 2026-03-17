use cozy_chess::{Board, Color, GameStatus, Piece, Square};
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashMap;
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

/// Number of floats per perspective half: 12 piece planes × 64 squares + 2 castling rights.
const HALF_SIZE: usize = 770;

/// Total tensor size: two perspective halves.
pub const TENSOR_SIZE: usize = HALF_SIZE * 2; // 1540

/// Encode a board position as a flat [1540] float32 tensor (dual perspective).
///
/// Layout: two 770-element halves.  Each half contains:
///   - 768 floats: 12 piece planes × 64 squares
///   - 2 floats:   castling rights (kingside, queenside) for that half's color
///
/// First 770 (STM perspective):
/// - Channels 0–5:  side-to-move's pieces (P, N, B, R, Q, K)
/// - Channels 6–11: opponent's pieces      (P, N, B, R, Q, K)
/// - [768]: STM can castle kingside  (1.0 / 0.0)
/// - [769]: STM can castle queenside (1.0 / 0.0)
/// - Ranks flipped when STM is Black.
///
/// Last 770 (NSTM perspective):
/// - Channels 0–5:  non-side-to-move's pieces (P, N, B, R, Q, K)
/// - Channels 6–11: side-to-move's pieces      (P, N, B, R, Q, K)
/// - [1538]: NSTM can castle kingside  (1.0 / 0.0)
/// - [1539]: NSTM can castle queenside (1.0 / 0.0)
/// - Ranks flipped when NSTM is Black.
pub fn board_to_tensor(game: &GameState) -> Vec<f32> {
    let board = &game.board;
    let us = board.side_to_move();
    let them = !us;

    let mut tensor = vec![0.0f32; TENSOR_SIZE];

    // First half: STM perspective (our pieces ch 0-5, their pieces ch 6-11)
    let stm_flip = us == Color::Black;
    for (ch, &piece) in PIECE_TYPES.iter().enumerate() {
        for sq in board.colored_pieces(us, piece) {
            tensor[ch * 64 + square_idx(sq, stm_flip)] = 1.0;
        }
        for sq in board.colored_pieces(them, piece) {
            tensor[(ch + 6) * 64 + square_idx(sq, stm_flip)] = 1.0;
        }
    }
    // STM castling rights
    let stm_rights = board.castle_rights(us);
    if stm_rights.short.is_some() {
        tensor[768] = 1.0;
    }
    if stm_rights.long.is_some() {
        tensor[769] = 1.0;
    }

    // Second half: NSTM perspective (their pieces ch 0-5, our pieces ch 6-11)
    let nstm_flip = them == Color::Black;
    for (ch, &piece) in PIECE_TYPES.iter().enumerate() {
        for sq in board.colored_pieces(them, piece) {
            tensor[HALF_SIZE + ch * 64 + square_idx(sq, nstm_flip)] = 1.0;
        }
        for sq in board.colored_pieces(us, piece) {
            tensor[HALF_SIZE + (ch + 6) * 64 + square_idx(sq, nstm_flip)] = 1.0;
        }
    }
    // NSTM castling rights
    let nstm_rights = board.castle_rights(them);
    if nstm_rights.short.is_some() {
        tensor[HALF_SIZE + 768] = 1.0;
    }
    if nstm_rights.long.is_some() {
        tensor[HALF_SIZE + 769] = 1.0;
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
// NnEvalBot — scalar eval network with depth-1 + batched quiescence search
// ---------------------------------------------------------------------------

const MATE_SCORE_F: f32 = 100_000.0;
const DRAW_SCORE_F: f32 = 0.0;

/// A chess bot that runs an ONNX scalar evaluation network with depth-1
/// search plus quiescence (follows captures to quiet positions).
///
/// Uses batched inference: collects all positions in the quiescence trees
/// for every legal move, evaluates them in a single ONNX call, then runs
/// alpha-beta on the cached results.
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

    /// Max positions per ONNX batch call (limits memory for the flat tensor).
    const BATCH_CHUNK: usize = 4096;

    /// Evaluate a batch of positions, chunking if necessary.
    fn nn_eval_batch(
        &self,
        tensors: &[Vec<f32>],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let n = tensors.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let mut all_results = Vec::with_capacity(n);
        for chunk in tensors.chunks(Self::BATCH_CHUNK) {
            let cn = chunk.len();
            let mut flat = Vec::with_capacity(cn * TENSOR_SIZE);
            for t in chunk {
                flat.extend_from_slice(t);
            }

            let input = Tensor::<f32>::from_array(([cn, TENSOR_SIZE], flat))?;

            let mut session = self
                .session
                .lock()
                .map_err(|_| "session mutex poisoned")?;

            let outputs = session.run(ort::inputs!["board" => input])?;
            let (_, raw) = outputs[0].try_extract_tensor::<f32>()?;
            all_results.extend_from_slice(&raw.to_vec());
        }
        Ok(all_results)
    }

    /// Max positions to collect across all quiescence trees (safety valve).
    const MAX_COLLECT: usize = 50_000;
    /// Max depth for quiescence tree collection.
    const MAX_QS_DEPTH: u32 = 8;

    /// Recursively walk the quiescence tree, collecting every position
    /// that needs NN evaluation. Uses board hash for deduplication.
    fn collect_quiescence_positions(
        board: &Board,
        depth: u32,
        tensors: &mut Vec<Vec<f32>>,
        hash_to_idx: &mut HashMap<u64, usize>,
    ) {
        if tensors.len() >= Self::MAX_COLLECT || depth >= Self::MAX_QS_DEPTH {
            return;
        }

        let hash = board.hash();
        if hash_to_idx.contains_key(&hash) {
            return; // already collected
        }

        let idx = tensors.len();
        let game = GameState::from_board(board.clone());
        tensors.push(board_to_tensor(&game));
        hash_to_idx.insert(hash, idx);

        // Recurse into captures
        for mv in capture_moves(board) {
            if tensors.len() >= Self::MAX_COLLECT {
                break;
            }
            let mut child = board.clone();
            child.play_unchecked(mv);
            if child.status() == GameStatus::Ongoing {
                Self::collect_quiescence_positions(&child, depth + 1, tensors, hash_to_idx);
            }
        }
    }

    /// Alpha-beta quiescence search using pre-computed cached evaluations.
    fn quiescence_cached(
        board: &Board,
        mut alpha: f32,
        beta: f32,
        cache: &HashMap<u64, f32>,
    ) -> f32 {
        match board.status() {
            GameStatus::Won => return -MATE_SCORE_F,
            GameStatus::Drawn => return DRAW_SCORE_F,
            GameStatus::Ongoing => {}
        }

        let stand_pat = cache.get(&board.hash()).copied().unwrap_or(0.0);
        if stand_pat >= beta {
            return beta;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        for mv in capture_moves(board) {
            let mut child = board.clone();
            child.play_unchecked(mv);
            let score = -Self::quiescence_cached(&child, -beta, -alpha, cache);
            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }

    /// Depth-1 search with batched quiescence.
    ///
    /// 1. Expand all legal moves and walk their quiescence trees to collect
    ///    every position that needs NN evaluation.
    /// 2. Evaluate all positions in one batched ONNX call.
    /// 3. Run alpha-beta quiescence on each move using the cached values.
    fn try_choose_move(
        &self,
        game: &GameState,
    ) -> Result<Option<Move>, Box<dyn std::error::Error>> {
        let legal = game.legal_moves();
        if legal.is_empty() {
            return Ok(None);
        }

        // Phase 1: Collect all positions that need NN evaluation
        let mut tensors: Vec<Vec<f32>> = Vec::new();
        let mut hash_to_idx: HashMap<u64, usize> = HashMap::new();

        for &mv in &legal {
            let mut child = game.board.clone();
            child.play_unchecked(mv);
            if child.status() == GameStatus::Ongoing {
                Self::collect_quiescence_positions(&child, 0, &mut tensors, &mut hash_to_idx);
            }
        }

        // Phase 2: Batch NN evaluation
        let evals = self.nn_eval_batch(&tensors)?;

        // Build hash → eval lookup
        let mut cache: HashMap<u64, f32> = HashMap::with_capacity(hash_to_idx.len());
        for (&hash, &idx) in &hash_to_idx {
            cache.insert(hash, evals[idx]);
        }

        // Phase 3: Score each move using cached quiescence search
        let mut best_mv: Option<Move> = None;
        let mut best_eval = f32::NEG_INFINITY;

        for &mv in &legal {
            let mut child_board = game.board.clone();
            child_board.play_unchecked(mv);

            let eval = match child_board.status() {
                GameStatus::Won => MATE_SCORE_F,
                GameStatus::Drawn => DRAW_SCORE_F,
                GameStatus::Ongoing => {
                    -Self::quiescence_cached(
                        &child_board,
                        f32::NEG_INFINITY,
                        f32::INFINITY,
                        &cache,
                    )
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
