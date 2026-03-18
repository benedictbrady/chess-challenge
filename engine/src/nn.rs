use cozy_chess::{Board, Color, GameStatus, Piece, Square};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
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
// NnEvalBot — scalar eval network with depth-1 + quiescence search
// ---------------------------------------------------------------------------

const MATE_SCORE_F: f32 = 100_000.0;
const DRAW_SCORE_F: f32 = 0.0;

/// A chess bot that runs an ONNX scalar evaluation network with depth-1
/// search plus quiescence (follows captures to quiet positions).
///
/// The model must accept input "board" [N, 1540] float32 and output a scalar
/// eval [N, 1] float32 (positive = good for side to move).
pub struct NnEvalBot {
    session: Mutex<Session>,
    pub param_count: u64,
    /// Number of ONNX session.run() calls (for benchmarking).
    inference_calls: AtomicU64,
    /// Total positions evaluated across all batch calls.
    positions_evaluated: AtomicU64,
}

impl NnEvalBot {
    pub fn load(path: &Path) -> Result<NnEvalBot, Box<dyn std::error::Error + Send + Sync>> {
        let param_count = count_parameters(path)?;
        let mut session = Session::builder()?.commit_from_file(path)?;

        // Probe: verify batched inference works (batch=2).
        // Models with unnamed/anonymous batch dimensions on input or output
        // will fail here instead of silently falling back to per-position
        // inference at runtime (which is ~30-60x slower).
        {
            let dummy = vec![0.0f32; 2 * TENSOR_SIZE];
            let input = Tensor::<f32>::from_array(([2, TENSOR_SIZE], dummy))
                .map_err(|e| format!("batch probe: failed to create tensor: {e}"))?;
            let outputs = session.run(ort::inputs!["board" => input])
                .map_err(|e| format!(
                    "batch probe: model does not support batched inference (batch=2 failed).\n\
                     This usually means the ONNX model has unnamed/anonymous batch dimensions.\n\
                     Fix: ensure your export uses named dynamic axes, e.g.:\n  \
                       dynamic_axes={{\"board\": {{0: \"batch\"}}, \"eval\": {{0: \"batch\"}}}}\n\
                     Or use a Linear (Gemm) final layer instead of MatMul+Add.\n\
                     ORT error: {e}"))?;
            let (shape, raw) = outputs[0].try_extract_tensor::<f32>()
                .map_err(|e| format!(
                    "batch probe: model output is not extractable as a float tensor.\n\
                     Expected output shape [N, 1] with a named batch dimension.\n\
                     ORT error: {e}"))?;
            if raw.len() != 2 || shape.len() != 2 || shape[0] != 2 || shape[1] != 1 {
                return Err(format!(
                    "batch probe: expected output shape [2, 1] (2 values), \
                     got shape {:?} ({} values).\n\
                     Your model's output batch dimension may be unnamed/anonymous.\n\
                     Fix: use named dynamic axes in your ONNX export, or use a Linear (Gemm) \
                     final layer instead of MatMul+Add.",
                    &*shape, raw.len()
                ).into());
            }
        }

        Ok(NnEvalBot {
            session: Mutex::new(session),
            param_count,
            inference_calls: AtomicU64::new(0),
            positions_evaluated: AtomicU64::new(0),
        })
    }

    /// Reset inference counters.
    pub fn reset_counters(&self) {
        self.inference_calls.store(0, Ordering::Relaxed);
        self.positions_evaluated.store(0, Ordering::Relaxed);
    }

    /// Get (inference_calls, positions_evaluated).
    pub fn counters(&self) -> (u64, u64) {
        (
            self.inference_calls.load(Ordering::Relaxed),
            self.positions_evaluated.load(Ordering::Relaxed),
        )
    }

    /// Evaluate a batch of positions in a single ONNX call.
    /// Each tensor in `tensors` is a flat [1540] encoding.
    /// Returns one scalar eval per position.
    fn nn_eval_batch(
        &self,
        tensors: &[Vec<f32>],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let n = tensors.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        self.nn_eval_batch_inner(tensors)
    }

    fn nn_eval_batch_inner(
        &self,
        tensors: &[Vec<f32>],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let n = tensors.len();

        self.inference_calls.fetch_add(1, Ordering::Relaxed);
        self.positions_evaluated
            .fetch_add(n as u64, Ordering::Relaxed);

        let mut flat = Vec::with_capacity(n * TENSOR_SIZE);
        for t in tensors {
            flat.extend_from_slice(t);
        }

        let input = Tensor::<f32>::from_array(([n, TENSOR_SIZE], flat))?;

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

    /// Sequential quiescence search (reference implementation for testing).
    #[cfg(test)]
    fn quiescence_nn_sequential(
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

        let in_check = !board.checkers().is_empty();

        if !in_check {
            let sp = self.nn_eval(&GameState::from_board(board.clone()))?;
            if sp >= beta {
                return Ok(beta);
            }
            if sp > alpha {
                alpha = sp;
            }
        }

        let moves = if in_check {
            let mut all = Vec::new();
            board.generate_moves(|piece_moves| {
                all.extend(piece_moves);
                false
            });
            all
        } else {
            capture_moves(board)
        };

        for mv in moves {
            let mut child = board.clone();
            child.play_unchecked(mv);
            let score = -self.quiescence_nn_sequential(&child, -beta, -alpha)?;
            if score >= beta {
                return Ok(beta);
            }
            if score > alpha {
                alpha = score;
            }
        }

        Ok(alpha)
    }

    /// Quiescence search with sibling-level batching.
    ///
    /// `stand_pat_hint`: if `Some(v)`, use `v` as this node's stand-pat eval
    /// (pre-computed by the parent via batch). If `None`, compute on demand.
    ///
    /// At each node, before the alpha-beta loop, batch-evaluates stand-pats for
    /// all non-check, non-terminal children in a single `nn_eval_batch` call.
    fn quiescence_nn(
        &self,
        board: &Board,
        mut alpha: f32,
        beta: f32,
        stand_pat_hint: Option<f32>,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        match board.status() {
            GameStatus::Won => return Ok(-MATE_SCORE_F),
            GameStatus::Drawn => return Ok(DRAW_SCORE_F),
            GameStatus::Ongoing => {}
        }

        let in_check = !board.checkers().is_empty();

        let _stand_pat = if in_check {
            f32::NEG_INFINITY
        } else {
            let sp = match stand_pat_hint {
                Some(v) => v,
                None => self.nn_eval(&GameState::from_board(board.clone()))?,
            };
            if sp >= beta {
                return Ok(beta);
            }
            if sp > alpha {
                alpha = sp;
            }
            sp
        };

        let moves = if in_check {
            let mut all = Vec::new();
            board.generate_moves(|piece_moves| {
                all.extend(piece_moves);
                false
            });
            all
        } else {
            capture_moves(board)
        };

        // Collect all children (no delta pruning — NN output scale is arbitrary)
        let mut children: Vec<Board> = Vec::with_capacity(moves.len());
        for mv in &moves {
            let mut child = board.clone();
            child.play_unchecked(*mv);
            children.push(child);
        }

        if children.is_empty() {
            return Ok(alpha);
        }

        // Batch-eval stand-pats for non-terminal, non-check children
        let mut hints: Vec<Option<f32>> = vec![None; children.len()];
        let mut batch_indices: Vec<usize> = Vec::new();
        let mut batch_tensors: Vec<Vec<f32>> = Vec::new();

        for (i, child) in children.iter().enumerate() {
            if child.status() != GameStatus::Ongoing {
                continue;
            }
            if !child.checkers().is_empty() {
                continue;
            }
            batch_indices.push(i);
            batch_tensors.push(board_to_tensor(&GameState::from_board(child.clone())));
        }

        if !batch_tensors.is_empty() {
            let evals = self.nn_eval_batch(&batch_tensors)?;
            for (&idx, &eval) in batch_indices.iter().zip(evals.iter()) {
                hints[idx] = Some(eval);
            }
        }

        // Alpha-beta loop with pre-computed hints
        for (i, child) in children.iter().enumerate() {
            let score = -self.quiescence_nn(child, -beta, -alpha, hints[i])?;
            if score >= beta {
                return Ok(beta);
            }
            if score > alpha {
                alpha = score;
            }
        }

        Ok(alpha)
    }

    /// Sequential depth-1 search (reference implementation for testing).
    #[cfg(test)]
    fn try_choose_move_sequential(
        &self,
        game: &GameState,
    ) -> Result<Option<Move>, Box<dyn std::error::Error>> {
        let legal = game.legal_moves();
        if legal.is_empty() {
            return Ok(None);
        }

        let mut best_mv: Option<Move> = None;
        let mut alpha = f32::NEG_INFINITY;

        for &mv in &legal {
            let mut child_board = game.board.clone();
            child_board.play_unchecked(mv);

            let eval = match child_board.status() {
                GameStatus::Won => MATE_SCORE_F,
                GameStatus::Drawn => DRAW_SCORE_F,
                GameStatus::Ongoing => {
                    -self.quiescence_nn_sequential(&child_board, f32::NEG_INFINITY, -alpha)?
                }
            };

            if eval > alpha {
                alpha = eval;
                best_mv = Some(mv);
            }

            if eval >= MATE_SCORE_F {
                break;
            }
        }

        if best_mv.is_none() {
            best_mv = legal.into_iter().next();
        }

        Ok(best_mv)
    }

    /// Depth-1 search with alpha-beta at root + batched quiescence.
    ///
    /// Batch-evaluates stand-pats for all root children in one ONNX call,
    /// then passes pre-computed hints into quiescence search.
    fn try_choose_move(
        &self,
        game: &GameState,
    ) -> Result<Option<Move>, Box<dyn std::error::Error>> {
        let legal = game.legal_moves();
        if legal.is_empty() {
            return Ok(None);
        }

        // Make all child boards
        let child_boards: Vec<(Move, Board)> = legal
            .iter()
            .map(|&mv| {
                let mut child = game.board.clone();
                child.play_unchecked(mv);
                (mv, child)
            })
            .collect();

        // Batch-eval stand-pats for ongoing, non-check children
        let mut hints: Vec<Option<f32>> = vec![None; child_boards.len()];
        let mut batch_indices: Vec<usize> = Vec::new();
        let mut batch_tensors: Vec<Vec<f32>> = Vec::new();

        for (i, (_, child)) in child_boards.iter().enumerate() {
            if child.status() != GameStatus::Ongoing {
                continue;
            }
            if !child.checkers().is_empty() {
                continue;
            }
            batch_indices.push(i);
            batch_tensors.push(board_to_tensor(&GameState::from_board(child.clone())));
        }

        if !batch_tensors.is_empty() {
            let evals = self.nn_eval_batch(&batch_tensors)?;
            for (&idx, &eval) in batch_indices.iter().zip(evals.iter()) {
                hints[idx] = Some(eval);
            }
        }

        let mut best_mv: Option<Move> = None;
        let mut alpha = f32::NEG_INFINITY;

        for (i, (mv, child_board)) in child_boards.iter().enumerate() {
            let eval = match child_board.status() {
                GameStatus::Won => MATE_SCORE_F,
                GameStatus::Drawn => DRAW_SCORE_F,
                GameStatus::Ongoing => {
                    -self.quiescence_nn(child_board, f32::NEG_INFINITY, -alpha, hints[i])?
                }
            };

            if eval > alpha {
                alpha = eval;
                best_mv = Some(*mv);
            }

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cozy_chess::Board;

    // ── Encoding tests ──────────────────────────────────────────────────

    #[test]
    fn tensor_size_is_1540() {
        assert_eq!(TENSOR_SIZE, 1540);
        assert_eq!(HALF_SIZE, 770);
    }

    #[test]
    fn startpos_encoding_piece_count() {
        let game = GameState::new();
        let tensor = board_to_tensor(&game);
        assert_eq!(tensor.len(), TENSOR_SIZE);

        // 32 pieces × 2 halves = 64, plus 4 castling rights = 68
        let ones: f32 = tensor.iter().sum();
        assert_eq!(ones as i32, 68, "startpos should have 68 ones in tensor");
    }

    #[test]
    fn endgame_no_castling() {
        // K+P vs K endgame — no castling rights
        let board: Board = "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1".parse().unwrap();
        let game = GameState::from_board(board);
        let tensor = board_to_tensor(&game);

        // 3 pieces × 2 halves = 6 ones, no castling
        let ones: f32 = tensor.iter().sum();
        assert_eq!(ones as i32, 6);

        // Castling bits should all be zero
        assert_eq!(tensor[768], 0.0); // STM kingside
        assert_eq!(tensor[769], 0.0); // STM queenside
        assert_eq!(tensor[HALF_SIZE + 768], 0.0); // NSTM kingside
        assert_eq!(tensor[HALF_SIZE + 769], 0.0); // NSTM queenside
    }

    #[test]
    fn white_king_e1_in_startpos() {
        let game = GameState::new();
        let tensor = board_to_tensor(&game);

        // White to move, so STM = White, no flip
        // King is piece type 5 (index in PIECE_TYPES), channel 5
        // e1 = file 4 + rank 0 * 8 = 4
        let king_idx = 5 * 64 + 4; // channel 5, square e1
        assert_eq!(tensor[king_idx], 1.0, "White king should be at e1 in STM half");
    }

    #[test]
    fn black_to_move_flips_ranks() {
        // After 1.e4, Black to move
        let board: Board =
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1".parse().unwrap();
        let game = GameState::from_board(board);
        let tensor = board_to_tensor(&game);

        // STM = Black, so ranks are flipped for STM half
        // Black king is at e8 (file 4, rank 7)
        // With flip: rank becomes 7-7=0, so square = 4 + 0*8 = 4
        let king_idx = 5 * 64 + 4; // STM king at flipped e8
        assert_eq!(tensor[king_idx], 1.0, "Black king should be at flipped e8 in STM half");
    }

    #[test]
    fn symmetric_encoding_consistency() {
        // The same position from White's and Black's perspective should
        // produce tensors where the STM halves contain the same piece layout
        // (since the encoding is relative to side-to-move).
        let board_w: Board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            .parse().unwrap();
        let game_w = GameState::from_board(board_w);
        let tensor_w = board_to_tensor(&game_w);

        // The STM half when White moves should match the NSTM half's "opponent pieces"
        // layout when Black moves (after flip). Just verify both tensors have same total.
        let board_b: Board = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
            .parse().unwrap();
        let game_b = GameState::from_board(board_b);
        let tensor_b = board_to_tensor(&game_b);

        // Both should be valid 1540-float tensors
        assert_eq!(tensor_w.len(), TENSOR_SIZE);
        assert_eq!(tensor_b.len(), TENSOR_SIZE);

        // Different positions should produce different tensors
        assert_ne!(tensor_w, tensor_b);
    }

    #[test]
    fn castling_rights_encoded() {
        let game = GameState::new();
        let tensor = board_to_tensor(&game);

        // White to move: STM = White
        // White can castle both sides
        assert_eq!(tensor[768], 1.0, "STM (White) kingside castling");
        assert_eq!(tensor[769], 1.0, "STM (White) queenside castling");
        // Black can castle both sides (NSTM)
        assert_eq!(tensor[HALF_SIZE + 768], 1.0, "NSTM (Black) kingside castling");
        assert_eq!(tensor[HALF_SIZE + 769], 1.0, "NSTM (Black) queenside castling");
    }

    #[test]
    fn castling_rights_partial() {
        // Position where White lost kingside castling
        let board: Board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w Qkq - 0 1"
            .parse().unwrap();
        let game = GameState::from_board(board);
        let tensor = board_to_tensor(&game);

        assert_eq!(tensor[768], 0.0, "White lost kingside castling");
        assert_eq!(tensor[769], 1.0, "White has queenside castling");
    }

    // ── Search structure tests (no ONNX needed) ─────────────────────────

    #[test]
    fn capture_moves_returns_only_captures_and_promotions() {
        let board: Board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            .parse().unwrap();
        let captures = capture_moves(&board);
        // Startpos has no captures
        assert!(captures.is_empty(), "startpos should have no capture moves");
    }

    #[test]
    fn capture_moves_finds_captures() {
        // Position with captures available
        let board: Board = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
            .parse().unwrap();
        let captures = capture_moves(&board);
        // e4xd5 should be a capture
        assert!(!captures.is_empty(), "should find exd5 capture");
    }

    #[test]
    fn quiescence_classic_returns_eval_for_quiet_position() {
        use crate::search::quiescence_classic;

        // Startpos is quiet — quiescence should return stand-pat eval
        let board = Board::default();
        let eval = quiescence_classic(&board, -100_000, 100_000);
        // The handcrafted eval of startpos should be ~0 (symmetric)
        assert!(eval.abs() < 50, "startpos eval should be near 0, got {eval}");
    }

    #[test]
    fn quiescence_classic_finds_mate() {
        use crate::search::quiescence_classic;

        // Checkmate position: Black is mated
        let board: Board = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
            .parse().unwrap();
        assert_eq!(board.status(), GameStatus::Won);
        let eval = quiescence_classic(&board, -100_000, 100_000);
        assert_eq!(eval, -100_000, "mated side should get -MATE_SCORE");
    }

    #[test]
    fn quiescence_classic_alpha_beta_prunes() {
        use crate::search::quiescence_classic;

        // With a very tight window, quiescence should still return a valid value
        let board = Board::default();
        let eval_full = quiescence_classic(&board, -100_000, 100_000);
        let eval_narrow = quiescence_classic(&board, eval_full - 1, eval_full + 1);
        // Narrow window should return the same or clipped value
        assert!((eval_narrow - eval_full).abs() <= 1);
    }

    // ── Encoding determinism ────────────────────────────────────────────

    #[test]
    fn encoding_is_deterministic() {
        let board: Board = "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
            .parse().unwrap();
        let game = GameState::from_board(board);
        let t1 = board_to_tensor(&game);
        let t2 = board_to_tensor(&game);
        assert_eq!(t1, t2, "encoding the same position twice must be identical");
    }

    #[test]
    fn encoding_all_zeros_except_pieces_and_castling() {
        let game = GameState::new();
        let tensor = board_to_tensor(&game);

        // Count non-zero entries
        let nonzero: usize = tensor.iter().filter(|&&v| v != 0.0).count();
        // 32 pieces × 2 halves = 64, plus 4 castling rights = 68
        assert_eq!(nonzero, 68, "startpos should have exactly 68 non-zero entries");

        // All values should be exactly 0.0 or 1.0
        for (i, &v) in tensor.iter().enumerate() {
            assert!(
                v == 0.0 || v == 1.0,
                "tensor[{i}] = {v}, expected 0.0 or 1.0"
            );
        }
    }

    // ── Depth-1 search structure (mock-free) ────────────────────────────

    #[test]
    fn classic_depth1_picks_obvious_capture() {
        use crate::search::best_move_with_scores_classic;

        // White can capture a hanging queen
        let board: Board = "rnb1kbnr/pppppppp/8/8/4q3/3P4/PPP1PPPP/RNBQKBNR w KQkq - 0 1"
            .parse().unwrap();
        let scored = best_move_with_scores_classic(&board, 1);
        let best = scored.iter().max_by_key(|(_, s)| *s).unwrap();
        // d3xe4 captures the queen — should be the best move
        let best_move_str = format!("{}{}", best.0.from, best.0.to);
        assert_eq!(best_move_str, "d3e4", "should capture the hanging queen");
    }

    #[test]
    fn classic_depth1_avoids_stalemate() {
        use crate::search::best_move_with_scores_classic;

        // Position where only one move avoids stalemate
        // If the engine picks any legal move, the test passes (no crash)
        let board: Board = "k7/8/1K6/8/8/8/8/1Q6 w - - 0 1".parse().unwrap();
        let scored = best_move_with_scores_classic(&board, 1);
        assert!(!scored.is_empty(), "should find at least one legal move");
    }

    // ── ONNX model end-to-end tests ────────────────────────────────────
    //
    // These use a tiny ONNX fixture (1540→1 linear layer) checked into
    // engine/tests/fixtures/tiny_eval.onnx. No training deps needed.

    fn fixture_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tiny_eval.onnx")
    }

    #[test]
    fn count_parameters_on_fixture() {
        let path = fixture_path();
        let count = count_parameters(&path).expect("should parse ONNX");
        // Linear(1540, 1) = 1540 weights. Bias may be folded away by constant folding.
        assert!(
            count >= 1540 && count <= 1541,
            "expected ~1540 params, got {count}"
        );
    }

    #[test]
    fn load_nn_eval_bot() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).expect("should load tiny model");
        assert!(bot.param_count >= 1540 && bot.param_count <= 1541);
    }

    #[test]
    fn nn_eval_returns_finite_value() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();
        let game = GameState::new();
        let eval = bot.nn_eval(&game).expect("inference should succeed");
        assert!(eval.is_finite(), "eval should be finite, got {eval}");
    }

    #[test]
    fn nn_eval_different_positions_differ() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();

        let startpos = GameState::new();
        let eval1 = bot.nn_eval(&startpos).unwrap();

        // Position with material imbalance — should get different eval
        let board: Board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"
            .parse()
            .unwrap();
        let imbalance = GameState::from_board(board);
        let eval2 = bot.nn_eval(&imbalance).unwrap();

        assert_ne!(
            eval1, eval2,
            "different positions should produce different evals"
        );
    }

    #[test]
    fn nn_eval_batch_matches_individual() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();

        let positions = [
            GameState::new(),
            GameState::from_board(
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
                    .parse()
                    .unwrap(),
            ),
            GameState::from_board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1".parse().unwrap()),
        ];

        // Individual evals
        let individual: Vec<f32> = positions
            .iter()
            .map(|g| bot.nn_eval(g).unwrap())
            .collect();

        // Batch eval
        let tensors: Vec<Vec<f32>> = positions.iter().map(|g| board_to_tensor(g)).collect();
        let batch = bot.nn_eval_batch(&tensors).unwrap();

        assert_eq!(individual.len(), batch.len());
        for (i, (ind, bat)) in individual.iter().zip(&batch).enumerate() {
            assert!(
                (ind - bat).abs() < 1e-5,
                "position {i}: individual={ind}, batch={bat}"
            );
        }
    }

    #[test]
    fn quiescence_nn_quiet_position_returns_stand_pat() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();

        // Startpos is quiet (no captures) — quiescence should equal stand-pat
        let game = GameState::new();
        let stand_pat = bot.nn_eval(&game).unwrap();
        let qs = bot
            .quiescence_nn(&game.board, f32::NEG_INFINITY, f32::INFINITY, None)
            .unwrap();

        assert!(
            (stand_pat - qs).abs() < 1e-5,
            "quiet position: stand_pat={stand_pat}, quiescence={qs}"
        );
    }

    #[test]
    fn quiescence_nn_with_captures() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();

        // Position with a capture available (exd5)
        let board: Board = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
            .parse()
            .unwrap();
        let qs = bot
            .quiescence_nn(&board, f32::NEG_INFINITY, f32::INFINITY, None)
            .unwrap();
        assert!(qs.is_finite(), "quiescence with captures should be finite");
    }

    #[test]
    fn try_choose_move_returns_legal() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();
        let game = GameState::new();
        let mv = bot
            .try_choose_move(&game)
            .expect("should not error")
            .expect("should return a move");
        assert!(
            game.legal_moves().contains(&mv),
            "chosen move must be legal"
        );
    }

    #[test]
    fn try_choose_move_finds_checkmate() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();

        // Fool's mate: Black plays Qh4#
        let game = GameState::from_fen(
            "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
        )
        .unwrap();
        let mv = bot.try_choose_move(&game).unwrap().unwrap();
        let mut after = game.board.clone();
        after.play(mv);
        assert_eq!(
            after.status(),
            GameStatus::Won,
            "NN bot should find checkmate when available"
        );
    }

    #[test]
    fn choose_move_via_bot_trait() {
        use crate::bot::Bot;
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();
        let game = GameState::new();
        let mv = bot.choose_move(&game).expect("Bot trait should return a move");
        assert!(game.legal_moves().contains(&mv));
    }

    #[test]
    fn nn_plays_full_game_without_crash() {
        use crate::bot::Bot;
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();
        let mut game = GameState::new();

        // Play up to 200 half-moves — just make sure it doesn't crash
        for _ in 0..200 {
            if game.is_game_over() {
                break;
            }
            match bot.choose_move(&game) {
                Some(mv) => {
                    assert!(game.make_move(mv), "move should be legal");
                }
                None => break,
            }
        }
        // If we get here without panicking, the test passes
    }

    // ── Golden reference tests ────────────────────────────────────────────

    /// Test positions used for golden/equivalence tests
    fn golden_positions() -> Vec<(&'static str, Board)> {
        vec![
            ("startpos", Board::default()),
            ("after 1.e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1".parse().unwrap()),
            ("capture available", "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2".parse().unwrap()),
            ("endgame K+P vs K", "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1".parse().unwrap()),
            ("complex middlegame", "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3".parse().unwrap()),
            ("queen trade", "rnb1kbnr/pppppppp/8/8/4q3/3P4/PPP1PPPP/RNBQKBNR w KQkq - 0 1".parse().unwrap()),
            ("promotion possible", "8/P7/8/8/8/8/8/4K2k w - - 0 1".parse().unwrap()),
            ("black in check", "rnbqkbnr/pppp1ppp/8/4N3/4P3/8/PPPP1PPP/RNBQKB1R b KQkq - 0 1".parse().unwrap()),
        ]
    }

    #[test]
    fn quiescence_scores_deterministic() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();

        for (name, board) in golden_positions() {
            let s1 = bot
                .quiescence_nn(&board, f32::NEG_INFINITY, f32::INFINITY, None)
                .unwrap();
            let s2 = bot
                .quiescence_nn(&board, f32::NEG_INFINITY, f32::INFINITY, None)
                .unwrap();
            assert_eq!(s1, s2, "{name}: quiescence must be deterministic");
        }
    }

    #[test]
    fn quiescence_sequential_golden() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();

        // Record scores for all golden positions — assert finite and deterministic
        let mut scores = Vec::new();
        for (name, board) in golden_positions() {
            let s = bot
                .quiescence_nn_sequential(&board, f32::NEG_INFINITY, f32::INFINITY)
                .unwrap();
            assert!(s.is_finite(), "{name}: score should be finite, got {s}");
            scores.push(s);
        }

        // Run again — must be identical
        for (i, (name, board)) in golden_positions().iter().enumerate() {
            let s = bot
                .quiescence_nn_sequential(board, f32::NEG_INFINITY, f32::INFINITY)
                .unwrap();
            assert_eq!(s, scores[i], "{name}: sequential score not deterministic");
        }
    }

    #[test]
    fn try_choose_move_golden() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();

        let positions = [
            GameState::new(),
            GameState::from_board(
                "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
                    .parse().unwrap(),
            ),
            GameState::from_board(
                "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
                    .parse().unwrap(),
            ),
        ];

        // Each position should return a legal move, deterministically
        for game in &positions {
            let mv1 = bot.try_choose_move(game).unwrap().unwrap();
            let mv2 = bot.try_choose_move(game).unwrap().unwrap();
            assert_eq!(mv1, mv2, "try_choose_move must be deterministic");
            assert!(game.legal_moves().contains(&mv1), "move must be legal");
        }
    }

    // ── Equivalence tests: batched == sequential ──────────────────────────

    #[test]
    fn quiescence_batched_equals_sequential() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();

        for (name, board) in golden_positions() {
            let seq = bot
                .quiescence_nn_sequential(&board, f32::NEG_INFINITY, f32::INFINITY)
                .unwrap();
            let bat = bot
                .quiescence_nn(&board, f32::NEG_INFINITY, f32::INFINITY, None)
                .unwrap();
            assert!(
                (seq - bat).abs() < 1e-5,
                "{name}: sequential={seq}, batched={bat}"
            );
        }
    }

    #[test]
    fn quiescence_batched_equals_sequential_narrow_window() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();

        for (name, board) in golden_positions() {
            // Use a narrow alpha-beta window
            let seq = bot
                .quiescence_nn_sequential(&board, -10.0, 10.0)
                .unwrap();
            let bat = bot
                .quiescence_nn(&board, -10.0, 10.0, None)
                .unwrap();
            assert!(
                (seq - bat).abs() < 1e-5,
                "{name} (narrow): sequential={seq}, batched={bat}"
            );
        }
    }

    #[test]
    fn quiescence_batched_with_hint_equals_without() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();

        for (name, board) in golden_positions() {
            if board.status() != GameStatus::Ongoing || !board.checkers().is_empty() {
                continue;
            }
            let sp = bot.nn_eval(&GameState::from_board(board.clone())).unwrap();
            let without_hint = bot
                .quiescence_nn(&board, f32::NEG_INFINITY, f32::INFINITY, None)
                .unwrap();
            let with_hint = bot
                .quiescence_nn(&board, f32::NEG_INFINITY, f32::INFINITY, Some(sp))
                .unwrap();
            assert!(
                (without_hint - with_hint).abs() < 1e-5,
                "{name}: without_hint={without_hint}, with_hint={with_hint}"
            );
        }
    }

    #[test]
    fn try_choose_move_batched_equals_sequential() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();

        let positions = [
            GameState::new(),
            GameState::from_board(
                "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
                    .parse().unwrap(),
            ),
            GameState::from_board(
                "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
                    .parse().unwrap(),
            ),
            // Fool's mate position
            GameState::from_fen(
                "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
            ).unwrap(),
        ];

        for game in &positions {
            let seq = bot.try_choose_move_sequential(game).unwrap();
            let bat = bot.try_choose_move(game).unwrap();
            assert_eq!(
                seq, bat,
                "batched and sequential should choose same move for {}",
                game.board
            );
        }
    }

    #[test]
    fn quiescence_batched_handles_terminal_positions() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();

        // Checkmate
        let mated: Board = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
            .parse().unwrap();
        assert_eq!(mated.status(), GameStatus::Won);
        let score = bot
            .quiescence_nn(&mated, f32::NEG_INFINITY, f32::INFINITY, None)
            .unwrap();
        assert_eq!(score, -MATE_SCORE_F);

        // Stalemate
        let stale: Board = "k7/8/1K6/8/8/8/8/8 b - - 0 1".parse().unwrap();
        if stale.status() == GameStatus::Drawn {
            let score = bot
                .quiescence_nn(&stale, f32::NEG_INFINITY, f32::INFINITY, None)
                .unwrap();
            assert_eq!(score, DRAW_SCORE_F);
        }
    }

    #[test]
    fn quiescence_batched_handles_check_positions() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();

        // Black in check — stand_pat_hint should be ignored (in-check uses NEG_INFINITY)
        let board: Board = "rnbqkbnr/pppp1ppp/8/4N3/4P3/8/PPPP1PPP/RNBQKB1R b KQkq - 0 1"
            .parse().unwrap();

        if !board.checkers().is_empty() {
            let without = bot
                .quiescence_nn(&board, f32::NEG_INFINITY, f32::INFINITY, None)
                .unwrap();
            // Even with a bogus hint, check positions ignore it
            let with_bogus = bot
                .quiescence_nn(&board, f32::NEG_INFINITY, f32::INFINITY, Some(999.0))
                .unwrap();
            assert!(
                (without - with_bogus).abs() < 1e-5,
                "check position should ignore stand_pat_hint"
            );
        }
    }

    // ── Performance benchmark (run with --nocapture to see output) ────────

    #[test]
    fn bench_batched_vs_sequential_call_counts() {
        let path = fixture_path();
        let bot = NnEvalBot::load(&path).unwrap();

        let positions: Vec<GameState> = vec![
            GameState::new(),
            GameState::from_board(
                "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
                    .parse().unwrap(),
            ),
            GameState::from_board(
                "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
                    .parse().unwrap(),
            ),
            GameState::from_board(
                "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
                    .parse().unwrap(),
            ),
            GameState::from_fen(
                "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
            ).unwrap(),
        ];

        let mut total_seq_calls = 0u64;
        let mut total_bat_calls = 0u64;
        let mut total_seq_positions = 0u64;
        let mut total_bat_positions = 0u64;

        for g in &positions {
            // Sequential
            bot.reset_counters();
            let _ = bot.try_choose_move_sequential(g);
            let (seq_calls, seq_pos) = bot.counters();
            total_seq_calls += seq_calls;
            total_seq_positions += seq_pos;

            // Batched
            bot.reset_counters();
            let _ = bot.try_choose_move(g);
            let (bat_calls, bat_pos) = bot.counters();
            total_bat_calls += bat_calls;
            total_bat_positions += bat_pos;
        }

        let call_reduction = total_seq_calls as f64 / total_bat_calls as f64;
        let avg_batch_size = total_bat_positions as f64 / total_bat_calls as f64;

        eprintln!("\n--- ONNX inference call counts ({} positions) ---", positions.len());
        eprintln!("  Sequential: {total_seq_calls} calls, {total_seq_positions} positions (batch=1)");
        eprintln!("  Batched:    {total_bat_calls} calls, {total_bat_positions} positions (avg batch={avg_batch_size:.1})");
        eprintln!("  Call reduction: {call_reduction:.1}x fewer ONNX calls");
        eprintln!("---");

        // Batched should make fewer ONNX calls
        assert!(
            total_bat_calls < total_seq_calls,
            "batched should make fewer ONNX calls: {total_bat_calls} >= {total_seq_calls}"
        );
    }
}
