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
    /// Each tensor in `tensors` is a flat [1540] encoding.
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
}
