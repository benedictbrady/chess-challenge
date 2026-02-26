/// Single-move player: given a FEN, make one bot move and output JSON.
///
/// Usage:
///   play-move <model.onnx> <fen>       # NN model mode
///   play-move --baseline <fen>          # baseline bot mode
///
/// Output (JSON to stdout):
///   {"uci":"e2e4","fen":"...after move...","gameOver":false,"outcome":null}

use engine::bot::Bot;
use engine::game::{GameState, Outcome};
use engine::{BaselineBot, Color, Move, NnEvalBot, Piece};
use std::path::Path;

fn format_move(mv: Move) -> String {
    let promo = match mv.promotion {
        Some(Piece::Queen) => "q",
        Some(Piece::Rook) => "r",
        Some(Piece::Bishop) => "b",
        Some(Piece::Knight) => "n",
        _ => "",
    };
    format!("{}{}{}", mv.from, mv.to, promo)
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("Usage: play-move <model.onnx> <fen>");
        eprintln!("       play-move --baseline <fen>");
        std::process::exit(1);
    }

    let is_baseline = args[1] == "--baseline";
    let fen = if is_baseline { &args[2] } else { &args[2] };
    let model_path = if is_baseline { None } else { Some(&args[1]) };

    // Load bot
    let baseline = BaselineBot::default();
    let nn: Option<NnEvalBot>;
    let bot: &dyn Bot = if let Some(path) = model_path {
        nn = Some(match NnEvalBot::load(Path::new(path)) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Failed to load model: {e}");
                std::process::exit(1);
            }
        });
        nn.as_ref().unwrap()
    } else {
        &baseline
    };

    // Parse FEN
    let mut game = match GameState::from_fen(fen) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Invalid FEN: {e}");
            std::process::exit(1);
        }
    };

    // Check if game is already over
    if game.is_game_over() {
        let (game_over, outcome) = outcome_json(&game);
        println!(
            "{{\"uci\":null,\"fen\":\"{}\",\"gameOver\":{},\"outcome\":{}}}",
            escape_json(&game.board.to_string()),
            game_over,
            outcome,
        );
        return;
    }

    // Choose and make move
    match bot.choose_move(&game) {
        Some(mv) => {
            let uci = format_move(mv);
            game.make_move(mv);
            let new_fen = game.board.to_string();
            let (game_over, outcome) = outcome_json(&game);
            println!(
                "{{\"uci\":\"{}\",\"fen\":\"{}\",\"gameOver\":{},\"outcome\":{}}}",
                uci,
                escape_json(&new_fen),
                game_over,
                outcome,
            );
        }
        None => {
            // No legal moves (shouldn't happen if game isn't over, but handle gracefully)
            let (game_over, outcome) = outcome_json(&game);
            println!(
                "{{\"uci\":null,\"fen\":\"{}\",\"gameOver\":{},\"outcome\":{}}}",
                escape_json(&game.board.to_string()),
                game_over,
                outcome,
            );
        }
    }
}

fn outcome_json(game: &GameState) -> (bool, String) {
    if !game.is_game_over() {
        return (false, "null".to_string());
    }
    match game.outcome() {
        Some(Outcome::Checkmate { winner }) => {
            let w = if winner == Color::White { "white" } else { "black" };
            (true, format!("\"checkmate-{w}\""))
        }
        Some(Outcome::Draw) => (true, "\"draw\"".to_string()),
        None => (true, "\"draw\"".to_string()),
    }
}
