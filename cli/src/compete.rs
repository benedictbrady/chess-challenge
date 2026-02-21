/// Competition runner: pit an ONNX policy network against BaselineBot.
///
/// Usage:
///   compete <model.onnx> [--games N]
///
/// The NN plays N games (alternating colors, default 10).
/// It must win ALL games to pass. Draws count as losses.
/// Models with >10 000 000 parameters are rejected.

use engine::bot::{Bot, BaselineBot};
use engine::game::{GameState, Outcome};
use engine::nn::count_parameters;
use engine::{Color, NnBot};
use std::path::Path;

const MAX_PARAMS: u64 = 10_000_000;
const MAX_PLIES: usize = 500; // safety valve against infinite games

fn run_game(white: &dyn Bot, black: &dyn Bot) -> (Outcome, usize) {
    let mut game = GameState::new();
    let mut plies = 0;

    loop {
        if game.is_game_over() {
            let outcome = game.outcome().unwrap_or(Outcome::Draw);
            return (outcome, plies);
        }

        if plies >= MAX_PLIES {
            return (Outcome::Draw, plies);
        }

        let side = game.side_to_move();
        let mv = if side == Color::White {
            white.choose_move(&game)
        } else {
            black.choose_move(&game)
        };

        match mv {
            Some(mv) => {
                game.make_move(mv);
                plies += 1;
            }
            None => {
                // Bot returned None mid-game → forfeit
                let winner = !side;
                return (Outcome::Checkmate { winner }, plies);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("Usage: compete <model.onnx> [--games N]");
        eprintln!();
        eprintln!("  model.onnx   ONNX policy network (input: board [1,768], output: policy [1,4096])");
        eprintln!("  --games N    number of games to play (default: 10, must win all N)");
        std::process::exit(1);
    }

    let model_path = Path::new(&args[1]);

    let num_games: usize = {
        let mut n = 10usize;
        let mut i = 2;
        while i < args.len() {
            if args[i] == "--games" {
                if let Some(val) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    n = val;
                    i += 1;
                }
            }
            i += 1;
        }
        n
    };

    // Count parameters before loading the session (fast file parse)
    println!("Loading: {}", model_path.display());
    let param_count = match count_parameters(model_path) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Failed to read model: {e}");
            std::process::exit(1);
        }
    };

    println!("Parameters: {:>12}", format_num(param_count));
    println!("Limit:      {:>12}", format_num(MAX_PARAMS));

    if param_count > MAX_PARAMS {
        eprintln!("REJECTED — model exceeds the {} parameter limit.", format_num(MAX_PARAMS));
        std::process::exit(1);
    }

    // Now load the ONNX Runtime session
    let nn = match NnBot::load(model_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            std::process::exit(1);
        }
    };

    println!();
    println!("Running {num_games} games vs BaselineBot…");
    println!("{}", "─".repeat(52));

    let spicy = BaselineBot::default();

    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut draws = 0usize;

    for game_idx in 0..num_games {
        // Alternate: even games NN=White, odd games NN=Black
        let nn_is_white = game_idx % 2 == 0;

        let (outcome, plies) = if nn_is_white {
            run_game(&nn, &spicy)
        } else {
            run_game(&spicy, &nn)
        };

        let nn_color = if nn_is_white { Color::White } else { Color::Black };

        let result_str = match outcome {
            Outcome::Checkmate { winner } => {
                if winner == nn_color {
                    wins += 1;
                    "WIN "
                } else {
                    losses += 1;
                    "LOSS"
                }
            }
            Outcome::Draw => {
                draws += 1;
                "DRAW"
            }
        };

        println!(
            "Game {:>2}/{num_games}  NN={}  {:4}  ({} plies)",
            game_idx + 1,
            if nn_is_white { "White" } else { "Black" },
            result_str,
            plies,
        );
    }

    println!("{}", "─".repeat(52));
    println!("Results: {}W / {}L / {}D", wins, losses, draws);
    println!();

    if wins == num_games {
        println!("PASS ✓  — beat BaselineBot {}/{num_games} games!", wins);
        std::process::exit(0);
    } else {
        println!(
            "FAIL ✗  — won {}/{num_games} (need all {num_games})",
            wins
        );
        std::process::exit(1);
    }
}

fn format_num(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}
