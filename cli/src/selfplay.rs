/// Self-play testing: pit BaselineBot at different depths against each other.
/// This tests whether the eval function produces meaningful strength differences.

use engine::bot::{BaselineBot, Bot};
use engine::game::{GameState, Outcome};
use engine::openings::load_opening_fens;
use engine::{Color, Move, Piece};
use std::path::Path;
use std::time::Instant;

const MAX_PLIES: usize = 500;

fn run_game(
    white: &dyn Bot,
    black: &dyn Bot,
    starting_fen: Option<&str>,
) -> (Outcome, usize) {
    let mut game = match starting_fen {
        Some(fen) => match GameState::from_fen(fen) {
            Ok(g) => g,
            Err(_) => GameState::new(),
        },
        None => GameState::new(),
    };

    let mut plies = 0;
    loop {
        if game.is_game_over() {
            return (game.outcome().unwrap_or(Outcome::Draw), plies);
        }
        if plies >= MAX_PLIES {
            return (Outcome::Draw, plies);
        }

        let mv = if game.side_to_move() == Color::White {
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
                return (Outcome::Checkmate { winner: !game.side_to_move() }, plies);
            }
        }
    }
}

struct MatchResult {
    white_name: String,
    black_name: String,
    wins_w: u32,
    wins_b: u32,
    draws: u32,
    total_plies: u64,
}

impl MatchResult {
    fn total(&self) -> u32 { self.wins_w + self.wins_b + self.draws }
    fn score_w(&self) -> f64 {
        (self.wins_w as f64 + 0.5 * self.draws as f64) / self.total() as f64
    }
}

fn run_match(
    bot_a: &BaselineBot,
    name_a: &str,
    bot_b: &BaselineBot,
    name_b: &str,
    n_games: usize,
    openings: &[String],
) -> MatchResult {
    let mut result = MatchResult {
        white_name: name_a.to_string(),
        black_name: name_b.to_string(),
        wins_w: 0,
        wins_b: 0,
        draws: 0,
        total_plies: 0,
    };

    for i in 0..n_games {
        let a_is_white = i % 2 == 0;
        let fen = if openings.is_empty() {
            None
        } else {
            Some(openings[(i / 2) % openings.len()].as_str())
        };

        let (outcome, plies) = if a_is_white {
            run_game(bot_a, bot_b, fen)
        } else {
            run_game(bot_b, bot_a, fen)
        };

        result.total_plies += plies as u64;

        match outcome {
            Outcome::Checkmate { winner } => {
                let a_won = (winner == Color::White && a_is_white)
                    || (winner == Color::Black && !a_is_white);
                if a_won {
                    if a_is_white { result.wins_w += 1; } else { result.wins_b += 1; }
                } else {
                    if a_is_white { result.wins_b += 1; } else { result.wins_w += 1; }
                }
            }
            Outcome::Draw => result.draws += 1,
        }

        let result_str = match outcome {
            Outcome::Checkmate { winner } => {
                let a_won = (winner == Color::White && a_is_white)
                    || (winner == Color::Black && !a_is_white);
                if a_won { format!("{} wins", name_a) } else { format!("{} wins", name_b) }
            }
            Outcome::Draw => "Draw".to_string(),
        };
        print!("  Game {:>2}/{}: {} ({} plies)\n", i + 1, n_games, result_str, plies);
    }

    result
}

fn main() {
    let openings = match load_opening_fens(Path::new("data/openings.txt")) {
        Ok(fens) => {
            println!("Loaded {} openings", fens.len());
            fens
        }
        Err(_) => {
            println!("No openings file, using startpos");
            Vec::new()
        }
    };

    println!("\n=== Self-Play Depth Experiments ===\n");

    // Test configs: (name, depth, window, blunder_rate)
    let configs: Vec<(&str, u32, i32, f64)> = vec![
        ("Depth3", 3, 0, 0.0),
        ("Depth4", 4, 0, 0.0),
        ("Depth5", 5, 0, 0.0),
        ("Depth3-Blunder10%", 3, 80, 0.10),
        ("Depth2-Blunder15%", 2, 80, 0.15),
    ];

    let n_games = 20; // 10 pairs (alternating colors)

    // Match 1: Depth 5 vs Depth 3 (should be a clear win for depth 5)
    println!("--- Match 1: Depth5 vs Depth3 ({} games) ---", n_games);
    let bot_a = BaselineBot { depth: 5, candidate_window: 0, blunder_rate: 0.0 };
    let bot_b = BaselineBot { depth: 3, candidate_window: 0, blunder_rate: 0.0 };
    let t = Instant::now();
    let r = run_match(&bot_a, "Depth5", &bot_b, "Depth3", n_games, &openings);
    println!("  Result: Depth5 scored {:.1}% (+{} ={} -{}) [{:.1}s]\n",
        r.score_w() * 100.0, r.wins_w, r.draws, r.wins_b, t.elapsed().as_secs_f64());

    // Match 2: Depth 5 vs Depth 4
    println!("--- Match 2: Depth5 vs Depth4 ({} games) ---", n_games);
    let bot_b = BaselineBot { depth: 4, candidate_window: 0, blunder_rate: 0.0 };
    let t = Instant::now();
    let r = run_match(&bot_a, "Depth5", &bot_b, "Depth4", n_games, &openings);
    println!("  Result: Depth5 scored {:.1}% (+{} ={} -{}) [{:.1}s]\n",
        r.score_w() * 100.0, r.wins_w, r.draws, r.wins_b, t.elapsed().as_secs_f64());

    // Match 3: Depth 5 vs Depth 3 with 10% blunder (simulating ~1200 Elo)
    println!("--- Match 3: Depth5 vs Depth3-Blunder10% ({} games) ---", n_games);
    let bot_b = BaselineBot { depth: 3, candidate_window: 80, blunder_rate: 0.10 };
    let t = Instant::now();
    let r = run_match(&bot_a, "Depth5", &bot_b, "D3-Blunder", n_games, &openings);
    println!("  Result: Depth5 scored {:.1}% (+{} ={} -{}) [{:.1}s]\n",
        r.score_w() * 100.0, r.wins_w, r.draws, r.wins_b, t.elapsed().as_secs_f64());

    // Match 4: Depth 5 vs Depth 2 with 15% blunder (simulating ~1000 Elo)
    println!("--- Match 4: Depth5 vs Depth2-Blunder15% ({} games) ---", n_games);
    let bot_b = BaselineBot { depth: 2, candidate_window: 80, blunder_rate: 0.15 };
    let t = Instant::now();
    let r = run_match(&bot_a, "Depth5", &bot_b, "D2-Blunder", n_games, &openings);
    println!("  Result: Depth5 scored {:.1}% (+{} ={} -{}) [{:.1}s]\n",
        r.score_w() * 100.0, r.wins_w, r.draws, r.wins_b, t.elapsed().as_secs_f64());

    // Match 5: Depth 4 vs Depth 3 (calibration: how much is 1 ply worth?)
    println!("--- Match 5: Depth4 vs Depth3 ({} games) ---", n_games);
    let bot_a = BaselineBot { depth: 4, candidate_window: 0, blunder_rate: 0.0 };
    let bot_b = BaselineBot { depth: 3, candidate_window: 0, blunder_rate: 0.0 };
    let t = Instant::now();
    let r = run_match(&bot_a, "Depth4", &bot_b, "Depth3", n_games, &openings);
    println!("  Result: Depth4 scored {:.1}% (+{} ={} -{}) [{:.1}s]\n",
        r.score_w() * 100.0, r.wins_w, r.draws, r.wins_b, t.elapsed().as_secs_f64());

    println!("=== Summary ===");
    println!("If Depth5 draws everything against Depth3, the eval is too flat.");
    println!("If Depth5 crushes Depth3 but draws Depth4, depth matters more than eval.");
    println!("The blunder matches show if the engine can punish mistakes.");
}
