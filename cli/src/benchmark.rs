/// Benchmark: play the enhanced engine against the classic (original) engine
/// and infer Elo from the score differential.
///
/// Usage:
///   benchmark [--games N] [--depth N]

use engine::bot::{BaselineBot, Bot};
use engine::game::{GameState, Outcome};
use engine::openings::load_opening_fens;
use engine::Color;
use std::path::Path;
use std::time::Instant;

const DEFAULT_GAMES: usize = 100;
const DEFAULT_OPENINGS: &str = "data/openings.txt";
const MAX_PLIES: usize = 500;

// Assumed Elo of the classic depth-4 bot (from previous Stockfish calibration)
const CLASSIC_ELO: f64 = 1550.0;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut n_games = DEFAULT_GAMES;
    let mut enhanced_depth = 4u32;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                i += 1;
                n_games = args[i].parse().unwrap();
            }
            "--depth" => {
                i += 1;
                enhanced_depth = args[i].parse().unwrap();
            }
            _ => {}
        }
        i += 1;
    }

    // Must be even for color fairness
    if n_games % 2 != 0 {
        n_games += 1;
    }

    let openings = load_opening_fens(Path::new(DEFAULT_OPENINGS)).unwrap_or_default();

    let enhanced = BaselineBot::new(enhanced_depth, 0, 0.0, true);
    let classic = BaselineBot::classic(4);

    println!("════════════════════════════════════════════════════");
    println!("  Engine Benchmark: Enhanced vs Classic");
    println!("────────────────────────────────────────────────────");
    println!("  Enhanced: {}", enhanced.description());
    println!("  Classic:  {}", classic.description());
    println!("  Games:    {} ({} positions x 2 colors)", n_games, n_games / 2);
    println!("  Classic assumed Elo: {:.0}", CLASSIC_ELO);
    println!("════════════════════════════════════════════════════");
    println!();

    let timer = Instant::now();

    let mut enhanced_wins = 0u32;
    let mut draws = 0u32;
    let mut classic_wins = 0u32;

    let n_pairs = n_games / 2;

    for pair in 0..n_pairs {
        let fen = if openings.is_empty() {
            None
        } else {
            Some(openings[pair % openings.len()].as_str())
        };

        // Game 1: enhanced=White, classic=Black
        enhanced.reset();
        let r1 = play_game(&enhanced, &classic, fen, true);
        match r1 {
            GameResult::EnhancedWin => enhanced_wins += 1,
            GameResult::Draw => draws += 1,
            GameResult::ClassicWin => classic_wins += 1,
        }

        // Game 2: classic=White, enhanced=Black
        enhanced.reset();
        let r2 = play_game(&enhanced, &classic, fen, false);
        match r2 {
            GameResult::EnhancedWin => enhanced_wins += 1,
            GameResult::Draw => draws += 1,
            GameResult::ClassicWin => classic_wins += 1,
        }

        let total = (pair + 1) as u32 * 2;
        let score = enhanced_wins as f64 + 0.5 * draws as f64;
        let pct = score / total as f64 * 100.0;
        print!(
            "\r  Pair {:>3}/{}  +{} ={} -{}  ({:.1}%)",
            pair + 1,
            n_pairs,
            enhanced_wins,
            draws,
            classic_wins,
            pct,
        );
    }
    println!();

    let elapsed = timer.elapsed();
    let total = n_games as u32;
    let score = enhanced_wins as f64 + 0.5 * draws as f64;
    let pct = score / total as f64;

    // Elo calculation: E_enhanced = E_classic + 400 * log10(S / (1 - S))
    let elo = if pct <= 0.001 {
        CLASSIC_ELO - 800.0
    } else if pct >= 0.999 {
        CLASSIC_ELO + 800.0
    } else {
        CLASSIC_ELO + 400.0 * (pct / (1.0 - pct)).log10()
    };

    // Confidence interval
    let se = (pct * (1.0 - pct) / total as f64).sqrt();
    let lo_pct = (pct - 1.96 * se).clamp(0.001, 0.999);
    let hi_pct = (pct + 1.96 * se).clamp(0.001, 0.999);
    let elo_lo = CLASSIC_ELO + 400.0 * (lo_pct / (1.0 - lo_pct)).log10();
    let elo_hi = CLASSIC_ELO + 400.0 * (hi_pct / (1.0 - hi_pct)).log10();

    println!();
    println!("════════════════════════════════════════════════════");
    println!("  RESULTS");
    println!("────────────────────────────────────────────────────");
    println!(
        "  Enhanced: +{} ={} -{}  ({:.1}%)",
        enhanced_wins,
        draws,
        classic_wins,
        pct * 100.0
    );
    println!(
        "  Elo delta: {:+.0}  (enhanced vs classic)",
        elo - CLASSIC_ELO
    );
    println!(
        "  Enhanced Elo: ~{:.0}  [{:.0} .. {:.0}]",
        elo, elo_lo, elo_hi,
    );
    println!(
        "  Time: {:.1}s ({:.2}s/game)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / n_games as f64,
    );
    println!("════════════════════════════════════════════════════");
}

enum GameResult {
    EnhancedWin,
    Draw,
    ClassicWin,
}

fn play_game(
    enhanced: &BaselineBot,
    classic: &BaselineBot,
    fen: Option<&str>,
    enhanced_is_white: bool,
) -> GameResult {
    let mut game = match fen {
        Some(f) => GameState::from_fen(f).unwrap_or_else(|_| GameState::new()),
        None => GameState::new(),
    };

    for _ in 0..MAX_PLIES {
        if game.is_game_over() {
            break;
        }

        let side = game.side_to_move();
        let is_enhanced_turn =
            (side == Color::White && enhanced_is_white)
                || (side == Color::Black && !enhanced_is_white);

        let mv = if is_enhanced_turn {
            enhanced.choose_move(&game)
        } else {
            classic.choose_move(&game)
        };

        match mv {
            Some(m) => {
                game.make_move(m);
            }
            None => break,
        }
    }

    match game.outcome() {
        Some(Outcome::Checkmate { winner }) => {
            let enhanced_color = if enhanced_is_white {
                Color::White
            } else {
                Color::Black
            };
            if winner == enhanced_color {
                GameResult::EnhancedWin
            } else {
                GameResult::ClassicWin
            }
        }
        _ => GameResult::Draw,
    }
}
