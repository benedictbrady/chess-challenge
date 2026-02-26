/// Competition runner: pit an ONNX eval network against baseline bots at multiple levels.
///
/// Usage:
///   compete <model.onnx> [--level N] [--openings <path>]
///
/// The NN plays 50 games (25 positions x 2 colors) per level against increasingly
/// strong baselines. Scoring: 1 for win, 0.5 for draw, 0 for loss. Must reach 70%.
/// Models with >10 000 000 parameters are rejected.

use engine::bot::Bot;
use engine::game::{GameState, Outcome};
use engine::nn::count_parameters;
use engine::openings::load_opening_fens;
use engine::{BaselineBot, Color, Level, Move, NnEvalBot, Piece, ALL_LEVELS};
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

const MAX_PARAMS: u64 = 10_000_000;
const MAX_PLIES: usize = 500;
const NUM_POSITIONS: usize = 25;
const TOTAL_GAMES: usize = 50; // NUM_POSITIONS * 2
const PASS_THRESHOLD: f64 = 0.70;

// ---------------------------------------------------------------------------
// Move formatting (UCI style)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Diversity tracking
// ---------------------------------------------------------------------------

struct DiversityTracker {
    all_nn_moves: Vec<String>,
    first_moves: Vec<String>,
    four_move_seqs: Vec<String>,
    total_games: usize,
}

impl DiversityTracker {
    fn new() -> Self {
        DiversityTracker {
            all_nn_moves: Vec::new(),
            first_moves: Vec::new(),
            four_move_seqs: Vec::new(),
            total_games: 0,
        }
    }

    fn record_game(&mut self, nn_moves: &[String]) {
        self.total_games += 1;

        if let Some(first) = nn_moves.first() {
            self.first_moves.push(first.clone());
        }

        let seq: Vec<&str> = nn_moves.iter().take(4).map(|s| s.as_str()).collect();
        self.four_move_seqs.push(seq.join(","));

        self.all_nn_moves.extend(nn_moves.iter().cloned());
    }

    fn report(&self) {
        println!();
        println!("--- Playing Diversity ---");

        let distinct_first: HashSet<&str> = self.first_moves.iter().map(|s| s.as_str()).collect();
        let mut sorted_first: Vec<&str> = distinct_first.iter().copied().collect();
        sorted_first.sort();
        println!(
            "First moves:         {} distinct ({})",
            distinct_first.len(),
            sorted_first.join(", "),
        );

        let distinct_seqs: HashSet<&str> =
            self.four_move_seqs.iter().map(|s| s.as_str()).collect();
        println!(
            "4-move sequences:    {}/{} unique",
            distinct_seqs.len(),
            self.total_games,
        );

        let entropy = self.move_entropy();
        println!("Move entropy:        {:.1} bits", entropy);
    }

    fn move_entropy(&self) -> f64 {
        if self.all_nn_moves.is_empty() {
            return 0.0;
        }
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for mv in &self.all_nn_moves {
            *counts.entry(mv.as_str()).or_insert(0) += 1;
        }
        let total = self.all_nn_moves.len() as f64;
        let mut entropy = 0.0;
        for &count in counts.values() {
            let p = count as f64 / total;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
        entropy
    }
}

// ---------------------------------------------------------------------------
// Game runner
// ---------------------------------------------------------------------------

fn run_game(
    white: &dyn Bot,
    black: &dyn Bot,
    starting_fen: Option<&str>,
    nn_is_white: bool,
) -> (Outcome, usize, Vec<String>) {
    let mut game = match starting_fen {
        Some(fen) => match GameState::from_fen(fen) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Warning: bad FEN ({e}), falling back to startpos");
                GameState::new()
            }
        },
        None => GameState::new(),
    };

    let mut plies = 0;
    let mut nn_moves: Vec<String> = Vec::new();

    loop {
        if game.is_game_over() {
            let outcome = game.outcome().unwrap_or(Outcome::Draw);
            return (outcome, plies, nn_moves);
        }

        if plies >= MAX_PLIES {
            return (Outcome::Draw, plies, nn_moves);
        }

        let side = game.side_to_move();
        let is_nn_turn =
            (side == Color::White && nn_is_white) || (side == Color::Black && !nn_is_white);

        let mv = if side == Color::White {
            white.choose_move(&game)
        } else {
            black.choose_move(&game)
        };

        match mv {
            Some(mv) => {
                if is_nn_turn {
                    nn_moves.push(format_move(mv));
                }
                game.make_move(mv);
                plies += 1;
            }
            None => {
                // Bot returned None mid-game -> forfeit
                let winner = !side;
                return (Outcome::Checkmate { winner }, plies, nn_moves);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Opening loading & position selection
// ---------------------------------------------------------------------------

fn load_openings_or_fallback(path: &Path) -> Vec<String> {
    match load_opening_fens(path) {
        Ok(fens) => {
            println!("Loaded {} openings from {}", fens.len(), path.display());
            fens
        }
        Err(e) => {
            eprintln!("Note: {e} \u{2014} using standard startpos for all games.");
            Vec::new()
        }
    }
}

/// Randomly sample `n` positions without replacement (cycles if fewer available).
fn select_positions(openings: &[String], n: usize) -> Vec<String> {
    if openings.is_empty() {
        return vec!["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(); n];
    }

    let mut rng = rand::thread_rng();

    if openings.len() >= n {
        // Sample without replacement
        let mut indices: Vec<usize> = (0..openings.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(n);
        indices.iter().map(|&i| openings[i].clone()).collect()
    } else {
        // Cycle through available openings
        let mut result = Vec::with_capacity(n);
        let mut pool: Vec<usize> = Vec::new();
        while result.len() < n {
            if pool.is_empty() {
                pool = (0..openings.len()).collect();
                pool.shuffle(&mut rng);
            }
            result.push(openings[pool.pop().unwrap()].clone());
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

fn score_outcome(outcome: &Outcome, nn_color: Color) -> f64 {
    match outcome {
        Outcome::Checkmate { winner } => {
            if *winner == nn_color {
                1.0
            } else {
                0.0
            }
        }
        Outcome::Draw => 0.5,
    }
}

// ---------------------------------------------------------------------------
// Level result
// ---------------------------------------------------------------------------

struct LevelResult {
    level: Level,
    score: f64,
    wins: usize,
    draws: usize,
    losses: usize,
    passed: bool,
    elapsed: std::time::Duration,
}

// ---------------------------------------------------------------------------
// Run a single level
// ---------------------------------------------------------------------------

fn run_level(
    level: Level,
    nn: &NnEvalBot,
    positions: &[String],
) -> LevelResult {
    let baseline = BaselineBot::from_level(level);
    let pass_points = (TOTAL_GAMES as f64 * PASS_THRESHOLD).ceil() as usize;

    println!();
    println!(
        "\u{2550}\u{2550}\u{2550} Level {} \u{2014} {} \u{2550}\u{2550}\u{2550}",
        level.value(),
        level.name(),
    );
    println!("  {}", level.description());
    println!("  Baseline: {}", baseline.description());
    println!(
        "  {} games, need {:.0}% = {}/{} points",
        TOTAL_GAMES,
        PASS_THRESHOLD * 100.0,
        pass_points,
        TOTAL_GAMES,
    );
    println!();

    let mut diversity = DiversityTracker::new();
    let mut total_score: f64 = 0.0;
    let mut wins = 0usize;
    let mut draws = 0usize;
    let mut losses = 0usize;

    let timer = Instant::now();

    for (pos_idx, fen) in positions.iter().enumerate() {
        // Game A: NN=White vs Baseline=Black
        baseline.reset();
        let (outcome_a, plies_a, nn_moves_a) =
            run_game(nn, &baseline, Some(fen), true);
        diversity.record_game(&nn_moves_a);
        let score_a = score_outcome(&outcome_a, Color::White);
        total_score += score_a;
        match score_a as u32 {
            1 => wins += 1,
            0 => losses += 1,
            _ => draws += 1,
        }

        // Game B: Baseline=White vs NN=Black
        baseline.reset();
        let (outcome_b, plies_b, nn_moves_b) =
            run_game(&baseline, nn, Some(fen), false);
        diversity.record_game(&nn_moves_b);
        let score_b = score_outcome(&outcome_b, Color::Black);
        total_score += score_b;
        match score_b as u32 {
            1 => wins += 1,
            0 => losses += 1,
            _ => draws += 1,
        }

        let result_a = match score_a as u32 {
            1 => "WIN ",
            0 => "LOSS",
            _ => "DRAW",
        };
        let result_b = match score_b as u32 {
            1 => "WIN ",
            0 => "LOSS",
            _ => "DRAW",
        };

        println!(
            "  Pos {:>2}/{}  W:{} ({}pl)  B:{} ({}pl)  running={:.1}/{:.0}",
            pos_idx + 1,
            NUM_POSITIONS,
            result_a,
            plies_a,
            result_b,
            plies_b,
            total_score,
            pass_points,
        );
    }

    let elapsed = timer.elapsed();

    diversity.report();

    let passed = total_score >= pass_points as f64;
    let pct = total_score / TOTAL_GAMES as f64 * 100.0;

    println!();
    if passed {
        println!(
            "  Level {} PASS \u{2713}  {:.1}/{} ({:.0}%) in {:.1}s",
            level.value(),
            total_score,
            TOTAL_GAMES,
            pct,
            elapsed.as_secs_f64(),
        );
    } else {
        println!(
            "  Level {} FAIL \u{2717}  {:.1}/{} ({:.0}%) in {:.1}s",
            level.value(),
            total_score,
            TOTAL_GAMES,
            pct,
            elapsed.as_secs_f64(),
        );
    }

    LevelResult {
        level,
        score: total_score,
        wins,
        draws,
        losses,
        passed,
        elapsed,
    }
}

// ---------------------------------------------------------------------------
// Scorecard
// ---------------------------------------------------------------------------

fn print_scorecard(results: &[LevelResult], param_count: u64) {
    let best_level = results
        .iter()
        .rev()
        .find(|r| r.passed)
        .map(|r| r.level.value());

    println!();
    println!(
        "\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}"
    );
    println!(
        "                          SCORECARD"
    );
    println!(
        "\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}"
    );
    println!(
        "  Level  Name            Score     Record      Result"
    );
    println!(
        "  \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}  \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}  \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}  \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}  \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}"
    );

    for r in results {
        let result_str = if r.passed { "PASS" } else { "FAIL" };
        println!(
            "    {}    {:<14}  {:>4.1}/{}   {:>2}W/{:>2}D/{:>2}L  {}",
            r.level.value(),
            r.level.name(),
            r.score,
            TOTAL_GAMES,
            r.wins,
            r.draws,
            r.losses,
            result_str,
        );
    }

    println!();
    println!("  Parameters: {}", format_num(param_count));

    match best_level {
        Some(lv) => println!(
            "  Rating: Level {} / {} params",
            lv,
            format_num(param_count),
        ),
        None => println!("  Rating: no levels passed"),
    }

    let total_time: f64 = results.iter().map(|r| r.elapsed.as_secs_f64()).sum();
    println!("  Time: {:.1}s total", total_time);
    println!(
        "\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}"
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("Usage: compete <model.onnx> [--level N] [--openings <path>]");
        eprintln!();
        eprintln!("  model.onnx          ONNX eval network (input: board [1,768], output: eval [1,1])");
        eprintln!("  --level N           Run only level N (1-5). Omit to run all levels.");
        eprintln!("  --openings <path>   opening FEN file (default: data/openings.txt)");
        eprintln!();
        eprintln!("Levels:");
        for lv in &ALL_LEVELS {
            eprintln!("  {}  {:<14}  {}", lv.value(), lv.name(), lv.description());
        }
        std::process::exit(1);
    }

    let model_path = Path::new(&args[1]);

    // Parse CLI flags
    let mut openings_path = String::from("data/openings.txt");
    let mut single_level: Option<u8> = None;
    {
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--openings" => {
                    if let Some(val) = args.get(i + 1) {
                        openings_path = val.clone();
                        i += 1;
                    }
                }
                "--level" => {
                    if let Some(val) = args.get(i + 1) {
                        match val.parse::<u8>() {
                            Ok(n) if (1..=5).contains(&n) => single_level = Some(n),
                            _ => {
                                eprintln!("Error: --level must be 1-5");
                                std::process::exit(1);
                            }
                        }
                        i += 1;
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }

    // Count parameters before loading the session
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
        eprintln!(
            "REJECTED \u{2014} model exceeds the {} parameter limit.",
            format_num(MAX_PARAMS)
        );
        std::process::exit(1);
    }

    // Load the ONNX Runtime session
    let nn = match NnEvalBot::load(model_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            std::process::exit(1);
        }
    };

    // Load openings and select positions
    let openings = load_openings_or_fallback(Path::new(&openings_path));
    let positions = select_positions(&openings, NUM_POSITIONS);

    // Determine which levels to run
    let levels: Vec<Level> = match single_level {
        Some(n) => vec![Level::new(n).unwrap()],
        None => ALL_LEVELS.to_vec(),
    };

    // Run levels
    let mut results: Vec<LevelResult> = Vec::new();

    for level in &levels {
        let result = run_level(*level, &nn, &positions);
        let failed = !result.passed;
        results.push(result);

        // Early termination: if running all levels and this one failed, stop
        if failed && single_level.is_none() {
            println!();
            println!("Stopping \u{2014} failed level {}, skipping harder levels.", level.value());
            break;
        }
    }

    // Print scorecard
    print_scorecard(&results, param_count);

    // Exit code: 0 if any level passed, 1 if none
    let any_passed = results.iter().any(|r| r.passed);
    if any_passed {
        std::process::exit(0);
    } else {
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
