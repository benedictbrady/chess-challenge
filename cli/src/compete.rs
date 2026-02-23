/// Competition runner: pit an ONNX policy network against a fleet of 5 challenger bots.
///
/// Usage:
///   compete <model.onnx> [--games-per-bot N] [--openings <path>]
///
/// The NN plays N games against each of 5 challengers (default 5, total 25 games).
/// It must win at least 3/N against each opponent to pass. Draws count as losses.
/// Models with >10 000 000 parameters are rejected.
///
/// --openings <path>  Load opening FENs from file (default: data/openings.txt).
///                    Each bot faces different openings via offset into the book.
///                    Games are paired: games (0,1) share an opening (NN plays both
///                    colors), games (2,3) share another, game 4 gets its own.

use engine::bot::Bot;
use engine::game::{GameState, Outcome};
use engine::nn::count_parameters;
use engine::openings::load_opening_fens;
use engine::{Color, Move, NnBot, Piece, CHALLENGERS};
use std::collections::{HashMap, HashSet};
use std::path::Path;

const MAX_PARAMS: u64 = 10_000_000;
const MAX_PLIES: usize = 500; // safety valve against infinite games
const DEFAULT_GAMES_PER_BOT: usize = 5;
const WINS_REQUIRED_FRACTION_NUM: usize = 3; // need 3 out of 5

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

/// Collects NN move statistics across all games for diversity reporting.
struct DiversityTracker {
    /// Every NN move played, formatted as UCI string.
    all_nn_moves: Vec<String>,
    /// The first NN move in each game (UCI string).
    first_moves: Vec<String>,
    /// The first 4 NN moves per game, joined as a sequence string.
    four_move_seqs: Vec<String>,
    /// Total games tracked.
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

    /// Record NN moves from a single game.
    fn record_game(&mut self, nn_moves: &[String]) {
        self.total_games += 1;

        if let Some(first) = nn_moves.first() {
            self.first_moves.push(first.clone());
        }

        let seq: Vec<&str> = nn_moves.iter().take(4).map(|s| s.as_str()).collect();
        self.four_move_seqs.push(seq.join(","));

        self.all_nn_moves.extend(nn_moves.iter().cloned());
    }

    /// Print the diversity report.
    fn report(&self) {
        println!();
        println!("--- Playing Diversity ---");

        // 1. First-move diversity
        let distinct_first: HashSet<&str> = self.first_moves.iter().map(|s| s.as_str()).collect();
        let mut sorted_first: Vec<&str> = distinct_first.iter().copied().collect();
        sorted_first.sort();
        println!(
            "First moves:         {} distinct ({})",
            distinct_first.len(),
            sorted_first.join(", "),
        );

        // 2. Opening sequence diversity (first 4 NN moves)
        let distinct_seqs: HashSet<&str> =
            self.four_move_seqs.iter().map(|s| s.as_str()).collect();
        println!(
            "4-move sequences:    {}/{} unique",
            distinct_seqs.len(),
            self.total_games,
        );

        // 3. Move entropy (Shannon entropy over NN move distribution)
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

/// Run a single game, optionally from a starting FEN.
/// Returns (outcome, ply_count, list_of_nn_moves_as_uci_strings).
///
/// `nn_is_white` tells us which side is the NN so we can track its moves.
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
// Opening loading
// ---------------------------------------------------------------------------

/// Load openings from the given path, falling back to no openings if the file
/// is missing (games will start from the standard position).
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

/// Given a game index within a bot's set, a bot offset, and the list of opening FENs,
/// return the FEN to use.
/// Games are paired: (0,1) share opening 0, (2,3) share opening 1, etc.
/// Each bot uses a different offset into the book for diversity.
fn opening_for_game(
    game_idx: usize,
    bot_offset: usize,
    openings: &[String],
) -> Option<&str> {
    if openings.is_empty() {
        return None;
    }
    let opening_idx = (bot_offset + game_idx / 2) % openings.len();
    Some(openings[opening_idx].as_str())
}

// ---------------------------------------------------------------------------
// Per-opponent result tracking
// ---------------------------------------------------------------------------

struct OpponentResult {
    name: &'static str,
    wins: usize,
    losses: usize,
    draws: usize,
    games_per_bot: usize,
}

impl OpponentResult {
    fn passed(&self) -> bool {
        let wins_needed = wins_required(self.games_per_bot);
        self.wins >= wins_needed
    }
}

/// Calculate wins required: 3 out of 5 (scales: ceil(games * 3/5))
fn wins_required(games_per_bot: usize) -> usize {
    // For 5 games: 3. For other values: ceil(games * 3 / 5)
    (games_per_bot * WINS_REQUIRED_FRACTION_NUM + DEFAULT_GAMES_PER_BOT - 1)
        / DEFAULT_GAMES_PER_BOT
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!(
            "Usage: compete <model.onnx> [--games-per-bot N] [--openings <path>]"
        );
        eprintln!();
        eprintln!("  model.onnx          ONNX policy network (input: board [1,768], output: policy [1,4096])");
        eprintln!(
            "  --games-per-bot N   games per opponent (default: {}, must win {}/{})",
            DEFAULT_GAMES_PER_BOT,
            wins_required(DEFAULT_GAMES_PER_BOT),
            DEFAULT_GAMES_PER_BOT
        );
        eprintln!("  --openings <path>   opening FEN file (default: data/openings.txt)");
        std::process::exit(1);
    }

    let model_path = Path::new(&args[1]);

    // Parse CLI flags
    let mut games_per_bot: usize = DEFAULT_GAMES_PER_BOT;
    let mut openings_path = String::from("data/openings.txt");

    {
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--games-per-bot" => {
                    if let Some(val) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                        games_per_bot = val;
                        i += 1;
                    }
                }
                "--openings" => {
                    if let Some(val) = args.get(i + 1) {
                        openings_path = val.clone();
                        i += 1;
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }

    let total_games = games_per_bot * CHALLENGERS.len();
    let wins_needed = wins_required(games_per_bot);

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
        eprintln!(
            "REJECTED \u{2014} model exceeds the {} parameter limit.",
            format_num(MAX_PARAMS)
        );
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

    // Load openings
    let openings = load_openings_or_fallback(Path::new(&openings_path));

    println!();
    println!(
        "Running {} games ({} per opponent \u{00d7} {} opponents, need {}/{} wins each)\u{2026}",
        total_games,
        games_per_bot,
        CHALLENGERS.len(),
        wins_needed,
        games_per_bot,
    );

    let mut diversity = DiversityTracker::new();
    let mut opponent_results: Vec<OpponentResult> = Vec::new();

    for (bot_idx, challenger) in CHALLENGERS.iter().enumerate() {
        println!();
        println!("--- vs {} ---", challenger.name);
        println!("    {}", challenger.description);

        let bot = challenger.to_bot();
        let mut wins = 0usize;
        let mut losses = 0usize;
        let mut draws = 0usize;

        // Each bot gets a different offset into the opening book
        let bot_offset = bot_idx * ((games_per_bot + 1) / 2);

        for game_idx in 0..games_per_bot {
            let nn_is_white = game_idx % 2 == 0;
            let fen = opening_for_game(game_idx, bot_offset, &openings);

            let (outcome, plies, nn_moves) = if nn_is_white {
                run_game(&nn, &bot, fen, true)
            } else {
                run_game(&bot, &nn, fen, false)
            };

            // Track diversity
            diversity.record_game(&nn_moves);

            let nn_color = if nn_is_white {
                Color::White
            } else {
                Color::Black
            };

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
                "Game {:>2}/{}  NN={}  {:4}  ({} plies)",
                game_idx + 1,
                games_per_bot,
                if nn_is_white { "White" } else { "Black" },
                result_str,
                plies,
            );
        }

        let result = OpponentResult {
            name: challenger.name,
            wins,
            losses,
            draws,
            games_per_bot,
        };
        let pass_str = if result.passed() { "PASS" } else { "FAIL" };
        println!(
            "    Result: {}W / {}L / {}D  {}",
            wins, losses, draws, pass_str
        );

        opponent_results.push(result);
    }

    // Playing diversity report (across all games)
    diversity.report();

    // Results summary
    println!();
    println!(
        "\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}"
    );
    println!("             RESULTS SUMMARY");
    println!(
        "\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}"
    );
    println!(
        "  {:<12} {:>2}  {:>2}  {:>2}  Result",
        "Opponent", "W", "L", "D"
    );

    let mut opponents_passed = 0usize;
    for r in &opponent_results {
        let pass_str = if r.passed() { "PASS" } else { "FAIL" };
        println!(
            "  {:<12} {:>2}  {:>2}  {:>2}  {}",
            r.name, r.wins, r.losses, r.draws, pass_str
        );
        if r.passed() {
            opponents_passed += 1;
        }
    }

    println!(
        "\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}"
    );
    println!(
        "  Overall: {}/{} opponents defeated",
        opponents_passed,
        CHALLENGERS.len()
    );
    println!();

    let all_passed = opponent_results.iter().all(|r| r.passed());

    if all_passed {
        println!(
            "PASS \u{2713}  \u{2014} beat all {} opponents ({}/{} each)!",
            CHALLENGERS.len(),
            wins_needed,
            games_per_bot
        );
        std::process::exit(0);
    } else {
        println!(
            "FAIL \u{2717}  \u{2014} must beat all {} opponents ({}/{} each)",
            CHALLENGERS.len(),
            wins_needed,
            games_per_bot
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
