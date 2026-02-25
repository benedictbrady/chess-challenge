/// Bot ELO validation using Stockfish as a calibrated reference.
///
/// Stockfish 18 supports `UCI_LimitStrength` + `UCI_Elo` (range 1320–3190).
/// We measure a bot's win rate against multiple ELO levels and solve for
/// the rating that gives ~50% expected score using the ELO formula:
///
///   ELO_self = R_opp + 400 * log10(S / (1 - S))
///
/// where S is the bot's score (win=1, draw=0.5, loss=0).
///
/// Usage:
///   validate [--games N] [--openings <path>]
use engine::bot::{BaselineBot, Bot};
use engine::game::{GameState, Outcome};
use engine::openings::load_opening_fens;
use engine::{Color, File, Move, Piece, Rank, Square};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

const STOCKFISH_PATH: &str = "/tmp/stockfish/stockfish-macos-m1-apple-silicon";
const MOVETIME_MS: u32 = 100; // ms per Stockfish move (fast for testing)
const MAX_HALFMOVES: usize = 300;
const DEFAULT_GAMES: usize = 100;
const DEFAULT_OPENINGS_PATH: &str = "data/openings.txt";
const TARGET_LO: f64 = 1600.0;
const TARGET_HI: f64 = 1800.0;

// ── CLI argument parsing ──────────────────────────────────────────────────────

struct CliArgs {
    n_games: usize,
    openings_path: Option<String>,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut n_games = DEFAULT_GAMES;
    let mut openings_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                i += 1;
                if i < args.len() {
                    n_games = args[i].parse::<usize>().unwrap_or_else(|_| {
                        eprintln!("Invalid --games value: {}", args[i]);
                        std::process::exit(1);
                    });
                } else {
                    eprintln!("--games requires a value");
                    std::process::exit(1);
                }
            }
            "--openings" => {
                i += 1;
                if i < args.len() {
                    openings_path = Some(args[i].clone());
                } else {
                    eprintln!("--openings requires a path");
                    std::process::exit(1);
                }
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    CliArgs {
        n_games,
        openings_path,
    }
}

// ── UCI bridge to Stockfish ───────────────────────────────────────────────────

struct StockfishProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl StockfishProcess {
    fn spawn(elo: u32) -> Self {
        let mut child = Command::new(STOCKFISH_PATH)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn Stockfish");

        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut sf = StockfishProcess {
            child,
            stdin,
            stdout,
        };

        sf.send("uci");
        sf.wait_for("uciok");
        sf.send("setoption name UCI_LimitStrength value true");
        sf.send(&format!("setoption name UCI_Elo value {}", elo));
        sf.send("setoption name Threads value 1");
        sf.send("isready");
        sf.wait_for("readyok");
        sf
    }

    fn send(&mut self, cmd: &str) {
        writeln!(self.stdin, "{}", cmd).unwrap();
        self.stdin.flush().unwrap();
    }

    fn read_line(&mut self) -> String {
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        line.trim_end().to_string()
    }

    fn wait_for(&mut self, token: &str) {
        loop {
            let line = self.read_line();
            if line.contains(token) {
                break;
            }
        }
    }

    fn get_move(&mut self, move_history: &[Move]) -> Option<Move> {
        let moves_str: Vec<String> = move_history.iter().map(|m| uci_move(*m)).collect();
        let pos_cmd = if moves_str.is_empty() {
            "position startpos".to_string()
        } else {
            format!("position startpos moves {}", moves_str.join(" "))
        };
        self.send(&pos_cmd);
        self.send(&format!("go movetime {}", MOVETIME_MS));

        loop {
            let line = self.read_line();
            if line.starts_with("bestmove") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 && parts[1] != "(none)" {
                    return parse_uci_move(parts[1]);
                }
                return None;
            }
        }
    }

    fn get_move_from_fen(&mut self, fen: &str, move_history: &[Move]) -> Option<Move> {
        let moves_str: Vec<String> = move_history.iter().map(|m| uci_move(*m)).collect();
        let pos_cmd = if moves_str.is_empty() {
            format!("position fen {}", fen)
        } else {
            format!("position fen {} moves {}", fen, moves_str.join(" "))
        };
        self.send(&pos_cmd);
        self.send(&format!("go movetime {}", MOVETIME_MS));

        loop {
            let line = self.read_line();
            if line.starts_with("bestmove") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 && parts[1] != "(none)" {
                    return parse_uci_move(parts[1]);
                }
                return None;
            }
        }
    }

    fn new_game(&mut self) {
        self.send("ucinewgame");
        self.send("isready");
        self.wait_for("readyok");
    }
}

impl Drop for StockfishProcess {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let _ = self.child.wait();
    }
}

// ── Move formatting ───────────────────────────────────────────────────────────

fn uci_move(mv: Move) -> String {
    let promo = mv
        .promotion
        .map(|p| match p {
            Piece::Queen => "q",
            Piece::Rook => "r",
            Piece::Bishop => "b",
            Piece::Knight => "n",
            _ => "",
        })
        .unwrap_or("");
    format!("{}{}{}", mv.from, mv.to, promo)
}

fn parse_file(c: char) -> Option<File> {
    match c {
        'a' => Some(File::A),
        'b' => Some(File::B),
        'c' => Some(File::C),
        'd' => Some(File::D),
        'e' => Some(File::E),
        'f' => Some(File::F),
        'g' => Some(File::G),
        'h' => Some(File::H),
        _ => None,
    }
}

fn parse_rank(c: char) -> Option<Rank> {
    match c {
        '1' => Some(Rank::First),
        '2' => Some(Rank::Second),
        '3' => Some(Rank::Third),
        '4' => Some(Rank::Fourth),
        '5' => Some(Rank::Fifth),
        '6' => Some(Rank::Sixth),
        '7' => Some(Rank::Seventh),
        '8' => Some(Rank::Eighth),
        _ => None,
    }
}

fn parse_uci_move(s: &str) -> Option<Move> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 4 {
        return None;
    }
    let from = Square::new(parse_file(chars[0])?, parse_rank(chars[1])?);
    let to = Square::new(parse_file(chars[2])?, parse_rank(chars[3])?);
    let promotion = if chars.len() == 5 {
        match chars[4] {
            'q' => Some(Piece::Queen),
            'r' => Some(Piece::Rook),
            'b' => Some(Piece::Bishop),
            'n' => Some(Piece::Knight),
            _ => return None,
        }
    } else {
        None
    };
    Some(Move {
        from,
        to,
        promotion,
    })
}

// ── Game runner ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum GameResult {
    WhiteWins,
    BlackWins,
    Draw,
}

fn play_one_game(
    bot: &BaselineBot,
    sf: &mut StockfishProcess,
    bot_is_white: bool,
    starting_fen: Option<&str>,
) -> GameResult {
    let mut game = match starting_fen {
        Some(fen) => GameState::from_fen(fen).unwrap_or_else(|e| {
            eprintln!(
                "Warning: bad FEN '{}': {}, falling back to startpos",
                fen, e
            );
            GameState::new()
        }),
        None => GameState::new(),
    };
    sf.new_game();

    for _ in 0..MAX_HALFMOVES {
        if game.is_game_over() {
            break;
        }
        let side = game.side_to_move();
        let bot_plays = (side == Color::White) == bot_is_white;

        let mv = if bot_plays {
            match bot.choose_move(&game) {
                Some(m) => m,
                None => break,
            }
        } else {
            match starting_fen {
                Some(fen) => sf.get_move_from_fen(fen, &game.history),
                None => sf.get_move(&game.history),
            }
            .unwrap_or_else(|| {
                Move {
                    from: Square::new(File::A, Rank::First),
                    to: Square::new(File::A, Rank::First),
                    promotion: None,
                }
            })
        };

        if !game.make_move(mv) {
            break;
        }
    }

    match game.outcome() {
        Some(Outcome::Checkmate {
            winner: Color::White,
        }) => GameResult::WhiteWins,
        Some(Outcome::Checkmate {
            winner: Color::Black,
        }) => GameResult::BlackWins,
        _ => GameResult::Draw,
    }
}

// ── ELO calculation ───────────────────────────────────────────────────────────

fn elo_from_score(score: f64, opp_elo: f64) -> f64 {
    let clamped = score.clamp(0.001, 0.999);
    opp_elo + 400.0 * (clamped / (1.0 - clamped)).log10()
}

fn elo_confidence_interval(score: f64, n: u32, opp_elo: f64) -> (f64, f64) {
    let se = (score * (1.0 - score) / n as f64).sqrt();
    let score_lo = (score - 1.96 * se).clamp(0.001, 0.999);
    let score_hi = (score + 1.96 * se).clamp(0.001, 0.999);
    let elo_lo = elo_from_score(score_lo, opp_elo);
    let elo_hi = elo_from_score(score_hi, opp_elo);
    (elo_lo, elo_hi)
}

// ── Matchup ───────────────────────────────────────────────────────────────────

struct MatchResult {
    wins: u32,
    draws: u32,
    losses: u32,
}

impl MatchResult {
    fn total(&self) -> u32 {
        self.wins + self.draws + self.losses
    }
    fn score(&self) -> f64 {
        (self.wins as f64 + 0.5 * self.draws as f64) / self.total() as f64
    }
}

fn run_match(
    bot: &BaselineBot,
    sf_elo: u32,
    n_games: usize,
    openings: &[String],
) -> MatchResult {
    let mut sf = StockfishProcess::spawn(sf_elo);
    let mut wins = 0u32;
    let mut draws = 0u32;
    let mut losses = 0u32;

    if openings.is_empty() {
        let half = n_games / 2;
        for i in 0..n_games {
            let bot_is_white = i < half;
            match play_one_game(bot, &mut sf, bot_is_white, None) {
                GameResult::WhiteWins if bot_is_white => wins += 1,
                GameResult::BlackWins if !bot_is_white => wins += 1,
                GameResult::Draw => draws += 1,
                _ => losses += 1,
            }
            print!(".");
            std::io::stdout().flush().unwrap();
        }
    } else {
        let n_pairs = n_games / 2;
        let remainder = n_games % 2;
        for pair_idx in 0..n_pairs {
            let fen = &openings[pair_idx % openings.len()];
            match play_one_game(bot, &mut sf, true, Some(fen)) {
                GameResult::WhiteWins => wins += 1,
                GameResult::BlackWins => losses += 1,
                GameResult::Draw => draws += 1,
            }
            print!(".");
            std::io::stdout().flush().unwrap();
            match play_one_game(bot, &mut sf, false, Some(fen)) {
                GameResult::BlackWins => wins += 1,
                GameResult::WhiteWins => losses += 1,
                GameResult::Draw => draws += 1,
            }
            print!(".");
            std::io::stdout().flush().unwrap();
        }
        if remainder > 0 {
            let fen = &openings[n_pairs % openings.len()];
            match play_one_game(bot, &mut sf, true, Some(fen)) {
                GameResult::WhiteWins => wins += 1,
                GameResult::BlackWins => losses += 1,
                GameResult::Draw => draws += 1,
            }
            print!(".");
            std::io::stdout().flush().unwrap();
        }
    }
    println!();
    MatchResult {
        wins,
        draws,
        losses,
    }
}

// ── Opening loading ───────────────────────────────────────────────────────────

fn load_openings_or_empty(path_override: &Option<String>) -> Vec<String> {
    let path_str = path_override.as_deref().unwrap_or(DEFAULT_OPENINGS_PATH);
    let path = Path::new(path_str);
    match load_opening_fens(path) {
        Ok(fens) => {
            println!("  Loaded {} openings from {}", fens.len(), path.display());
            fens
        }
        Err(e) => {
            if path_override.is_some() {
                eprintln!("  WARNING: {}", e);
                eprintln!("  Falling back to startpos for all games.");
            } else {
                println!("  No openings file at {}; using startpos.", path.display());
            }
            Vec::new()
        }
    }
}

// ── Benchmark ───────────────────────────────────────────────────────────────

struct BenchmarkResult {
    weighted_elo: f64,
    ci_lo: f64,
    ci_hi: f64,
}

fn benchmark_bot(
    bot: &BaselineBot,
    n_games: usize,
    openings: &[String],
) -> BenchmarkResult {
    let levels = [1500u32, 1600, 1700, 1800, 1900];
    let mut results: Vec<(u32, MatchResult)> = Vec::new();

    for &sf_elo in &levels {
        print!("  vs SF@{sf_elo}  ");
        std::io::stdout().flush().unwrap();
        let t = Instant::now();
        let r = run_match(bot, sf_elo, n_games, openings);
        let score = r.score();
        let my_elo = elo_from_score(score, sf_elo as f64);
        let (elo_lo, elo_hi) = elo_confidence_interval(score, r.total(), sf_elo as f64);
        println!(
            "  +{:2} ={:2} -{:2}  score={:5.1}%  Elo\u{2248}{:5.0} [{:.0}..{:.0}]  [{:.1}s]",
            r.wins,
            r.draws,
            r.losses,
            score * 100.0,
            my_elo,
            elo_lo,
            elo_hi,
            t.elapsed().as_secs_f64()
        );
        results.push((sf_elo, r));
    }

    // Weighted estimate
    let scored: Vec<(f64, f64, u32)> = results
        .iter()
        .map(|(sf_elo, r)| {
            let score = r.score();
            let elo = elo_from_score(score, *sf_elo as f64);
            (score, elo, r.total())
        })
        .collect();

    let weights: Vec<f64> = scored
        .iter()
        .map(|(s, _, _)| 1.0 - (2.0 * s - 1.0).abs())
        .collect();
    let total_weight: f64 = weights.iter().sum();
    let weighted_elo: f64 = scored
        .iter()
        .zip(&weights)
        .map(|((_, e, _), w)| e * w)
        .sum::<f64>()
        / total_weight;

    let total_games: u32 = results.iter().map(|(_, r)| r.total()).sum();
    let total_score: f64 = results
        .iter()
        .map(|(_, r)| r.wins as f64 + 0.5 * r.draws as f64)
        .sum::<f64>();
    let agg_score = total_score / total_games as f64;
    let agg_opp_elo: f64 = results
        .iter()
        .map(|(sf_elo, r)| *sf_elo as f64 * r.total() as f64)
        .sum::<f64>()
        / total_games as f64;
    let (ci_lo, ci_hi) = elo_confidence_interval(agg_score, total_games, agg_opp_elo);

    BenchmarkResult {
        weighted_elo,
        ci_lo,
        ci_hi,
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let cli = parse_args();
    let n_games = cli.n_games;
    let openings = load_openings_or_empty(&cli.openings_path);

    let bot = BaselineBot::default();

    println!("\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");
    println!(
        "  BaselineBot ELO Validation  (vs Stockfish 18, {} games per level)",
        n_games
    );
    println!("  {}", BaselineBot::description());
    println!(
        "  Config: depth={}, window={}cp, blunder_rate={}%",
        bot.depth,
        bot.candidate_window,
        (bot.blunder_rate * 100.0) as u32
    );
    println!("  Target range: {:.0}\u{2013}{:.0} Elo", TARGET_LO, TARGET_HI);
    println!("  Stockfish movetime: {}ms/move", MOVETIME_MS);
    if !openings.is_empty() {
        println!(
            "  Opening book: {} positions (each played as both colors)",
            openings.len()
        );
    }
    println!("\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");
    println!();

    let result = benchmark_bot(&bot, n_games, &openings);

    println!();
    let verdict = if result.weighted_elo < TARGET_LO {
        "BELOW TARGET \u{2014} try increasing depth"
    } else if result.weighted_elo > TARGET_HI {
        "ABOVE TARGET \u{2014} try reducing depth"
    } else {
        "IN RANGE"
    };
    println!(
        "  Weighted Elo: ~{:.0}  [{:.0}..{:.0}]  {}",
        result.weighted_elo, result.ci_lo, result.ci_hi, verdict
    );
    println!();
    println!("  Note: Stockfish ELO is calibrated; these estimates are reliable");
    println!(
        "  when the bot's score is between 15% and 85% against an opponent."
    );
}
