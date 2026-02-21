/// Bot ELO validation using Stockfish as a calibrated reference.
///
/// Stockfish 18 supports `UCI_LimitStrength` + `UCI_Elo` (range 1320–3190).
/// We measure SpicyBot's win rate against multiple ELO levels and solve for
/// the rating that gives ~50% expected score using the ELO formula:
///
///   ELO_self = R_opp + 400 * log10(S / (1 - S))
///
/// where S is SpicyBot's score (win=1, draw=0.5, loss=0).
use engine::bot::{Bot, SpicyBot};
use engine::game::{GameState, Outcome};
use engine::{Color, File, Move, Piece, Rank, Square};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

const STOCKFISH_PATH: &str = "/tmp/stockfish/stockfish-macos-m1-apple-silicon";
const MOVETIME_MS: u32 = 100; // ms per Stockfish move (fast for testing)
const MAX_HALFMOVES: usize = 300;

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
        let mut sf = StockfishProcess { child, stdin, stdout };

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

    /// Ask Stockfish for a move given the move history from startpos.
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
    let promo = mv.promotion.map(|p| match p {
        Piece::Queen => "q",
        Piece::Rook => "r",
        Piece::Bishop => "b",
        Piece::Knight => "n",
        _ => "",
    }).unwrap_or("");
    format!("{}{}{}", mv.from, mv.to, promo)
}

fn parse_file(c: char) -> Option<File> {
    match c {
        'a' => Some(File::A), 'b' => Some(File::B),
        'c' => Some(File::C), 'd' => Some(File::D),
        'e' => Some(File::E), 'f' => Some(File::F),
        'g' => Some(File::G), 'h' => Some(File::H),
        _ => None,
    }
}

fn parse_rank(c: char) -> Option<Rank> {
    match c {
        '1' => Some(Rank::First),  '2' => Some(Rank::Second),
        '3' => Some(Rank::Third),  '4' => Some(Rank::Fourth),
        '5' => Some(Rank::Fifth),  '6' => Some(Rank::Sixth),
        '7' => Some(Rank::Seventh),'8' => Some(Rank::Eighth),
        _ => None,
    }
}

fn parse_uci_move(s: &str) -> Option<Move> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 4 { return None; }
    let from = Square::new(parse_file(chars[0])?, parse_rank(chars[1])?);
    let to   = Square::new(parse_file(chars[2])?, parse_rank(chars[3])?);
    let promotion = if chars.len() == 5 {
        match chars[4] {
            'q' => Some(Piece::Queen), 'r' => Some(Piece::Rook),
            'b' => Some(Piece::Bishop),'n' => Some(Piece::Knight),
            _ => return None,
        }
    } else { None };
    Some(Move { from, to, promotion })
}

// ── Game runner ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum GameResult { WhiteWins, BlackWins, Draw }

/// Play one game. spicy_is_white determines color assignment.
fn play_one_game(spicy: &SpicyBot, sf: &mut StockfishProcess, spicy_is_white: bool) -> GameResult {
    let mut game = GameState::new();
    sf.new_game();

    for _ in 0..MAX_HALFMOVES {
        if game.is_game_over() { break; }
        let side = game.side_to_move();
        let spicy_plays = (side == Color::White) == spicy_is_white;

        let mv = if spicy_plays {
            match spicy.choose_move(&game) {
                Some(m) => m,
                None => break,
            }
        } else {
            match sf.get_move(&game.history) {
                Some(m) => m,
                None => break,
            }
        };

        if !game.make_move(mv) { break; }
    }

    match game.outcome() {
        Some(Outcome::Checkmate { winner: Color::White }) => GameResult::WhiteWins,
        Some(Outcome::Checkmate { winner: Color::Black }) => GameResult::BlackWins,
        _ => GameResult::Draw,
    }
}

// ── ELO calculation ───────────────────────────────────────────────────────────

fn elo_from_score(score: f64, opp_elo: f64) -> f64 {
    let clamped = score.clamp(0.001, 0.999);
    opp_elo + 400.0 * (clamped / (1.0 - clamped)).log10()
}

// ── Matchup ───────────────────────────────────────────────────────────────────

struct MatchResult { wins: u32, draws: u32, losses: u32 }

impl MatchResult {
    fn total(&self) -> u32 { self.wins + self.draws + self.losses }
    fn score(&self) -> f64 {
        (self.wins as f64 + 0.5 * self.draws as f64) / self.total() as f64
    }
}

fn run_match(spicy: &SpicyBot, sf_elo: u32, n_games: usize) -> MatchResult {
    let mut sf = StockfishProcess::spawn(sf_elo);
    let half = n_games / 2;
    let mut wins = 0u32; let mut draws = 0u32; let mut losses = 0u32;

    for i in 0..n_games {
        let spicy_is_white = i < half; // alternate color each half
        match play_one_game(spicy, &mut sf, spicy_is_white) {
            GameResult::WhiteWins if spicy_is_white  => wins += 1,
            GameResult::BlackWins if !spicy_is_white => wins += 1,
            GameResult::Draw => draws += 1,
            _ => losses += 1,
        }
        print!(".");
        std::io::stdout().flush().unwrap();
    }
    println!();
    MatchResult { wins, draws, losses }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let spicy = SpicyBot::default();
    let n_games = 20; // 20 per ELO level — fast but sufficient for rough estimate

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  SpicyBot ELO Validation  (vs Stockfish 18, {} games per level)", n_games);
    println!("  SpicyBot: depth=3, window=80cp, blunder_rate=15%");
    println!("  Stockfish movetime: {}ms/move", MOVETIME_MS);
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // Test at ELO levels bracketing the expected ~1000 range.
    // Stockfish 18 minimum is 1320.
    let levels = [1320u32, 1400, 1500, 1600];
    let mut results: Vec<(u32, f64, f64)> = Vec::new(); // (sf_elo, score, my_elo_estimate)

    for &sf_elo in &levels {
        print!("  vs Stockfish@{sf_elo}  ");
        std::io::stdout().flush().unwrap();
        let t = Instant::now();
        let r = run_match(&spicy, sf_elo, n_games);
        let score = r.score();
        let my_elo = elo_from_score(score, sf_elo as f64);
        println!(
            "  +{:2} ={:2} -{:2}  score={:5.1}%  → ELO≈{:5.0}  [{:.1}s]",
            r.wins, r.draws, r.losses,
            score * 100.0, my_elo, t.elapsed().as_secs_f64()
        );
        results.push((sf_elo, score, my_elo));
    }

    println!();

    // Best estimate: weight by how close to 50% each score is (most informative).
    let weights: Vec<f64> = results.iter()
        .map(|(_, s, _)| 1.0 - (2.0 * s - 1.0).abs())
        .collect();
    let total_weight: f64 = weights.iter().sum();
    let weighted_elo: f64 = results.iter().zip(&weights)
        .map(|((_, _, e), w)| e * w)
        .sum::<f64>() / total_weight;

    println!("  ┌──────────────────────────────────────────────────────────┐");
    println!("  │ Weighted ELO estimate: {:5.0}                            │", weighted_elo);
    println!("  │ Target range:          900–1100                          │");
    let verdict = if weighted_elo < 800.0 {
        "Too weak"
    } else if weighted_elo < 950.0 {
        "Slightly below — minor tuning needed"
    } else if weighted_elo <= 1150.0 {
        "On target for ~1000 ELO"
    } else {
        "Above target — reduce randomness or depth"
    };
    println!("  │ Verdict: {:<50}│", verdict);
    println!("  └──────────────────────────────────────────────────────────┘");
    println!();
    println!("  Note: Stockfish ELO is calibrated; these estimates are reliable");
    println!("  when SpicyBot's score is between 15% and 85% against an opponent.");
    println!("  Scores outside that range saturate the formula.");
}
