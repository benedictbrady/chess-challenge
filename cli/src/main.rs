use engine::bot::{Bot, BaselineBot};
use engine::game::{GameState, Outcome};
use engine::uci::{format_move, parse_file, parse_rank, piece_unicode};
use engine::{Color, File, Move, Piece, Rank, Square};
use std::io::{self, Write};

fn print_board(game: &GameState) {
    let board = &game.board;
    println!("  a b c d e f g h");
    for rank in (0..8usize).rev() {
        print!("{} ", rank + 1);
        for file in 0..8usize {
            let sq = Square::new(File::index(file), Rank::index(rank));
            if let Some(piece) = board.piece_on(sq) {
                let color = if board.colors(Color::White).has(sq) {
                    Color::White
                } else {
                    Color::Black
                };
                print!("{} ", piece_unicode(piece, color));
            } else {
                print!(". ");
            }
        }
        println!("{}", rank + 1);
    }
    println!("  a b c d e f g h");
}

fn parse_move(input: &str, game: &GameState) -> Option<Move> {
    let input = input.trim();
    if input.len() < 4 || input.len() > 5 {
        return None;
    }

    let chars: Vec<char> = input.chars().collect();

    let from_file = parse_file(chars[0])?;
    let from_rank = parse_rank(chars[1])?;
    let to_file = parse_file(chars[2])?;
    let to_rank = parse_rank(chars[3])?;

    let from = Square::new(from_file, from_rank);
    let to = Square::new(to_file, to_rank);

    let promotion = if chars.len() == 5 {
        match chars[4].to_ascii_lowercase() {
            'q' => Some(Piece::Queen),
            'r' => Some(Piece::Rook),
            'b' => Some(Piece::Bishop),
            'n' => Some(Piece::Knight),
            _ => return None,
        }
    } else {
        None
    };

    let mv = Move { from, to, promotion };

    // Verify the move is legal
    if game.legal_moves().contains(&mv) {
        Some(mv)
    } else {
        None
    }
}

fn main() {
    let mut game = GameState::new();
    let bot = BaselineBot::default();
    let human_color = Color::White;

    println!("Chess vs BaselineBot");
    println!("You play as White. Enter moves in UCI format (e.g. e2e4, e7e8q).");
    println!("Type 'quit' to exit.");
    println!();

    loop {
        print_board(&game);
        println!();

        if game.is_game_over() {
            match game.outcome() {
                Some(Outcome::Checkmate { winner }) => println!(
                    "Checkmate! {} wins!",
                    if winner == Color::White { "White" } else { "Black" }
                ),
                Some(Outcome::Draw) => println!("Draw!"),
                None => println!("Game over."),
            }
            break;
        }

        let side = game.side_to_move();

        if side == human_color {
            print!("{} to move: ", if side == Color::White { "White" } else { "Black" });
            io::stdout().flush().unwrap();

            let mut input = String::new();
            match io::stdin().read_line(&mut input) {
                Ok(0) => break, // EOF
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error reading input: {}", e);
                    break;
                }
            }

            let trimmed = input.trim();
            if trimmed == "quit" || trimmed == "exit" {
                println!("Goodbye!");
                break;
            }

            match parse_move(trimmed, &game) {
                Some(mv) => {
                    game.make_move(mv);
                    println!("You played: {}", format_move(mv));
                }
                None => {
                    println!("Illegal move. Try again (e.g. e2e4).");
                    continue;
                }
            }
        } else {
            println!("Bot is thinking...");
            match bot.choose_move(&game) {
                Some(mv) => {
                    game.make_move(mv);
                    println!("Bot played: {}", format_move(mv));
                }
                None => {
                    println!("Bot has no moves.");
                    break;
                }
            }
        }
        println!();
    }
}
