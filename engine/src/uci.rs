use crate::{Color, File, Move, Piece, Rank, Square};

/// Format a move in UCI notation (e.g. "e2e4", "e7e8q").
pub fn format_move(mv: Move) -> String {
    let promo = match mv.promotion {
        Some(Piece::Queen) => "q",
        Some(Piece::Rook) => "r",
        Some(Piece::Bishop) => "b",
        Some(Piece::Knight) => "n",
        _ => "",
    };
    format!("{}{}{}", mv.from, mv.to, promo)
}

pub fn parse_file(c: char) -> Option<File> {
    match c.to_ascii_lowercase() {
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

pub fn parse_rank(c: char) -> Option<Rank> {
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

pub fn parse_uci_move(s: &str) -> Option<Move> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 4 {
        return None;
    }
    let from = Square::new(parse_file(chars[0])?, parse_rank(chars[1])?);
    let to = Square::new(parse_file(chars[2])?, parse_rank(chars[3])?);
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
    Some(Move {
        from,
        to,
        promotion,
    })
}

pub fn piece_unicode(piece: Piece, color: Color) -> &'static str {
    match (piece, color) {
        (Piece::King, Color::White) => "♔",
        (Piece::Queen, Color::White) => "♕",
        (Piece::Rook, Color::White) => "♖",
        (Piece::Bishop, Color::White) => "♗",
        (Piece::Knight, Color::White) => "♘",
        (Piece::Pawn, Color::White) => "♙",
        (Piece::King, Color::Black) => "♚",
        (Piece::Queen, Color::Black) => "♛",
        (Piece::Rook, Color::Black) => "♜",
        (Piece::Bishop, Color::Black) => "♝",
        (Piece::Knight, Color::Black) => "♞",
        (Piece::Pawn, Color::Black) => "♟",
    }
}
