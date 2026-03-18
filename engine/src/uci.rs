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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn format_parse_round_trip() {
        let cases = [
            (File::E, Rank::Second, File::E, Rank::Fourth, None),
            (File::G, Rank::First, File::F, Rank::Third, None),
            (File::E, Rank::Seventh, File::E, Rank::Eighth, Some(Piece::Queen)),
            (File::A, Rank::Second, File::A, Rank::First, Some(Piece::Rook)),
            (File::H, Rank::Seventh, File::H, Rank::Eighth, Some(Piece::Bishop)),
            (File::C, Rank::Second, File::C, Rank::First, Some(Piece::Knight)),
        ];
        for (ff, fr, tf, tr, promo) in cases {
            let mv = Move {
                from: Square::new(ff, fr),
                to: Square::new(tf, tr),
                promotion: promo,
            };
            let uci = format_move(mv);
            let parsed = parse_uci_move(&uci).expect(&format!("failed to parse '{uci}'"));
            assert_eq!(parsed.from, mv.from, "from mismatch for '{uci}'");
            assert_eq!(parsed.to, mv.to, "to mismatch for '{uci}'");
            assert_eq!(parsed.promotion, mv.promotion, "promo mismatch for '{uci}'");
        }
    }

    #[test]
    fn format_move_lengths() {
        let regular = Move {
            from: Square::new(File::E, Rank::Second),
            to: Square::new(File::E, Rank::Fourth),
            promotion: None,
        };
        assert_eq!(format_move(regular).len(), 4);

        let promo = Move {
            from: Square::new(File::E, Rank::Seventh),
            to: Square::new(File::E, Rank::Eighth),
            promotion: Some(Piece::Queen),
        };
        assert_eq!(format_move(promo).len(), 5);
    }

    #[test]
    fn parse_rejects_garbage() {
        assert!(parse_uci_move("").is_none());
        assert!(parse_uci_move("e2").is_none());
        assert!(parse_uci_move("z2e4").is_none());
        assert!(parse_uci_move("e0e4").is_none());
        assert!(parse_uci_move("e9e4").is_none());
        assert!(parse_uci_move("e7e8x").is_none());
    }

    #[test]
    fn parse_corner_squares() {
        let mv = parse_uci_move("a1h8").unwrap();
        assert_eq!(mv.from, Square::new(File::A, Rank::First));
        assert_eq!(mv.to, Square::new(File::H, Rank::Eighth));

        let mv = parse_uci_move("h8a1").unwrap();
        assert_eq!(mv.from, Square::new(File::H, Rank::Eighth));
        assert_eq!(mv.to, Square::new(File::A, Rank::First));
    }

    #[test]
    fn piece_unicode_all_distinct() {
        let pieces = [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King];
        let mut seen = HashSet::new();
        for &color in &[Color::White, Color::Black] {
            for &piece in &pieces {
                let s = piece_unicode(piece, color);
                assert!(!s.is_empty());
                assert!(seen.insert(s), "duplicate unicode for {piece:?}/{color:?}");
            }
        }
        assert_eq!(seen.len(), 12);
    }
}
