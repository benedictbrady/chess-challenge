use crate::game::GameState;
use std::path::Path;

/// Load opening FENs from a text file (one per line, `#` comments and blank lines skipped).
pub fn load_opening_fens(path: &Path) -> Result<Vec<String>, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read {}: {}", path.display(), e))?;
    let fens: Vec<String> = contents
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.to_string())
        .collect();
    if fens.is_empty() {
        return Err(format!("No FENs found in {}", path.display()));
    }
    Ok(fens)
}

/// Load opening FENs and parse each into a GameState (validates all FENs).
pub fn load_openings(path: &Path) -> Result<Vec<GameState>, String> {
    let fens = load_opening_fens(path)?;
    fens.iter()
        .enumerate()
        .map(|(i, fen)| {
            GameState::from_fen(fen)
                .map_err(|e| format!("Opening #{} invalid: {}", i + 1, e))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_openings_all_valid() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/openings.txt");
        let openings = load_openings(&path).expect("all openings should parse");
        assert!(
            openings.len() >= 40,
            "expected at least 40 openings, got {}",
            openings.len()
        );
        for (i, game) in openings.iter().enumerate() {
            assert!(
                !game.is_game_over(),
                "opening #{} should not be a finished game",
                i + 1
            );
        }
    }
}
