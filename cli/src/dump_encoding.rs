/// Dump tensor encodings for FEN positions — used to validate Python encoding matches Rust.
///
/// Usage: cargo run -p cli --bin dump-encoding -- "fen string"
use engine::game::GameState;
use engine::nn::board_to_tensor;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: dump-encoding <FEN>");
        eprintln!("Example: dump-encoding \"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\"");
        std::process::exit(1);
    }

    let fen = &args[1];
    let game = match GameState::from_fen(fen) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Invalid FEN: {e}");
            std::process::exit(1);
        }
    };

    let tensor = board_to_tensor(&game);

    // Print non-zero indices and values
    println!("FEN: {fen}");
    println!("Tensor size: {}", tensor.len());
    println!("Non-zero count: {}", tensor.iter().filter(|&&v| v != 0.0).count());
    println!("Sum: {}", tensor.iter().sum::<f32>());
    println!("Non-zero indices:");
    for (i, &v) in tensor.iter().enumerate() {
        if v != 0.0 {
            println!("  [{i}] = {v}");
        }
    }
}
