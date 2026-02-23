# Chess Challenge

**Build the smallest neural network that can beat our chess bot fleet.**

Train an ONNX policy network and run it against 5 challenger bots. To pass, you must win at least **3 out of 5 games against every challenger**. The winning submission is the one with the fewest parameters.

---

## The Fleet

Your model faces 5 opponents built on the same negamax + alpha-beta engine with material and piece-square-table evaluation, each tuned differently:

| Challenger | Depth | Window | Blunder | Character |
|------------|-------|--------|---------|-----------|
| **Grunt** | 2 | 50cp | 10% | Shallow but principled. Misses deep tactics but doesn't give away free pieces often. |
| **Fortress** | 3 | 0cp | 12% | Always plays the engine's best move when not blundering. Deterministic in good moves, unpredictable via blunders. |
| **Scholar** | 4 | 40cp | 20% | Deep calculator — sees tactics your NN can't. High blunder rate brings Elo down. |
| **Chaos** | 2 | 200cp | 8% | Extremely wide candidate window makes play highly unpredictable even without blunders. |
| **Wall** | 3 | 60cp | 0% | Zero blunders. Must outplay on pure merit. Hardest of the five. |

All challengers play roughly 1000–1200 Elo. Your model plays **without any search** — pure policy, no MCTS, no alpha-beta. The network must encode enough chess understanding to beat each opponent style.

---

## Model Spec

Your ONNX model must conform to this interface:

| Tensor | Name | Shape | dtype |
|--------|------|-------|-------|
| Input  | `board` | `[1, 768]` | float32 |
| Output | `policy` | `[1, 4096]` | float32 |

### Board Encoding

The board is encoded as **12 binary planes x 64 squares = 768 floats**, always from the current player's perspective:

| Channel | Contents |
|---------|----------|
| 0–5 | Current player's Pawns, Knights, Bishops, Rooks, Queens, King |
| 6–11 | Opponent's Pawns, Knights, Bishops, Rooks, Queens, King |

**Square indexing:** `a1=0, b1=1, …, h1=7, a2=8, …, h8=63`
**Flat index:** `channel * 64 + square`

When it is Black's turn, ranks are flipped (a1↔a8) so the network always sees its own pawns advancing up the board. Files are not flipped.

### Policy Encoding

4096 raw logits, one per (from, to) square pair:

```
index = from_square * 64 + to_square
```

Square indices use the same flipped coordinate system as the board input. The harness masks illegal moves to −∞, takes argmax, and auto-promotes pawns to Queen. The output does not need to be softmaxed.

### ONNX Requirements

- Input named `board`, output named `policy`
- `ir_version = 8`, opset 17 recommended
- All weights stored as ONNX initializers (the harness counts parameters from these)
- Max **10,000,000 parameters**

---

## Rules

1. **25 games total** — 5 games against each of 5 challengers
2. **Win 3/5 against every opponent** — draws count as losses
3. **10M parameter limit** — models exceeding this are rejected
4. **Score = parameter count** — lower is better
5. Games use openings from the book (`data/openings.txt`), with color alternation across the 5 games per opponent

---

## Quick Start

### Prerequisites

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build

```bash
cargo build --workspace
```

### Run the Competition

```bash
cargo run -p cli --bin compete -- path/to/your_model.onnx
```

A reference model is included for testing the pipeline:

```bash
cargo run -p cli --bin compete -- models/capture_policy.onnx
```

This is a hand-crafted capture heuristic (~315K params) — it prefers moving to squares occupied by valuable opponent pieces. It won't pass the competition but verifies everything works end to end.

You can also adjust games per opponent for quicker iteration:

```bash
cargo run -p cli --bin compete -- my_model.onnx --games-per-bot 3
```

### Play Against the Bots

ASCII terminal:
```bash
cargo run -p cli
```

Desktop GUI:
```bash
cargo run -p gui
```

### Watch Bot vs Bot

Watch your model play against a challenger in the GUI:
```bash
cargo run -p gui -- models/capture_policy.onnx
cargo run -p gui -- models/capture_policy.onnx --delay 300  # slower playback
```

### Validate Bot Elo (vs Stockfish)

Benchmark the challengers against Stockfish to verify they're in the target Elo range:

```bash
cargo run -p cli --bin validate -- --all --games 50
cargo run -p cli --bin validate -- --bot Wall --games 100
```

---

## Training Tips

A pure policy net needs to be roughly **1300+ Elo** to reliably beat all 5 challengers. Some approaches:

- **Supervised on human games** — train on the [Lichess open database](https://database.lichess.org/), filtering to games ≥1800 Elo. A small MLP or CNN trained for a few hours on a GPU should reach ~1400–1600.
- **Supervised on engine evals** — generate positions labeled with Stockfish's best move. Cleaner signal than human games.
- **Self-play RL** — AlphaZero-style training with MCTS. Expensive but can exceed human-level play.

---

## Repository Layout

```
chess-challenge/
├── engine/              # Core library
│   └── src/
│       ├── bot.rs       # BaselineBot, 5 challenger configs
│       ├── eval.rs      # Material + piece-square-table evaluation
│       ├── game.rs      # GameState, move generation, repetition detection
│       ├── nn.rs        # NnBot: ONNX inference, board encoding, parameter counting
│       ├── openings.rs  # Opening book loader
│       └── search.rs    # Negamax + alpha-beta pruning
├── cli/                 # Command-line tools
│   └── src/
│       ├── main.rs      # Human vs bot (ASCII board)
│       ├── compete.rs   # Competition runner
│       └── validate.rs  # Elo validation vs Stockfish
├── gui/                 # egui desktop GUI (human play + watch mode)
├── models/              # Reference ONNX model
│   └── capture_policy.onnx
└── data/
    └── openings.txt     # ~50 opening positions
```

---

## License

MIT
