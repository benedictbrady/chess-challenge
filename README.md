# Chess Challenge

**Who can build the smallest neural network that beats our chess bot fleet?**

This repo hosts a Rust chess engine and a competition harness. The goal is simple: train an ONNX policy network with as few parameters as possible that can beat **all 5 challenger bots** — each requiring **3/5 wins**.

The winning submission is the one with the lowest parameter count that defeats every challenger.

---

## The Fleet You're Beating

Your model faces 5 opponents, each with a different playing personality. All are built on the same negamax engine with alpha-beta pruning, material + piece-square-table evaluation, but tuned differently:

| Challenger | Depth | Window | Blunder | Character |
|------------|-------|--------|---------|-----------|
| **Grunt** | 2 | 50cp | 10% | Shallow but principled. Misses deep tactics but doesn't give away free pieces often. |
| **Fortress** | 3 | 0cp | 12% | Always plays the engine's best move when not blundering. Deterministic in good moves, unpredictable via blunders. |
| **Scholar** | 4 | 40cp | 20% | Deep calculator — sees tactics your NN can't. High blunder rate brings Elo down. |
| **Chaos** | 2 | 200cp | 8% | Extremely wide candidate window makes play highly unpredictable even without blunders. |
| **Wall** | 3 | 60cp | 0% | Zero blunders. Must outplay on pure merit. Hardest of the five. |

All challengers play roughly 1000–1200 Elo. Your model plays **without any search** — pure policy, no MCTS, no alpha-beta. The network must encode enough chess understanding to beat each opponent style.

---

## Model Interface (Competition Spec)

Your ONNX model must have exactly these inputs and outputs:

| Tensor | Name | Shape | dtype |
|--------|------|-------|-------|
| Input  | `board` | `[1, 768]` | float32 |
| Output | `policy` | `[1, 4096]` | float32 |

### Board Encoding (`board`)

The board is encoded as **12 binary planes × 64 squares = 768 floats**, always from the **current player's perspective**:

| Channel | Contents |
|---------|----------|
| 0 | Current player's Pawns |
| 1 | Current player's Knights |
| 2 | Current player's Bishops |
| 3 | Current player's Rooks |
| 4 | Current player's Queens |
| 5 | Current player's King |
| 6 | Opponent's Pawns |
| 7 | Opponent's Knights |
| 8 | Opponent's Bishops |
| 9 | Opponent's Rooks |
| 10 | Opponent's Queens |
| 11 | Opponent's King |

**Square indexing:** `a1=0, b1=1, …, h1=7, a2=8, …, h8=63`
**Flat index:** `channel × 64 + square`

**Perspective:** when it is Black's turn to move, ranks are flipped (a1↔a8, etc.) so the network always sees its own pawns advancing up the board. Files are not flipped.

### Policy Encoding (`policy`)

4096 raw logits, one per (from-square, to-square) pair:

```
index = from_square × 64 + to_square
```

Square indices use the **same flipped coordinate system** as the board input. The harness:
- masks illegal moves to −∞
- takes argmax over remaining legal moves
- auto-promotes pawns reaching the back rank to **Queen** (under-promotions are skipped)

The output does **not** need to be softmaxed — the harness handles that.

### ONNX Requirements

- Input named `board`, output named `policy`
- `ir_version = 8` (ONNX IR v8, supported by ort 2.0.0-rc.11)
- Opset 17 recommended
- All weights must be stored as ONNX initializers (parameter counting reads these)
- Max **10,000,000 parameters** (weights + biases, everything counts)

---

## Competition Rules

1. Your model plays **5 games** against each of 5 challengers = **25 total games**
2. You must win **at least 3/5** against each opponent — draws count as losses
3. Models exceeding **10 million parameters** are rejected outright
4. **Score:** total parameter count — lower is better
5. Tie-break: if two submissions both pass, the smaller parameter count wins
6. Opening diversity: each bot faces different openings via offset into the book
7. Color alternation: games 0,1 share an opening (NN plays both colors), games 2,3 share another, game 4 gets its own

---

## Running the Competition

### Prerequisites

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python + ONNX (for creating / training models)
pip install onnx numpy
```

### Build

```bash
cargo build --workspace
```

### Run the Competition

```bash
cargo run -p cli --bin compete -- path/to/your_model.onnx
```

Example output:
```
Loading: models/capture_policy.onnx
Parameters:      315,456
Limit:        10,000,000

Running 25 games (5 per opponent × 5 opponents, need 3/5 wins each)…

--- vs Grunt ---
    Shallow but principled (depth=2, window=50cp, blunder=10%)
Game  1/5  NN=White  WIN   (84 plies)
Game  2/5  NN=Black  LOSS  (162 plies)
Game  3/5  NN=White  WIN   (95 plies)
Game  4/5  NN=Black  WIN   (140 plies)
Game  5/5  NN=White  DRAW  (500 plies)
    Result: 3W / 1L / 1D  PASS

--- vs Wall ---
    Zero blunders, must outplay on pure merit (depth=3, window=60cp, blunder=0%)
Game  1/5  NN=White  LOSS  (200 plies)
...
    Result: 1W / 3L / 1D  FAIL

════════════════════════════════════════════════
             RESULTS SUMMARY
────────────────────────────────────────────────
  Opponent      W   L   D  Result
  Grunt         3   1   1  PASS
  Fortress      3   1   1  PASS
  Scholar       4   1   0  PASS
  Chaos         3   2   0  PASS
  Wall          1   3   1  FAIL
────────────────────────────────────────────────
  Overall: 4/5 opponents defeated

FAIL ✗  — must beat all 5 opponents (3/5 each)
```

You can adjust games per opponent for quick iteration:
```bash
cargo run -p cli --bin compete -- my_model.onnx --games-per-bot 3
```

### Validate Bot Elo (vs Stockfish)

Benchmark all 5 challengers against Stockfish to verify they're in the target Elo range:

```bash
cargo run -p cli --bin validate -- --all --games 50
```

Or benchmark a single challenger:
```bash
cargo run -p cli --bin validate -- --bot Wall --games 100
```

### Generate Test Models

The `scripts/` directory contains a Python script that creates two reference models:

```bash
python3 scripts/create_test_models.py
```

This produces:
- `models/random_policy.onnx` — near-random play (~315K params, loses badly)
- `models/capture_policy.onnx` — hand-crafted capture heuristic (~315K params, draws occasionally)

These are useful for verifying the pipeline works before you start training.

### Play Against the Bots Yourself

```bash
cargo run -p cli
```

Or launch the GUI:
```bash
cargo run -p gui
```

---

## Training a Competitive Model

A pure policy net needs to be roughly **1300+ Elo** to reliably beat all 5 challengers (3/5 each). Here are the main approaches, roughly ordered by difficulty:

### Supervised Learning on Human Games

Train on the [Lichess open database](https://database.lichess.org/) (free, ~500M games):
- Policy target: the move played
- Filter to games ≥ 1800 Elo and longer time controls
- A small CNN or ResNet trained for a few hours on a GPU should reach ~1400–1600 Elo

### Supervised Learning on Engine Evaluations

Generate positions and label them with Stockfish evaluations:
- Policy target: Stockfish's best move
- Value target: `tanh(centipawn_score / 400)`
- Cleaner signal than human games; avoids learning human biases

### Self-Play Reinforcement Learning

AlphaZero-style training:
- Start from a pre-trained or random network
- Generate games with MCTS guided by current policy
- Train on outcomes; repeat
- Expensive but can exceed human-level play

---

## Repository Layout

```
chess-challenge/
├── engine/          # Core library: board logic, challenger bots, NnBot
│   └── src/
│       ├── bot.rs   # Bot trait, BaselineBot, ChallengerConfig, CHALLENGERS fleet
│       ├── eval.rs  # Material + PST evaluation
│       ├── game.rs  # GameState, Outcome, repetition detection
│       ├── nn.rs    # NnBot: ONNX inference, board encoding, parameter counting
│       └── search.rs# Negamax + alpha-beta
├── cli/             # Command-line binaries
│   └── src/
│       ├── main.rs  # Human vs bot (ASCII board)
│       ├── compete.rs # Competition runner (5-bot fleet)
│       └── validate.rs # Elo validation vs Stockfish (--bot, --all)
├── gui/             # egui desktop GUI
├── models/          # Pre-built test models (.onnx)
└── scripts/
    └── create_test_models.py  # Generates reference models
```

---

## License

MIT
