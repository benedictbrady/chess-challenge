# Chess Challenge

**Who can build the smallest neural network that beats our chess bot?**

This repo hosts a Rust chess engine and a competition harness. The goal is simple: train an ONNX policy network with as few parameters as possible that can beat **SpicyBot** (our ~1000 ELO bot) **10 games out of 10**.

The winning submission is the one with the lowest parameter count that achieves a perfect 10/10 record.

---

## The Bot You're Beating

**SpicyBot** is a depth-3 negamax engine with alpha-beta pruning, material + piece-square-table evaluation, a candidate-move window (±80 cp), and a 15% random blunder rate. It plays roughly 1000–1200 ELO.

Your model plays **without any search** — pure policy, no MCTS, no alpha-beta. The network must encode enough chess understanding to beat SpicyBot move-by-move on its own.

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

1. Your model plays **10 games** against SpicyBot — 5 as White, 5 as Black (alternating)
2. You must **win all 10** — draws count as losses for the challenger
3. Models exceeding **10 million parameters** are rejected outright
4. **Score:** total parameter count — lower is better
5. Tie-break: if two submissions both win 10/10, the smaller parameter count wins

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

Running 10 games vs SpicyBot…
────────────────────────────────────────────────────
Game  1/10  NN=White  LOSS  (162 plies)
Game  2/10  NN=Black  DRAW  (229 plies)
...
────────────────────────────────────────────────────
Results: 0W / 7L / 3D

FAIL ✗  — won 0/10 (need all 10)
```

You can also run fewer games for quick iteration:
```bash
cargo run -p cli --bin compete -- my_model.onnx --games 3
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

### Play Against SpicyBot Yourself

```bash
cargo run -p cli
```

Or launch the GUI:
```bash
cargo run -p gui
```

---

## Training a Competitive Model

A pure policy net needs to be roughly **1300+ ELO** to reliably win 10/10 against a ~1000 ELO opponent. Here are the main approaches, roughly ordered by difficulty:

### Supervised Learning on Human Games

Train on the [Lichess open database](https://database.lichess.org/) (free, ~500M games):
- Policy target: the move played
- Filter to games ≥ 1800 ELO and longer time controls
- A small CNN or ResNet trained for a few hours on a GPU should reach ~1400–1600 ELO

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

### Architecture Starting Points

| Params | Architecture | Expected ELO (pure policy) |
|--------|--------------|---------------------------|
| ~50K   | 2-layer MLP  | Random / ~600             |
| ~500K  | 3-layer MLP  | ~1000–1200 (if trained well) |
| ~2M    | Small CNN    | ~1400–1600                |
| ~10M   | ResNet-5     | ~1800–2000                |

---

## Repository Layout

```
chess-challenge/
├── engine/          # Core library: board logic, SpicyBot, NnBot
│   └── src/
│       ├── bot.rs   # Bot trait, SpicyBot
│       ├── eval.rs  # Material + PST evaluation
│       ├── game.rs  # GameState, Outcome, repetition detection
│       ├── nn.rs    # NnBot: ONNX inference, board encoding, parameter counting
│       └── search.rs# Negamax + alpha-beta
├── cli/             # Command-line binaries
│   └── src/
│       ├── main.rs  # Human vs SpicyBot (ASCII board)
│       └── compete.rs # Competition runner
├── gui/             # egui desktop GUI
├── models/          # Pre-built test models (.onnx)
└── scripts/
    └── create_test_models.py  # Generates reference models
```

---

## Tips

- **Validate your encoding first.** Before training, write a quick test: set up a known position, call your encoder, and verify the tensor values are what you expect.
- **Auto-queen is fine.** The harness always promotes to Queen. Don't waste capacity trying to learn under-promotions.
- **Castling:** the board encoding doesn't include castling rights or en passant. Your network can still learn to castle — it will see the king and rook on their starting squares and infer castling is likely available. Adding those features as extra channels could improve play.
- **The competition is deterministic.** SpicyBot has a 15% random blunder rate, so results will vary slightly across runs. Test with `--games 20` or more before submitting to make sure your model wins consistently.

---

## License

MIT
