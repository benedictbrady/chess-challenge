# Chess Challenge

**Build the smallest neural network that can beat a strong chess engine.**

Train an ONNX evaluation network and run it against an alpha-beta baseline (~1500-1600 Elo). Your net uses only 1-ply search — it evaluates every legal move one step ahead and picks the best. To pass, you must score **70% or higher across 50 games**. The winning submission is the one with the fewest parameters.

---

## The Challenge

Can a neural network's position understanding substitute for search depth?

| | Baseline | Your NN |
|---|---|---|
| **Eval** | Handcrafted (material, PSTs, king safety, passed pawns, mobility, pawn structure) | Learned (your ONNX model) |
| **Search** | Alpha-beta depth 1–4 + quiescence | 1-ply (evaluate all legal moves, pick best) |
| **Target Elo** | ~1500–1600 (Level 5) | Must beat baseline at 70% |

The baseline sees several moves ahead with a handcrafted eval. Your network sees 1 move ahead but with (hopefully) a much stronger learned eval. Who wins?

### Levels

Your model is tested against increasingly strong baselines:

| Level | Name | Depth | Mode | Description |
|-------|------|-------|------|-------------|
| 1 | Beginner | 1 | classic | Pure eval — picks best immediate position |
| 2 | Novice | 2 | classic | 2-ply alpha-beta + quiescence |
| 3 | Intermediate | 3 | classic | 3-ply alpha-beta + quiescence |
| 4 | Advanced | 3 | enhanced | 3-ply + TT/PVS/NMP/delta pruning |
| 5 | Expert | 4 | enhanced | Full strength baseline (~1500–1600 Elo) |

Levels 3 and 4 are both depth 3, but enhanced mode adds transposition tables, null-move pruning, and other techniques — a meaningful jump in difficulty. By default the runner tests all levels and stops at the first failure.

---

## Model Spec

Your ONNX model must conform to this interface:

| Tensor | Name | Shape | dtype |
|--------|------|-------|-------|
| Input  | `board` | `[1, 768]` | float32 |
| Output | (any) | `[1, 1]` | float32 |

The output is a **scalar evaluation**: positive = good for side to move.

### Board Encoding

The board is encoded as **12 binary planes × 64 squares = 768 floats**, always from the current player's perspective:

| Channel | Contents |
|---------|----------|
| 0–5 | Current player's Pawns, Knights, Bishops, Rooks, Queens, King |
| 6–11 | Opponent's Pawns, Knights, Bishops, Rooks, Queens, King |

**Square indexing:** `a1=0, b1=1, …, h1=7, a2=8, …, h8=63`
**Flat index:** `channel * 64 + square`

When it is Black's turn, ranks are flipped (a1↔a8) so the network always sees its own pawns advancing up the board. Files are not flipped.

### How Your Model Is Used

For each move the NN makes:
1. Generate all legal moves
2. For each move, apply it to get a child position
3. Encode all child positions as `[N, 768]` tensors (batched)
4. Run one ONNX inference call → `[N, 1]` output
5. Negate each eval (child is from opponent's perspective)
6. Pick the move with the highest eval

Checkmates are detected immediately (no NN needed). Draws evaluate to 0.0.

### ONNX Requirements

- Input named `board`
- Output can have any name (accessed by index)
- `ir_version = 8`, opset 17 recommended
- All weights stored as ONNX initializers (the harness counts parameters from these)
- Max **1,000,000 parameters**

---

## Rules

1. **50 games per level** — 25 opening positions × 2 colors (NN plays both sides)
2. **Score 70% or higher** — win=1, draw=0.5, loss=0 (need 35/50 points)
3. **1M parameter limit** — models exceeding this are rejected
4. **Two-axis rating** — highest level passed, then fewest parameters (lower is better)
5. Games use openings from the book (`data/openings.txt`)

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

Run all levels (stops at first failure):
```bash
cargo run -p cli --bin compete -- path/to/your_model.onnx
```

Run a single level:
```bash
cargo run -p cli --bin compete -- path/to/your_model.onnx --level 1
```

A reference model is included for testing the pipeline:

```bash
cargo run -p cli --bin compete -- models/capture_policy.onnx
```

This is a hand-crafted capture heuristic (~315K params) — it won't pass the competition but verifies the pipeline works end to end.

### Play Against the Bot

ASCII terminal:
```bash
cargo run -p cli
```

Desktop GUI:
```bash
cargo run -p gui
```

### Watch Bot vs Bot

Watch your model play against the baseline in the GUI:
```bash
cargo run -p gui -- models/capture_policy.onnx
cargo run -p gui -- models/capture_policy.onnx --delay 300  # slower playback
```

### Validate Bot Elo (vs Stockfish)

Benchmark the baseline against Stockfish to verify it's in the target Elo range:

```bash
cargo run -p cli --bin validate -- --games 50
```

---

## License

MIT
