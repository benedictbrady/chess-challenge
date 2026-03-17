# Chess Challenge

**Build the smallest neural network that can beat a strong chess engine.**

Train an ONNX evaluation network and run it against increasingly strong alpha-beta baselines (levels 1–4). Your net always uses **depth-1 search** — it evaluates every legal move one step ahead and picks the best. This is fixed and cannot be changed. Each level requires **70% or higher across 50 games**. Submissions are rated by highest level passed, then fewest parameters.

---

## The Challenge

Can a neural network's position understanding substitute for search depth?

| | Baseline | Your NN |
|---|---|---|
| **Eval** | Handcrafted (material, PSTs, king safety, passed pawns, mobility, pawn structure) | Learned (your ONNX model) |
| **Search** | Alpha-beta depth 1–4 + quiescence | Depth 1 + quiescence (follows captures to quiet positions) |
| **Target Elo** | ~1500–1600 (Level 4) | Must beat baseline at 70% per level |

The baseline sees several moves ahead with a handcrafted eval. Your network sees 1 move ahead but with (hopefully) a much stronger learned eval. Who wins?

### Levels

Your model is tested against increasingly strong baselines:

| Level | Name | Depth | Mode | Description |
|-------|------|-------|------|-------------|
| 1 | Beginner | 1 | classic | Depth-1 + quiescence — both sides follow captures |
| 2 | Novice | 2 | classic | Depth-2 alpha-beta + quiescence |
| 3 | Advanced | 3 | enhanced | Depth-3 + TT/PVS/NMP/delta pruning |
| 4 | Expert | 4 | enhanced | Full strength baseline (~1500–1600 Elo) |

Levels 1–2 use classic alpha-beta. Levels 3–4 switch to enhanced mode with transposition tables, null-move pruning, PVS, and delta pruning. By default the runner tests all levels and stops at the first failure.

---

## Model Spec

Your ONNX model must conform to this interface:

| Tensor | Name | Shape | dtype |
|--------|------|-------|-------|
| Input  | `board` | `[1, 1540]` | float32 |
| Output | (any) | `[1, 1]` | float32 |

The output is a **scalar evaluation**: positive = good for side to move.

### Board Encoding

The board is encoded as **dual perspective**: two 770-element halves = **1540 floats**. Each half encodes 12 piece planes × 64 squares + 2 castling rights from one side's viewpoint.

**First 770 — Side-to-move (STM) perspective:**

| Index | Contents |
|-------|----------|
| 0–383 | STM's Pawns, Knights, Bishops, Rooks, Queens, King (6 channels × 64 squares) |
| 384–767 | Opponent's Pawns, Knights, Bishops, Rooks, Queens, King (6 channels × 64 squares) |
| 768 | STM can castle kingside (1.0 / 0.0) |
| 769 | STM can castle queenside (1.0 / 0.0) |

**Last 770 — Non-side-to-move (NSTM) perspective:**

| Index | Contents |
|-------|----------|
| 770–1153 | NSTM's Pawns, Knights, Bishops, Rooks, Queens, King |
| 1154–1537 | STM's Pawns, Knights, Bishops, Rooks, Queens, King |
| 1538 | NSTM can castle kingside (1.0 / 0.0) |
| 1539 | NSTM can castle queenside (1.0 / 0.0) |

**Square indexing:** `a1=0, b1=1, …, h1=7, a2=8, …, h8=63`
**Flat index:** `channel * 64 + square` (offset by 770 for the NSTM half)

Each half flips ranks (a1↔a8) when its perspective's color is Black, so the network always sees that side's pawns advancing up the board. Files are not flipped.

Castling rights are included because they carry information the piece positions alone cannot express — a king on e1 with castling rights is fundamentally different from one without. Every competitive NNUE encodes them.

This dual encoding enables NNUE-style architectures where a shared feature transformer processes both perspectives through the same weights, then output layers compare the two views.

### How Your Model Is Used

For each move the NN makes:
1. Generate all legal moves
2. For each move, apply it to get a child position
3. If the child is checkmate or draw, score it immediately
4. Otherwise, run **quiescence search** on the child — follow all captures until the position is quiet, then evaluate with the NN
5. Negate the eval (child is from opponent's perspective)
6. Pick the move with the highest eval

Both the NN bot and the baseline use quiescence search at every level, so the comparison is fair: both sides follow captures to quiet positions before evaluating.

### ONNX Requirements

- Input named `board`
- Output can have any name (accessed by index)
- `ir_version = 8`, opset 17 recommended
- All weights stored as ONNX initializers (the harness counts parameters from these)
- Max **10,000,000 parameters**

---

## Rules

1. **Depth-1 search only** — your model always uses depth-1 search (see "How Your Model Is Used" above). This is enforced by the harness and cannot be changed.
2. **50 games per level** — 25 opening positions × 2 colors (NN plays both sides)
3. **Score 70% or higher** — win=1, draw=0.5, loss=0 (need 35/50 points)
4. **10M parameter limit** — models exceeding this are rejected
5. **Two-axis rating** — highest level passed, then fewest parameters (lower is better)
6. Games use openings from the book (`data/openings.txt`)

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
cargo run -p gui -- path/to/your_model.onnx
cargo run -p gui -- path/to/your_model.onnx --delay 300  # slower playback
```

---

## License

MIT
