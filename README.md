# Chess Challenge

**Build the smallest neural network that can beat a strong chess engine.**

Train an ONNX evaluation network and run it against increasingly strong baselines (levels 1–4). Your net always uses **depth-1 search**. Each level requires **70% or higher across 50 games**. Ranked by highest level passed, then fewest parameters.

---

## Levels

| Level | Name | Baseline |
|-------|------|----------|
| 1 | Beginner | Depth 1 — same search as your NN, just handcrafted eval |
| 2 | Novice | Depth 2 — sees your response to each move |
| 3 | Advanced | Depth 3 — with search optimizations |
| 4 | Expert | Depth 4 — with search optimizations (~1500–1600 Elo) |

Your NN always uses depth 1. All levels use the same quiescence search (follow captures to quiet positions, then evaluate). The runner tests all levels and stops at the first failure.

---

## Model Spec

| Tensor | Name | Shape | dtype |
|--------|------|-------|-------|
| Input  | `board` | `[N, 1540]` | float32 |
| Output | (any) | `[N, 1]` | float32 |

Output is a **scalar evaluation**: positive = good for side to move.

### Board Encoding (1540 floats)

Two 770-element halves (dual perspective). Each half: 12 piece planes × 64 squares + 2 castling rights.

**First 770 — Side-to-move (STM) perspective:**

| Index | Contents |
|-------|----------|
| 0–383 | STM's Pawns, Knights, Bishops, Rooks, Queens, King (6 × 64) |
| 384–767 | Opponent's pieces (same order) |
| 768 | STM can castle kingside (1.0 / 0.0) |
| 769 | STM can castle queenside (1.0 / 0.0) |

**Last 770 — Non-side-to-move (NSTM) perspective:**

| Index | Contents |
|-------|----------|
| 770–1153 | NSTM's pieces (6 × 64) |
| 1154–1537 | STM's pieces (6 × 64) |
| 1538 | NSTM can castle kingside (1.0 / 0.0) |
| 1539 | NSTM can castle queenside (1.0 / 0.0) |

**Square indexing:** `a1=0, b1=1, …, h1=7, a2=8, …, h8=63`. Ranks flip when that half's color is Black.

### How Your Model Is Used

1. Generate all legal moves
2. For each move, apply it to get a child position
3. If checkmate or draw, score immediately
4. Otherwise, run **quiescence search** (follow captures to quiet position, then evaluate with your NN)
5. Negate the eval (opponent's perspective)
6. Pick the move with the highest eval

### ONNX Requirements

- Input named `board`, shape `[N, 1540]`
- Output shape `[N, 1]`
- `ir_version = 8`, opset 17
- Max **10,000,000 parameters**
- **Batch dimension must be named** (e.g. `dim_param="batch"`) on both input and output

The harness uses batched inference for performance. Models with unnamed/anonymous batch dimensions will be **rejected at load time** with a diagnostic error.

**Common pitfall:** Using `MatMul + Add` for the final layer produces an output with an anonymous batch dimension, even if `dynamic_axes` names it. Use `nn.Linear` (which exports as a `Gemm` op) instead, or verify your export with `validate_onnx()`.

**Correct export example:**
```python
torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=["board"],
    output_names=["eval"],
    dynamic_axes={"board": {0: "batch"}, "eval": {0: "batch"}},
    opset_version=17,
)
```

---

## Rules

1. **Depth-1 search only** — enforced by the harness, cannot be changed
2. **50 games per level** — 25 openings × 2 colors
3. **Score 70%+** — win=1, draw=0.5, loss=0 (need 35/50)
4. **10M parameter limit**
5. **Ranked by** highest level passed, then fewest parameters

---

## Quick Start

```bash
# Build
cargo build --workspace

# Run all levels
cargo run -p cli --release --bin compete -- path/to/model.onnx

# Run a single level
cargo run -p cli --release --bin compete -- path/to/model.onnx --level 1

# Watch bot vs bot in GUI
cargo run -p gui -- path/to/model.onnx
```

---

## License

MIT
