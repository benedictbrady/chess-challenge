# Chess NNUE Training — Research Program

## Challenge

Train a neural network chess evaluation function that scores **35/50 (70%)** against Stockfish level 1 (depth-4 search). The NN gets a depth-1 search + quiescence.

**Constraints:**
- Max 10M parameters
- ONNX format, 1540-float input (dual-perspective encoding)
- Must integrate with existing Rust `compete` CLI

**Current best: ~22-24/50 (44-49%) — exp40** (nnue_2048x2_64_32, 500K SF data, sigmoid_mse, eval_scale=400, 60 epochs + alpha-beta search fix)

**Previous best: 19.5/50 (39%) — exp09** (nnue_1024x2_32_32, 1.5M outcome data, outcome_bce, eval_scale=1.0, early checkpoint)

## What we know

### Works
- **outcome_bce with eval_scale=1.0** is the only loss that produces a playing model
- **nnue_1024x2_32_32** (856K params) is the sweet spot architecture for current data sizes
- **1.5M positions from Lichess 2015+2016** is the best dataset so far
- **Early stopping is critical** — best playing strength comes well before best val loss (exp07: 10/50 early vs 1/50 final; exp09 tested from ~epoch 15)

### Doesn't work
- **eval_scale=400 with outcome_bce** — squashes all outputs to ~0.5, killing gradients (exp01-06)
- **sigmoid_mse with any eval_scale (50, 400)** — causes model collapse (constant output ~0.267). The fundamental issue: default PyTorch init produces near-zero outputs, sigmoid(0/400)=0.5, gradient ≈ 0. Even with proper init code in model.py, it hasn't been tested yet.
- **Self-play random positions** — unrealistic positions, total failure (exp11)
- **Reconstructed SF evals from tensors** — ~50% reconstruction errors corrupted exp12/13/23/24/25
- **More data beyond 1.5M** — 3.5M and 5.5M both performed worse than 1.5M (noise dilution? data diversity issues?)
- **Fine-tuning from exp09** — degraded performance (exp18: 9.5, exp19: 12.0), base checkpoint was past its sweet spot
- **Larger models (4.2M params) on <2M data** — overfit badly (exp08)

### Core blocker

**outcome_bce learns game-outcome prediction, not position evaluation.** A position where White is up a pawn but lost the game gets trained as "bad for White." This fundamentally caps performance because move selection requires knowing which positions are objectively better, not which positions correlate with eventual victory.

The fix is training on Stockfish centipawn labels (sigmoid_mse or direct_mse), but all attempts have collapsed. The _init_weights code in model.py was written to fix this but **has never been tested with a fresh training run on real SF data**.

## Experiment protocol

1. **Pick the top idea** from the queue below
2. **Run on Modal** with `modal run modal_app.py::train ...`
3. **Test at level 1** with `cargo run -p cli --bin compete -- <model.onnx> --level 1`
4. **Log result** in `results.tsv` with score, params, and notes
5. **Update this file** — move idea to "tried" if done, add new insights
6. **Commit** the updated results + program

### Testing notes
- Always test from the **best checkpoint** (not final), since overfitting degrades play
- A quick sanity check: if the model outputs near-constant values for different positions, it's collapsed
- Score variance is ~±2 games at 50-game sample size

## Ideas queue (ranked by expected impact)

### 0. DONE — Breakthroughs from training-run-2 session

**sigmoid_mse with weight init fix works** — exp33 was the first functional sigmoid_mse model (15/50). The _init_weights code in model.py was the key.

**nnue_2048x2_64_32 is the sweet spot** — 1.8M params. Smaller (856K) scores 15/50, larger (4.2M) overfits on 500K data and slows inference.

**Alpha-beta pruning at search root** — 4x speedup. Passes best-so-far alpha to quiescence calls. Changed engine/src/nn.rs.

**Data filtering to |eval| < 500cp** — improves val loss but doesn't change playing strength.

**Curriculum from outcome models is a dead end** — sigmoid(x/400) gradient = 0 when model outputs [-2, 2].

**The plateau at 20-24/50** — All configs (500K-1M data, 1.8M-4.2M params, various hyperparams) converge to this range. Val loss improvements don't translate to better play. The gap to 35/50 requires fundamentally better training data (higher SF depth or 10x more positions).

### 1. Higher depth SF labels (HIGH — next priority)

Depth-10 SF evals are noisy. The model plateau at 20-24/50 may be because the training signal quality caps out. Depth 15-20 gives much more accurate positional evaluations.

**Status:** `lichess_2024_03_sf_d15_500k` ingestion started (depth 15, March 2024). PGN downloaded, SF labeling in progress (~4-6 hours).

**What to do when ready:**
- `modal run modal_app.py::train --dataset-name lichess_2024_03_sf_d15_500k --arch nnue_2048x2_64_32 --experiment exp_d15_2048 --loss-type sigmoid_mse --eval-scale 400 --epochs 60 --batch-size 4096 --learning-rate 5e-4`
- Compare against depth-10 models at same data size
- If better, ingest more depth-15 data from other months

### 2. Much more training data (HIGH)

1M positions → 5-10M positions. The handcrafted baseline has ~600 lines of domain knowledge (material, PST, king safety, mobility, passed pawns, pawn structure, bishop pair, rook files). Our NN must learn ALL of this from data. 1M positions may not be enough to accurately learn subtle features like king safety or passed pawn bonuses.

### 3. Ranking/contrastive loss between moves (MEDIUM)

Instead of training position evaluation accuracy, train move selection quality. For each position, evaluate all legal moves → rank by SF eval. Train model to preserve ranking. Directly optimizes what matters for playing strength.

### 4. Ensemble of multiple random inits (MEDIUM-LOW)

Scores vary ±3-4 between runs with same config (exp40: 24.5, exp43: 21.5). Training multiple models and picking the best init might get a lucky 26-28.

## Datasets on volume

| Name | Size | Labels | Source |
|------|------|--------|--------|
| lichess_2015_01_outcomes | 500K | outcomes | Lichess Jan 2015, min Elo 1600 |
| lichess_2016_01_outcomes | 1M | outcomes | Lichess Jan 2016, min Elo 1600 |
| lichess_2017_01_outcomes | 2M | outcomes | Lichess Jan 2017, min Elo 1600 |
| lichess_2024_06_outcomes | 2M | outcomes | Lichess Jun 2024, min Elo 1800 |
| lichess_2024_01_sf_100k | 100K | evals+outcomes | Lichess Jan 2024, SF depth 10 |
| lichess_2024_01_sf_500k | 500K | evals+outcomes | Lichess Jan 2024, SF depth 10 |
| selfplay_sf_200k | 200K | evals | Random self-play, SF depth 8 |
| lichess_2015_01_sf_labeled | 200K | evals+outcomes | Reconstructed from tensors (CORRUPT) |
| merged_1.5M_outcomes | 1.5M | outcomes | 2015+2016 merge |
| merged_3.5M_outcomes | 3.5M | outcomes | 2015+2016+2017 merge |
