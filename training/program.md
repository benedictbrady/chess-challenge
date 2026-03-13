# Chess NNUE Training — Research Program

## Challenge

Train a neural network chess evaluation function that scores **35/50 (70%)** against Stockfish level 1 (depth-4 search). The NN gets a depth-1 search + quiescence.

**Constraints:**
- Max 10M parameters
- ONNX format, 1540-float input (dual-perspective encoding)
- Must integrate with existing Rust `compete` CLI

**Current best: 19.5/50 (39%) — exp09** (nnue_1024x2_32_32, 1.5M outcome data, outcome_bce, eval_scale=1.0, early checkpoint ~epoch 15)

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

### 1. Fix sigmoid_mse via weight init (HIGH — directly addresses core blocker)

The _init_weights code in model.py scales the final layer to produce outputs in [-300, 300] cp range initially. This should prevent sigmoid(x/400) from starting at the flat 0.5 plateau.

**What to do:**
- Ingest fresh SF-labeled data: `modal run modal_app.py::ingest_lichess_sf --pgn-url "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst" --dataset-name lichess_2024_01_sf_500k --max-positions 500000 --sf-depth 10`
- Train with sigmoid_mse, eval_scale=400: `modal run modal_app.py::train --dataset-name lichess_2024_01_sf_500k --arch nnue_1024x2_32_32 --experiment exp33_sigmoid_mse_init --loss-type sigmoid_mse --eval-scale 400 --epochs 40 --batch-size 4096 --learning-rate 1e-3`
- **Verify** the model's initial output distribution spans ±300cp (not constant ~0)
- If it works, this unlocks position-evaluation training and likely jumps past 39%

### 2. Direct MSE on clamped centipawns (HIGH — avoids sigmoid entirely)

Skip the sigmoid transformation. Train with raw MSE on centipawn values clamped to [-2000, 2000]. The model learns centipawn-scale outputs directly. Simpler gradient landscape.

**What to do:**
- Use existing `direct_mse` loss in train.py (already implemented)
- `modal run modal_app.py::train --dataset-name lichess_2024_01_sf_500k --arch nnue_1024x2_32_32 --experiment exp_direct_mse --loss-type direct_mse --epochs 40 --batch-size 4096 --learning-rate 1e-3`
- No sigmoid at all — model output IS the centipawn eval

### 3. Curriculum: outcome pretrain → SF fine-tune (MEDIUM)

Use outcome training (which works) to learn coarse features, then switch to SF sigmoid_mse (with fixed init) for fine-grained eval. The outcome-pretrained model starts with meaningful features instead of random init.

**What to do:**
- Take exp09's best checkpoint (the 19.5/50 model)
- Fine-tune on SF data with sigmoid_mse at very low LR (1e-5)
- The model already has useful features; SF loss refines the evaluation

### 4. Outcome training with better early stopping (MEDIUM)

exp09's 19.5/50 came from an early checkpoint (~epoch 15 of 100). We never systematically explored which epoch is optimal. Train with frequent checkpointing and test every 5 epochs.

**What to do:**
- Retrain exp09's config but save checkpoints every 5 epochs
- Test each checkpoint at level 1
- Find the optimal training duration (might be epoch 8, might be epoch 20)
- This could squeeze a few more wins out of the current approach

### 5. Ranking/contrastive loss between moves (MEDIUM-LOW)

For each position, evaluate all legal moves by making each move and evaluating the resulting position. Train the model to rank moves in the same order as Stockfish. This directly optimizes what matters: move selection quality.

**Challenges:** requires generating move-comparison data, more complex training loop.

### 6. More/better Lichess data for outcome training (LOW)

The 1.5M→5.5M scaling didn't help — but the extra data was from different eras/Elo ranges. Try:
- Higher min_elo (2000+) for stronger game outcomes
- Same era/month, more positions (increase max_positions)
- Decisive games only (remove draws)

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
