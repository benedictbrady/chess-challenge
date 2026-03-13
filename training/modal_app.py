"""Modal app for chess NN training pipeline — 1540 dual-perspective encoding.

Data ingestion and training are separated. Data is ingested once to a volume,
then training runs read from the volume and can iterate quickly.

Usage:
  # Data ingestion (run once, results persist on volume)
  modal run modal_app.py::ingest_lichess --pgn-url "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst" --dataset-name lichess_2024_01 --max-positions 2000000
  modal run modal_app.py::ingest_lichess_sf --pgn-url "..." --dataset-name lichess_sf_500k --max-positions 500000 --sf-depth 10

  # Training (reads from volume, fast iteration)
  modal run modal_app.py::train --dataset-name lichess_2024_01 --arch nnue_256x2_32_32 --experiment exp01_baseline --epochs 30
  modal run modal_app.py::train --dataset-name lichess_sf_500k --arch nnue_512x2_32_32 --experiment exp02_sf --loss-type sigmoid_mse --epochs 50

  # Export (pull best checkpoint → ONNX)
  modal run modal_app.py::export --experiment exp01_baseline --arch nnue_256x2_32_32

  # List datasets and experiments
  modal run modal_app.py::list_data
"""

import modal

app = modal.App("chess-training-v2")

volume = modal.Volume.from_name("chess-training-v2", create_if_missing=True)
VOLUME_PATH = "/data"

# Images
_pip_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "python-chess>=1.999",
        "numpy>=1.24",
        "onnx>=1.14",
        "onnxruntime>=1.16",
        "pyyaml>=6.0",
        "tqdm>=4.65",
        "onnxscript>=0.1",
        "zstandard>=0.21",
    )
)

base_image = _pip_image.add_local_dir("src", "/app/src")

datagen_image = (
    _pip_image
    .apt_install("git", "make", "g++", "curl")
    .run_commands(
        "git clone --depth 1 --branch sf_16.1 https://github.com/official-stockfish/Stockfish.git /tmp/sf"
        " && cd /tmp/sf/src && make -j$(nproc) build ARCH=x86-64"
        " && cp /tmp/sf/src/stockfish /usr/local/bin/stockfish"
        " && rm -rf /tmp/sf"
    )
    .add_local_dir("src", "/app/src")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "numpy>=1.24",
        "onnx>=1.14",
        "onnxruntime>=1.16",
        "pyyaml>=6.0",
        "onnxscript>=0.1",
    )
    .add_local_dir("src", "/app/src")
)


# ── Data ingestion: Lichess positions with game outcomes ──────────────────────


@app.function(
    image=datagen_image,
    volumes={VOLUME_PATH: volume},
    cpu=2,
    memory=8192,
    timeout=7200,
)
def ingest_lichess(
    pgn_url: str,
    dataset_name: str,
    max_positions: int = 2_000_000,
    min_elo: int = 1800,
    sample_every_n: int = 4,
):
    """Download Lichess PGN, extract positions with game outcomes, save to volume.

    This creates outcome-labeled data (no Stockfish needed) — fast and scalable.
    """
    import sys
    sys.path.insert(0, "/app")
    import os
    import subprocess
    import numpy as np
    from src.data_gen import extract_positions_from_pgn

    out_dir = f"{VOLUME_PATH}/datasets/{dataset_name}"
    final_path = f"{out_dir}/data.npz"

    if os.path.exists(final_path):
        d = np.load(final_path)
        print(f"Dataset already exists: {d['positions'].shape[0]} positions")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Download PGN
    pgn_path = f"/tmp/games.pgn.zst"
    print(f"Downloading {pgn_url}...")
    subprocess.run(["curl", "-L", "-o", pgn_path, pgn_url], check=True)
    print(f"Download complete, extracting positions...")

    positions, outcomes = extract_positions_from_pgn(
        pgn_path,
        max_positions=max_positions,
        min_elo=min_elo,
        sample_every_n=sample_every_n,
        include_outcomes=True,
    )

    np.savez_compressed(final_path, positions=positions, outcomes=outcomes)
    volume.commit()

    print(f"\nSaved {positions.shape[0]} positions to {final_path}")
    print(f"  Shape: {positions.shape}")
    print(f"  Outcomes: mean={outcomes.mean():.3f} (0.5 = balanced)")


@app.function(
    image=datagen_image,
    volumes={VOLUME_PATH: volume},
    cpu=4,
    memory=8192,
    timeout=14400,  # 4 hours — SF labeling is slow
)
def ingest_lichess_sf(
    pgn_url: str,
    dataset_name: str,
    max_positions: int = 500_000,
    min_elo: int = 1800,
    sf_depth: int = 10,
):
    """Download Lichess PGN, extract positions, label with Stockfish + outcomes."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import io
    import subprocess
    import random
    import time
    import chess
    import chess.pgn
    import chess.engine
    import numpy as np
    from src.encoding import board_to_tensor

    out_dir = f"{VOLUME_PATH}/datasets/{dataset_name}"
    final_path = f"{out_dir}/data.npz"

    if os.path.exists(final_path):
        d = np.load(final_path)
        print(f"Dataset already exists: {d['positions'].shape[0]} positions")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Download
    pgn_path = "/tmp/games.pgn.zst"
    print(f"Downloading {pgn_url}...")
    subprocess.run(["curl", "-L", "-o", pgn_path, pgn_url], check=True)

    # Open PGN
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    raw_fh = open(pgn_path, "rb")
    reader = dctx.stream_reader(raw_fh)
    text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")

    # Start Stockfish
    engine = chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish")
    engine.configure({"Threads": 2, "Hash": 128})

    positions_list = []
    evals_list = []
    outcomes_list = []
    rng = random.Random(42)
    games_parsed = 0
    t0 = time.time()

    try:
        while len(positions_list) < max_positions:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break

            games_parsed += 1
            if games_parsed % 5000 == 0:
                print(f"  {games_parsed} games, {len(positions_list)} positions, "
                      f"{time.time() - t0:.0f}s")

            try:
                w_elo = int(game.headers.get("WhiteElo", "0"))
                b_elo = int(game.headers.get("BlackElo", "0"))
            except ValueError:
                continue
            if (w_elo + b_elo) / 2 < min_elo:
                continue
            if game.headers.get("Variant", "Standard") != "Standard":
                continue

            result = game.headers.get("Result", "*")
            if result == "1-0":
                w_outcome = 1.0
            elif result == "0-1":
                w_outcome = 0.0
            elif result == "1/2-1/2":
                w_outcome = 0.5
            else:
                continue

            board = game.board()
            ply = 0
            for node in game.mainline():
                board.push(node.move)
                ply += 1

                if ply < 10 or ply > 200:
                    continue
                if ply % 4 != 0:
                    continue
                if len(board.piece_map()) < 6 or board.is_game_over():
                    continue
                if rng.random() < 0.3:
                    continue

                # Stockfish eval
                try:
                    info = engine.analyse(board, chess.engine.Limit(depth=sf_depth))
                    score = info["score"].pov(board.turn)
                    cp = score.score(mate_score=15000)
                    if cp is None:
                        continue
                except Exception:
                    continue

                positions_list.append(board_to_tensor(board))
                evals_list.append(float(cp))
                outcome = w_outcome if board.turn == chess.WHITE else 1.0 - w_outcome
                outcomes_list.append(outcome)

                if len(positions_list) >= max_positions:
                    break

            # Periodic save (resume-friendly)
            if len(positions_list) > 0 and len(positions_list) % 50000 == 0:
                _save_checkpoint(out_dir, positions_list, evals_list, outcomes_list)

    finally:
        engine.quit()
        text_stream.close()
        raw_fh.close()

    # Final save
    positions = np.array(positions_list, dtype=np.float32)
    evals = np.array(evals_list, dtype=np.float32)
    outcomes = np.array(outcomes_list, dtype=np.float32)

    np.savez_compressed(final_path, positions=positions, evals=evals, outcomes=outcomes)
    volume.commit()

    print(f"\nSaved {len(positions)} positions to {final_path}")
    print(f"  Eval range: [{evals.min():.0f}, {evals.max():.0f}] cp")
    print(f"  Outcome mean: {outcomes.mean():.3f}")


def _save_checkpoint(out_dir, positions_list, evals_list, outcomes_list):
    import numpy as np
    path = f"{out_dir}/checkpoint.npz"
    np.savez_compressed(
        path,
        positions=np.array(positions_list, dtype=np.float32),
        evals=np.array(evals_list, dtype=np.float32),
        outcomes=np.array(outcomes_list, dtype=np.float32),
    )
    volume.commit()
    print(f"  Checkpoint: {len(positions_list)} positions saved")


# ── Re-label existing dataset with Stockfish ─────────────────────────────────


@app.function(
    image=datagen_image,
    volumes={VOLUME_PATH: volume},
    cpu=4,
    memory=8192,
    timeout=14400,
)
def relabel_with_stockfish(
    source_dataset: str,
    target_dataset: str,
    sf_depth: int = 8,
    max_positions: int = 500_000,
):
    """Take existing positions from a dataset and add Stockfish eval labels.

    Reads positions from source dataset, reconstructs boards, evaluates with
    Stockfish, and saves positions + evals + outcomes to target dataset.
    """
    import sys
    sys.path.insert(0, "/app")
    import os
    import time
    import chess
    import chess.engine
    import numpy as np
    from src.encoding import board_to_tensor, HALF_SIZE, PIECE_ORDER

    volume.reload()

    out_dir = f"{VOLUME_PATH}/datasets/{target_dataset}"
    final_path = f"{out_dir}/data.npz"
    if os.path.exists(final_path):
        d = np.load(final_path)
        print(f"Target dataset exists: {d['positions'].shape[0]} positions")
        return

    # Load source
    src_path = f"{VOLUME_PATH}/datasets/{source_dataset}/data.npz"
    src = np.load(src_path)
    src_positions = src["positions"]
    src_outcomes = src.get("outcomes", None)
    n_total = min(len(src_positions), max_positions)
    print(f"Source: {len(src_positions)} positions, will label {n_total}")

    # Reconstruct boards from tensor encoding
    def tensor_to_fen_approx(tensor):
        """Approximate FEN reconstruction from 1540 tensor.

        Since the tensor encodes from STM perspective with rank flips,
        exact reconstruction requires knowing which side is to move.
        We use the STM half and assume White to move (good enough for eval).
        """
        board = chess.Board.empty()
        board.turn = chess.WHITE

        piece_map = {
            0: chess.PAWN, 1: chess.KNIGHT, 2: chess.BISHOP,
            3: chess.ROOK, 4: chess.QUEEN, 5: chess.KING,
        }

        # STM pieces (channels 0-5 = White)
        for ch in range(6):
            for sq in range(64):
                if tensor[ch * 64 + sq] > 0.5:
                    board.set_piece_at(sq, chess.Piece(piece_map[ch], chess.WHITE))

        # NSTM pieces (channels 6-11 = Black)
        for ch in range(6):
            for sq in range(64):
                if tensor[(ch + 6) * 64 + sq] > 0.5:
                    board.set_piece_at(sq, chess.Piece(piece_map[ch], chess.BLACK))

        # Castling from STM half
        castling = ""
        if tensor[768] > 0.5: castling += "K"
        if tensor[769] > 0.5: castling += "Q"
        # NSTM castling
        if tensor[HALF_SIZE + 768] > 0.5: castling += "k"
        if tensor[HALF_SIZE + 769] > 0.5: castling += "q"

        if not castling: castling = "-"
        board.set_castling_fen(castling)

        return board

    os.makedirs(out_dir, exist_ok=True)

    # Start Stockfish
    engine = chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish")
    engine.configure({"Threads": 2, "Hash": 128})

    positions_out = []
    evals_out = []
    outcomes_out = []
    t0 = time.time()

    for i in range(n_total):
        if i % 5000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            print(f"  {i}/{n_total} ({rate:.0f} pos/s), {elapsed:.0f}s")

        tensor = src_positions[i]

        # Try to reconstruct board and evaluate
        try:
            board = tensor_to_fen_approx(tensor)
            if not board.is_valid() or board.is_game_over():
                continue

            info = engine.analyse(board, chess.engine.Limit(depth=sf_depth))
            score = info["score"].pov(board.turn)
            cp = score.score(mate_score=15000)
            if cp is None:
                continue

            # Re-encode from the reconstructed board (ensures consistency)
            new_tensor = board_to_tensor(board)
            positions_out.append(new_tensor)
            evals_out.append(float(cp))

            if src_outcomes is not None:
                outcomes_out.append(src_outcomes[i])
            else:
                outcomes_out.append(0.5)

        except Exception as e:
            if i < 5:
                print(f"  Skip {i}: {e}")
            continue

        # Periodic checkpoint
        if len(positions_out) > 0 and len(positions_out) % 50000 == 0:
            _save_sf_checkpoint(out_dir, positions_out, evals_out, outcomes_out)

    engine.quit()

    if not positions_out:
        print("ERROR: No positions labeled!")
        return

    positions = np.array(positions_out, dtype=np.float32)
    evals = np.array(evals_out, dtype=np.float32)
    outcomes = np.array(outcomes_out, dtype=np.float32)

    np.savez_compressed(final_path, positions=positions, evals=evals, outcomes=outcomes)
    volume.commit()

    elapsed = time.time() - t0
    print(f"\nLabeled {len(positions)} positions in {elapsed:.0f}s")
    print(f"  Eval range: [{evals.min():.0f}, {evals.max():.0f}] cp")
    print(f"  Eval mean: {evals.mean():.1f}, std: {evals.std():.1f}")


def _save_sf_checkpoint(out_dir, positions_list, evals_list, outcomes_list):
    import numpy as np
    path = f"{out_dir}/checkpoint.npz"
    np.savez_compressed(
        path,
        positions=np.array(positions_list, dtype=np.float32),
        evals=np.array(evals_list, dtype=np.float32),
        outcomes=np.array(outcomes_list, dtype=np.float32),
    )
    volume.commit()
    print(f"  Checkpoint: {len(positions_list)} positions")


# ── Fast data gen: random self-play + Stockfish labeling (no PGN download) ───


@app.function(
    image=datagen_image,
    volumes={VOLUME_PATH: volume},
    cpu=4,
    memory=4096,
    timeout=7200,
)
def generate_sf_data(
    dataset_name: str,
    num_positions: int = 500_000,
    sf_depth: int = 8,
    min_ply: int = 8,
    max_ply: int = 80,
    seed: int = 42,
):
    """Generate positions by random self-play and label with Stockfish.

    No PGN download needed — starts immediately. Good for quick experiments.
    Produces positions + evals (centipawns from STM perspective).
    """
    import sys
    sys.path.insert(0, "/app")
    import os
    import numpy as np
    from src.data_gen import generate_random_positions

    out_dir = f"{VOLUME_PATH}/datasets/{dataset_name}"
    final_path = f"{out_dir}/data.npz"

    if os.path.exists(final_path):
        d = np.load(final_path)
        print(f"Dataset already exists: {d['positions'].shape[0]} positions")
        return

    os.makedirs(out_dir, exist_ok=True)

    positions, evals = generate_random_positions(
        num_positions=num_positions,
        stockfish_path="/usr/local/bin/stockfish",
        depth=sf_depth,
        min_ply=min_ply,
        max_ply=max_ply,
        seed=seed,
    )

    np.savez_compressed(final_path, positions=positions, evals=evals)
    volume.commit()

    print(f"\nSaved {len(positions)} positions to {final_path}")
    print(f"  Eval range: [{evals.min():.0f}, {evals.max():.0f}] cp")
    print(f"  Eval mean: {evals.mean():.1f}, std: {evals.std():.1f}")


# ── Training ──────────────────────────────────────────────────────────────────


@app.function(
    image=gpu_image,
    volumes={VOLUME_PATH: volume},
    gpu="T4",
    memory=65536,
    timeout=14400,
)
def train(
    dataset_name: str,
    experiment: str,
    arch: str = "nnue_256x2_32_32",
    epochs: int = 30,
    batch_size: int = 16384,
    learning_rate: float = 1e-3,
    loss_type: str = "outcome_bce",
    eval_scale: float = 400.0,
    blend_lambda: float = 0.75,
    weight_decay: float = 1e-5,
    warmup_epochs: int = 2,
    grad_clip: float = 1.0,
    resume_from: str = "",
    max_abs_eval: float = 10000.0,
):
    """Train a model on data from the volume.

    dataset_name can be comma-separated to load multiple datasets without merging.
    E.g. --dataset-name "lichess_2015_01_outcomes,lichess_2016_01_outcomes"
    """
    import sys
    sys.path.insert(0, "/app")

    volume.reload()

    # Support comma-separated dataset names → comma-separated data paths
    names = [n.strip() for n in dataset_name.split(",")]
    if len(names) > 1:
        data_path = ",".join(f"{VOLUME_PATH}/datasets/{n}/data.npz" for n in names)
        print(f"Multi-dataset mode: {len(names)} datasets")
        for n in names:
            print(f"  - {n}")
    else:
        data_path = f"{VOLUME_PATH}/datasets/{dataset_name}/data.npz"

    ckpt_dir = f"{VOLUME_PATH}/checkpoints/{experiment}"
    model_dir = f"{VOLUME_PATH}/models/{experiment}"

    from src.train import train as run_training

    config = {
        "data_path": data_path,
        "architecture": arch,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "loss_type": loss_type,
        "eval_scale": eval_scale,
        "blend_lambda": blend_lambda,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "grad_clip": grad_clip,
        "checkpoint_dir": ckpt_dir,
        "num_workers": 2,
        "seed": 42,
        "max_abs_eval": max_abs_eval,
    }
    if resume_from:
        config["resume_from"] = f"{VOLUME_PATH}/checkpoints/{resume_from}/best.pt"

    model, best_val_loss = run_training(config)
    volume.commit()

    # Also export ONNX immediately
    import os
    os.makedirs(model_dir, exist_ok=True)
    from src.export_onnx import export_onnx, validate_onnx
    onnx_path = f"{model_dir}/model.onnx"

    # Load best checkpoint
    import torch
    best_path = f"{ckpt_dir}/best.pt"
    state = torch.load(best_path, map_location="cpu", weights_only=False)
    from src.model import build_model
    best_model = build_model(arch)
    best_model.load_state_dict(state["model_state_dict"])

    export_onnx(best_model, onnx_path)
    validate_onnx(onnx_path)
    volume.commit()

    print(f"\nExperiment: {experiment}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  ONNX: {onnx_path}")
    print(f"  Params: {best_model.num_params:,}")

    return {"experiment": experiment, "val_loss": best_val_loss, "params": best_model.num_params}


# ── Export ────────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    cpu=1,
    memory=4096,
    timeout=600,
)
def export(experiment: str, arch: str):
    """Export best checkpoint to ONNX and validate."""
    import sys
    sys.path.insert(0, "/app")
    import os

    volume.reload()

    ckpt_path = f"{VOLUME_PATH}/checkpoints/{experiment}/best.pt"
    model_dir = f"{VOLUME_PATH}/models/{experiment}"
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = f"{model_dir}/model.onnx"

    from src.export_onnx import export_from_checkpoint
    export_from_checkpoint(ckpt_path, arch, onnx_path)
    volume.commit()

    print(f"Exported: {onnx_path}")


@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    cpu=1,
    memory=2048,
    timeout=300,
)
def download_model(experiment: str) -> bytes:
    """Download ONNX model bytes from volume."""
    volume.reload()
    path = f"{VOLUME_PATH}/models/{experiment}/model.onnx"
    with open(path, "rb") as f:
        return f.read()


# ── Utilities ─────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    cpu=1,
    memory=2048,
    timeout=300,
)
def list_data():
    """List all datasets and experiments on the volume."""
    import os
    volume.reload()

    print("=== Datasets ===")
    ds_dir = f"{VOLUME_PATH}/datasets"
    if os.path.exists(ds_dir):
        for name in sorted(os.listdir(ds_dir)):
            data_path = f"{ds_dir}/{name}/data.npz"
            if os.path.exists(data_path):
                import numpy as np
                d = np.load(data_path)
                n = d["positions"].shape[0]
                keys = list(d.keys())
                print(f"  {name}: {n:,} positions, keys={keys}")
            else:
                print(f"  {name}: (no data.npz)")

    print("\n=== Experiments ===")
    ckpt_dir = f"{VOLUME_PATH}/checkpoints"
    if os.path.exists(ckpt_dir):
        for name in sorted(os.listdir(ckpt_dir)):
            best = f"{ckpt_dir}/{name}/best.pt"
            has_best = os.path.exists(best)
            model_path = f"{VOLUME_PATH}/models/{name}/model.onnx"
            has_onnx = os.path.exists(model_path)
            print(f"  {name}: best={'yes' if has_best else 'no'}, onnx={'yes' if has_onnx else 'no'}")


@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    cpu=2,
    memory=65536,
    timeout=1200,
)
def merge_datasets(dataset_names: str, target_name: str):
    """Merge multiple datasets into one. dataset_names is comma-separated."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import numpy as np

    volume.reload()
    names = [n.strip() for n in dataset_names.split(",")]
    out_dir = f"{VOLUME_PATH}/datasets/{target_name}"
    os.makedirs(out_dir, exist_ok=True)

    all_positions = []
    all_outcomes = []
    all_evals = []

    for name in names:
        path = f"{VOLUME_PATH}/datasets/{name}/data.npz"
        d = np.load(path)
        all_positions.append(d["positions"])
        if "outcomes" in d:
            all_outcomes.append(d["outcomes"])
        if "evals" in d:
            all_evals.append(d["evals"])
        print(f"  {name}: {d['positions'].shape[0]:,} positions")

    positions = np.concatenate(all_positions, axis=0)
    out_data = {"positions": positions}

    if all_outcomes:
        out_data["outcomes"] = np.concatenate(all_outcomes, axis=0)
    if all_evals:
        out_data["evals"] = np.concatenate(all_evals, axis=0)

    # Use uncompressed savez for large datasets (compressed can corrupt on large arrays)
    np.savez(f"{out_dir}/data.npz", **out_data)
    volume.commit()
    print(f"\nMerged {len(positions):,} positions into {target_name}")


@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    cpu=1,
    memory=2048,
    timeout=300,
)
def dataset_stats(dataset_name: str):
    """Print statistics about a dataset."""
    import sys
    sys.path.insert(0, "/app")
    import numpy as np

    volume.reload()
    path = f"{VOLUME_PATH}/datasets/{dataset_name}/data.npz"
    d = np.load(path)

    print(f"Dataset: {dataset_name}")
    print(f"  Positions: {d['positions'].shape}")

    if "evals" in d:
        evals = d["evals"]
        print(f"  Evals: mean={evals.mean():.1f}, std={evals.std():.1f}, "
              f"range=[{evals.min():.0f}, {evals.max():.0f}]")
        # Distribution
        for thresh in [100, 200, 500, 1000, 2000]:
            pct = (np.abs(evals) < thresh).mean() * 100
            print(f"    |eval| < {thresh}: {pct:.1f}%")

    if "outcomes" in d:
        outcomes = d["outcomes"]
        wins = (outcomes > 0.7).mean()
        draws = ((outcomes > 0.3) & (outcomes < 0.7)).mean()
        losses = (outcomes < 0.3).mean()
        print(f"  Outcomes: W={wins:.1%}, D={draws:.1%}, L={losses:.1%}")
