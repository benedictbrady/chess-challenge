"""Modal app for chess NN training pipeline.

Data generation is fanned out as many small parallel jobs (shards).
Each shard generates ~5K positions and saves to the volume independently.
A merge step combines shards into the final dataset.

Usage:
  modal run modal_app.py::generate_data --num-positions 10000000 --depth 12 --dataset-name v2_quiet --quiet-only
  modal run modal_app.py::generate_lichess_data --pgn-url "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst" --dataset-name v3_lichess
  modal run modal_app.py::train_model --config configs/level1_v3_lichess.yaml
  modal run modal_app.py::export_model --experiment level1_v3_lichess
"""

import modal

app = modal.App("chess-nn-training")

volume = modal.Volume.from_name("chess-training-data", create_if_missing=True)
VOLUME_PATH = "/data"

SHARD_SIZE = 5_000  # positions per shard — small enough to survive preemption (SF depth 12 + quiet ~10-15 min/shard)

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

base_image = (
    _pip_image
    .add_local_dir("src", "/app/src")
    .add_local_dir("configs", "/app/configs")
)

datagen_image = (
    _pip_image
    .apt_install("git", "make", "g++", "curl")
    .run_commands(
        # Build Stockfish 16.1 from source (prebuilt binary needs newer libstdc++)
        "git clone --depth 1 --branch sf_16.1 https://github.com/official-stockfish/Stockfish.git /tmp/sf"
        " && cd /tmp/sf/src && make -j$(nproc) build ARCH=x86-64"
        " && cp /tmp/sf/src/stockfish /usr/local/bin/stockfish"
        " && rm -rf /tmp/sf"
    )
    .add_local_dir("src", "/app/src")
    .add_local_dir("configs", "/app/configs")
    .add_local_file("../data/openings.txt", "/app/data/openings.txt")
)


# ── Data generation: fan-out / fan-in ────────────────────────────────────────


@app.function(
    image=datagen_image,
    volumes={VOLUME_PATH: volume},
    cpu=1,
    memory=2048,
    timeout=3600,  # 60 min per shard — allows for quiet filtering + deeper search
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=5.0),
)
def generate_shard(
    shard_id: int,
    num_positions: int,
    dataset_name: str,
    depth: int,
    min_ply: int,
    max_ply: int,
    quiet_only: bool = False,
):
    """Generate one shard of training data and save it to the volume."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import numpy as np
    from src.data_gen import generate_positions

    shard_dir = f"{VOLUME_PATH}/datasets/{dataset_name}/shards"
    shard_path = f"{shard_dir}/shard_{shard_id:04d}.npz"

    # Skip if shard already exists (resume after preemption)
    if os.path.exists(shard_path):
        existing = np.load(shard_path)
        n = existing["positions"].shape[0]
        print(f"Shard {shard_id} already exists ({n} positions), skipping.")
        return shard_id, n

    os.makedirs(shard_dir, exist_ok=True)

    positions, evals, _fens = generate_positions(
        num_positions=num_positions,
        stockfish_path="/usr/local/bin/stockfish",
        depth=depth,
        min_ply=min_ply,
        max_ply=max_ply,
        num_threads=1,
        hash_mb=16,
        quiet_only=quiet_only,
    )

    np.savez_compressed(shard_path, positions=positions, evals=evals)
    volume.commit()
    print(f"Shard {shard_id}: saved {len(positions)} positions to {shard_path}")
    return shard_id, len(positions)


@app.function(
    image=datagen_image,
    volumes={VOLUME_PATH: volume},
    cpu=1,
    memory=2048,
    timeout=3600,
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=5.0),
)
def generate_dual_shard(
    shard_id: int,
    num_positions: int,
    dataset_name: str,
    depth: int,
    min_ply: int,
    max_ply: int,
):
    """Generate one shard with BOTH baseline and SF labels per position."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import numpy as np
    from src.data_gen import generate_dual_positions

    shard_dir = f"{VOLUME_PATH}/datasets/{dataset_name}/shards"
    shard_path = f"{shard_dir}/shard_{shard_id:04d}.npz"

    if os.path.exists(shard_path):
        existing = np.load(shard_path)
        n = existing["positions"].shape[0]
        print(f"Shard {shard_id} already exists ({n} positions), skipping.")
        return shard_id, n

    os.makedirs(shard_dir, exist_ok=True)

    positions, bl_evals, sf_evals = generate_dual_positions(
        num_positions=num_positions,
        stockfish_path="/usr/local/bin/stockfish",
        depth=depth,
        min_ply=min_ply,
        max_ply=max_ply,
    )

    np.savez_compressed(shard_path, positions=positions, baseline_evals=bl_evals, sf_evals=sf_evals)
    volume.commit()
    print(f"Shard {shard_id}: saved {len(positions)} dual-labeled positions to {shard_path}")
    return shard_id, len(positions)


@app.function(
    image=datagen_image,
    volumes={VOLUME_PATH: volume},
    cpu=2,
    memory=32768,
    timeout=7200,
)
def merge_dual_shards(dataset_name: str, num_shards: int):
    """Merge dual-labeled shards preserving both eval columns."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import json
    import numpy as np

    shard_dir = f"{VOLUME_PATH}/datasets/{dataset_name}/shards"
    volume.reload()

    all_positions = []
    all_bl = []
    all_sf = []

    for i in range(num_shards):
        path = f"{shard_dir}/shard_{i:04d}.npz"
        if not os.path.exists(path):
            print(f"WARNING: shard {i} missing at {path}")
            continue
        data = np.load(path)
        all_positions.append(data["positions"])
        all_bl.append(data["baseline_evals"])
        all_sf.append(data["sf_evals"])
        if i % 100 == 0:
            print(f"  Loaded shard {i}: {data['positions'].shape[0]} positions")

    positions = np.concatenate(all_positions, axis=0)
    bl_evals = np.concatenate(all_bl, axis=0)
    sf_evals = np.concatenate(all_sf, axis=0)

    output_dir = f"{VOLUME_PATH}/datasets/{dataset_name}"
    np.savez_compressed(f"{output_dir}/data.npz",
                        positions=positions, baseline_evals=bl_evals, sf_evals=sf_evals)

    # Also save single-label version (evals = baseline) for compatibility
    np.savez_compressed(f"{output_dir}/data_baseline_only.npz",
                        positions=positions, evals=bl_evals)

    metadata = {
        "num_positions": int(positions.shape[0]),
        "num_shards": num_shards,
        "dual_label": True,
        "baseline_mean": float(bl_evals.mean()), "baseline_std": float(bl_evals.std()),
        "sf_mean": float(sf_evals.mean()), "sf_std": float(sf_evals.std()),
    }
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    volume.commit()
    print(f"\nMerged {positions.shape[0]} dual-labeled positions")
    print(f"Baseline: mean={bl_evals.mean():.1f}, std={bl_evals.std():.1f}")
    print(f"SF: mean={sf_evals.mean():.1f}, std={sf_evals.std():.1f}")
    return int(positions.shape[0])


@app.local_entrypoint()
def generate_dual_data(
    num_positions: int = 5_000_000,
    depth: int = 8,
    dataset_name: str = "v4_dual_bl_sf",
    min_ply: int = 8,
    max_ply: int = 80,
):
    """Generate dual-labeled data (baseline + SF), then merge."""
    num_shards = (num_positions + SHARD_SIZE - 1) // SHARD_SIZE
    per_shard = SHARD_SIZE

    print(f"Generating {num_positions} dual-labeled positions (SF d{depth}) as {num_shards} shards")

    shard_args = [
        (i, per_shard, dataset_name, depth, min_ply, max_ply)
        for i in range(num_shards)
    ]

    total = 0
    for result in generate_dual_shard.starmap(shard_args):
        shard_id, count = result
        total += count
        if shard_id % 50 == 0:
            print(f"  Shard {shard_id} done: {count} positions (running total: {total})")

    print(f"\nAll shards complete. Merging...")
    final_count = merge_dual_shards.remote(dataset_name, num_shards)
    print(f"Dataset ready: {final_count} dual-labeled positions in {dataset_name}")


@app.function(
    image=datagen_image,
    volumes={VOLUME_PATH: volume},
    cpu=2,
    memory=32768,
    timeout=7200,
)
def merge_shards(dataset_name: str, num_shards: int):
    """Merge all shards into a single dataset .npz file."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import json
    import numpy as np

    shard_dir = f"{VOLUME_PATH}/datasets/{dataset_name}/shards"
    volume.reload()  # pick up newly written shards

    all_positions = []
    all_evals = []

    for i in range(num_shards):
        path = f"{shard_dir}/shard_{i:04d}.npz"
        if not os.path.exists(path):
            print(f"WARNING: shard {i} missing at {path}")
            continue
        data = np.load(path)
        all_positions.append(data["positions"])
        all_evals.append(data["evals"])
        print(f"  Loaded shard {i}: {data['positions'].shape[0]} positions")

    positions = np.concatenate(all_positions, axis=0)
    evals = np.concatenate(all_evals, axis=0)

    output_dir = f"{VOLUME_PATH}/datasets/{dataset_name}"
    np.savez_compressed(f"{output_dir}/data.npz", positions=positions, evals=evals)

    metadata = {
        "num_positions": int(positions.shape[0]),
        "num_shards": num_shards,
        "eval_mean": float(evals.mean()),
        "eval_std": float(evals.std()),
        "eval_min": float(evals.min()),
        "eval_max": float(evals.max()),
    }
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    volume.commit()
    print(f"\nMerged {positions.shape[0]} positions into {output_dir}/data.npz")
    print(f"Eval range: [{evals.min():.0f}, {evals.max():.0f}] cp")
    print(f"Eval mean: {evals.mean():.1f}, std: {evals.std():.1f}")
    return int(positions.shape[0])


@app.local_entrypoint()
def generate_data(
    num_positions: int = 2_000_000,
    depth: int = 10,
    dataset_name: str = "v1",
    min_ply: int = 8,
    max_ply: int = 80,
    quiet_only: bool = False,
):
    """Fan out shard generation, then merge."""
    num_shards = (num_positions + SHARD_SIZE - 1) // SHARD_SIZE
    per_shard = SHARD_SIZE

    quiet_str = " (quiet only)" if quiet_only else ""
    print(f"Generating {num_positions} positions{quiet_str} as {num_shards} shards of {per_shard}")

    shard_args = [
        (i, per_shard, dataset_name, depth, min_ply, max_ply, quiet_only)
        for i in range(num_shards)
    ]

    # Fan out: run all shards in parallel
    total = 0
    for result in generate_shard.starmap(shard_args):
        shard_id, count = result
        total += count
        print(f"  Shard {shard_id} done: {count} positions (running total: {total})")

    print(f"\nAll shards complete. Merging...")
    final_count = merge_shards.remote(dataset_name, num_shards)
    print(f"Dataset ready: {final_count} positions in {dataset_name}")


# ── Baseline eval data generation (no Stockfish needed) ─────────────────────

BASELINE_SHARD_SIZE = 5_000  # same shard size, but much faster per position


@app.function(
    image=base_image,  # no Stockfish needed — pure Python eval
    volumes={VOLUME_PATH: volume},
    cpu=1,
    memory=2048,
    timeout=1800,  # 30 min — baseline eval is ~10x faster than SF
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=5.0),
)
def generate_baseline_shard(
    shard_id: int,
    num_positions: int,
    dataset_name: str,
    min_ply: int,
    max_ply: int,
):
    """Generate one shard using the baseline eval (no Stockfish)."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import numpy as np
    from src.data_gen import generate_positions

    shard_dir = f"{VOLUME_PATH}/datasets/{dataset_name}/shards"
    shard_path = f"{shard_dir}/shard_{shard_id:04d}.npz"

    if os.path.exists(shard_path):
        try:
            existing = np.load(shard_path)
            n = existing["positions"].shape[0]
            print(f"Shard {shard_id} already exists ({n} positions), skipping.")
            return shard_id, n
        except Exception:
            print(f"Shard {shard_id} corrupted, regenerating...")
            os.remove(shard_path)

    os.makedirs(shard_dir, exist_ok=True)

    positions, evals, _fens = generate_positions(
        num_positions=num_positions,
        depth=0,  # unused for baseline
        min_ply=min_ply,
        max_ply=max_ply,
        num_threads=1,
        eval_engine="baseline",
    )

    np.savez_compressed(shard_path, positions=positions, evals=evals)
    volume.commit()
    print(f"Shard {shard_id}: saved {len(positions)} positions to {shard_path}")
    return shard_id, len(positions)


@app.local_entrypoint()
def generate_baseline_data(
    num_positions: int = 10_000_000,
    dataset_name: str = "v4_baseline_distill",
    min_ply: int = 8,
    max_ply: int = 80,
):
    """Generate data using baseline eval (no Stockfish), then merge."""
    num_shards = (num_positions + BASELINE_SHARD_SIZE - 1) // BASELINE_SHARD_SIZE
    per_shard = BASELINE_SHARD_SIZE

    print(f"Generating {num_positions} baseline-eval positions as {num_shards} shards of {per_shard}")

    shard_args = [
        (i, per_shard, dataset_name, min_ply, max_ply)
        for i in range(num_shards)
    ]

    total = 0
    for result in generate_baseline_shard.starmap(shard_args):
        shard_id, count = result
        total += count
        if shard_id % 100 == 0:
            print(f"  Shard {shard_id} done: {count} positions (running total: {total})")

    print(f"\nAll shards complete. Merging...")
    final_count = merge_shards.remote(dataset_name, num_shards)
    print(f"Dataset ready: {final_count} positions in {dataset_name}")


# ── Search distillation data generation (depth-N minimax, no SF) ─────────────


@app.function(
    image=base_image,  # no Stockfish needed — pure Python search
    volumes={VOLUME_PATH: volume},
    cpu=1,
    memory=2048,
    timeout=3600,  # depth-1: ~3min/shard, depth-2: ~38min/shard
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=5.0),
)
def generate_search_shard(
    shard_id: int,
    num_positions: int,
    dataset_name: str,
    search_depth: int,
    min_ply: int,
    max_ply: int,
):
    """Generate one shard using search-based eval (depth-N minimax with baseline eval)."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import numpy as np
    from src.data_gen import generate_positions

    shard_dir = f"{VOLUME_PATH}/datasets/{dataset_name}/shards"
    shard_path = f"{shard_dir}/shard_{shard_id:04d}.npz"

    if os.path.exists(shard_path):
        existing = np.load(shard_path)
        n = existing["positions"].shape[0]
        print(f"Shard {shard_id} already exists ({n} positions), skipping.")
        return shard_id, n

    os.makedirs(shard_dir, exist_ok=True)

    positions, evals, _fens = generate_positions(
        num_positions=num_positions,
        depth=0,  # unused
        min_ply=min_ply,
        max_ply=max_ply,
        num_threads=1,
        eval_engine=f"search_d{search_depth}",
    )

    np.savez_compressed(shard_path, positions=positions, evals=evals)
    volume.commit()
    print(f"Shard {shard_id}: saved {len(positions)} search-d{search_depth} positions")
    return shard_id, len(positions)


@app.local_entrypoint()
def generate_search_data(
    num_positions: int = 5_000_000,
    search_depth: int = 1,
    dataset_name: str = "v4_search_d1",
    min_ply: int = 8,
    max_ply: int = 80,
):
    """Generate data using search-based eval (minimax + quiescence), then merge."""
    num_shards = (num_positions + BASELINE_SHARD_SIZE - 1) // BASELINE_SHARD_SIZE
    per_shard = BASELINE_SHARD_SIZE

    print(f"Generating {num_positions} search-d{search_depth} positions as {num_shards} shards of {per_shard}")

    shard_args = [
        (i, per_shard, dataset_name, search_depth, min_ply, max_ply)
        for i in range(num_shards)
    ]

    total = 0
    for result in generate_search_shard.starmap(shard_args):
        shard_id, count = result
        total += count
        if shard_id % 50 == 0:
            print(f"  Shard {shard_id} done: {count} positions (running total: {total})")

    print(f"\nAll shards complete. Merging...")
    final_count = merge_shards.remote(dataset_name, num_shards)
    print(f"Dataset ready: {final_count} positions in {dataset_name}")


# ── Opening-based data generation ────────────────────────────────────────────


@app.function(
    image=datagen_image,
    volumes={VOLUME_PATH: volume},
    cpu=1,
    memory=2048,
    timeout=3600,
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=5.0),
)
def generate_opening_shard(
    shard_id: int,
    num_positions: int,
    dataset_name: str,
    depth: int,
    quiet_only: bool = False,
):
    """Generate one shard of training data from opening continuations."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import numpy as np
    from src.opening_data_gen import generate_from_openings

    shard_dir = f"{VOLUME_PATH}/datasets/{dataset_name}/shards"
    shard_path = f"{shard_dir}/shard_{shard_id:04d}.npz"

    if os.path.exists(shard_path):
        existing = np.load(shard_path)
        n = existing["positions"].shape[0]
        print(f"Shard {shard_id} already exists ({n} positions), skipping.")
        return shard_id, n

    os.makedirs(shard_dir, exist_ok=True)

    positions, evals, _fens = generate_from_openings(
        num_positions=num_positions,
        openings_path="/app/data/openings.txt",
        stockfish_path="/usr/local/bin/stockfish",
        depth=depth,
        quiet_only=quiet_only,
    )

    np.savez_compressed(shard_path, positions=positions, evals=evals)
    volume.commit()
    print(f"Shard {shard_id}: saved {len(positions)} positions to {shard_path}")
    return shard_id, len(positions)


@app.local_entrypoint()
def generate_opening_data(
    num_positions: int = 2_000_000,
    depth: int = 10,
    dataset_name: str = "v3_openings_sf",
    quiet_only: bool = False,
):
    """Generate data from opening continuations, then merge."""
    num_shards = (num_positions + SHARD_SIZE - 1) // SHARD_SIZE
    per_shard = SHARD_SIZE

    quiet_str = " (quiet only)" if quiet_only else ""
    print(f"Generating {num_positions} opening positions{quiet_str} as {num_shards} shards of {per_shard}")

    shard_args = [
        (i, per_shard, dataset_name, depth, quiet_only)
        for i in range(num_shards)
    ]

    total = 0
    for result in generate_opening_shard.starmap(shard_args):
        shard_id, count = result
        total += count
        print(f"  Shard {shard_id} done: {count} positions (running total: {total})")

    print(f"\nAll shards complete. Merging...")
    final_count = merge_shards.remote(dataset_name, num_shards)
    print(f"Dataset ready: {final_count} positions in {dataset_name}")


# ── Lichess data generation: download PGN → extract positions → label ────────


@app.function(
    image=datagen_image,
    volumes={VOLUME_PATH: volume},
    cpu=2,
    memory=8192,
    timeout=7200,  # 2 hours for PGN download + extraction
)
def download_and_extract_lichess(
    pgn_url: str,
    dataset_name: str,
    max_positions: int,
    min_elo: int,
    sample_moves: str,
    seed: int,
):
    """Download Lichess PGN and extract positions (no Stockfish labeling yet)."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import subprocess
    import numpy as np
    from src.encoding import board_to_tensor_dual as board_to_tensor
    from src.lichess_data import extract_positions_from_pgn

    extract_dir = f"{VOLUME_PATH}/datasets/{dataset_name}"
    positions_path = f"{extract_dir}/extracted_positions.npz"

    # Skip if already done
    if os.path.exists(positions_path):
        data = np.load(positions_path)
        n = data["fens"].shape[0]
        print(f"Positions already extracted ({n}), skipping download.")
        return n

    os.makedirs(extract_dir, exist_ok=True)

    # Download PGN
    pgn_local = f"/tmp/lichess.pgn.zst"
    print(f"Downloading {pgn_url}...")
    subprocess.run(["curl", "-L", "-o", pgn_local, pgn_url], check=True)

    file_size = os.path.getsize(pgn_local) / (1024 * 1024)
    print(f"Downloaded {file_size:.0f} MB")

    # Extract positions
    moves_list = [int(x) for x in sample_moves.split(",")]
    positions = extract_positions_from_pgn(
        pgn_path=pgn_local,
        max_positions=max_positions,
        min_elo=min_elo,
        sample_moves=moves_list,
        seed=seed,
    )

    # Save positions as FENs (to be labeled by shards)
    fens = [b.fen() for b in positions]
    np.savez_compressed(positions_path, fens=np.array(fens))
    volume.commit()

    print(f"Extracted {len(fens)} positions, saved to {positions_path}")
    return len(fens)


LICHESS_LABEL_SHARD_SIZE = 10_000  # positions per labeling shard


@app.function(
    image=datagen_image,
    volumes={VOLUME_PATH: volume},
    cpu=1,
    memory=2048,
    timeout=3600,
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=5.0),
)
def label_lichess_shard(
    shard_id: int,
    dataset_name: str,
    start_idx: int,
    end_idx: int,
    depth: int,
):
    """Label a shard of extracted positions with Stockfish."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import chess
    import chess.engine
    import numpy as np
    from src.encoding import board_to_tensor_dual as board_to_tensor

    shard_dir = f"{VOLUME_PATH}/datasets/{dataset_name}/label_shards"
    shard_path = f"{shard_dir}/shard_{shard_id:04d}.npz"

    if os.path.exists(shard_path):
        existing = np.load(shard_path)
        n = existing["positions"].shape[0]
        print(f"Label shard {shard_id} already exists ({n} positions), skipping.")
        return shard_id, n

    os.makedirs(shard_dir, exist_ok=True)

    # Load extracted FENs
    volume.reload()
    positions_path = f"{VOLUME_PATH}/datasets/{dataset_name}/extracted_positions.npz"
    data = np.load(positions_path, allow_pickle=True)
    all_fens = data["fens"]
    fens = all_fens[start_idx:end_idx]

    # Label with Stockfish
    engine = chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish")
    engine.configure({"Threads": 1, "Hash": 16})

    encoded = []
    evals = []

    for fen_str in fens:
        fen = str(fen_str)
        try:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].pov(board.turn)
            cp = score.score(mate_score=15000)
            if cp is None:
                continue
            encoded.append(board_to_tensor(board))
            evals.append(float(cp))
        except Exception:
            continue

    engine.quit()

    if encoded:
        np.savez_compressed(
            shard_path,
            positions=np.array(encoded, dtype=np.float32),
            evals=np.array(evals, dtype=np.float32),
        )
        volume.commit()

    print(f"Label shard {shard_id}: labeled {len(encoded)} / {len(fens)} positions")
    return shard_id, len(encoded)


@app.local_entrypoint()
def generate_lichess_data(
    pgn_url: str = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst",
    dataset_name: str = "v3_lichess",
    max_positions: int = 10_000_000,
    min_elo: int = 1500,
    sample_moves: str = "10,15,20,25,30",
    depth: int = 12,
    seed: int = 42,
):
    """Download Lichess PGN, extract positions, label with SF, merge."""
    # Step 1: Download and extract positions
    print(f"Step 1: Downloading and extracting positions from Lichess...")
    num_positions = download_and_extract_lichess.remote(
        pgn_url, dataset_name, max_positions, min_elo, sample_moves, seed
    )
    print(f"Extracted {num_positions} positions")

    # Step 2: Fan out labeling
    num_shards = (num_positions + LICHESS_LABEL_SHARD_SIZE - 1) // LICHESS_LABEL_SHARD_SIZE
    print(f"\nStep 2: Labeling {num_positions} positions in {num_shards} shards (depth {depth})...")

    shard_args = []
    for i in range(num_shards):
        start = i * LICHESS_LABEL_SHARD_SIZE
        end = min(start + LICHESS_LABEL_SHARD_SIZE, num_positions)
        shard_args.append((i, dataset_name, start, end, depth))

    total = 0
    for result in label_lichess_shard.starmap(shard_args):
        shard_id, count = result
        total += count
        if shard_id % 50 == 0:
            print(f"  Shard {shard_id} done: {count} positions (running total: {total})")

    print(f"\nAll labeling done. Total labeled: {total}")

    # Step 3: Merge (reuse existing merge logic with label_shards)
    print("\nStep 3: Merging labeled shards...")
    final_count = merge_lichess_shards.remote(dataset_name, num_shards)
    print(f"Dataset ready: {final_count} positions in {dataset_name}")


@app.function(
    image=datagen_image,
    volumes={VOLUME_PATH: volume},
    cpu=2,
    memory=16384,
    timeout=1800,
)
def merge_lichess_shards(dataset_name: str, num_shards: int):
    """Merge labeled shards into final dataset."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import json
    import numpy as np

    shard_dir = f"{VOLUME_PATH}/datasets/{dataset_name}/label_shards"
    volume.reload()

    all_positions = []
    all_evals = []

    for i in range(num_shards):
        path = f"{shard_dir}/shard_{i:04d}.npz"
        if not os.path.exists(path):
            print(f"WARNING: label shard {i} missing at {path}")
            continue
        data = np.load(path)
        all_positions.append(data["positions"])
        all_evals.append(data["evals"])
        if i % 100 == 0:
            print(f"  Loaded shard {i}: {data['positions'].shape[0]} positions")

    positions = np.concatenate(all_positions, axis=0)
    evals = np.concatenate(all_evals, axis=0)

    output_dir = f"{VOLUME_PATH}/datasets/{dataset_name}"
    np.savez_compressed(f"{output_dir}/data.npz", positions=positions, evals=evals)

    metadata = {
        "num_positions": int(positions.shape[0]),
        "num_shards": num_shards,
        "source": "lichess",
        "eval_mean": float(evals.mean()),
        "eval_std": float(evals.std()),
        "eval_min": float(evals.min()),
        "eval_max": float(evals.max()),
    }
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    volume.commit()
    print(f"\nMerged {positions.shape[0]} positions into {output_dir}/data.npz")
    print(f"Eval range: [{evals.min():.0f}, {evals.max():.0f}] cp")
    print(f"Eval mean: {evals.mean():.1f}, std: {evals.std():.1f}")
    return int(positions.shape[0])


# ── Lichess baseline labeling (no Stockfish — uses Python baseline eval) ──────


@app.function(
    image=base_image,  # no Stockfish needed
    volumes={VOLUME_PATH: volume},
    cpu=1,
    memory=2048,
    timeout=3600,
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=5.0),
)
def label_lichess_baseline_shard(
    shard_id: int,
    dataset_name: str,
    start_idx: int,
    end_idx: int,
):
    """Label a shard of extracted Lichess positions with baseline eval (no SF)."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import chess
    import numpy as np
    from src.encoding import board_to_tensor_dual as board_to_tensor
    from src.baseline_eval import evaluate as baseline_evaluate

    shard_dir = f"{VOLUME_PATH}/datasets/{dataset_name}/bl_label_shards"
    shard_path = f"{shard_dir}/shard_{shard_id:04d}.npz"

    if os.path.exists(shard_path):
        existing = np.load(shard_path)
        n = existing["positions"].shape[0]
        print(f"Shard {shard_id} already exists ({n} positions), skipping.")
        return shard_id, n

    os.makedirs(shard_dir, exist_ok=True)

    volume.reload()
    positions_path = f"{VOLUME_PATH}/datasets/{dataset_name}/extracted_positions.npz"
    data = np.load(positions_path, allow_pickle=True)
    all_fens = data["fens"]
    fens = all_fens[start_idx:end_idx]

    encoded = []
    evals = []

    for fen_str in fens:
        fen = str(fen_str)
        try:
            board = chess.Board(fen)
            if board.is_game_over() or len(board.piece_map()) < 4:
                continue
            cp = float(baseline_evaluate(board))
            if abs(cp) > 10000:
                continue
            encoded.append(board_to_tensor(board))
            evals.append(cp)
        except Exception:
            continue

    if encoded:
        np.savez_compressed(
            shard_path,
            positions=np.array(encoded, dtype=np.float32),
            evals=np.array(evals, dtype=np.float32),
        )
        volume.commit()

    print(f"Shard {shard_id}: labeled {len(encoded)} / {len(fens)} with baseline eval")
    return shard_id, len(encoded)


@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    cpu=2,
    memory=16384,
    timeout=1800,
)
def merge_lichess_baseline_shards(dataset_name: str, num_shards: int):
    """Merge baseline-labeled Lichess shards into final dataset."""
    import os
    import json
    import numpy as np

    shard_dir = f"{VOLUME_PATH}/datasets/{dataset_name}/bl_label_shards"
    volume.reload()

    all_positions = []
    all_evals = []

    for i in range(num_shards):
        path = f"{shard_dir}/shard_{i:04d}.npz"
        if not os.path.exists(path):
            continue
        data = np.load(path)
        all_positions.append(data["positions"])
        all_evals.append(data["evals"])
        if i % 100 == 0:
            print(f"  Loaded shard {i}: {data['positions'].shape[0]} positions")

    positions = np.concatenate(all_positions, axis=0)
    evals = np.concatenate(all_evals, axis=0)

    output_dir = f"{VOLUME_PATH}/datasets/{dataset_name}"
    np.savez_compressed(f"{output_dir}/data_baseline.npz", positions=positions, evals=evals)

    metadata = {
        "num_positions": int(positions.shape[0]),
        "source": "lichess_baseline",
        "eval_mean": float(evals.mean()),
        "eval_std": float(evals.std()),
    }
    with open(f"{output_dir}/metadata_baseline.json", "w") as f:
        json.dump(metadata, f, indent=2)

    volume.commit()
    print(f"\nMerged {positions.shape[0]} baseline-labeled Lichess positions")
    return int(positions.shape[0])


@app.local_entrypoint()
def generate_lichess_baseline_data(
    pgn_url: str = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst",
    dataset_name: str = "v5_lichess_baseline",
    max_positions: int = 5_000_000,
    min_elo: int = 1500,
    sample_moves: str = "10,15,20,25,30,35,40",
    seed: int = 42,
):
    """Download Lichess PGN, extract positions, label with BASELINE eval, merge."""
    # Step 1: Download and extract positions (reuses existing function)
    print(f"Step 1: Downloading and extracting positions from Lichess...")
    num_positions = download_and_extract_lichess.remote(
        pgn_url, dataset_name, max_positions, min_elo, sample_moves, seed
    )
    print(f"Extracted {num_positions} positions")

    # Step 2: Fan out baseline labeling
    label_shard_size = 10_000
    num_shards = (num_positions + label_shard_size - 1) // label_shard_size
    print(f"\nStep 2: Labeling {num_positions} positions with baseline eval in {num_shards} shards...")

    shard_args = []
    for i in range(num_shards):
        start = i * label_shard_size
        end = min(start + label_shard_size, num_positions)
        shard_args.append((i, dataset_name, start, end))

    total = 0
    for result in label_lichess_baseline_shard.starmap(shard_args):
        shard_id, count = result
        total += count
        if shard_id % 50 == 0:
            print(f"  Shard {shard_id} done: {count} positions (total: {total})")

    # Step 3: Merge
    print("\nStep 3: Merging...")
    final_count = merge_lichess_baseline_shards.remote(dataset_name, num_shards)
    print(f"Dataset ready: {final_count} baseline-labeled Lichess positions in {dataset_name}")


# ── Move-pair data generation (for ranking loss) ────────────────────────────


@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    cpu=1,
    memory=2048,
    timeout=3600,
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=5.0),
)
def generate_move_pair_shard(
    shard_id: int,
    num_pairs: int,
    dataset_name: str,
    min_ply: int,
    max_ply: int,
):
    """Generate one shard of move-pair data for ranking loss."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import numpy as np
    from src.data_gen import generate_move_pairs

    shard_dir = f"{VOLUME_PATH}/datasets/{dataset_name}/shards"
    shard_path = f"{shard_dir}/shard_{shard_id:04d}.npz"

    if os.path.exists(shard_path):
        try:
            existing = np.load(shard_path)
            n = existing["pos_good"].shape[0]
            print(f"Shard {shard_id} already exists ({n} pairs), skipping.")
            return shard_id, n
        except Exception:
            print(f"Shard {shard_id} corrupted, regenerating...")
            os.remove(shard_path)

    os.makedirs(shard_dir, exist_ok=True)

    pos_good, pos_bad, margins = generate_move_pairs(
        num_pairs=num_pairs,
        min_ply=min_ply,
        max_ply=max_ply,
    )

    np.savez_compressed(shard_path, pos_good=pos_good, pos_bad=pos_bad, margins=margins)
    volume.commit()
    print(f"Shard {shard_id}: saved {len(pos_good)} pairs")
    return shard_id, len(pos_good)


@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    cpu=2,
    memory=32768,
    timeout=7200,
)
def merge_pair_shards(dataset_name: str, num_shards: int):
    """Merge move-pair shards into final dataset."""
    import os
    import json
    import numpy as np

    shard_dir = f"{VOLUME_PATH}/datasets/{dataset_name}/shards"
    volume.reload()

    all_good = []
    all_bad = []
    all_margins = []

    for i in range(num_shards):
        path = f"{shard_dir}/shard_{i:04d}.npz"
        if not os.path.exists(path):
            print(f"WARNING: shard {i} missing")
            continue
        data = np.load(path)
        all_good.append(data["pos_good"])
        all_bad.append(data["pos_bad"])
        all_margins.append(data["margins"])
        if i % 100 == 0:
            print(f"  Loaded shard {i}: {data['pos_good'].shape[0]} pairs")

    pos_good = np.concatenate(all_good, axis=0)
    pos_bad = np.concatenate(all_bad, axis=0)
    margins = np.concatenate(all_margins, axis=0)

    output_dir = f"{VOLUME_PATH}/datasets/{dataset_name}"
    np.savez_compressed(
        f"{output_dir}/data.npz",
        pos_good=pos_good, pos_bad=pos_bad, margins=margins,
    )

    metadata = {
        "num_pairs": int(pos_good.shape[0]),
        "margin_mean": float(margins.mean()),
        "margin_std": float(margins.std()),
    }
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    volume.commit()
    print(f"\nMerged {pos_good.shape[0]} move pairs")
    print(f"Margin: mean={margins.mean():.1f}, std={margins.std():.1f}")
    return int(pos_good.shape[0])


PAIR_SHARD_SIZE = 2_000  # pairs per shard (slower due to ~30 evals per position)


@app.local_entrypoint()
def generate_move_pair_data(
    num_pairs: int = 5_000_000,
    dataset_name: str = "v5_move_pairs",
    min_ply: int = 8,
    max_ply: int = 80,
):
    """Generate move-pair data for ranking loss, then merge."""
    num_shards = (num_pairs + PAIR_SHARD_SIZE - 1) // PAIR_SHARD_SIZE
    per_shard = PAIR_SHARD_SIZE

    print(f"Generating {num_pairs} move pairs as {num_shards} shards of {per_shard}")

    shard_args = [
        (i, per_shard, dataset_name, min_ply, max_ply)
        for i in range(num_shards)
    ]

    total = 0
    for result in generate_move_pair_shard.starmap(shard_args):
        shard_id, count = result
        total += count
        if shard_id % 100 == 0:
            print(f"  Shard {shard_id} done: {count} pairs (total: {total})")

    print(f"\nAll shards complete. Merging...")
    final_count = merge_pair_shards.remote(dataset_name, num_shards)
    print(f"Dataset ready: {final_count} move pairs in {dataset_name}")


# ── Self-play game-outcome data generation ───────────────────────────────────


@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    cpu=1,
    memory=4096,
    timeout=3600,
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=5.0),
)
def generate_selfplay_shard(
    shard_id: int,
    onnx_model_path: str,
    openings: list[str],
    games_per_opening: int,
    dataset_name: str,
):
    """Play games and record positions + outcomes."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import numpy as np
    from src.selfplay import play_games_batch

    shard_dir = f"{VOLUME_PATH}/datasets/{dataset_name}/shards"
    shard_path = f"{shard_dir}/shard_{shard_id:04d}.npz"

    if os.path.exists(shard_path):
        try:
            existing = np.load(shard_path)
            n = existing["positions"].shape[0]
            print(f"Shard {shard_id} already exists ({n} positions), skipping.")
            return shard_id, n
        except Exception:
            print(f"Shard {shard_id} corrupted, regenerating...")
            os.remove(shard_path)

    os.makedirs(shard_dir, exist_ok=True)

    # Download model from volume
    volume.reload()
    local_model = f"/tmp/model_shard_{shard_id}.onnx"
    import shutil
    shutil.copy(onnx_model_path, local_model)

    positions, evals, outcomes = play_games_batch(
        nn_onnx_path=local_model,
        openings=openings,
        games_per_opening=games_per_opening,
    )

    np.savez_compressed(shard_path, positions=positions, evals=evals, outcomes=outcomes)
    volume.commit()
    print(f"Shard {shard_id}: saved {len(positions)} positions from {len(openings) * games_per_opening} games")
    return shard_id, len(positions)


@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    cpu=2,
    memory=32768,
    timeout=3600,
)
def merge_selfplay_shards(dataset_name: str, num_shards: int):
    """Merge self-play shards into final dataset."""
    import os
    import json
    import numpy as np

    shard_dir = f"{VOLUME_PATH}/datasets/{dataset_name}/shards"
    volume.reload()

    all_positions = []
    all_evals = []
    all_outcomes = []

    for i in range(num_shards):
        path = f"{shard_dir}/shard_{i:04d}.npz"
        if not os.path.exists(path):
            print(f"WARNING: shard {i} missing")
            continue
        data = np.load(path)
        all_positions.append(data["positions"])
        if "evals" in data:
            all_evals.append(data["evals"])
        all_outcomes.append(data["outcomes"])
        if i % 10 == 0:
            print(f"  Loaded shard {i}: {data['positions'].shape[0]} positions")

    positions = np.concatenate(all_positions, axis=0)
    outcomes = np.concatenate(all_outcomes, axis=0)

    save_dict = {"positions": positions, "outcomes": outcomes}
    if all_evals:
        evals = np.concatenate(all_evals, axis=0)
        save_dict["evals"] = evals

    output_dir = f"{VOLUME_PATH}/datasets/{dataset_name}"
    np.savez_compressed(f"{output_dir}/data.npz", **save_dict)

    wins = (outcomes > 0.7).sum()
    losses = (outcomes < 0.3).sum()
    draws = ((outcomes >= 0.3) & (outcomes <= 0.7)).sum()

    metadata = {
        "num_positions": int(positions.shape[0]),
        "num_shards": num_shards,
        "outcome_wins": int(wins),
        "outcome_losses": int(losses),
        "outcome_draws": int(draws),
        "outcome_mean": float(outcomes.mean()),
    }
    if all_evals:
        metadata["eval_mean"] = float(evals.mean())
        metadata["eval_std"] = float(evals.std())
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    volume.commit()
    print(f"\nMerged {positions.shape[0]} positions")
    print(f"Outcomes: {wins} wins, {draws} draws, {losses} losses (mean={outcomes.mean():.3f})")
    return int(positions.shape[0])


@app.local_entrypoint()
def generate_selfplay_data(
    model_experiment: str = "level1_v4_distill_long",
    dataset_name: str = "v6_selfplay",
    num_shards: int = 100,
    games_per_shard: int = 200,
):
    """Generate self-play data: NN vs baseline from competition openings."""
    import os

    # Load openings
    openings_path = "/app/data/openings.txt" if os.path.exists("/app/data/openings.txt") else "../data/openings.txt"
    with open(openings_path) as f:
        all_openings = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    print(f"Loaded {len(all_openings)} openings")

    # Model path on volume
    onnx_path = f"{VOLUME_PATH}/models/{model_experiment}/model.onnx"

    # Each shard plays games_per_shard games across all openings
    games_per_opening = max(1, games_per_shard // len(all_openings))
    actual_games = games_per_opening * len(all_openings)
    print(f"Each shard: {actual_games} games ({games_per_opening} per opening)")
    print(f"Total: {num_shards} shards × {actual_games} games = {num_shards * actual_games} games")

    shard_args = [
        (i, onnx_path, all_openings, games_per_opening, dataset_name)
        for i in range(num_shards)
    ]

    total = 0
    for result in generate_selfplay_shard.starmap(shard_args):
        shard_id, count = result
        total += count
        if shard_id % 10 == 0:
            print(f"  Shard {shard_id} done: {count} positions (total: {total})")

    print(f"\nAll shards complete. Merging...")
    final_count = merge_selfplay_shards.remote(dataset_name, num_shards)
    print(f"Dataset ready: {final_count} self-play positions in {dataset_name}")


# ── Dataset merging ──────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    cpu=2,
    memory=32768,
    timeout=1800,
)
def merge_datasets_remote(
    source_datasets: list[str],
    output_dataset: str,
):
    """Merge multiple datasets into one."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import json
    import numpy as np

    volume.reload()

    all_positions = []
    all_evals = []

    for ds_name in source_datasets:
        path = f"{VOLUME_PATH}/datasets/{ds_name}/data.npz"
        data = np.load(path)
        print(f"  {ds_name}: {data['positions'].shape[0]:,} positions")
        all_positions.append(data["positions"])
        all_evals.append(data["evals"])

    positions = np.concatenate(all_positions, axis=0)
    evals = np.concatenate(all_evals, axis=0)

    # Shuffle
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(positions))
    positions = positions[indices]
    evals = evals[indices]

    output_dir = f"{VOLUME_PATH}/datasets/{output_dataset}"
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(f"{output_dir}/data.npz", positions=positions, evals=evals)

    metadata = {
        "num_positions": int(positions.shape[0]),
        "sources": source_datasets,
        "eval_mean": float(evals.mean()),
        "eval_std": float(evals.std()),
        "eval_min": float(evals.min()),
        "eval_max": float(evals.max()),
    }
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    volume.commit()
    print(f"\nMerged {positions.shape[0]:,} positions into {output_dir}/data.npz")
    print(f"Eval range: [{evals.min():.0f}, {evals.max():.0f}] cp")
    return int(positions.shape[0])


@app.local_entrypoint()
def merge_datasets_cmd(
    sources: str = "v3_random_sf,v3_openings_sf",
    output: str = "v3_combined_sf",
):
    """Merge datasets. Sources is comma-separated list of dataset names."""
    source_list = [s.strip() for s in sources.split(",")]
    print(f"Merging datasets: {source_list} → {output}")
    count = merge_datasets_remote.remote(source_list, output)
    print(f"Done: {count:,} positions")


# ── Training ─────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=4 * 3600,
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=10.0),
)
def train_model(
    config_path: str = "configs/level1_baseline.yaml",
    resume: bool = True,
):
    """Train the neural network on GPU."""
    import sys
    sys.path.insert(0, "/app")

    import yaml
    from src.train import train

    with open(f"/app/{config_path}") as f:
        config = yaml.safe_load(f)

    dataset_name = config.get("dataset_name", "v1")
    experiment_name = config.get("experiment_name", "default")

    config["data_path"] = f"{VOLUME_PATH}/datasets/{dataset_name}/data.npz"
    config["checkpoint_dir"] = f"{VOLUME_PATH}/checkpoints/{experiment_name}"
    config["resume"] = resume

    # For fine-tuning: resolve resume_from experiment name to best checkpoint path
    if config.get("resume_from_experiment"):
        src_exp = config["resume_from_experiment"]
        best_path = f"{VOLUME_PATH}/checkpoints/{src_exp}/best.pt"
        config["resume_from"] = best_path
        print(f"Fine-tuning from {best_path}")

    # Ranking loss: resolve pair dataset name to path
    if config.get("pair_dataset_name"):
        config["pair_data_path"] = f"{VOLUME_PATH}/datasets/{config['pair_dataset_name']}/data.npz"

    model, ckpt_mgr = train(config)
    volume.commit()
    print(f"Training complete. Checkpoints at {config['checkpoint_dir']}")


# ── Export ───────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    cpu=2,
    timeout=600,
)
def export_model(
    experiment_name: str = "level1_baseline_v1",
    architecture: str = "mlp_512_256_128",
):
    """Export best checkpoint to ONNX and validate."""
    import sys
    sys.path.insert(0, "/app")
    import os
    import torch
    from src.export_onnx import export_onnx, validate_onnx
    from src.checkpoint import CheckpointManager
    from src.model import build_model

    checkpoint_dir = f"{VOLUME_PATH}/checkpoints/{experiment_name}"
    output_dir = f"{VOLUME_PATH}/models/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)

    mgr = CheckpointManager(checkpoint_dir)
    ckpt_path = mgr.best_checkpoint() or mgr.latest_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = build_model(architecture)
    model.load_state_dict(state["model_state_dict"])

    output_path = f"{output_dir}/model.onnx"
    export_onnx(model, output_path)

    valid = validate_onnx(output_path)
    if not valid:
        raise RuntimeError("ONNX validation failed!")

    volume.commit()
    print(f"Model exported to {output_path}")


@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    cpu=2,
    timeout=600,
)
def export_ensemble(
    experiment_names: list[str],
    architectures: list[str],
    output_name: str = "ensemble_v5",
):
    """Export an ensemble of trained models as a single ONNX file."""
    import sys
    sys.path.insert(0, "/app")
    import os
    from src.export_onnx import export_ensemble_onnx
    from src.checkpoint import CheckpointManager

    checkpoint_paths = []
    for exp_name in experiment_names:
        ckpt_dir = f"{VOLUME_PATH}/checkpoints/{exp_name}"
        mgr = CheckpointManager(ckpt_dir)
        ckpt_path = mgr.best_checkpoint() or mgr.latest_checkpoint()
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
        checkpoint_paths.append(str(ckpt_path))
        print(f"  {exp_name}: {ckpt_path}")

    output_dir = f"{VOLUME_PATH}/models/{output_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/model.onnx"

    export_ensemble_onnx(checkpoint_paths, architectures, output_path)

    volume.commit()
    print(f"Ensemble exported to {output_path}")


@app.local_entrypoint()
def export_ensemble_cmd(
    output_name: str = "ensemble_v5",
):
    """Export the 3-member ensemble as a single ONNX file."""
    experiment_names = ["level1_v5_ens_a", "level1_v5_ens_b", "level1_v5_ens_c"]
    architectures = ["ens_a", "ens_b_relu", "ens_c_relu"]
    export_ensemble.remote(experiment_names, architectures, output_name)
