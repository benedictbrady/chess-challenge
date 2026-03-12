"""Merge multiple datasets into one combined dataset."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse


def merge_datasets(paths: list[str], output_path: str):
    all_positions = []
    all_evals = []

    for path in paths:
        data = np.load(path)
        positions = data["positions"]
        evals = data["evals"]
        print(f"  {path}: {positions.shape[0]:,} positions, eval range [{evals.min():.0f}, {evals.max():.0f}]")
        all_positions.append(positions)
        all_evals.append(evals)

    positions = np.concatenate(all_positions, axis=0)
    evals = np.concatenate(all_evals, axis=0)

    # Shuffle
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(positions))
    positions = positions[indices]
    evals = evals[indices]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, positions=positions, evals=evals)

    print(f"\nMerged: {positions.shape[0]:,} positions")
    print(f"Eval range: [{evals.min():.0f}, {evals.max():.0f}]")
    print(f"Eval mean: {evals.mean():.1f}, std: {evals.std():.1f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merge_datasets(args.inputs, args.output)
