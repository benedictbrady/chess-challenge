"""Run cargo compete locally and parse results.

Usage:
    python eval_local.py model.onnx --level 1
    python eval_local.py --download <experiment_name> --level 1
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


def download_from_modal(experiment_name: str, output_path: str):
    """Download model from Modal volume."""
    remote_path = f"models/{experiment_name}/model.onnx"
    cmd = [
        "modal", "volume", "get",
        "chess-training-data",
        remote_path,
        output_path,
    ]
    print(f"Downloading: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Download failed: {result.stderr}")
        sys.exit(1)
    print(f"Downloaded to {output_path}")


def run_compete(model_path: str, level: int | None = None) -> dict | None:
    """Run cargo compete and parse results."""
    # Build the command
    cmd = [
        "cargo", "run", "-p", "cli", "--bin", "compete", "--release",
        "--", model_path,
    ]

    if level is not None:
        cmd.extend(["--level", str(level)])

    # JSON output for machine parsing
    json_path = tempfile.mktemp(suffix=".json")
    cmd.extend(["--json-output", json_path])

    print(f"Running: {' '.join(cmd)}")
    print()

    # Run from project root (one level up from training/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(cmd, cwd=project_root)

    # Parse JSON results
    results = None
    if os.path.exists(json_path):
        with open(json_path) as f:
            results = json.load(f)
        os.unlink(json_path)

    if results:
        print("\n--- Results Summary ---")
        print(f"Parameters: {results['param_count']:,}")
        print(f"Best level: {results.get('best_level', 'none')}")
        for level_result in results.get("levels", []):
            status = "PASS" if level_result["passed"] else "FAIL"
            print(
                f"  Level {level_result['level']}: "
                f"{level_result['score']:.1f}/50 ({level_result['score_pct']:.0f}%) "
                f"W:{level_result['wins']} D:{level_result['draws']} L:{level_result['losses']} "
                f"[{status}]"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate chess model locally")
    parser.add_argument("model", nargs="?", help="Path to ONNX model file")
    parser.add_argument("--download", type=str, help="Download model from Modal experiment")
    parser.add_argument("--level", type=int, help="Run only this level (1-5)")
    args = parser.parse_args()

    model_path = args.model

    if args.download:
        model_path = f"models/{args.download}.onnx"
        os.makedirs("models", exist_ok=True)
        download_from_modal(args.download, model_path)
    elif model_path is None:
        parser.error("Must provide model path or --download <experiment_name>")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    run_compete(model_path, level=args.level)


if __name__ == "__main__":
    main()
