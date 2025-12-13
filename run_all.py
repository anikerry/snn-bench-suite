"""Run multiple SNN benchmarks and export canonical JSON models.

Usage:
    python run_all.py --bench xor N1 S1 S2 E1 A3 ...

All available benchmark IDs are listed in registry.json.
"""

import argparse
import importlib
from pathlib import Path
from typing import List

import torch

from common.train_manager import Trainer
from common.canonical_export import export_canonical
import json

def load_registry(path: str = "registry.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_benchmark(bench_id: str, registry: dict) -> None:
    if bench_id not in registry:
        raise ValueError(f"Unknown benchmark id '{bench_id}'. Check registry.json for valid IDs.")

    entry = registry[bench_id]
    module_base = entry["path"]

    cfg_mod = importlib.import_module(f"{module_base}.config")
    model_mod = importlib.import_module(f"{module_base}.model")
    gen_mod = importlib.import_module(f"{module_base}.generator")

    cfg = cfg_mod.get_config()
    ModelClass = getattr(model_mod, entry["model_class"])

    # Select generator: prefer `generate`, then `generate_data`, then task-specific
    gen_fn = None
    if hasattr(gen_mod, "generate" ):
        gen_fn = getattr(gen_mod, "generate")
    elif hasattr(gen_mod, "generate_data"):
        gen_fn = getattr(gen_mod, "generate_data")
    elif hasattr(gen_mod, "generate_xor_data"):
        gen_fn = getattr(gen_mod, "generate_xor_data")
    else:
        raise RuntimeError(f"No suitable generator function found for benchmark '{bench_id}'.")

    # Call generator, trying with timesteps if supported
    try:
        X_train, y_train, X_test, y_test = gen_fn(timesteps=cfg.timesteps)
    except TypeError:
        X_train, y_train, X_test, y_test = gen_fn()

    model = ModelClass(timesteps=cfg.timesteps)
    trainer = Trainer(model, cfg)

    print(f"\n=== Running benchmark {bench_id} ({entry['series']}) ===")
    trainer.train(X_train, y_train)
    loss, acc = trainer.evaluate(X_test, y_test)
    print(f"[{bench_id}] Final test loss: {loss:.4f}, accuracy: {acc:.3f}")

    out_dir = Path("artifacts") / bench_id
    out_dir.mkdir(parents=True, exist_ok=True)
    export_canonical(model, cfg, str(out_dir / "canonical_model.json"))

def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Run SNN benchmark suite.")
    parser.add_argument(
        "--bench",
        nargs="+",
        required=True,
        help="Benchmark IDs to run. See registry.json for all options.",
    )
    args = parser.parse_args(argv)

    registry = load_registry()

    for bench_id in args.bench:
        run_benchmark(bench_id, registry)

if __name__ == "__main__":
    main()
