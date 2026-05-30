import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from ase.io import read

from gptff.model.mpredict import ASECalculator


def parse_repeat(text):
    values = [int(x) for x in text.lower().replace(",", "x").split("x")]
    if len(values) != 3 or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("repeat must be formatted like 4x4x2")
    return tuple(values)


def cuda_snapshot():
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "cuda_allocated_bytes": 0,
            "cuda_reserved_bytes": 0,
            "cuda_peak_allocated_bytes": 0,
            "cuda_peak_reserved_bytes": 0,
        }
    torch.cuda.synchronize()
    return {
        "cuda_available": True,
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def run_mode(atoms, calc, mode, warmup, steps):
    atoms.calc = calc
    proc = psutil.Process()
    vm = psutil.virtual_memory()
    rss_peak = proc.memory_info().rss
    system_available_min = vm.available

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    durations = []
    for idx in range(warmup + steps):
        atoms.positions[0, 0] += 1.0e-5 if idx % 2 == 0 else -1.0e-5
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if mode == "energy":
            value = atoms.get_potential_energy()
        elif mode == "forces":
            value = atoms.get_forces()
        elif mode == "stress":
            value = atoms.get_stress(voigt=False)
        else:
            raise ValueError(f"unknown mode: {mode}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        rss_peak = max(rss_peak, proc.memory_info().rss)
        system_available_min = min(system_available_min, psutil.virtual_memory().available)
        if idx >= warmup:
            durations.append(dt)

    value_array = np.asarray(value)
    durations = np.asarray(durations, dtype=np.float64)
    result = {
        "mode": mode,
        "atoms": len(atoms),
        "warmup": warmup,
        "steps": steps,
        "total_s": float(durations.sum()),
        "mean_s": float(durations.mean()),
        "median_s": float(np.median(durations)),
        "std_s": float(durations.std()),
        "steps_per_s": float(1.0 / durations.mean()),
        "atom_steps_per_s": float(len(atoms) / durations.mean()),
        "rss_peak_bytes": int(rss_peak),
        "system_available_min_bytes": int(system_available_min),
        "value_shape": list(value_array.shape),
    }
    result.update(cuda_snapshot())
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPTFF ASECalculator call paths.")
    parser.add_argument("--structure", required=True, help="Input structure readable by ASE.")
    parser.add_argument("--checkpoint", default="pretrained/gptff_v1.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeat", type=parse_repeat, default=(1, 1, 1))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--modes", default="energy,forces,stress")
    parser.add_argument("--use-checkpoint", action="store_true", help="Compatibility flag for whole-model checkpointing.")
    parser.add_argument("--checkpoint-mode", choices=["none", "model", "layer"], default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    atoms = read(args.structure) * args.repeat
    calc = ASECalculator(args.checkpoint, device=args.device, use_checkpoint=args.use_checkpoint, checkpoint_mode=args.checkpoint_mode)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    metadata = {
        "structure": str(Path(args.structure).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": args.device,
        "repeat": list(args.repeat),
        "atoms": len(atoms),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "use_checkpoint": args.use_checkpoint,
        "checkpoint_mode": calc.checkpoint_mode,
    }

    rows = []
    for mode in modes:
        mode_atoms = atoms.copy()
        row = run_mode(mode_atoms, calc, mode, args.warmup, args.steps)
        row.update(metadata)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    with out_path.open("w") as f:
        json.dump({"metadata": metadata, "results": rows}, f, indent=2)

    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
