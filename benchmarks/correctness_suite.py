import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase import Atoms, units
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import FIRE
from pymatgen.core import Lattice, Structure

from gptff.model.mpredict import ASECalculator


def parse_repeat(text):
    values = [int(x) for x in text.lower().replace(",", "x").split("x")]
    if len(values) != 3 or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("repeat must be formatted like 2x2x1")
    return tuple(values)


def sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def metric(status, name, **kwargs):
    row = {"name": name, "status": status}
    row.update(kwargs)
    return row


def calculator(checkpoint, device):
    return ASECalculator(str(checkpoint), device=device)


def run_efs(atoms, checkpoint, device):
    rows = []

    atoms_e = atoms.copy()
    atoms_e.calc = calculator(checkpoint, device)
    energy = atoms_e.get_potential_energy()

    atoms_f = atoms.copy()
    atoms_f.calc = calculator(checkpoint, device)
    forces = atoms_f.get_forces()
    energy_f = atoms_f.calc.results["energy"]

    atoms_s = atoms.copy()
    atoms_s.calc = calculator(checkpoint, device)
    stress = atoms_s.get_stress(voigt=False)
    energy_s = atoms_s.calc.results["energy"]
    forces_s = atoms_s.calc.results["forces"]

    ok = (
        np.isfinite(energy)
        and np.all(np.isfinite(forces))
        and np.all(np.isfinite(stress))
        and np.isclose(energy, energy_f, rtol=5.0e-5, atol=5.0e-5)
        and np.isclose(energy, energy_s, rtol=5.0e-5, atol=5.0e-5)
        and np.allclose(forces, forces_s, rtol=5.0e-4, atol=5.0e-4)
    )
    rows.append(
        metric(
            "passed" if ok else "failed",
            "efs_consistency",
            energy=float(energy),
            force_abs_max=float(np.abs(forces).max()),
            stress_abs_max=float(np.abs(stress).max()),
            energy_force_delta=float(abs(energy - energy_f)),
            energy_stress_delta=float(abs(energy - energy_s)),
            force_stress_max_delta=float(np.abs(forces - forces_s).max()),
        )
    )
    return rows


def run_finite_difference(atoms, checkpoint, device):
    h = 1.0e-3
    atoms_f = atoms.copy()
    atoms_f.calc = calculator(checkpoint, device)
    force = float(atoms_f.get_forces()[0, 0])

    atoms_p = atoms.copy()
    atoms_p.positions[0, 0] += h
    atoms_p.calc = calculator(checkpoint, device)
    e_plus = atoms_p.get_potential_energy()

    atoms_m = atoms.copy()
    atoms_m.positions[0, 0] -= h
    atoms_m.calc = calculator(checkpoint, device)
    e_minus = atoms_m.get_potential_energy()

    fd_force = float(-(e_plus - e_minus) / (2.0 * h))
    ok = np.isclose(force, fd_force, rtol=5.0e-2, atol=5.0e-2)
    return [
        metric(
            "passed" if ok else "failed",
            "finite_difference_force",
            force=force,
            finite_difference_force=fd_force,
            abs_delta=float(abs(force - fd_force)),
        )
    ]


def run_fixed_cell_optimization(atoms, checkpoint, device, steps):
    atoms = atoms.copy()
    atoms.calc = calculator(checkpoint, device)
    initial_fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    t0 = time.perf_counter()
    FIRE(atoms, logfile=None).run(fmax=0.01, steps=steps)
    dt = time.perf_counter() - t0
    final_forces = atoms.get_forces()
    final_fmax = float(np.linalg.norm(final_forces, axis=1).max())
    ok = np.all(np.isfinite(atoms.positions)) and np.all(np.isfinite(final_forces))
    return [
        metric(
            "passed" if ok else "failed",
            "fixed_cell_optimization",
            steps=steps,
            seconds=dt,
            initial_fmax=initial_fmax,
            final_fmax=final_fmax,
        )
    ]


def run_cell_optimization(atoms, checkpoint, device, steps):
    atoms = atoms.copy()
    atoms.calc = calculator(checkpoint, device)
    t0 = time.perf_counter()
    filtered = FrechetCellFilter(atoms)
    FIRE(filtered, logfile=None).run(fmax=0.01, steps=steps)
    dt = time.perf_counter() - t0
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=False)
    ok = np.all(np.isfinite(atoms.positions)) and np.all(np.isfinite(forces)) and np.all(np.isfinite(stress))
    return [
        metric(
            "passed" if ok else "failed",
            "cell_optimization_smoke",
            steps=steps,
            seconds=dt,
            volume=float(atoms.get_volume()),
            stress_abs_max=float(np.abs(stress).max()),
        )
    ]


def run_md(atoms, checkpoint, device, steps):
    atoms = atoms.copy()
    atoms.calc = calculator(checkpoint, device)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, force_temp=True)
    dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs, logfile=None)
    t0 = time.perf_counter()
    dyn.run(steps)
    sync(device)
    dt = time.perf_counter() - t0
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    ok = np.isfinite(energy) and np.all(np.isfinite(atoms.positions)) and np.all(np.isfinite(forces))
    return [
        metric(
            "passed" if ok else "failed",
            "short_md_nve",
            steps=steps,
            seconds=dt,
            steps_per_s=float(steps / dt) if dt > 0 else None,
            energy=float(energy),
            force_abs_max=float(np.abs(forces).max()),
        )
    ]


def write_training_files(workdir, device):
    structure = Structure(
        Lattice.cubic(4.8),
        ["Li", "O", "O"],
        [[0.0, 0.0, 0.0], [0.45, 0.45, 0.0], [0.45, 0.0, 0.45]],
    )
    rows = []
    for idx, fold in enumerate([-1, -1, 0]):
        rows.append(
            {
                "struct_id": idx,
                "energy": 0.0,
                "forces": str(np.zeros((len(structure), 3)).tolist()),
                "stress": str(np.zeros((3, 3)).tolist()),
                "structure": str(structure.as_dict()),
                "fold": fold,
                "ref_energy": 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(workdir / "smoke.csv", index=False)
    config = {
        "training": {
            "workers": 0,
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "node_feature_len": 8,
            "edge_feature_len": 8,
            "n_layers": 1,
            "n_readout_layers": 1,
            "warmup_steps": 0,
            "device": device,
            "val_fold": 0,
            "resume": False,
            "transformer_activate": False,
            "start_epoch": 0,
            "weight_energy": 0.0,
            "weight_force": 0.0,
            "weight_stress": 0.0,
        },
        "data": {"data_path": str(workdir), "data_file": "smoke.csv"},
    }
    config_path = workdir / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path


def run_training_smoke(device, timeout):
    with tempfile.TemporaryDirectory(prefix="gptff_training_smoke_") as tmp:
        workdir = Path(tmp)
        config = write_training_files(workdir, device)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "gptff.trainer.trainer", str(config)],
            cwd=workdir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        dt = time.perf_counter() - t0
        ok = proc.returncode == 0 and (workdir / "curr_checkpoint.pth").exists() and (workdir / "best_checkpoint.pth").exists()
        return [
            metric(
                "passed" if ok else "failed",
                "training_smoke",
                seconds=dt,
                returncode=proc.returncode,
                stdout_tail=proc.stdout[-1000:],
                stderr_tail=proc.stderr[-1000:],
            )
        ]


def main():
    parser = argparse.ArgumentParser(description="Run GPTFF correctness workflow suite.")
    parser.add_argument("--structure", default="notebooks/data/NaCl.cif")
    parser.add_argument("--checkpoint", default="pretrained/gptff_v1.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--training-device", default="cpu")
    parser.add_argument("--repeat", type=parse_repeat, default=(1, 1, 1))
    parser.add_argument("--opt-steps", type=int, default=3)
    parser.add_argument("--cell-opt-steps", type=int, default=1)
    parser.add_argument("--md-steps", type=int, default=5)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--training-timeout", type=int, default=180)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    structure_path = Path(args.structure)
    checkpoint = Path(args.checkpoint)
    atoms = read(structure_path) * args.repeat

    rows = []
    for fn in [
        run_efs,
        run_finite_difference,
        lambda a, c, d: run_fixed_cell_optimization(a, c, d, args.opt_steps),
        lambda a, c, d: run_cell_optimization(a, c, d, args.cell_opt_steps),
        lambda a, c, d: run_md(a, c, d, args.md_steps),
    ]:
        rows.extend(fn(atoms, checkpoint, args.device))

    if not args.skip_training:
        rows.extend(run_training_smoke(args.training_device, args.training_timeout))

    failed = [row for row in rows if row["status"] != "passed"]
    payload = {
        "metadata": {
            "structure": str(structure_path.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "device": args.device,
            "training_device": args.training_device,
            "repeat": list(args.repeat),
            "atoms": len(atoms),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "results": rows,
        "failed": failed,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, ensure_ascii=False), flush=True)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
