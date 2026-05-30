import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from gptff.model.mpredict import ASECalculator, _pair_and_triple_features, collate_fn


def parse_repeat(text):
    values = [int(x) for x in text.lower().replace(",", "x").split("x")]
    if len(values) != 3 or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("repeat must be formatted like 6x6x2")
    return tuple(values)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def memory_snapshot(prefix):
    if not torch.cuda.is_available():
        return {
            f"{prefix}_allocated_bytes": 0,
            f"{prefix}_reserved_bytes": 0,
            f"{prefix}_peak_allocated_bytes": 0,
            f"{prefix}_peak_reserved_bytes": 0,
        }
    sync()
    return {
        f"{prefix}_allocated_bytes": int(torch.cuda.memory_allocated()),
        f"{prefix}_reserved_bytes": int(torch.cuda.memory_reserved()),
        f"{prefix}_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        f"{prefix}_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def timed(section, timings, fn):
    sync()
    t0 = time.perf_counter()
    value = fn()
    sync()
    timings[f"{section}_s"] = time.perf_counter() - t0
    timings.update(memory_snapshot(section))
    return value


def profile_once(atoms, calc, mode):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    timings = {"mode": mode, "atoms": len(atoms), "use_checkpoint": calc.use_checkpoint}

    data = timed("graph", timings, lambda: calc.graph.transform(atoms))
    data = timed("collate", timings, lambda: collate_fn(data))
    data = timed("to_device", timings, lambda: [x.to(calc.device) for x in data])

    atom_fea, coords, _d_ij, offsets, lattice, n_atoms, pairs_count, nbr_atoms, bond_pairs_indices, _n_bond_pairs_struc, _n_bond_pairs_atom, n_bond_pairs_bond, ref_energy = data
    coords = coords.requires_grad_(True)

    if mode == "stress":
        strain = torch.zeros_like(lattice, dtype=torch.float32).requires_grad_(True)
    else:
        strain = None

    features = timed(
        "features",
        timings,
        lambda: _pair_and_triple_features(
            coords,
            offsets,
            lattice,
            n_atoms,
            pairs_count,
            nbr_atoms,
            bond_pairs_indices,
            calc.device,
            strain,
        ),
    )
    _coords_out, lattices, pair_dist_ij, triple_dist_ij, triple_dist_ik, triple_a_jik = features

    energy = timed(
        "forward",
        timings,
        lambda: calc._model_energy(
            atom_fea,
            pair_dist_ij,
            n_atoms,
            triple_dist_ij,
            triple_dist_ik,
            triple_a_jik,
            nbr_atoms,
            n_bond_pairs_bond,
            bond_pairs_indices,
            ref_energy,
        ),
    )

    if mode == "stress":
        volumes = torch.linalg.det(lattices)

        def backward():
            forces, stress = torch.autograd.grad(energy, [coords, strain], torch.ones_like(energy), retain_graph=False, create_graph=False)
            return -forces, stress / volumes[:, None, None] * 160.21766208

    else:

        def backward():
            forces = torch.autograd.grad(energy, coords, torch.ones_like(energy), retain_graph=False, create_graph=False)[0]
            return -forces

    result = timed("backward", timings, backward)
    timings.update(memory_snapshot("final"))
    timings["energy"] = float(energy.detach().cpu().numpy().ravel()[0])
    if mode == "stress":
        forces, stress = result
        timings["force_abs_max"] = float(forces.detach().abs().max().cpu())
        timings["stress_abs_max"] = float(stress.detach().abs().max().cpu())
    else:
        timings["force_abs_max"] = float(result.detach().abs().max().cpu())
    return timings


def main():
    parser = argparse.ArgumentParser(description="Profile GPTFF force/stress backward phases.")
    parser.add_argument("--structure", required=True)
    parser.add_argument("--checkpoint", default="pretrained/gptff_v1.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeat", type=parse_repeat, default=(1, 1, 1))
    parser.add_argument("--mode", choices=["forces", "stress"], default="forces")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    atoms = read(args.structure) * args.repeat
    calc = ASECalculator(args.checkpoint, device=args.device, use_checkpoint=args.use_checkpoint)

    for idx in range(args.warmup):
        profile_once(atoms.copy(), calc, args.mode)

    rows = []
    for idx in range(args.steps):
        row = profile_once(atoms.copy(), calc, args.mode)
        row["iteration"] = idx
        rows.append(row)
        print(json.dumps(row), flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": rows}, indent=2))
    with out_path.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
