import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import FIRE
from pymatgen.core import Lattice, Structure

from gptff.model.mpredict import ASECalculator


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "pretrained" / "gptff_v1.pth"


def make_atoms():
    return Atoms(
        "Li2O",
        positions=[[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [1.1, 1.6, 0.0]],
        cell=[12.0, 12.0, 12.0],
        pbc=True,
    )


def attach_calc(atoms):
    atoms.calc = ASECalculator(str(V1), device="cpu")
    return atoms


def test_energy_force_stress_paths_are_consistent():
    atoms_e = attach_calc(make_atoms())
    energy_only = atoms_e.get_potential_energy()

    atoms_f = attach_calc(make_atoms())
    forces = atoms_f.get_forces()
    energy_with_forces = atoms_f.calc.results["energy"]

    atoms_s = attach_calc(make_atoms())
    stress = atoms_s.get_stress(voigt=False)
    energy_with_stress = atoms_s.calc.results["energy"]
    forces_with_stress = atoms_s.calc.results["forces"]

    assert np.isfinite(energy_only)
    assert np.all(np.isfinite(forces))
    assert np.all(np.isfinite(stress))
    assert np.isclose(energy_only, energy_with_forces, rtol=1.0e-5, atol=1.0e-5)
    assert np.isclose(energy_only, energy_with_stress, rtol=1.0e-5, atol=1.0e-5)
    assert np.allclose(forces, forces_with_stress, rtol=1.0e-4, atol=1.0e-4)


def test_force_matches_finite_difference_direction():
    h = 1.0e-3
    atoms = attach_calc(make_atoms())
    force = atoms.get_forces()[0, 0]

    atoms_plus = attach_calc(make_atoms())
    atoms_plus.positions[0, 0] += h
    e_plus = atoms_plus.get_potential_energy()

    atoms_minus = attach_calc(make_atoms())
    atoms_minus.positions[0, 0] -= h
    e_minus = atoms_minus.get_potential_energy()

    fd_force = -(e_plus - e_minus) / (2.0 * h)
    assert np.isfinite(fd_force)
    assert np.isclose(force, fd_force, rtol=5.0e-2, atol=5.0e-2)


def test_fixed_cell_optimization_smoke():
    atoms = attach_calc(make_atoms())
    initial_fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    opt = FIRE(atoms, logfile=None)
    opt.run(fmax=0.01, steps=3)
    final_forces = atoms.get_forces()

    assert np.all(np.isfinite(atoms.positions))
    assert np.all(np.isfinite(final_forces))
    assert float(np.linalg.norm(final_forces, axis=1).max()) < max(initial_fmax * 10.0, 100.0)


def test_short_md_smoke():
    atoms = attach_calc(make_atoms())
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, force_temp=True)
    dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs, logfile=None)
    dyn.run(3)

    assert np.all(np.isfinite(atoms.positions))
    assert np.all(np.isfinite(atoms.get_forces()))
    assert np.isfinite(atoms.get_potential_energy())


def write_training_smoke_files(tmp_path):
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
    pd.DataFrame(rows).to_csv(tmp_path / "smoke.csv", index=False)
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
            "device": "cpu",
            "val_fold": 0,
            "resume": False,
            "transformer_activate": False,
            "start_epoch": 0,
            "weight_energy": 0.0,
            "weight_force": 0.0,
            "weight_stress": 0.0,
        },
        "data": {"data_path": str(tmp_path), "data_file": "smoke.csv"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path / "config.json"


@pytest.mark.skipif(os.environ.get("GPTFF_RUN_TRAINING_SMOKE") != "1", reason="set GPTFF_RUN_TRAINING_SMOKE=1 to run training smoke")
def test_training_entrypoint_smoke(tmp_path):
    config = write_training_smoke_files(tmp_path)
    subprocess.run(
        [sys.executable, "-m", "gptff.trainer.trainer", str(config)],
        cwd=tmp_path,
        check=True,
        timeout=120,
    )
    assert (tmp_path / "curr_checkpoint.pth").exists()
    assert (tmp_path / "best_checkpoint.pth").exists()
