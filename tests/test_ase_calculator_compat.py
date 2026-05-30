from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from gptff.model.mpredict import ASECalculator


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = [
    ROOT / "pretrained" / "gptff_v1.pth",
    ROOT / "pretrained" / "gptff_v2.pth",
]


@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
@pytest.mark.parametrize(
    "atoms",
    [
        Atoms("Li", positions=[[0.0, 0.0, 0.0]], cell=[20.0, 20.0, 20.0], pbc=True),
        Atoms("Li2", positions=[[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]], cell=[20.0, 20.0, 20.0], pbc=True),
    ],
)
def test_pretrained_calculator_handles_small_systems(checkpoint, atoms):
    calc = ASECalculator(str(checkpoint), device="cpu")
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.all(np.isfinite(forces))


@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_pretrained_calculator_stress_shape(checkpoint):
    calc = ASECalculator(str(checkpoint), device="cpu")
    atoms = Atoms("Li2", positions=[[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]], cell=[20.0, 20.0, 20.0], pbc=True)
    atoms.calc = calc

    stress = atoms.get_stress(voigt=False)

    assert stress.shape == (3, 3)
    assert np.all(np.isfinite(stress))


def test_calculator_uses_property_specific_paths():
    calc = ASECalculator(str(CHECKPOINTS[0]), device="cpu")
    atoms = Atoms("Li2", positions=[[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]], cell=[20.0, 20.0, 20.0], pbc=True)
    atoms.calc = calc

    atoms.get_potential_energy()
    assert "energy" in calc.results
    assert "forces" not in calc.results
    assert "stress" not in calc.results

    atoms.positions[0, 0] += 1.0e-5
    atoms.get_forces()
    assert "forces" in calc.results
    assert "stress" not in calc.results

    atoms.positions[0, 0] -= 1.0e-5
    atoms.get_stress(voigt=False)
    assert "stress" in calc.results
