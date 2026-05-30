import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

from gptff.utils_.data import Mydataset, collate_fn, structure_arrays


def make_row(structure, idx, fold=-1):
    return {
        "struct_id": idx,
        "energy": float(idx),
        "forces": str(np.zeros((len(structure), 3), dtype=float).tolist()),
        "stress": str(np.zeros((3, 3), dtype=float).tolist()),
        "structure": str(structure.as_dict()),
        "fold": fold,
        "ref_energy": 0.0,
    }


def assert_items_close(left, right):
    assert len(left) == len(right)
    for left_value, right_value in zip(left, right):
        assert np.allclose(np.asarray(left_value), np.asarray(right_value))


def test_structure_arrays_match_pymatgen_structure():
    structure = Structure(
        Lattice.cubic(5.0),
        ["Li", "O", "O"],
        [[0.0, 0.0, 0.0], [0.42, 0.42, 0.0], [0.42, 0.0, 0.42]],
    )

    atom_fea, coords, lattice = structure_arrays(str(structure.as_dict()))

    assert atom_fea.shape == (3, 1)
    assert atom_fea.ravel().tolist() == [3, 8, 8]
    assert np.allclose(coords, structure.cart_coords)
    assert np.allclose(lattice, structure.lattice.matrix)


def test_dataset_precompute_matches_lazy_cache():
    structure = Structure(
        Lattice.cubic(5.0),
        ["Li", "O", "O"],
        [[0.0, 0.0, 0.0], [0.42, 0.42, 0.0], [0.42, 0.0, 0.42]],
    )
    df = pd.DataFrame([make_row(structure, idx=0)])

    lazy = Mydataset(df, cache_graphs=True, precompute_graphs=False)
    precomputed = Mydataset(df, cache_graphs=True, precompute_graphs=True)

    assert precomputed.cache_info() == {"enabled": True, "cached": 1, "size": 1}
    assert_items_close(lazy[0], precomputed[0])
    assert lazy.cache_info() == {"enabled": True, "cached": 1, "size": 1}


def test_collate_handles_cached_and_no_neighbor_items():
    isolated = Structure(Lattice.cubic(20.0), ["Li"], [[0.0, 0.0, 0.0]])
    compact = Structure(
        Lattice.cubic(5.0),
        ["Li", "O", "O"],
        [[0.0, 0.0, 0.0], [0.42, 0.42, 0.0], [0.42, 0.0, 0.42]],
    )
    df = pd.DataFrame([make_row(isolated, idx=0), make_row(compact, idx=1)])
    dataset = Mydataset(df, cache_graphs=True, precompute_graphs=True)

    batch = collate_fn([dataset[0], dataset[1]])
    atom_fea, coords, offsets, lattice, n_atoms, pairs_count, nbr_atoms, bond_pairs_indices, n_bond_pairs_bond, energy, forces, stress, ref_energy = batch

    assert atom_fea.shape == (4, 1)
    assert coords.shape == (4, 3)
    assert offsets.shape[1] == 3
    assert lattice.shape == (2, 3, 3)
    assert n_atoms.tolist() == [1, 3]
    assert pairs_count.shape == (2,)
    assert nbr_atoms.shape[1] == 2
    assert bond_pairs_indices.shape[1] == 2
    assert n_bond_pairs_bond.shape[0] == pairs_count.sum().item()
    assert energy.shape == (2,)
    assert forces.shape == (4, 3)
    assert stress.shape == (2, 3, 3)
    assert ref_energy.shape == (2,)
