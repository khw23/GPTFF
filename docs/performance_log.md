# GPTFF Performance Log

## 2026-05-30 - ASECalculator property-specific paths

### Change

- Split `ASECalculator.calculate()` into separate paths for energy-only, force-only, and
  full stress evaluation.
- Keep the full `energy + forces + stress` path for `atoms.get_stress()`.
- Use `torch.no_grad()` for `atoms.get_potential_energy()`.
- Avoid strain construction and coordinate strain transforms for force-only calls.
- Read positions, cell, and atomic numbers directly from ASE `Atoms` in
  `custom_graph.transform()` instead of converting to pymatgen `Structure` each call.

### Benchmark

- Machine: DGX Spark, `NVIDIA GB10`
- Image: `gptff-dev-modern:ase3.28.0-py3.12-cuda13.0.1-arm64`
- Structure: LiCoO2, `6x6x2`, `864` atoms
- Checkpoint: `pretrained/gptff_v1.pth`
- Steps: `3` warmup + `15` timed calls per mode
- Baseline: `45df59e`

| mode | mean before (s) | mean after (s) | speedup | CUDA peak before | CUDA peak after |
|---|---:|---:|---:|---:|---:|
| energy | 0.3252 | 0.1506 | 2.16x | 5.18 GiB | 1.36 GiB |
| forces | 0.3297 | 0.3095 | 1.07x | 5.18 GiB | 5.18 GiB |
| stress | 0.3264 | 0.3234 | 1.01x | 5.18 GiB | 5.18 GiB |

### Notes

- Energy-only inference benefits most because it no longer builds autograd graphs for
  forces and stress.
- Force-only inference is modestly faster, but CUDA peak is still dominated by the
  coordinate-gradient graph.
- Stress inference intentionally remains close to the baseline because it still needs
  strain autograd.
- Detailed local results are stored in the parent DGX benchmark project under
  `results/gptff_optimization/2026-05-30/`.

## 2026-05-30 - Correctness suite and force backward profiling

### Correctness

Added:

- `tests/test_calculator_workflows.py`
- `benchmarks/correctness_suite.py`

Coverage:

- EFS path consistency
- finite-difference force smoke
- fixed-cell structure optimization
- short NVE MD
- `FrechetCellFilter` cell-optimization smoke in the full suite
- training entrypoint smoke with generated tiny CSV/config

Validation:

- ASE `3.26.0`: `12 passed, 1 skipped`
- ASE `3.28.0`: `12 passed, 1 skipped`
- Training smoke explicitly enabled: `1 passed`
- Full CUDA suite on LiCoO2 primitive cell: all `6/6` workflows passed.

### Force backward profile

Added `benchmarks/force_backward_profile.py`.

LiCoO2 `6x6x2` (`864` atoms), V1 checkpoint, force-only path:

| variant | graph | collate | features | forward | backward | final peak |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.025 s | 0.009 s | 0.001 s | 0.270 s | 0.161 s | 5.18 GiB |
| whole-model checkpoint | 0.025 s | 0.011 s | 0.001 s | 0.184 s | 0.380 s | 5.18 GiB |

Whole-model checkpointing reduces forward-phase peak memory but does not reduce final
force-backward peak for this workload, and increases total profiled time by about
`1.29x`. It is exposed as an explicit `use_checkpoint=True` experiment but should not
be enabled by default for current GPTFF MD benchmarks.

## 2026-05-30 - Fine-grained model-internal optimization

### Change

- Removed the duplicate pair radial-basis evaluation in V1/V2 model forward. The same
  `bonds_dist` tensor now feeds both initial edge construction and distance gates.
- Added a non-persistent radial-basis filter buffer, avoiding repeated `torch.arange`
  construction while keeping pretrained V1/V2 state dicts compatible.
- Precomputed layer-invariant neighbor and three-body index tensors once per forward
  instead of reconstructing them inside every message-passing layer.
- Built V2 padding masks directly on the target device and reused batch indices for
  final structure pooling.
- Added explicit `ASECalculator(..., checkpoint_mode="none"|"model"|"layer")`.
  `use_checkpoint=True` remains a compatibility alias for the old whole-model
  checkpoint path; `checkpoint_mode="layer"` enables the new block-level checkpoint.

### Validation

- ASE `3.26.0`: `15 passed, 1 skipped`
- ASE `3.28.0`: `15 passed, 1 skipped`
- Checkpoint correctness now covers both `pretrained/gptff_v1.pth` and
  `pretrained/gptff_v2.pth` for whole-model and layer checkpoint modes.
- Full CUDA correctness suite on LiCoO2 primitive cell, V1 checkpoint: all `6/6`
  workflows passed.
- CUDA smoke on LiCoO2 primitive cell passed for V1/V2 with `checkpoint_mode="none"`
  and `checkpoint_mode="layer"`.

### Benchmark

LiCoO2 `6x6x2` (`864` atoms), V1 checkpoint, DGX Spark / `NVIDIA GB10`.

End-to-end ASE force calls (`5` warmup + `20` timed calls):

| variant | force mean | atom-steps/s | CUDA peak allocated | time vs default | peak vs default |
|---|---:|---:|---:|---:|---:|
| default | 0.3070 s | 2814 | 5.16 GiB | 1.00x | 1.00x |
| whole-model checkpoint | 0.4202 s | 2056 | 5.16 GiB | 1.37x | 1.00x |
| layer checkpoint | 0.4217 s | 2049 | 2.10 GiB | 1.37x | 0.41x |

Force backward profile (`2` warmup + `8` timed profiles):

| variant | total profiled | final peak allocated | time vs default | peak vs default |
|---|---:|---:|---:|---:|
| default | 0.470 s | 5.16 GiB | 1.00x | 1.00x |
| whole-model checkpoint | 0.598 s | 5.16 GiB | 1.27x | 1.00x |
| layer checkpoint | 0.499 s | 2.10 GiB | 1.06x | 0.41x |

The profiling table includes per-section synchronization and memory snapshot overhead,
so the end-to-end ASE table is the better proxy for MD wall time. The profile is still
useful for locating the allocator peak.

### Conclusion

- The default path keeps the previous P1/P2 speed and memory behavior essentially
  unchanged: `energy/forces/stress` were measured at `0.1504/0.3070/0.3213 s`.
- Whole-model checkpointing is still not useful for production MD: it slows force calls
  and does not lower the final force-backward peak.
- Layer checkpointing is a real memory knob. For this 864-atom LiCoO2 force case it
  reduces CUDA peak allocation by about `59%`, from `5.16 GiB` to `2.10 GiB`, but slows
  end-to-end force calls by about `37%`. Keep it opt-in for larger or memory-limited
  systems rather than enabling it by default.

## 2026-05-30 - Training data and graph pipeline optimization

### Change

- Reworked `gptff.utils_.data.Mydataset`.
  - Original path: each `__getitem__` read pandas columns, parsed string literals,
    built a full `pymatgen.Structure`, then extracted atomic numbers, Cartesian
    coordinates and the lattice before building neighbor and three-body graphs.
  - New path: for common ordered `Structure.as_dict()` rows, `structure_arrays()` reads
    `sites[*].xyz`, `sites[*].species[0].element` and `lattice.matrix` directly, and
    only falls back to `pymatgen.Structure.from_dict()` for uncommon rows.
  - Labels are read from column lists / NumPy arrays instead of repeated pandas scalar
    indexing.
- Added dataset-level graph caching.
  - `cache_graphs=True` keeps computed graph tuples in the dataset.
  - `precompute_graphs=True` builds all graph tuples once before the DataLoader starts.
  - `cache_info()` reports whether the cache is enabled and how many items are filled.
- Updated the training entrypoint.
  - New config keys: `cache_graphs`, `precompute_graphs`, `persistent_workers`,
    `pin_memory`, `prefetch_factor`.
  - Existing configs still work; defaults are conservative except that
    `persistent_workers` defaults to `True` when `workers > 0`.
- Lightened `collate_fn` by using `np.asarray` / `torch.as_tensor` where possible and
  by avoiding a redundant Python list conversion for `n_bond_pairs_struc`.

The collated batch contract is unchanged:

```text
atom_fea, coords, offsets, lattice, n_atoms, pairs_count, nbr_atoms,
bond_pairs_indices, n_bond_pairs_bond, target_energy, target_forces,
target_stress, ref_energy
```

### Validation

- ASE `3.26.0`: `18 passed, 1 skipped`
- ASE `3.28.0`: `18 passed, 1 skipped`
- Explicit training smoke: `1 passed`
- Full CUDA correctness suite on LiCoO2 primitive cell, V1 checkpoint: all `6/6`
  workflows passed.

### Benchmark

Dataset: `human/train/TiMgSbBi.csv`, `1799/200` train/validation structures.
Configuration: `batch=32`, `workers=4`, `5 epochs`, checkpoint writes disabled,
DGX Spark / `NVIDIA GB10`.

| variant | setup | training-loop total | incl. graph setup | steady train atoms/s | steady data wait | RSS peak |
|---|---|---:|---:|---:|---:|---:|
| baseline `22de21f` | old parser, no persistent workers | 34.44 s | 34.44 s | 7106 | 0.199 s/epoch | 9.22 GiB |
| optimized lazy | direct parser/cache, no persistent workers | 33.77 s | 33.77 s | 7218 | 0.192 s/epoch | 9.22 GiB |
| precompute + persistent | direct parser/cache, precompute graphs, persistent workers | 31.44 s | 33.41 s | 7429 | 0.016 s/epoch | 11.29 GiB |

`precompute_graphs=True` spent `1.97 s` building graph tuples before training. For only
`5` epochs, including that setup cost gives a modest `1.03x` full wall-time speedup
over the baseline. Excluding the one-time setup cost, the training loop itself is
`1.10x` faster. For long runs, the one-time graph setup is amortized.

### Conclusion

- Direct structure parsing and lighter collation are small but positive by themselves
  (`~2%` total speedup in this 5-epoch run).
- The main useful mode is `precompute_graphs=True` plus persistent workers. It cuts
  steady-state DataLoader wait by about `92%` and raises steady train throughput by
  about `4.5%` on this workload.
- The tradeoff is host/process RSS accounting: process-tree RSS rose from about
  `9.22 GiB` to `11.29 GiB`. DGX Spark host available memory stayed above `113 GiB`,
  so this is safe for the tested dataset, but very large datasets should either disable
  precompute or shard/cache to disk in a later optimization.
