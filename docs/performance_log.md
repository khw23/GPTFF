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
