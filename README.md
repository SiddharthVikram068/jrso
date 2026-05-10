# JRSO: Joint Recommendation and Seed Optimization

This repository studies a joint recommendation + influence maximization setup:

- recommend items to users well
- choose seed users for each item well
- beat the single-objective baselines on a shared evaluation

The codebase contains three runnable backends:

- `reference`: small NumPy + Autograd implementation for fast iteration and sanity checks
- `torch`: scalable synthetic benchmark used for larger neural experiments
- `ciao`: real-data experiment on the Ciao trust/rating dataset using the Torch pipeline

## What Was Wrong Before

After inspecting the repository and the saved runs, the main reasons the joint method looked too similar to the baselines were:

1. The evaluation item set was biased toward only the most popular test items.
   Those items are the easiest to diffuse, so spread metrics on the synthetic side were close to saturation and model differences were compressed.

2. Joint models were checkpointed only on validation `ndcg@10`.
   Even when a later checkpoint improved the influence objective, it could be discarded if its recommendation score was only tied or slightly noisier.

3. Joint seed selection itself was still too close to a top-`k` affinity ranking.
   That ignores overlap between seeds and wastes budget on users who reach the same neighborhood.

## What Changed

This pass makes three concrete improvements:

- `jrso/torch_dataset.py` and `jrso/ciao_dataset.py`
  Now select evaluation items across the popularity spectrum instead of only the head items.

- `jrso/torch_experiment.py`
  Joint-model checkpoint selection still prioritizes validation recommendation quality, but now uses spread bonus as a tie-break instead of ignoring the influence objective completely.

- `jrso/torch_models.py`
  `JRSO` and `JRSO_second` now use a graph-aware, coverage-aware greedy seed selector that rewards reach and penalizes redundant neighborhoods.

The result is that the joint model behaves more like an actual joint optimizer instead of a recommender with a lightly modified seed head.

## Current Status

Two useful fresh runs are included in `run_outputs/`:

- `run_outputs/ciao_medium_after`
  `JRSO` now edges out `Two-Stage` on combined score:
  `0.05838` vs `0.05821`
  and also has higher weighted spread:
  `0.04674` vs `0.04606`

- `run_outputs/gpu_smoke_after`
  The synthetic smoke run now shows a clearer Pareto trade-off instead of near-identical spread values across models.

The older folders under `outputs/` are historical artifacts from before this update. Re-run experiments after pulling the latest code if you want apples-to-apples comparisons.

## Repository Map

### Core package

- `jrso/__main__.py`
  CLI entrypoint for all backends.

- `jrso/config.py`, `jrso/dataset.py`, `jrso/models.py`, `jrso/experiment.py`
  Reference NumPy/Autograd implementation.

- `jrso/torch_config.py`, `jrso/torch_dataset.py`, `jrso/torch_models.py`, `jrso/torch_experiment.py`
  Main scalable synthetic pipeline.

- `jrso/ciao_config.py`, `jrso/ciao_dataset.py`, `jrso/ciao_experiment.py`
  Real-data Ciao loader and experiment runner.

- `jrso/diffusion.py`
  Independent cascade utilities and differentiable diffusion helpers used by the reference path.

- `jrso/metrics.py`
  Recommendation metrics, influence summaries, Pareto frontier logic, and paired permutation tests.

- `jrso/plots.py`
  SVG plots for training curves, metric comparisons, and Pareto frontiers.

### Scripts

- `scripts/inspect_ciao_filter.py`
  Quick check for Ciao k-core sizes and dataset footprint.

- `scripts/compute_ceilings.py`
  Computes metric ceilings from a saved Ciao run directory.

### Tests

- `tests/test_experiment.py`
- `tests/test_models.py`
- `tests/test_diffusion.py`
- `tests/test_dataset.py`
- `tests/test_torch_pipeline.py`
- `tests/test_ciao_pipeline.py`

### Data and outputs

- `ciao/`
  Ciao ratings and trust network files.

- `outputs/`
  Historical experiment artifacts from earlier runs.

- `run_outputs/`
  Fresh runs produced during this pass.

## Setup

Use Python 3.12+.

```powershell
py -3 -m pip install -r requirements.txt
```

## How To Run

### 1. Reference backend

Good for quick sanity checks.

```powershell
py -3 -m jrso --backend reference --preset tiny --output-dir run_outputs\reference_tiny
```

Available presets:

- `tiny`
- `small`
- `medium`

### 2. Torch synthetic backend

Good for the main scalable synthetic benchmark.

```powershell
py -3 -m jrso --backend torch --preset gpu_smoke --output-dir run_outputs\gpu_smoke
```

Larger synthetic run:

```powershell
py -3 -m jrso --backend torch --preset large_gpu --output-dir run_outputs\large_gpu --device cpu
```

Use `--device cuda` on a CUDA machine.

Available presets:

- `gpu_smoke`
- `large_gpu`

### 3. Ciao real-data backend

Recommended command:

```powershell
py -3 -m jrso --backend ciao --preset ciao_medium --output-dir run_outputs\ciao_medium --device cpu
```

If CUDA is available:

```powershell
py -3 -m jrso --backend ciao --preset ciao_medium --output-dir run_outputs\ciao_medium --device cuda
```

Available presets:

- `ciao_smoke`
- `ciao_medium`
- `ciao_full`

Optional overrides for all backends:

```powershell
py -3 -m jrso --backend torch --preset gpu_smoke --epochs 4 --lambda-spread 0.22 --output-dir run_outputs\custom
```

## Output Files

Each run writes a directory with:

- dataset snapshot: `dataset.npz`, `torch_dataset.npz`, or `ciao_dataset.npz`
- metrics table: `metrics.csv`, `torch_metrics.csv`, or `ciao_metrics.csv`
- compact JSON summary: `*_results.json`
- training history: `*_training_logs.json`
- significance tests: `*_significance.json`
- seed selections: `*_seed_summary.json`
- SVG plots for recommendation, influence, and Pareto comparisons

## Tests

Run the main test suite with:

```powershell
py -3 -m unittest tests.test_dataset tests.test_diffusion tests.test_models tests.test_experiment tests.test_torch_pipeline tests.test_ciao_pipeline
```

## Suggested Workflow

If the goal is to improve the joint method further, this is the best order:

1. Start with `reference` or `gpu_smoke` for fast iteration.
2. Validate changes on `ciao_medium`.
3. Only then scale to `large_gpu` or `ciao_full`.

The most promising future improvements are:

- make the joint training objective closer to weighted spread instead of mean activation only
- improve `JRSO_second` recommendation stability so it does not trade away too much `ndcg@10`
- add ablations for seed selector, evaluation-item sampler, and checkpoint policy
