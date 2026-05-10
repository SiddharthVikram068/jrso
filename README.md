# JRSO: Joint Recommendation and Seed Optimization

This repository studies a joint recommendation + influence maximization setup:

- recommend items to users well
- choose seed users for each item well
- beat the single-objective baselines on a shared evaluation

The codebase contains three runnable backends:

- `reference`: small NumPy + Autograd implementation for fast iteration and sanity checks
- `torch`: scalable synthetic benchmark used for larger neural experiments
- `ciao`: real-data experiment on the Ciao trust/rating dataset using the Torch pipeline

There are now two joint-model tracks in the Torch/Ciao runners:

- `JRSO` and `JRSO_second`: the original family, now given a stronger training budget and better seed selection
- `JRSO_third`: a new joint algorithm iteration with a different mathematical design

## What Was Wrong Before

After inspecting the repository and the saved runs, the main reasons the joint method looked too similar to the baselines were:

1. The evaluation item set was biased toward only the most popular test items.
   Those items are the easiest to diffuse, so spread metrics on the synthetic side were close to saturation and model differences were compressed.

2. Joint models were checkpointed only on validation `ndcg@10`.
   Even when a later checkpoint improved the influence objective, it could be discarded if its recommendation score was only tied or slightly noisier.

3. Joint seed selection itself was still too close to a top-`k` affinity ranking.
   That ignores overlap between seeds and wastes budget on users who reach the same neighborhood.

4. The scalable Torch models were recommendation-limited on the synthetic benchmark.
   They relied heavily on social structure and did not exploit interaction history strongly enough, which is especially costly on the `large_gpu` dataset.

## What Changed

This pass makes two kinds of improvements.

### 1. Stronger version of the original JRSO line

- `jrso/torch_config.py`
  The `large_gpu` Torch preset is now a submission-grade configuration with much larger training budgets:
  more epochs, more steps per epoch, larger hidden sizes, and more negative candidates.

- `jrso/torch_experiment.py`
  Training now uses cosine learning-rate decay, and joint-model checkpoint selection still prioritizes validation recommendation quality but now uses spread bonus as a tie-break.

- `jrso/torch_models.py`
  `JRSO` and `JRSO_second` now use a graph-aware, coverage-aware greedy seed selector that rewards reach and penalizes redundant neighborhoods.

### 2. A new algorithm iteration: `JRSO_third`

`JRSO_third` is not a rename of the old model. It is a separate joint method with a different mathematical structure:

- hybrid user encoding:
  social GNN representation + interaction-history representation

- gated fusion:
  the model learns how much to trust social structure versus user history

- weighted diffusion objective:
  instead of maximizing mean activation only, it optimizes a weighted spread objective that is closer to evaluation-time weighted influence

- overlap penalty:
  discourages the seed distribution from collapsing onto the same local neighborhood

The implementation lives in `jrso/torch_models.py` as `TorchJRSOThird`.

### 3. Better evaluation protocol

- `jrso/torch_dataset.py` and `jrso/ciao_dataset.py`
  Evaluation items are now sampled across the popularity spectrum instead of only from the most popular test items.

## Current Status

Two useful fresh runs are included in `run_outputs/`:

- `run_outputs/ciao_medium_hybrid`
  `JRSO_third` is the current strongest real-data result in this repository:
  `ndcg@10 = 0.0805`
  `weighted_spread_rate = 0.04705`
  `combined_score = 0.06376`

  Relative to `RecSys-Only`, that run gives:
  `+22.9%` on `ndcg@10`
  `+51.9%` on `weighted_spread_rate`
  `+32.3%` on combined score

- `run_outputs/gpu_smoke_hybrid_e6`
  On the Torch synthetic smoke run with a bit more training, the tuned original `JRSO` is already much stronger than before:
  `combined_score = 0.16711`
  vs `RecSys-Only = 0.13240`
  and `Two-Stage = 0.15622`

- `run_outputs/gpu_smoke_hybrid`
  Shows the new `JRSO_third` trade-off on the Torch path under a very short run.

The older folders under `outputs/` are historical artifacts from before this update. Re-run experiments after pulling the latest code if you want apples-to-apples comparisons.

## Repository Map

### Core package

- `jrso/__main__.py`
  CLI entrypoint for all backends.

- `jrso/config.py`, `jrso/dataset.py`, `jrso/models.py`, `jrso/experiment.py`
  Reference NumPy/Autograd implementation.

- `jrso/torch_config.py`, `jrso/torch_dataset.py`, `jrso/torch_models.py`, `jrso/torch_experiment.py`
  Main scalable synthetic pipeline, including tuned `JRSO`/`JRSO_second` and the new `JRSO_third`.

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

For the main submission-style synthetic benchmark, use:

```powershell
py -3 -m jrso --backend torch --preset large_gpu --output-dir run_outputs\large_gpu --device cpu
```

Important: `large_gpu` is now intentionally much heavier than before.
It uses more epochs, more steps, and larger models so the original `JRSO` line gets a fairer chance to improve.

Use `--device cuda` on a CUDA machine:

```powershell
py -3 -m jrso --backend torch --preset large_gpu --output-dir run_outputs\large_gpu --device cuda
```

If you want to push the original JRSO line even harder, you can further increase epochs:

```powershell
py -3 -m jrso --backend torch --preset large_gpu --epochs 120 --output-dir run_outputs\large_gpu_e120 --device cuda
```

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

## Model Names In Results

You will now see these model names in Torch/Ciao outputs:

- `RecSys-Only`
- `IM-Only`
- `Two-Stage`
- `JRSO`
- `JRSO_second`
- `JRSO_third`

Recommended interpretation:

- use `JRSO` to judge the tuned original method
- use `JRSO_third` to judge the new algorithm iteration

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

- add a true large-scale ablation on `large_gpu` comparing old and new presets after full CUDA training
- move from weighted one-hop coverage penalties to a stronger differentiable submodular objective
- improve `JRSO_second` or retire it if `JRSO_third` consistently dominates it
