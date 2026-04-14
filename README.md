# JRSO
The repository now contains a runnable synthetic benchmark, the baseline algorithms, the joint model, evaluation utilities, artifact generation, and tests.

## What This Repo Implements

- A **synthetic dataset generator** with:
  - community-structured social graphs
  - latent user and item embeddings
  - interaction generation
  - interaction-level train/validation/test splits
  - true influence probabilities
  - item-level cascade simulation
  - oracle seed sets and oracle adoption targets for evaluation
- Four algorithmic variants:
  - `RecSys-Only`
  - `IM-Only`
  - `Two-Stage`
  - `JRSO`
- An **experiment runner** that saves:
  - `dataset.npz`
  - `dataset_meta.json`
  - `metrics.csv`
  - `training_logs.json`
  - `pareto_frontier.json`
  - `significance.json`
  - SVG visualizations
- A **test suite** that validates the benchmark and the pipeline end to end

## Quickstart

Install the runtime dependencies:

```bash
python3.12 -m pip install -r requirements.txt
```

Run a fast smoke experiment:

```bash
python3.12 -m jrso --preset tiny --epochs 3 --output-dir outputs/tiny_run
```

Run the default benchmark:

```bash
python3.12 -m jrso --preset small --output-dir outputs/small_run
```

Run the test suite:

```bash
python3.12 -m unittest discover -s tests -v
```

## Repository Layout

```text
jrso/
  config.py        experiment and dataset configuration
  dataset.py       synthetic benchmark generation + oracle targets
  diffusion.py     true cascade simulator, CELF, differentiable surrogate
  metrics.py       recommendation and influence metrics
  models.py        RecSys-only, IM-only, Two-stage, and JRSO
  plots.py         lightweight SVG artifact generation
  experiment.py    end-to-end runner
tests/
  test_dataset.py
  test_diffusion.py
  test_models.py
  test_experiment.py
```

## Design Choices

The PDFs describe a GNN-based recommender plus a differentiable diffusion layer. This implementation keeps that spirit, but uses a lightweight stack:

- `numpy` for data and simulation
- `autograd` for differentiable optimization of the JRSO objective
- pure Python for evaluation, artifacts, and tests

That keeps the repository small and easy to run while still training a real joint objective instead of a stub.

## Important PDF Pitfalls Fixed

The longer implementation PDF is helpful, but several parts would have produced a broken or misleading system if copied literally. The repository fixes those points directly:

1. **Interaction splitting**
   The PDF template uses `train_test_split` on the full interaction matrix, which splits by user rows, not by interactions. This repo performs per-user interaction-level train/val/test masking.

2. **Invalid seed probabilities**
   The PDF normalizes seeds with `softmax(s) * k`, which can create values greater than `1` even though the diffusion state is treated like a probability. This repo uses a bounded continuous relaxation: `1 - exp(-k * softmax(logits))`.

3. **Item-agnostic joint loss**
   The PDF’s differentiable diffusion objective is not tied to a target item, so it can optimize spread in a way that is disconnected from recommendation quality for any specific item. This repo makes diffusion and seed scoring **item-conditioned**.

4. **Unstable adoption rule**
   The PDF cascade rule multiplies base preference by an influence factor, which can exceed `1`. This repo converts base preference to log-odds, adds social influence there, and maps back with a sigmoid.

5. **Evaluation target mismatch**
   The PDF precomputes “ground truth adoptions” from random seeds and then compares learned seeds against those adoptions. That is not a fair competitive benchmark. This repo evaluates every model under the same true simulator and compares them against **oracle item-level seed sets**.

6. **Weak recommendation-only baseline**
   The PDF suggests random seeds for the RecSys-only baseline. This repo uses the more meaningful recommendation-only policy of selecting top-affinity users for each item.

7. **Template code issues**
   The PDF code contains undefined variables, missing imports, neighbor-direction mistakes, and snippets that would not run as written. Those are resolved in the actual implementation.

8. **Practical scale**
   The PDF’s suggested `5000 x 2000` setup is not a good default for a lightweight reference implementation with oracle baselines and repeated cascade simulation. This repo ships with `tiny`, `small`, and `medium` presets so the full pipeline stays runnable.

## Model Summary

### `RecSys-Only`

- Socially-aware recommender trained with BPR loss
- Uses social propagation over the user graph
- Does not optimize spread directly
- Seeds are selected by top predicted user-item affinity

### `IM-Only`

- Classical influence maximization baseline
- Uses CELF over the true social simulator
- Does not model recommendation quality
- Produces one global seed set reused across items

### `Two-Stage`

- Trains the recommender first
- Derives item-specific influence estimates from the learned embeddings
- Runs CELF on those derived influence scores

### `JRSO`

- Trains recommendation and spread jointly
- Uses a differentiable item-conditioned diffusion surrogate
- Learns a global user seed bias plus item-conditioned affinity
- Selects seeds from the learned joint signal

## Outputs

Each experiment run writes a folder containing:

- `dataset.npz`: core benchmark arrays
- `dataset_meta.json`: config and oracle metadata
- `metrics.csv`: per-model summary metrics
- `training_logs.json`: per-epoch training traces
- `pareto_frontier.json`: lambda sweep points and non-dominated frontier
- `significance.json`: paired permutation test on spread ratio
- `training_curves.svg`
- `recommendation_comparison.svg`
- `influence_comparison.svg`
- `pareto_frontier.svg`
- `seed_selection_item_<id>.svg`

## Notes For Extension

- `SyntheticDatasetGenerator` is the right place to change the world model or add new benchmark targets.
- `diffusion.py` contains both the true simulator and the differentiable surrogate, so it is the main place to adjust influence assumptions.
- `models.py` is deliberately organized so that new baselines can reuse the same evaluation harness.

## Current Validation

The repository was validated with:

- `python3.12 -m unittest discover -s tests -v`
- `python3.12 -m jrso --preset tiny --epochs 3 --output-dir outputs/tiny_smoke`

Both passed in the working environment used for this implementation.

## Sparse Torch Backend

The repository also now includes a second backend for larger and more realistic experiments:

- sparse social graph storage instead of dense user-user matrices
- GPU-aware training with automatic `cuda` selection when available
- deeper residual GNNs
- a new improved joint model: `JRSO_second`
- direct simulator-based comparison against the baselines at larger scale

### New current-repo pitfalls fixed

The first implementation in this repository was intentionally lightweight, but it had practical limits for the benchmark you asked for:

1. Dense user-user tensors do not scale cleanly to `10,000` users.
2. `autograd`-based training is CPU-only and not suitable for real GPU benchmarking.
3. The original GNN depth and hidden size were too small for a serious large-run comparison.
4. Oracle-heavy evaluation is expensive at large scale, so the new backend compares models directly under the same simulator.

The sparse torch backend fixes those limits.

### New models in the torch backend

#### `JRSO`

- Sparse residual GNN encoder
- Item-conditioned differentiable diffusion
- GPU-capable training loop

#### `JRSO_second`

- More GNN layers and larger hidden state
- Item-conditioned seed MLP instead of the simple seed bias formulation
- Stronger item-aware diffusion head
- Harder negative competition during recommendation training
- Edge-alignment objective to better match the synthetic influence process

### Large GPU preset

Use this when you want the large benchmark the repository now supports:

```bash
python3.12 -m jrso --backend torch --preset large_gpu --output-dir outputs/large_gpu_run
```

The `large_gpu` preset uses:

- `10,000` users
- `1,000` items
- sparse social graph generation
- baseline comparisons against `RecSys-Only`, `IM-Only`, `Two-Stage`
- both `JRSO` and `JRSO_second`

### Torch smoke test

```bash
python3.12 -m jrso --backend torch --preset gpu_smoke --epochs 1 --output-dir outputs/torch_smoke
```

### Torch outputs

The sparse torch backend writes:

- `torch_dataset.npz`
- `torch_dataset_meta.json`
- `torch_metrics.csv`
- `torch_results.json`
- `torch_significance.json`
- `torch_training_logs.json`
- `torch_seed_summary.json`
- `torch_training_curves.svg`
- `torch_recommendation_comparison.svg`
- `torch_influence_comparison.svg`
- `torch_pareto_frontier.svg`

### GPU installation note

The code is written to use GPU when a CUDA build of PyTorch is installed. In the implementation environment here, the torch smoke test ran with a CPU wheel, but the backend itself is ready to run on GPU hardware.
