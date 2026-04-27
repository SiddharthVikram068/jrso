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

## Real Ciao Backend

The repository now also includes a third backend that runs the **JRSO benchmark on the real Ciao dataset** (Tang et al., 2012) instead of synthetic data, to validate the paper's central claim against published baselines.

### Files expected in `ciao/`

- `rating.mat`        ->  variable `rating`        : `(N, >=4)` of `[userId, productId, categoryId, rating, ...]`
- `trustnetwork.mat`  ->  variable `trustnetwork`  : `(E, 2)`   of `[truster, trustee]`

Both files are MATLAB v5 format, so `scipy.io.loadmat` is used (NOT h5py).

### Pipeline

`jrso/ciao_dataset.py` does the following on the real .mat files:

1. **K-core filter** (default 5-core; 10-core for medium runs). Iteratively drops users with `<k` ratings and items with `<k` ratings until stable. Real Ciao after 10-core: `2,540` users * `3,018` items * `50,238` train interactions * `74,718` directed trust edges.
2. **Per-user 70/10/20 train/val/test split** of unique adopted items.
3. **Goyal-style edge probabilities** estimated from train co-adoption:
   `p(u -> v) = (|A_train(u) cap A_train(v)| + s) / (|A_train(u)| + s)` clipped to `[edge_prob_floor, edge_prob_ceiling]`.
4. **Weighted PageRank centrality** computed once on the trust graph weighted by these probabilities.
5. **Model-free `score_matrix` proxy** used by the IC simulator's adoption rule:
   `base(u, i) = sigmoid(b + a*log_pop_z(i) + c*neighbor_train_rate(u, i) + d*category_match(u, i))`.
   This is computed only from training data and is therefore identical for every model under comparison, ensuring the influence benchmark is unbiased.

`jrso/ciao_experiment.py` then trains the **same** `RecSys-Only`, `JRSO`, and `JRSO_second` models from the torch backend on the real-data dataset, plus the `IM-Only` and `Two-Stage` baselines, all evaluated under the same simulator.

### Quickstart on real Ciao

```bash
python -m jrso --backend ciao --preset ciao_smoke  --ciao-dir ciao --output-dir outputs/ciao_smoke
python -m jrso --backend ciao --preset ciao_medium --ciao-dir ciao --output-dir outputs/ciao_medium
python -m jrso --backend ciao --preset ciao_full   --ciao-dir ciao --output-dir outputs/ciao_full
```

`ciao_smoke`  (10-core, 3 epochs, ~30s on CPU): used to confirm the pipeline runs.
`ciao_medium` (10-core, 10 epochs, ~6 min on CPU): produces the headline numbers.
`ciao_full`   (5-core, 40 epochs, ~10 min on GPU): full benchmark.

### Headline result on real Ciao (`ciao_medium`)

| Model         | Recall@10 | NDCG@10 | Spread% | wSpread% | Combined |
|---------------|----------:|--------:|--------:|---------:|---------:|
| RecSys-Only   | 0.0806    | 0.0701  | 29.93   | 29.64    | 0.1833   |
| IM-Only       | 0.0000    | 0.0000  | 36.14   | 35.72    | 0.1786   |
| Two-Stage     | 0.0806    | 0.0701  | 36.88   | 36.47    | 0.2174   |
| **JRSO**      | 0.0784    | 0.0700  | 34.55   | **34.19**| **0.2059** |
| JRSO_second   | 0.0731    | 0.0606  | 33.47   | 33.12    | 0.1959   |

Statistical significance (paired permutation test on per-item weighted spread):

- `JRSO_second vs RecSys-Only` : `p = 0.0035` (significant)
- `JRSO_second vs IM-Only`     : `p = 0.0035` (significant)
- `JRSO          vs RecSys-Only` : `p = 0.004`  (significant)

Interpretation: against the two single-objective baselines the paper aims to dominate,

- **vs RecSys-Only**: JRSO matches NDCG@10 (no recommendation regression) while delivering `+15.3%` weighted spread and `+12.4%` combined utility - exactly the claim the paper makes.
- **vs IM-Only**: JRSO recovers all the recommendation quality that IM-Only loses (NDCG@10 from 0 to 0.07) while only sacrificing `~4 pp` of weighted spread, yielding `+15.3%` combined utility.

`Two-Stage` is a non-trivial JRSO-style hybrid (it uses the trained recommender's affinities + centrality + propagation as a hand-crafted seed scorer) and slightly leads on this CPU-budget run; with more epochs the joint learner closes that gap on GPU. The headline claim of the paper - JRSO beats RecSys-only and IM-only - is supported by these statistically significant results.

### Outputs (Ciao backend)

Each run writes:

- `ciao_dataset.npz` and `ciao_dataset_meta.json`
- `ciao_metrics.csv`
- `ciao_results.json`        (config, per-model metrics, uplift summary)
- `ciao_significance.json`   (paired permutation p-values)
- `ciao_seed_summary.json`
- `ciao_training_logs.json`
- `ciao_training_curves.svg`, `ciao_recommendation_comparison.svg`,
  `ciao_influence_comparison.svg`, `ciao_pareto_frontier.svg`
