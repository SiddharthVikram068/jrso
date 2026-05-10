# JRSO: Joint Recommendation and Seed Optimization

This repository implements the JRSO framework from `recsys_report.pdf`: a single
end-to-end model that jointly optimises recommendation quality and influence
spread on a social graph. The torch / Ciao backends now ship with three pieces
that are central to the report:

1. a **bipartite + social GNN encoder** (LightGCN-style) that produces
   user and item embeddings from the user-user trust graph *and* the
   user-item interaction graph at the same time
2. an **end-to-end differentiable Independent Cascade** (Message-Passing
   Diffusion) layer where the influence probabilities `P_{v,u}(Theta)`
   are parameterised by the recommender's GNN embeddings
3. a **continuous seeding vector** `s = 1 - exp(-K * softmax(seed_logits))`
   that respects the budget `sum_u s_u ~ K` and is differentiable w.r.t.
   the recommender parameters

The joint training loss follows the report's specification:

```text
L_joint(Theta, s) = L_rec(Theta) - lambda * mean_u h^{(T)}_u(Theta, s)
```

with two extra regularisers used during training: an overlap penalty that
discourages seeds from sharing the same neighbourhood, and an edge-alignment
loss so that the learned edge probabilities stay close to the empirical
trust-network MLE estimates.

## Backends

- `reference` — small NumPy + autograd implementation, kept for fast sanity
  checks and the original tests.
- `torch` — sparse synthetic benchmark used for scaling experiments
  (`gpu_smoke`, `large_gpu`).
- `ciao` — real-data experiment on the Ciao trust + rating dataset
  (`ciao_smoke`, `ciao_medium`, `ciao_full`).

The `torch` and `ciao` backends share **all** model code: the only thing that
changes is the dataset loader.

## Models

The torch / ciao path runs six models in every experiment:

| Model         | Recommender                             | Seed selector                                                                    |
|---------------|-----------------------------------------|----------------------------------------------------------------------------------|
| `RecSys-Only` | LightGCN bipartite + social             | top-K affinity                                                                   |
| `IM-Only`     | none                                    | centrality-greedy with neighbour suppression                                     |
| `Two-Stage`   | uses `RecSys-Only` outputs              | greedy CELF over `affinity * (1 + propagation + centrality)`                     |
| `JRSO`        | LightGCN + edge prob head + spread head | greedy CELF over learned edge probabilities                                      |
| `JRSO_second` | LightGCN + attention edge head          | greedy CELF over learned attention edge probabilities                            |
| `JRSO_third`  | DiffNet++-style social-history fusion   | greedy CELF over learned edge probabilities + history-fused user representations |

`JRSO_third` is usually the strongest joint model on Ciao because the trust
graph is sparse and the user-item interaction history carries a lot of
the recommendation signal.

## Architectural changes vs the previous version

Earlier revisions of this repository used a social-only GNN encoder, a
unit-normalised score function, a softplus BPR loss with very few
negatives, and a CPU-only Python loop for seed selection. With those
choices `RecSys-Only` on `ciao_full` only reached `ndcg@10 ≈ 0.043` and
`recall@10 ≈ 0.05`. The current implementation rebuilds the encoder and
the seed selector around results that are well-known to work on Ciao /
Epinions / FilmTrust:

- LightGCN bipartite propagation (He et al., 2020) + social edges
  (SocialLGN; Liao et al., 2022)
- DiffNet++-style social/interest fusion (Wu et al., 2020) for
  `JRSO_third`
- BPR with hard negatives **plus** in-batch negatives and an L2
  regulariser on the raw embeddings
- a vectorised CELF-style greedy seed selector that uses the model's
  learned edge probabilities to pick a low-overlap, high-affinity seed
  set
- training-time edge subsampling for the differentiable diffusion step
  so the joint loss is fast even on the full Ciao trust graph

The result is a much stronger recommender baseline and a joint model
that improves both NDCG and weighted spread on Ciao.

## Setup

Use Python 3.9+. CPU and CUDA both work.

```bash
pip install -r requirements.txt
```

The Ciao `.mat` files must live under `ciao/` (or any directory
specified via `--ciao-dir`).

## How to run

### Reference backend (sanity checks)

```bash
python -m jrso --backend reference --preset tiny --output-dir run_outputs/reference_tiny
```

Available presets: `tiny`, `small`, `medium`.

### Torch synthetic backend

```bash
python -m jrso --backend torch --preset gpu_smoke --output-dir run_outputs/gpu_smoke
python -m jrso --backend torch --preset large_gpu --output-dir run_outputs/large_gpu --device cuda
```

### Ciao real-data backend

```bash
python -m jrso --backend ciao --preset ciao_smoke  --output-dir run_outputs/ciao_smoke   --device cpu
python -m jrso --backend ciao --preset ciao_medium --output-dir run_outputs/ciao_medium  --device cpu
python -m jrso --backend ciao --preset ciao_full   --output-dir run_outputs/ciao_full    --device cpu
```

CUDA versions just swap `--device cpu` for `--device cuda`. Common
optional overrides for any backend:

```bash
python -m jrso --backend ciao --preset ciao_full --epochs 90 --lambda-spread 0.30 \
    --output-dir run_outputs/ciao_full_lambda030 --device cuda
```

## Output files

Each run writes to `--output-dir`:

- `*_dataset.npz` — saved dataset snapshot (k-core filter, splits,
  edges, score matrix, etc.)
- `*_metrics.csv` — final per-model metrics
- `*_results.json` — same metrics in JSON, plus the run config and
  uplift summary
- `*_training_logs.json` — per-epoch training logs
- `*_seed_summary.json` — chosen seed users per evaluation item per
  model
- `*_significance.json` — paired permutation tests on weighted spread
- SVG plots: training curves, recommendation comparison, influence
  comparison, Pareto frontier

## Tests

```bash
python -m unittest tests.test_dataset tests.test_diffusion tests.test_models tests.test_experiment tests.test_torch_pipeline tests.test_ciao_pipeline
```

The torch / ciao tests run the entire pipeline end-to-end on a small
synthetic Ciao directory.

## File map

```text
jrso/
  __main__.py            CLI entrypoint
  config.py              reference NumPy/autograd dataclasses
  dataset.py             reference dataset generator
  diffusion.py           reference IC simulator
  experiment.py          reference experiment runner
  metrics.py             metrics, paired permutation test, Pareto frontier
  models.py              reference NumPy/autograd models
  optim.py               reference optimiser

  torch_config.py        torch dataset/training/experiment configs
  torch_dataset.py       sparse synthetic dataset generator
  torch_models.py        LightGCN encoder, JRSO/JRSO_second/JRSO_third,
                         IM-only and Two-Stage baselines, greedy seed selector
  torch_experiment.py    sparse-torch experiment runner

  ciao_config.py         Ciao dataset/training/experiment configs
  ciao_dataset.py        real-data Ciao loader (k-core, MLE edge probs,
                         model-free score-matrix proxy)
  ciao_experiment.py     Ciao experiment runner (re-uses torch_experiment)

  plots.py               SVG plots for training curves and comparisons

scripts/
  inspect_ciao_filter.py k-core size sweep over the raw Ciao data
  compute_ceilings.py    metric ceiling computation from a saved Ciao run

tests/
  test_dataset.py            reference dataset
  test_diffusion.py          reference diffusion
  test_models.py             reference models
  test_experiment.py         reference experiment
  test_torch_pipeline.py     torch backend end-to-end smoke
  test_ciao_pipeline.py      Ciao backend end-to-end smoke (synthetic .mat files)
```
