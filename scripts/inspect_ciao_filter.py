"""Quick utility to gauge K-core sizes for the real Ciao dataset."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jrso.ciao_config import CiaoDatasetConfig
from jrso.ciao_dataset import CiaoDatasetLoader


def main() -> None:
    for core in (5, 10, 15, 20):
        cfg = CiaoDatasetConfig(
            ciao_dir=Path("ciao"),
            user_core=core,
            item_core=core,
            num_eval_items=12,
        )
        dataset = CiaoDatasetLoader(cfg).load()
        score_size_mb = dataset.score_matrix.nbytes / (1024 * 1024)
        print(
            f"core={core:>2}: users={dataset.config.num_users:>6,}  "
            f"items={dataset.config.num_items:>6,}  "
            f"train_pairs={dataset.train_pairs.shape[0]:>8,}  "
            f"trust_edges={dataset.edge_index.shape[1]:>8,}  "
            f"score_matrix={score_size_mb:>7.1f} MB"
        )


if __name__ == "__main__":
    main()
