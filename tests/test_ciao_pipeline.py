import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.io

from jrso.ciao_config import CiaoExperimentConfig
from jrso.ciao_dataset import CiaoDatasetLoader
from jrso.ciao_experiment import run_ciao_experiment


def _write_synthetic_ciao(target_dir: Path, *, num_users: int = 80, num_items: int = 60, seed: int = 7) -> None:
    """Build a small ciao-format directory used by the smoke tests."""
    rng = np.random.default_rng(seed)
    target_dir.mkdir(parents=True, exist_ok=True)

    rating_rows: list[list[int]] = []
    num_categories = 4
    for user_idx in range(num_users):
        num_ratings = int(rng.integers(15, 26))
        items = rng.choice(num_items, size=num_ratings, replace=False)
        for item_idx in items:
            category = (int(item_idx) % num_categories) + 1
            score = int(rng.integers(3, 6))
            helpfulness = int(rng.integers(0, 6))
            rating_rows.append([int(user_idx) + 1, int(item_idx) + 1, category, score, helpfulness])
    ratings = np.asarray(rating_rows, dtype=np.int32)

    edges: set[tuple[int, int]] = set()
    avg_degree = 6
    for user_idx in range(num_users):
        peers = rng.choice(num_users, size=avg_degree, replace=False)
        for peer in peers:
            if int(peer) == user_idx:
                continue
            a, b = sorted((user_idx + 1, int(peer) + 1))
            edges.add((a, b))
    trust = np.asarray(sorted(edges), dtype=np.uint16)

    scipy.io.savemat(target_dir / "rating.mat", {"rating": ratings})
    scipy.io.savemat(target_dir / "trustnetwork.mat", {"trustnetwork": trust})


class CiaoLoaderTests(unittest.TestCase):
    def test_loads_synthetic_ciao_directory(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="ciao_loader_"))
        try:
            _write_synthetic_ciao(tmp)
            from jrso.ciao_config import CiaoDatasetConfig

            config = CiaoDatasetConfig(
                ciao_dir=tmp,
                user_core=4,
                item_core=4,
                k_seeds=4,
                cascade_steps=2,
                num_eval_items=4,
                evaluation_simulations=2,
                seed=11,
            )
            dataset = CiaoDatasetLoader(config).load()
            self.assertGreater(dataset.config.num_users, 0)
            self.assertGreater(dataset.config.num_items, 0)
            self.assertEqual(dataset.train_matrix.shape, (dataset.config.num_users, dataset.config.num_items))
            self.assertEqual(dataset.edge_index.shape[0], 2)
            self.assertEqual(dataset.edge_index.shape[1], dataset.edge_weight.shape[0])
            self.assertTrue(np.all(dataset.edge_weight > 0))
            self.assertEqual(dataset.score_matrix.shape, dataset.train_matrix.shape)
            self.assertGreaterEqual(dataset.evaluation_items.size, 1)
            self.assertGreater(dataset.train_pairs.shape[0], 0)
            self.assertTrue(np.isfinite(dataset.score_matrix).all())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class CiaoExperimentTests(unittest.TestCase):
    def test_full_pipeline_smoke(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="ciao_pipeline_"))
        ciao_dir = tmp / "ciao_data"
        out_dir = tmp / "out"
        try:
            _write_synthetic_ciao(ciao_dir)
            config = CiaoExperimentConfig.preset("ciao_smoke", out_dir, ciao_dir=ciao_dir)
            config = CiaoExperimentConfig(
                dataset=replace(config.dataset, user_core=4, item_core=4, k_seeds=4, num_eval_items=3, evaluation_simulations=2),
                recsys=config.recsys.replace(epochs=1, steps_per_epoch=1, amp=False),
                jrso=config.jrso.replace(epochs=1, steps_per_epoch=1, amp=False),
                jrso_second=config.jrso_second.replace(epochs=1, steps_per_epoch=1, amp=False),
                output_dir=out_dir,
                recommendation_k=config.recommendation_k,
                combined_alpha=config.combined_alpha,
                seed=config.seed,
            )
            result = run_ciao_experiment(config)
            self.assertEqual(result["backend"], "ciao")
            for required in ("RecSys-Only", "IM-Only", "Two-Stage", "JRSO", "JRSO_second"):
                self.assertIn(required, result["results"])
            self.assertTrue((out_dir / "ciao_metrics.csv").exists())
            self.assertTrue((out_dir / "ciao_results.json").exists())
            self.assertTrue((out_dir / "ciao_dataset.npz").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
