import shutil
import tempfile
import unittest
from pathlib import Path

from jrso.torch_config import TorchDatasetConfig, TorchExperimentConfig
from jrso.torch_dataset import SparseSyntheticDatasetGenerator
from jrso.torch_experiment import run_torch_experiment


class TorchPipelineTests(unittest.TestCase):
    def test_sparse_dataset_smoke_generation(self) -> None:
        dataset = SparseSyntheticDatasetGenerator(TorchDatasetConfig.preset("gpu_smoke")).generate()
        self.assertEqual(dataset.edge_index.shape[0], 2)
        self.assertEqual(dataset.edge_index.shape[1], dataset.edge_weight.shape[0])
        self.assertEqual(dataset.train_matrix.shape, (dataset.config.num_users, dataset.config.num_items))
        self.assertGreater(dataset.train_pairs.shape[0], 0)

    def test_torch_experiment_smoke(self) -> None:
        tmpdir = Path(tempfile.mkdtemp(prefix="jrso_torch_test_"))
        try:
            config = TorchExperimentConfig.preset("gpu_smoke", tmpdir)
            config = TorchExperimentConfig(
                dataset=config.dataset,
                recsys=config.recsys.replace(epochs=1, steps_per_epoch=1, amp=False),
                jrso=config.jrso.replace(epochs=1, steps_per_epoch=1, amp=False),
                jrso_second=config.jrso_second.replace(epochs=1, steps_per_epoch=1, amp=False),
                output_dir=config.output_dir,
                recommendation_k=config.recommendation_k,
                combined_alpha=config.combined_alpha,
                seed=config.seed,
            )
            result = run_torch_experiment(config)
            self.assertEqual(result["backend"], "torch")
            self.assertIn("JRSO", result["results"])
            self.assertIn("JRSO_second", result["results"])
            self.assertIn("Two-Stage", result["results"])
            self.assertTrue((tmpdir / "torch_metrics.csv").exists())
            self.assertTrue((tmpdir / "torch_training_logs.json").exists())
            self.assertTrue((tmpdir / "torch_results.json").exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
