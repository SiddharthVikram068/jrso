import shutil
import tempfile
import unittest
from pathlib import Path

from jrso.config import DatasetConfig, ExperimentConfig, TrainingConfig
from jrso.experiment import run_experiment


class ExperimentTests(unittest.TestCase):
    def test_tiny_experiment_writes_expected_artifacts(self) -> None:
        tmpdir = Path(tempfile.mkdtemp(prefix="jrso_test_"))
        try:
            dataset = DatasetConfig(
                **{
                    **DatasetConfig.preset("tiny").to_dict(),
                    "oracle_simulations": 6,
                    "evaluation_simulations": 4,
                }
            )
            training = TrainingConfig(epochs=1, batches_per_epoch=2, batch_size=16, lambda_spread=0.10)
            config = ExperimentConfig(
                dataset=dataset,
                training=training,
                lambda_sweep=(0.0, 0.10),
                output_dir=tmpdir,
            )
            result = run_experiment(config)
            self.assertIn("JRSO", result["results"])
            self.assertTrue((tmpdir / "dataset.npz").exists())
            self.assertTrue((tmpdir / "metrics.csv").exists())
            self.assertTrue((tmpdir / "training_logs.json").exists())
            self.assertTrue((tmpdir / "pareto_frontier.svg").exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
