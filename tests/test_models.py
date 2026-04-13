import unittest

from jrso.config import DatasetConfig, TrainingConfig
from jrso.dataset import SyntheticDatasetGenerator
from jrso.models import JRSOModel, RecSysOnlyModel, TwoStagePipeline


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset_config = DatasetConfig.preset("tiny")
        dataset_config = DatasetConfig(
            **{
                **dataset_config.to_dict(),
                "oracle_simulations": 8,
                "evaluation_simulations": 6,
            }
        )
        cls.dataset = SyntheticDatasetGenerator(dataset_config).generate()
        cls.training = TrainingConfig(epochs=2, batches_per_epoch=4, batch_size=24)

    def test_recsys_training_produces_score_matrix(self) -> None:
        model = RecSysOnlyModel(self.dataset, self.training, seed_offset=1).fit()
        scores = model.predict_score_matrix()
        self.assertEqual(scores.shape, (self.dataset.config.num_users, self.dataset.config.num_items))
        self.assertGreater(len(model.artifacts.history), 0)

    def test_jrso_selects_budgeted_seeds(self) -> None:
        model = JRSOModel(self.dataset, self.training.with_lambda(0.10), seed_offset=2).fit()
        item_idx = int(self.dataset.evaluation_items[0])
        seeds = model.select_seeds(item_idx)
        self.assertEqual(len(seeds), self.dataset.config.k_seeds)
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_two_stage_uses_trained_recommender(self) -> None:
        recommender = RecSysOnlyModel(self.dataset, self.training, seed_offset=3).fit()
        pipeline = TwoStagePipeline(recommender, self.dataset)
        seeds = pipeline.select_seeds(int(self.dataset.evaluation_items[0]))
        self.assertEqual(len(seeds), self.dataset.config.k_seeds)


if __name__ == "__main__":
    unittest.main()
