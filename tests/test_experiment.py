"""Checks for input semantics, fold isolation, and physical output constraints."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "prediction"))

import baselines
import nn_replicate_attention as rca
from run_experiment import validate_contexts
from spoilage_data import DATA_DIR, load_contexts


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.contexts, cls.observations = load_contexts(DATA_DIR, set())
        cls.contexts = validate_contexts(cls.contexts)
        cls.context_map = {(c.meat, c.temp_label): c for c in cls.contexts}
        torch.manual_seed(42)
        cls.model = rca.RCAGompertz().eval()
        cls.device = torch.device("cpu")

    def test_complete_dataset(self):
        self.assertEqual(len(self.contexts), 12)
        self.assertEqual(len(self.observations), 11100)
        self.assertEqual(sum(c.tensor.size for c in self.contexts), 55500)

    def test_parameter_count(self):
        self.assertEqual(rca.count_parameters(self.model), 32739)

    def test_all_heldout_data_excluded_from_training(self):
        for fold in range(5):
            altered = copy.deepcopy(self.context_map)
            for context in altered.values():
                context.tensor[:, :, :, fold] = np.nan
            expected = rca.build_training_batch(self.context_map, fold, self.device)
            actual = rca.build_training_batch(altered, fold, self.device)
            self.assertEqual(actual[0].shape, (1200, 25, 1))
            self.assertEqual(actual[1].shape, (1200, 4, 25, 1))
            self.assertTrue(torch.all(actual[3].sum(dim=1) == 3))
            for before, after in zip(expected, actual):
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_heldout_future_cannot_change_predictions(self):
        expected = pd.DataFrame(rca.predict_heldout(self.model, self.context_map, 0, self.device))
        altered = copy.deepcopy(self.context_map)
        for context in altered.values():
            context.tensor[25:, :, :, 0] = np.nan
        actual = pd.DataFrame(rca.predict_heldout(self.model, altered, 0, self.device))
        np.testing.assert_array_equal(expected.predicted_depth, actual.predicted_depth)
        self.assertTrue(actual.true_depth.isna().all())
        self.assertEqual(len(actual), 3600)
        self.assertEqual(sorted(actual.time.unique()), list(range(25, 37)))

    def test_support_order_and_padding(self):
        batch = list(rca.build_training_batch(self.context_map, 0, self.device))
        small = [value[:25] for value in batch[:-1]]
        with torch.no_grad():
            expected = self.model(*small)
            permutation = torch.tensor([2, 0, 3, 1])
            permuted = small.copy()
            for index in [1, 2, 3]:
                permuted[index] = permuted[index][:, permutation]
            actual = self.model(*permuted)
            torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
            padding_changed = [value.clone() for value in small]
            padding_changed[1][:, 3] = 123.0
            padding_changed[2][:, 3] = -456.0
            torch.testing.assert_close(expected, self.model(*padding_changed), rtol=1e-6, atol=1e-7)

    def test_output_bounds_and_monotonicity(self):
        batch = rca.build_training_batch(self.context_map, 0, self.device)
        with torch.no_grad():
            predicted = self.model(*batch[:-1])
        self.assertTrue(torch.isfinite(predicted).all())
        self.assertTrue(torch.all((predicted >= 0) & (predicted <= 1)))
        self.assertTrue(torch.all(predicted[:, 1:] >= predicted[:, :-1]))
        self.assertTrue(torch.all(predicted >= batch[0][:, -1, 0, None]))

    def test_reference_truth_matches_workbooks(self):
        reference = pd.read_csv(ROOT / "results" / "reference" / "predictions.csv")
        reference = reference[reference.method == "replicate_attention"]
        self.assertEqual(len(reference), 18000)
        for row in reference.itertuples():
            context = self.context_map[(row.meat, row.temperature)]
            self.assertAlmostEqual(row.true_depth,
                                   context.tensor[int(row.time), row.x - 1, row.y - 1, row.fold - 1],
                                   places=6)

    def test_reference_metrics_recomputed(self):
        predictions = pd.read_csv(ROOT / "results" / "reference" / "predictions.csv")
        expected = pd.read_csv(ROOT / "results" / "reference" / "summary_by_method_temp.csv")
        actual = baselines.aggregate(predictions)[2]
        pd.testing.assert_frame_equal(expected, actual, check_exact=False, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
