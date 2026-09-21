"""Focused regression tests: py -3.12 -m unittest discover -s tests -p test_*.py -v"""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models import build_model, count_parameters
from dpelm import DPELM, laplace_noise
from revision_data import fixed_tabular_features, NUMERIC_BOUNDS, CATEGORIES, load_revision_bundle
from revision_training import train_private_model
from revision_statistics import _interval, _signflip, write_analysis
from run_revisions import parser, validate, tasks_for, main
from train_nonprivate import make_criterion, _make_optimizer

torch.set_num_threads(2)


def model(key, shape=(6,), task="binary", classes=2, seed=12):
    torch.manual_seed(seed)
    return build_model(model_key=key, dataset="heart" if task == "binary" else "cifar10",
        input_shape=shape, task_type=task, num_classes=classes, seed=seed, privacy="private",
        n_random_features=7, max_input_dim=8)[0]


class ArchitectureTests(unittest.TestCase):
    def test_matched_trainable_counts_binary_and_multiclass(self):
        for shape, task, classes in [((6,), "binary", 2), ((3, 4, 4), "multiclass", 10)]:
            rvfl, frozen = model("rvfl", shape, task, classes), model("frozen_dense", shape, task, classes)
            self.assertEqual(count_parameters(rvfl)[0], count_parameters(frozen)[0])
            self.assertEqual(count_parameters(rvfl)[0], (min(math.prod(shape), 8)+7+1)*(1 if classes == 2 else classes))
            self.assertFalse(frozen.direct_link)

    def test_no_link_preserves_same_random_features(self):
        rvfl, elm = model("rvfl"), model("elm")
        self.assertTrue(torch.equal(rvfl.random_layer.weight, elm.random_layer.weight))
        self.assertTrue(torch.equal(rvfl.random_layer.bias, elm.random_layer.bias))
        self.assertEqual(rvfl.output_layer.in_features-elm.output_layer.in_features, 6)

    def test_logistic_projection_matches_rvfl(self):
        rvfl, linear, raw = [model(k, (12,)) for k in ["rvfl", "logistic", "logistic_raw"]]
        self.assertTrue(torch.equal(rvfl.input_reducer.weight, linear.input_reducer.weight))
        self.assertEqual(linear.output_layer.in_features, 8)
        self.assertEqual(raw.output_layer.in_features, 12)

    def test_frozen_features_do_not_receive_gradients(self):
        for key in ["rvfl", "elm", "frozen_dense"]:
            net = model(key)
            net(torch.randn(4, 6)).sum().backward()
            self.assertIsNone(net.random_layer.weight.grad)
            self.assertIsNotNone(net.output_layer.weight.grad)

    def test_nonprivate_sgd_momentum_is_applied(self):
        opt = _make_optimizer(model("logistic"), .01, "sgd", .01, .4)
        self.assertEqual(opt.param_groups[0]["momentum"], .4)


class DataTests(unittest.TestCase):
    def test_fixed_features_are_recordwise_and_bounded(self):
        for dataset in ["cardio", "heart"]:
            row = {k: (lo+hi)/2 for k, (lo, hi) in NUMERIC_BOUNDS[dataset].items()}
            row.update({k: vals[0] for k, vals in CATEGORIES[dataset].items()})
            altered = {k: np.nan for k in row}
            first, names = fixed_tabular_features(pd.DataFrame([row]), dataset)
            together, names2 = fixed_tabular_features(pd.DataFrame([row, altered]), dataset)
            np.testing.assert_array_equal(first[0], together[0])
            self.assertEqual(names, names2)
            self.assertTrue(np.isfinite(together).all())
            self.assertTrue((np.abs(together) <= 1).all())
            self.assertEqual(len(names), len(set(names)))

    def test_split_is_fixed_disjoint_and_training_seed_independent(self):
        a = load_revision_bundle("heart", ROOT / "data", 2026)
        b = load_revision_bundle("heart", ROOT / "data", 2026)
        self.assertEqual(a.dataset_summary, b.dataset_summary)
        partitions = [set(ds.indices) for ds in [a.train_dataset, a.validation_dataset, a.test_dataset]]
        self.assertFalse(partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2])

    def test_unweighted_criterion_does_not_read_training_labels(self):
        class Unreadable:
            def __iter__(self):
                raise AssertionError("Unaccounted label access")
        criterion = make_criterion("binary", Unreadable())
        self.assertIsNone(criterion.pos_weight)


class DPELMTests(unittest.TestCase):
    def setUp(self):
        self.data = TensorDataset(torch.randn(30, 4, generator=torch.Generator().manual_seed(10)), torch.arange(30) % 2)

    def test_closed_form_equals_normal_equations(self):
        net = DPELM((4,), 2, 3, 13)
        h = net.hidden(self.data.tensors[0]).numpy()
        y = np.eye(2)[self.data.tensors[1].numpy()]
        expected = np.linalg.solve(h.T @ h, h.T @ y)
        result = net.fit(self.data, epsilon=None)
        np.testing.assert_allclose(net.beta.numpy(), expected, rtol=1e-10)
        self.assertEqual(result["noise_scale_laplace"], 0)

    def test_published_scale_and_independent_entry_perturbations(self):
        net = DPELM((4,), 2, 3, 13)
        with patch("dpelm.laplace_noise", side_effect=lambda shape, scale, **kwargs: np.zeros(shape)) as noise:
            result = net.fit(self.data, epsilon=2, secure=False)
        self.assertEqual([call.args[0] for call in noise.call_args_list], [(3, 3), (3, 2)])
        self.assertTrue(all(call.args[1] == 7.5 for call in noise.call_args_list))
        self.assertEqual(result["delta"], 0)
        self.assertEqual(result["native_adjacency"], "replace_one")

    def test_pinv_fallback_only_on_singular_solve(self):
        net = DPELM((4,), 2, 3, 13)
        with patch("numpy.linalg.solve", side_effect=np.linalg.LinAlgError):
            result = net.fit(self.data)
        self.assertEqual(result["linear_solver"], "pinv_postprocessing_fallback")
        self.assertTrue(torch.isfinite(net.beta).all())

    def test_research_secure_noise_finite(self):
        noise = laplace_noise((100, 10), 3.0, secure=True, rng=None)
        self.assertTrue(np.isfinite(noise).all())
        self.assertGreater(np.std(noise), 0)

    def test_sensitivity_bounds_numerically(self):
        rng = np.random.default_rng(17)
        for width in [1, 3, 10]:
            for _ in range(30):
                h1, h2 = rng.random((2, width))
                y1, y2 = np.eye(4)[rng.integers(0, 4, 2)]
                t1, t2 = np.outer(h1, h1), np.outer(h2, h2)
                s1, s2 = np.outer(h1, y1), np.outer(h2, y2)
                self.assertLessEqual(np.abs(t1-t2).sum()+np.abs(s1-s2).sum(), width*(width+2))
                self.assertLessEqual(np.abs(t1).sum()+np.abs(s1).sum(), width*(width+1))


class PrivateTrainingTests(unittest.TestCase):
    def run_training(self, *, n=9, reference_size=9, batch_size=4, seed=4, epsilon=2, accountant="rdp", secure=False):
        torch.manual_seed(seed)
        net = model("rvfl")
        frozen = net.random_layer.weight.detach().clone()
        data = TensorDataset(torch.randn(n, 6), (torch.arange(n) % 2).float())
        class Unreadable:
            def __iter__(self):
                raise AssertionError("Validation/test must not be read during training")
        result = train_private_model(model=net, train_dataset=data, val_loader=Unreadable(), test_loader=Unreadable(),
            task_type="binary", epsilon=epsilon, delta=1e-5, max_grad_norm=1, epochs=1, lr=.001,
            seed=seed, batch_size=batch_size, num_workers=0, secure_mode=secure,
            rdp_alpha_mode="wide", accountant_name=accountant, public_reference_size=reference_size)
        self.assertTrue(torch.equal(frozen, net.random_layer.weight))
        self.assertTrue(torch.isfinite(net(torch.randn(3, 6))).all())
        self.assertFalse(hasattr(net, "autograd_grad_sample_hooks"))
        return result

    def test_exact_accounting_and_validation_not_accessed(self):
        result = self.run_training()
        self.assertEqual(result.steps_planned, 3)
        self.assertEqual(result.steps_completed, 3)
        self.assertEqual(result.sample_rate, 1/3)
        self.assertEqual(sum(item[2] for item in result.accountant_history), 3)
        self.assertLessEqual(result.epsilon_achieved, 2)
        self.assertAlmostEqual(result.epsilon_achieved, result.calibration_epsilon)

    def test_neighbor_sizes_keep_public_protocol_fixed(self):
        a = self.run_training(n=8, reference_size=9)
        b = self.run_training(n=9, reference_size=9)
        for field in ["sample_rate", "steps_planned", "expected_batch_size", "noise_multiplier"]:
            self.assertEqual(getattr(a, field), getattr(b, field))

    def test_empty_poisson_draws_are_counted(self):
        result = self.run_training(n=2, reference_size=16, batch_size=1)
        self.assertGreater(result.empty_batches, 0)
        self.assertEqual(result.steps_completed, 16)

    def test_low_epsilon_calibration(self):
        self.assertLessEqual(self.run_training(epsilon=.1).epsilon_achieved, .1)

    def test_prv_accountant_without_fallback(self):
        result = self.run_training(accountant="prv")
        self.assertEqual(result.accountant, "prv")
        self.assertLessEqual(result.epsilon_achieved, 2)

    def test_secure_opacus_path(self):
        result = self.run_training(secure=True)
        self.assertTrue(result.secure_rng_used)


class ProtocolTests(unittest.TestCase):
    def test_default_grid_has_unique_tasks(self):
        tasks = tasks_for(validate(parser().parse_args([])))
        self.assertEqual(len(tasks), 1540)
        self.assertEqual(len({t["task_id"] for t in tasks}), len(tasks))

    def test_resume_rejects_configuration_changes(self):
        with tempfile.TemporaryDirectory(prefix="dp_revision_test_") as output:
            args = ["--datasets", "heart", "--models", "rvfl", "--smoke", "--dry-run", "--development", "--output-dir", output]
            self.assertEqual(main(args), 0)
            self.assertEqual(main(args+["--resume"]), 0)
            with self.assertRaisesRegex(ValueError, "Resume refused"):
                main(args+["--resume", "--lr", ".002"])
            with self.assertRaisesRegex(ValueError, "not empty"):
                main(args)

    def test_statistics_do_not_invent_single_seed_variance(self):
        self.assertTrue(math.isnan(_interval([.8])[2]))
        self.assertTrue(math.isnan(_signflip(np.array([.1]))))
        self.assertEqual(_signflip(np.array([.1, .1])), .5)
        self.assertEqual(_signflip(np.zeros(5)), 1.0)

    def test_invalid_settings_rejected(self):
        for args in [["--epsilons", "0"], ["--lr", "nan"], ["--delta", "1"], ["--momentum", "2"]]:
            with self.assertRaises(ValueError):
                validate(parser().parse_args(args))


if __name__ == "__main__":
    unittest.main(verbosity=2)
