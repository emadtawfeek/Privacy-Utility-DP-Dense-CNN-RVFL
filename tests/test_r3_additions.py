"""Focused checks for the Reviewer 3 diagnostics and transfer audit."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dpelm import DPELM
from models import build_model
from review_complexity import estimate_forward_macs, parameter_storage_bytes
from revision_training import train_private_model
from train_nonprivate import train_nonprivate_model
from review_audit import audit
from run_revisions import parser, validate


class R3DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.data = TensorDataset(torch.randn(8, 6), (torch.arange(8) % 2).float())

    def model(self):
        return build_model(model_key="logistic_raw", dataset="heart", input_shape=(6,),
                           task_type="binary", num_classes=2, seed=42, privacy="private")[0]

    def test_complexity_and_storage(self):
        self.assertEqual(estimate_forward_macs(self.model(), (6,)), 6)
        self.assertEqual(estimate_forward_macs(DPELM((6,), 2, 3), (6,)), 24)
        self.assertGreater(parameter_storage_bytes(self.model()), 0)

    def test_nonprivate_history_opt_in(self):
        result = train_nonprivate_model(model=self.model(), train_dataset=self.data,
            task_type="binary", batch_size=4, epochs=2, lr=.001, seed=42, num_workers=0,
            benchmark_diagnostics=True)
        self.assertEqual(len(result.diagnostic_history), 2)
        self.assertEqual(result.diagnostic_history[0]["examples_seen"], 8)

    def test_private_clipping_diagnostics_opt_in(self):
        result = train_private_model(model=self.model(), train_dataset=self.data,
            val_loader=None, test_loader=None, task_type="binary", epsilon=2,
            delta=1e-5, max_grad_norm=1, epochs=1, lr=.001, seed=42,
            batch_size=4, num_workers=0, secure_mode=False,
            benchmark_diagnostics=True, public_reference_size=8)
        self.assertEqual(len(result.diagnostic_history), 1)
        history = result.diagnostic_history[0]
        self.assertGreater(history["poisson_record_draws"], 0)
        self.assertGreaterEqual(history["clipping_fraction"], 0)
        self.assertLessEqual(history["clipping_fraction"], 1)
        self.assertLessEqual(result.epsilon_achieved, 2)

    def test_audit_marks_missing_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "manifest.json").write_text(json.dumps({
                "identity_sha256": "test", "tasks": [{"task_id": "missing", "dataset": "heart",
                    "seed": 42, "model_key": "rvfl", "privacy": "private",
                    "epsilon": 2, "dpelm_width": None}]}), encoding="utf-8")
            result = audit(root, root / "audit")
            self.assertFalse(result["complete"])
            self.assertEqual(result["status_counts"]["missing"], 1)

    def test_diagnostics_reject_unbundled_data(self):
        with self.assertRaisesRegex(ValueError, "bundled public data"):
            validate(parser().parse_args(["--benchmark-diagnostics", "--data-dir", "C:/unverified"] ))


if __name__ == "__main__":
    unittest.main()
