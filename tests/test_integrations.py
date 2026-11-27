import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from asura.integrations.kaggle import _extract_archive_safely, validate_dataset_id
from asura.integrations.langsmith import log_eval_metrics, validate_eval_metrics
from asura.integrations.tracking import mlflow_enabled, validate_local_tracking_uri


class IntegrationSafetyTests(unittest.TestCase):
    def test_kaggle_requires_explicit_owner_and_dataset(self):
        self.assertEqual(validate_dataset_id("owner/public-text"), "owner/public-text")
        for dataset_id in ("", "../private", "owner", "owner/../../private", "https://kaggle.com/data"):
            with self.subTest(dataset_id=dataset_id), self.assertRaises(ValueError):
                validate_dataset_id(dataset_id)

    def test_kaggle_archive_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "dataset.zip"
            destination = root / "extracted"
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("../outside.txt", "not allowed")
            with self.assertRaises(ValueError):
                _extract_archive_safely(archive, destination)
            self.assertFalse((root / "outside.txt").exists())

    def test_mlflow_is_opt_in_and_local_only(self):
        self.assertFalse(mlflow_enabled({}))
        self.assertFalse(mlflow_enabled({"tracking": {"enabled": False}}))
        self.assertTrue(mlflow_enabled({"tracking": {"enabled": True}}))
        self.assertEqual(validate_local_tracking_uri("file:./mlruns"), "file:./mlruns")
        with self.assertRaises(ValueError):
            validate_local_tracking_uri("https://tracking.example")

    def test_langsmith_requires_public_or_synthetic_numeric_metrics(self):
        self.assertEqual(
            validate_eval_metrics({"accuracy": 0.5}, "synthetic"),
            {"accuracy": 0.5},
        )
        for policy in ("private", "unknown"):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                validate_eval_metrics({"accuracy": 0.5}, policy)
        with self.assertRaises(ValueError):
            validate_eval_metrics({"prompt": "private text"}, "public")
        with self.assertRaises(ValueError):
            validate_eval_metrics({"private prompt content": 1.0}, "public")

    def test_langsmith_does_not_send_without_explicit_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(log_eval_metrics({"accuracy": 0.5}, data_policy="synthetic"))


if __name__ == "__main__":
    unittest.main()
