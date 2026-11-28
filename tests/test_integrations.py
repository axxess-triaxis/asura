import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zipfile import ZipFile

from asura.integrations.kaggle import (
    _extract_archive_safely,
    _replace_extracted_archive,
    validate_dataset_id,
)
from asura.integrations.langsmith import log_eval_metrics, validate_eval_metrics
from asura.integrations.ollama import DEFAULT_MODEL, OLLAMA_BASE_URL, chat_completion
from asura.integrations.tracking import (
    _flatten_params,
    mlflow_enabled,
    require_mlflow,
    validate_local_tracking_uri,
)


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

    def test_kaggle_replaces_stale_extracted_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = root / "extracted"
            destination.mkdir()
            (destination / "old.txt").write_text("stale", encoding="utf-8")
            archive = root / "dataset.zip"
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("new.txt", "fresh")

            _replace_extracted_archive(archive, destination)

            self.assertFalse((destination / "old.txt").exists())
            self.assertEqual((destination / "new.txt").read_text(encoding="utf-8"), "fresh")

    def test_kaggle_replacement_rejects_archive_without_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = root / "extracted"
            destination.mkdir()
            (destination / "old.txt").write_text("stale", encoding="utf-8")
            archive = root / "dataset.zip"
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("image.png", b"not text")

            with self.assertRaisesRegex(ValueError, "no .txt files"):
                _replace_extracted_archive(archive, destination)
            self.assertTrue((destination / "old.txt").exists())

    def test_mlflow_is_opt_in_and_local_only(self):
        self.assertFalse(mlflow_enabled({}))
        self.assertFalse(mlflow_enabled({"tracking": {"enabled": False}}))
        self.assertTrue(mlflow_enabled({"tracking": {"enabled": True}}))
        self.assertEqual(validate_local_tracking_uri("file:./mlruns"), "file:./mlruns")
        with self.assertRaises(ValueError):
            validate_local_tracking_uri("https://tracking.example")

    def test_mlflow_logs_list_hyperparameters_as_json(self):
        params = _flatten_params({"train": {"betas": [0.9, 0.95]}})
        self.assertEqual(params["train.betas"], "[0.9, 0.95]")

    def test_mlflow_dependency_is_checked_before_training(self):
        with patch.dict(sys.modules, {"mlflow": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements-integrations.txt"):
                require_mlflow()

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

    def test_ollama_requires_public_or_synthetic_data(self):
        with self.assertRaises(ValueError):
            chat_completion([{"role": "user", "content": "private text"}], data_policy="private")

    def test_ollama_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OLLAMA_API_KEY"):
                chat_completion([{"role": "user", "content": "public text"}], data_policy="public")

    def test_ollama_uses_openai_compatible_endpoint_and_model(self):
        create = Mock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="response"))]
        ))
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        openai_module = SimpleNamespace(OpenAI=Mock(return_value=client))
        messages = [{"role": "user", "content": "informal public text"}]

        with patch.dict(sys.modules, {"openai": openai_module}), patch.dict(
            os.environ, {"OLLAMA_API_KEY": "test-key"}, clear=True
        ):
            result = chat_completion(messages, data_policy="public")

        self.assertEqual(result, "response")
        openai_module.OpenAI.assert_called_once_with(
            api_key="test-key", base_url=OLLAMA_BASE_URL
        )
        create.assert_called_once_with(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=256,
            temperature=0.2,
        )


if __name__ == "__main__":
    unittest.main()
