import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from asura.integrations.arc_agi import (
    evaluate_tasks,
    format_arc_prompt,
    parse_grid_response,
)
from asura.tokenizer import ByteTokenizer


class FakeModel:
    cfg = SimpleNamespace(context_len=4096)

    def __init__(self, response):
        self.response = response
        self.last_prompt = None

    def generate(self, input_ids, max_new_tokens, temperature):
        self.last_prompt = ByteTokenizer().decode(input_ids[0].tolist())
        response_ids = torch.tensor(
            [ByteTokenizer().encode(self.response)],
            dtype=torch.long,
            device=input_ids.device,
        )
        return torch.cat((input_ids, response_ids), dim=1)


class ArcAgiIntegrationTests(unittest.TestCase):
    def test_prompt_includes_training_pairs_but_not_test_answer(self):
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[9]]}],
        }

        prompt = format_arc_prompt(task, task["test"][0]["input"])

        self.assertIn("Output: [[2]]", prompt)
        self.assertIn("--Test Input--\n[[3]]", prompt)
        self.assertNotIn("[[9]]", prompt)

    def test_parser_extracts_grid_from_wrapped_response(self):
        self.assertEqual(parse_grid_response("Answer:\n```json\n[[1,2],[3,4]]\n```"), [[1, 2], [3, 4]])

    def test_parser_rejects_ragged_or_out_of_range_grids(self):
        for response in ("[[1],[2,3]]", "[[10]]", "no grid"):
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_grid_response(response)

    def test_evaluation_writes_benchmark_compatible_submission(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tasks = root / "tasks"
            outputs = root / "submissions"
            tasks.mkdir()
            (tasks / "task-1.json").write_text(
                json.dumps({
                    "train": [{"input": [[1]], "output": [[2]]}],
                    "test": [{"input": [[3]], "output": [[4]]}],
                }),
                encoding="utf-8",
            )
            model = FakeModel("[[4]]")

            summary = evaluate_tasks(
                model,
                ByteTokenizer(),
                torch.device("cpu"),
                tasks,
                outputs,
                checkpoint_name="tiny",
            )

            submission = json.loads((outputs / "task-1.json").read_text(encoding="utf-8"))
            attempt = submission[0]["attempt_1"]
            self.assertEqual(attempt["answer"], [[4]])
            self.assertTrue(attempt["correct"])
            self.assertEqual(attempt["metadata"]["provider"], "asura-local")
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertNotIn("[[4]]", model.last_prompt)


if __name__ == "__main__":
    unittest.main()
