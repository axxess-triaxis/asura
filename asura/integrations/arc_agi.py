"""Run local Asura checkpoints against ARC-AGI task files."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from ..device import get_device
from ..model import Asura, AsuraConfig
from ..tokenizer import ByteTokenizer


def _validate_grid(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty grid")
    if any(not isinstance(row, list) or not row for row in value):
        raise ValueError(f"{label} must contain non-empty rows")
    width = len(value[0])
    if any(len(row) != width for row in value):
        raise ValueError(f"{label} must be rectangular")
    if any(type(cell) is not int or not 0 <= cell <= 9 for row in value for cell in row):
        raise ValueError(f"{label} cells must be integers from 0 to 9")
    return value


def format_arc_prompt(task, test_input):
    """Format ARC examples without including a test pair's expected output."""
    examples = []
    for index, pair in enumerate(task.get("train", []), start=1):
        if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
            raise ValueError(f"training pair {index} must include input and output grids")
        input_grid = _validate_grid(pair["input"], f"training pair {index} input")
        output_grid = _validate_grid(pair["output"], f"training pair {index} output")
        examples.append(
            f"Input: {json.dumps(input_grid, separators=(',', ':'))}\n"
            f"Output: {json.dumps(output_grid, separators=(',', ':'))}"
        )
    if not examples:
        raise ValueError("ARC task must contain at least one training pair")
    test_grid = _validate_grid(test_input, "test input")
    return (
        "Infer the grid transformation from the training examples. "
        "Respond with only the output grid as a JSON array of arrays.\n"
        "--Training Examples--\n"
        + "\n".join(examples)
        + "\n--Test Input--\n"
        + json.dumps(test_grid, separators=(",", ":"))
        + "\nYour response:"
    )


def parse_grid_response(text):
    """Find the first rectangular 0-9 grid in a model response."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "[":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
            return _validate_grid(candidate, "generated output")
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError("response did not contain a valid rectangular grid")


def load_asura_checkpoint(checkpoint_path, device="auto"):
    target_device = get_device() if device == "auto" else torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
    if not isinstance(checkpoint, dict) or not {"model_cfg", "model"} <= checkpoint.keys():
        raise ValueError("checkpoint must contain model_cfg and model state")
    model = Asura(AsuraConfig(**checkpoint["model_cfg"])).to(target_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, ByteTokenizer(), target_device


def _attempt_record(
    answer,
    raw_response,
    task_id,
    pair_index,
    checkpoint_name,
    prompt_tokens,
    output_tokens,
    started_at,
    ended_at,
    correct,
):
    return {
        "answer": answer,
        "metadata": {
            "model": checkpoint_name,
            "provider": "asura-local",
            "start_timestamp": started_at.isoformat(),
            "end_timestamp": ended_at.isoformat(),
            "choices": [
                {"index": 0, "message": {"role": "user", "content": f"ARC-AGI task {task_id}, pair {pair_index}"}},
                {"index": 1, "message": {"role": "assistant", "content": raw_response}},
            ],
            "kwargs": {},
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens,
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": output_tokens,
                    "rejected_prediction_tokens": 0,
                },
            },
            "cost": {
                "prompt_cost": 0.0,
                "completion_cost": 0.0,
                "reasoning_cost": None,
                "total_cost": 0.0,
            },
            "task_id": task_id,
            "pair_index": pair_index,
            "test_id": checkpoint_name,
        },
        "correct": correct,
    }


def evaluate_tasks(
    model,
    tokenizer,
    device,
    task_dir,
    output_dir,
    checkpoint_name="asura",
    max_new_tokens=512,
    task_limit=None,
):
    task_root = Path(task_dir)
    output_root = Path(output_dir)
    task_files = sorted(task_root.glob("*.json"))
    if not task_files:
        raise FileNotFoundError(f"no ARC task JSON files found in {task_root}")
    if task_limit is not None:
        if task_limit < 1:
            raise ValueError("task_limit must be positive")
        task_files = task_files[:task_limit]
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    output_root.mkdir(parents=True, exist_ok=True)
    total_pairs = scored_pairs = correct_pairs = invalid_outputs = 0
    for task_file in task_files:
        task = json.loads(task_file.read_text(encoding="utf-8"))
        if not isinstance(task, dict) or not isinstance(task.get("test"), list):
            raise ValueError(f"{task_file} must contain a test-pair list")
        pair_submissions = []
        for pair_index, test_pair in enumerate(task["test"]):
            if not isinstance(test_pair, dict) or "input" not in test_pair:
                raise ValueError(f"{task_file} test pair {pair_index} is missing its input grid")
            prompt = format_arc_prompt(task, test_pair["input"])
            prompt_ids = tokenizer.encode(prompt)
            if len(prompt_ids) > model.cfg.context_len:
                raise ValueError(
                    f"{task_file.name} pair {pair_index} prompt is {len(prompt_ids)} bytes, "
                    f"exceeding model context_len={model.cfg.context_len}; increase the model context"
                )
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            started_at = datetime.now(timezone.utc)
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0,
            )
            ended_at = datetime.now(timezone.utc)
            output_ids = generated_ids[0, len(prompt_ids) :].tolist()
            raw_response = tokenizer.decode(output_ids)
            try:
                answer = parse_grid_response(raw_response)
            except ValueError:
                answer = raw_response.strip()
                invalid_outputs += 1

            expected = test_pair.get("output")
            correct = None
            if expected is not None:
                expected = _validate_grid(expected, f"{task_file.name} expected output")
                scored_pairs += 1
                correct = isinstance(answer, list) and answer == expected
                correct_pairs += int(correct)
            pair_submissions.append(
                {"attempt_1": _attempt_record(
                    answer,
                    raw_response,
                    task_file.stem,
                    pair_index,
                    checkpoint_name,
                    len(prompt_ids),
                    len(output_ids),
                    started_at,
                    ended_at,
                    correct,
                )}
            )
            total_pairs += 1

        (output_root / task_file.name).write_text(
            json.dumps(pair_submissions, indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "tasks": len(task_files),
        "pairs": total_pairs,
        "scored_pairs": scored_pairs,
        "correct_pairs": correct_pairs,
        "accuracy": correct_pairs / scored_pairs if scored_pairs else None,
        "invalid_outputs": invalid_outputs,
        "submission_dir": str(output_root),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate an Asura checkpoint on ARC-AGI tasks.")
    parser.add_argument("--checkpoint", required=True, help="Asura .pt checkpoint")
    parser.add_argument("--tasks-dir", required=True, help="Directory of ARC task JSON files")
    parser.add_argument("--output-dir", required=True, help="ARC benchmark-compatible submission directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch device such as cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--task-limit", type=int, help="Evaluate only the first N task files")
    args = parser.parse_args()

    model, tokenizer, device = load_asura_checkpoint(args.checkpoint, args.device)
    summary = evaluate_tasks(
        model,
        tokenizer,
        device,
        args.tasks_dir,
        args.output_dir,
        checkpoint_name=Path(args.checkpoint).stem,
        max_new_tokens=args.max_new_tokens,
        task_limit=args.task_limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
