"""Explicit, metrics-only LangSmith logging for synthetic or public evaluations."""

import math
import os
import re

_METRIC_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.~-]{0,99}$")


def validate_eval_metrics(metrics: dict, data_policy: str) -> dict[str, float]:
    if not isinstance(data_policy, str) or data_policy not in {"synthetic", "public"}:
        raise ValueError("LangSmith logging requires data_policy='synthetic' or 'public'")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics must be a non-empty mapping of numeric evaluation metrics")

    safe_metrics = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not _METRIC_NAME.fullmatch(name):
            raise ValueError("metric names must be short identifiers, not arbitrary text")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be a finite number")
        safe_metrics[name] = float(value)
    return safe_metrics


def log_eval_metrics(metrics: dict, *, data_policy: str) -> bool:
    """Send only numeric metrics when explicitly enabled; never send prompts or samples."""
    safe_metrics = validate_eval_metrics(metrics, data_policy)
    if os.environ.get("ASURA_LANGSMITH_TRACE") != "1":
        return False
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise RuntimeError("Set LANGSMITH_API_KEY in your environment or secret store; never in source control")

    try:
        from langsmith.run_trees import RunTree
    except ImportError as exc:
        raise RuntimeError(
            "LangSmith logging requires the optional dependencies in requirements-integrations.txt"
        ) from exc

    run = RunTree(
        name="asura-evaluation-metrics",
        run_type="chain",
        inputs={"data_policy": data_policy, "metrics": safe_metrics},
    )
    run.post()
    run.end(outputs={"metrics": safe_metrics})
    run.patch()
    return True
