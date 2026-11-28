"""Opt-in, local-only MLflow experiment tracking."""

import json
from urllib.parse import urlsplit


def mlflow_enabled(config: dict) -> bool:
    if "tracking" not in config:
        return False
    tracking = config["tracking"]
    if not isinstance(tracking, dict):
        raise ValueError("training config 'tracking' must be an object")
    enabled = tracking.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("training config tracking.enabled must be a boolean")
    return enabled


def validate_local_tracking_uri(uri: str) -> str:
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in ("", "localhost")
        or not parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("MLflow tracking URI must be a local file: URI; remote tracking is disabled")
    return uri


def _flatten_params(value, prefix=""):
    params = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in {"path", "api_key", "token", "password", "secret"}:
                continue
            child_prefix = f"{prefix}.{key}" if prefix else key
            params.update(_flatten_params(child, child_prefix))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        params[prefix] = str(value)
    elif isinstance(value, (list, tuple)):
        params[prefix] = json.dumps(list(value))
    return params


def require_mlflow() -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow tracking requires the optional dependencies in requirements-integrations.txt"
        ) from exc


def log_mlflow_run(config: dict, result: dict, seed: int) -> None:
    tracking = config.get("tracking", {})
    uri = validate_local_tracking_uri(tracking.get("uri", "file:./mlruns"))
    require_mlflow()
    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(tracking.get("experiment", config["name"]))
    with mlflow.start_run(run_name=f"{config['name']}-seed{seed}"):
        params = _flatten_params(
            {key: config[key] for key in ("model", "train", "eval", "data") if key in config}
        )
        params["seed"] = str(seed)
        mlflow.log_params(params)
        metrics = {
            key: value
            for key, value in result.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if metrics:
            mlflow.log_metrics(metrics)
