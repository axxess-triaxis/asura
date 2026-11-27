"""Opt-in download of public Kaggle text datasets for Asura training."""

import argparse
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_dataset_id(dataset_id: str) -> str:
    """Require an explicit Kaggle owner/dataset slug."""
    if not isinstance(dataset_id, str) or not _DATASET_ID.fullmatch(dataset_id):
        raise ValueError("dataset must be an explicit Kaggle owner/dataset identifier")
    return dataset_id


def _extract_archive_safely(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            member = item.filename.replace("\\", "/")
            member_path = PurePosixPath(member)
            if (
                not member_path.parts
                or member_path.is_absolute()
                or ".." in member_path.parts
                or ":" in member_path.parts[0]
            ):
                raise ValueError(f"unsafe path in Kaggle dataset archive: {item.filename!r}")
            target = (destination / Path(*member_path.parts)).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"unsafe path in Kaggle dataset archive: {item.filename!r}")
        zipped.extractall(destination)


def download_public_dataset(dataset_id: str, output_dir: str | Path) -> Path:
    """Download and safely extract a Kaggle dataset after verifying it is public."""
    dataset_id = validate_dataset_id(dataset_id)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle integration requires the optional dependencies in requirements-integrations.txt"
        ) from exc

    api = KaggleApi()
    api.authenticate()
    metadata = api.dataset_view(dataset_id)
    if getattr(metadata, "isPrivate", None) is not False:
        raise ValueError(f"Kaggle did not confirm that {dataset_id!r} is public; refusing download")

    dataset_dir = Path(output_dir).expanduser().resolve() / dataset_id.split("/", 1)[1]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(dataset_id, path=str(dataset_dir), quiet=False, unzip=False)
    archives = sorted(dataset_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not archives:
        raise FileNotFoundError(f"Kaggle API did not produce a dataset archive in {dataset_dir}")

    extracted = dataset_dir / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    _extract_archive_safely(archives[0], extracted)
    if not any(extracted.rglob("*.txt")):
        raise ValueError(
            f"{dataset_id!r} contains no .txt files; convert public text data to UTF-8 .txt files first"
        )
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="public Kaggle owner/dataset identifier")
    parser.add_argument("--config", required=True, help="Asura JSON training config")
    parser.add_argument("--output-dir", default="data/kaggle", help="download destination")
    parser.add_argument("--out-root", default="runs", help="training output root")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_dir = download_public_dataset(args.dataset, args.output_dir)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config["data"] = {"type": "text", "path": str(data_dir)}
    config["name"] = f"{config['name']}-kaggle"
    from ..train import train

    result = train(config, seed=args.seed, out_root=args.out_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
