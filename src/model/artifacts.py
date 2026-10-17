"""Read and validate the shared model bundle manifest contract."""

import hashlib
import json
from pathlib import Path


def resolve_bundle_manifest(manifest_path: Path) -> tuple[Path, Path]:
    """Resolve and checksum-check the ONNX/vectorizer pair in one manifest."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = manifest["run_id"]
        bundle_directory = manifest["bundle_directory"]
        if manifest_path.parent.name == run_id:
            bundle_path = manifest_path.parent
        else:
            bundle_path = (
                manifest_path.parent / bundle_directory
            ).resolve()
            bundle_path.relative_to(manifest_path.parent.resolve())
        model_path = _resolve_artifact(bundle_path, manifest["artifacts"]["model"])
        vectorizer_path = _resolve_artifact(
            bundle_path,
            manifest["artifacts"]["vectorizer"],
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid model bundle manifest {manifest_path}: {error}",
        ) from error
    return model_path, vectorizer_path


def _resolve_artifact(bundle_path: Path, artifact: dict[str, str]) -> Path:
    """Resolve one artifact filename and reject traversal or checksum mismatch."""
    filename = artifact["file"]
    if Path(filename).name != filename:
        raise ValueError("Bundle artifact paths must be simple filenames")
    artifact_path = bundle_path / filename
    expected_digest = artifact["sha256"]
    if _sha256(artifact_path) != expected_digest:
        raise ValueError(f"Bundle artifact checksum mismatch: {filename}")
    return artifact_path


def _sha256(path: Path) -> str:
    """Compute a streaming SHA-256 digest for an artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
