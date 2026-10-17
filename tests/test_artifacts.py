"""Tests for immutable model bundle manifest resolution."""

import hashlib
import json
from pathlib import Path

import pytest

from src.model.artifacts import resolve_bundle_manifest


def write_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create one bundle and its latest-run manifest for resolver tests."""
    artifact_root = tmp_path / "artifacts"
    bundle_path = artifact_root / "bundles" / "run-123"
    bundle_path.mkdir(parents=True)
    model_path = bundle_path / "model.onnx"
    vectorizer_path = bundle_path / "vectorizer.joblib"
    model_path.write_bytes(b"onnx model")
    vectorizer_path.write_bytes(b"vectorizer")
    manifest = {
        "run_id": "run-123",
        "bundle_directory": "bundles/run-123",
        "artifacts": {
            "model": {
                "file": model_path.name,
                "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            },
            "vectorizer": {
                "file": vectorizer_path.name,
                "sha256": hashlib.sha256(
                    vectorizer_path.read_bytes(),
                ).hexdigest(),
            },
        },
    }
    latest_manifest = artifact_root / "latest.json"
    latest_manifest.parent.mkdir(parents=True, exist_ok=True)
    latest_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    version_manifest = bundle_path / "manifest.json"
    version_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    return latest_manifest, version_manifest, bundle_path


def test_resolve_latest_and_versioned_manifests(tmp_path: Path) -> None:
    """Resolve artifact pairs from both latest and immutable manifests."""
    latest_manifest, version_manifest, bundle_path = write_bundle(tmp_path)

    assert resolve_bundle_manifest(latest_manifest) == (
        bundle_path / "model.onnx",
        bundle_path / "vectorizer.joblib",
    )
    assert resolve_bundle_manifest(version_manifest) == (
        bundle_path / "model.onnx",
        bundle_path / "vectorizer.joblib",
    )


def test_resolve_manifest_rejects_tampered_artifact(tmp_path: Path) -> None:
    """Refuse a bundle when an artifact no longer matches its checksum."""
    latest_manifest, _, bundle_path = write_bundle(tmp_path)
    (bundle_path / "model.onnx").write_bytes(b"changed model")

    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_bundle_manifest(latest_manifest)


def test_resolve_manifest_rejects_bundle_path_traversal(
    tmp_path: Path,
) -> None:
    """Prevent latest manifests from resolving files outside artifact root."""
    latest_manifest, _, _ = write_bundle(tmp_path)
    manifest = json.loads(latest_manifest.read_text(encoding="utf-8"))
    manifest["bundle_directory"] = "../../outside"
    latest_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid model bundle manifest"):
        resolve_bundle_manifest(latest_manifest)
