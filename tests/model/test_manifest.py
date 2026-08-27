from pathlib import Path

import pytest

from galbot_motion_studio.model.manifest import (
    CANONICAL_MANIFEST,
    ModelManifest,
    ModelVerificationError,
    verify_model,
)


def test_repository_relative_model_is_verified() -> None:
    verify_model()
    assert CANONICAL_MANIFEST.model_root.is_relative_to(Path.cwd())


def test_changed_hash_fails_closed() -> None:
    manifest = ModelManifest(
        model_root=CANONICAL_MANIFEST.model_root,
        package_tree_sha256=CANONICAL_MANIFEST.package_tree_sha256,
        floating_urdf_sha256="0" * 64,
        fixed_urdf_sha256=CANONICAL_MANIFEST.fixed_urdf_sha256,
        fixed_mjcf_sha256=CANONICAL_MANIFEST.fixed_mjcf_sha256,
    )
    with pytest.raises(ModelVerificationError, match="hash mismatch"):
        verify_model(manifest)


def test_changed_package_tree_fails_before_model_load() -> None:
    manifest = ModelManifest(
        model_root=CANONICAL_MANIFEST.model_root,
        package_tree_sha256="0" * 64,
        floating_urdf_sha256=CANONICAL_MANIFEST.floating_urdf_sha256,
        fixed_urdf_sha256=CANONICAL_MANIFEST.fixed_urdf_sha256,
        fixed_mjcf_sha256=CANONICAL_MANIFEST.fixed_mjcf_sha256,
    )
    with pytest.raises(ModelVerificationError, match="package-tree"):
        verify_model(manifest)
