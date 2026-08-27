"""MuJoCo load boundary for offline preview only."""

from __future__ import annotations

import mujoco

from galbot_motion_studio.model.manifest import CANONICAL_MANIFEST, ModelManifest, verify_model


def load_verified_fixed_base_model(manifest: ModelManifest = CANONICAL_MANIFEST) -> mujoco.MjModel:
    verify_model(manifest)
    return mujoco.MjModel.from_xml_path(str(manifest.fixed_mjcf))

