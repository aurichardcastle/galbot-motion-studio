"""Verify the exact model package before any simulator load."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isclose, pi, sqrt
from pathlib import Path
from re import S, search
from xml.etree import ElementTree


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ASSET_ROOT = Path(__file__).resolve().parents[1] / "_assets"
MODEL_ROOT = (
    PACKAGE_ASSET_ROOT / "galbot_one_golf_description"
    if (PACKAGE_ASSET_ROOT / "galbot_one_golf_description").is_dir()
    else PROJECT_ROOT / "third_party/galbot_one_golf_description"
)


@dataclass(frozen=True)
class ModelManifest:
    model_root: Path
    package_tree_sha256: str
    floating_urdf_sha256: str
    fixed_urdf_sha256: str
    fixed_mjcf_sha256: str

    @property
    def floating_urdf(self) -> Path:
        return self.model_root / "urdf/galbot_one_golf.urdf"

    @property
    def fixed_urdf(self) -> Path:
        return self.model_root / "urdf/galbot_one_golf_fixed_base.urdf"

    @property
    def fixed_mjcf(self) -> Path:
        return self.model_root / "mjcf/galbot_one_golf_fixed_base.xml"

    @property
    def tcp_validator(self) -> Path:
        return self.model_root / "scripts/validate_tcp_frames.py"


CANONICAL_MANIFEST = ModelManifest(
    model_root=MODEL_ROOT,
    package_tree_sha256="ab3707a6e3ff6027a3ed2a9d7bcbba7fbeda6e880a02ea664b4bea43ace9af10",
    floating_urdf_sha256="8e7722191495d8b96ca8d64cc5dff918fb9cd673c74b70728641856d757e3b48",
    fixed_urdf_sha256="b340d2491fcbb53ec19bf9df3779f12b10585e7130847386f18cb6f476bbdfce",
    fixed_mjcf_sha256="6a92e1bca62507c0d5e3ffbfaddf9fff50d8e1bc2b0a3bacd2c320f36159eea7",
)


class ModelVerificationError(RuntimeError):
    """Raised when the pinned assets do not match the approved Day 3 package."""


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Hash every vendored asset and path, including collision meshes and validator source."""
    digest = sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    if not files:
        raise ModelVerificationError(f"model package has no files: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def verify_model(manifest: ModelManifest = CANONICAL_MANIFEST) -> None:
    if sha256_tree(manifest.model_root) != manifest.package_tree_sha256:
        raise ModelVerificationError("model package-tree hash mismatch")
    expected = {
        manifest.floating_urdf: manifest.floating_urdf_sha256,
        manifest.fixed_urdf: manifest.fixed_urdf_sha256,
        manifest.fixed_mjcf: manifest.fixed_mjcf_sha256,
    }
    for path, approved_hash in expected.items():
        if not path.is_file():
            raise ModelVerificationError(f"required model asset is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != approved_hash:
            raise ModelVerificationError(
                f"model hash mismatch for {path.name}: expected {approved_hash}, got {actual_hash}"
            )
    validate_tcp_frames(manifest.model_root)


def _numbers(raw: str, count: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw.replace(",", " ").split())
    if len(values) != count:
        raise ModelVerificationError(f"expected {count} transform values, got {values}")
    return values


def _close(actual: tuple[float, ...], expected: tuple[float, ...]) -> bool:
    return len(actual) == len(expected) and all(
        isclose(left, right, abs_tol=1e-6) for left, right in zip(actual, expected, strict=True)
    )


def validate_tcp_frames(model_root: Path) -> None:
    """Verify transforms in-process; never execute model-supplied scripts during preflight."""
    expected_xyz = (0.13996, 0.0, 0.0)
    expected_rpy = (-pi / 2, 0.0, 0.0)
    expected_quat = (sqrt(0.5), -sqrt(0.5), 0.0, 0.0)
    for path in (model_root / "urdf/galbot_one_golf.urdf", model_root / "urdf/galbot_one_golf_fixed_base.urdf"):
        root = ElementTree.parse(path).getroot()
        for link_name in ("left_gripper_tcp_link", "right_gripper_tcp_link"):
            joints = []
            for joint in root.findall("joint"):
                child = joint.find("child")
                if child is not None and child.get("link") == link_name:
                    joints.append(joint)
            if len(joints) != 1:
                raise ModelVerificationError(f"{path.name}: missing unique {link_name} joint")
            origin = joints[0].find("origin")
            if origin is None or not _close(_numbers(origin.get("xyz", ""), 3), expected_xyz) or not _close(_numbers(origin.get("rpy", ""), 3), expected_rpy):
                raise ModelVerificationError(f"{path.name}: TCP transform mismatch for {link_name}")
    xacro = ElementTree.parse(model_root / "xacro/components/galbot_gripper.xacro").getroot()
    joint = next((entry for entry in xacro.iter("joint") if entry.get("name") == "${prefix}gripper_tcp_joint"), None)
    origin = None if joint is None else joint.find("origin")
    if origin is None or not _close(_numbers(origin.get("xyz", ""), 3), expected_xyz) or not _close(_numbers(origin.get("rpy", ""), 3), expected_rpy):
        raise ModelVerificationError("xacro TCP transform mismatch")
    for path in (
        model_root / "mjcf/galbot_one_golf.xml",
        model_root / "mjcf/galbot_one_golf_fixed_base.xml",
        model_root / "mjcf/galbot_one_golf_planar_base.xml",
    ):
        root = ElementTree.parse(path).getroot()
        for link_name in ("left_gripper_tcp_link", "right_gripper_tcp_link"):
            bodies = root.findall(f".//body[@name='{link_name}']")
            if len(bodies) != 1 or not _close(_numbers(bodies[0].get("pos", ""), 3), expected_xyz):
                raise ModelVerificationError(f"{path.name}: TCP position mismatch for {link_name}")
            quat = _numbers(bodies[0].get("quat", ""), 4)
            if not isclose(abs(sum(left * right for left, right in zip(quat, expected_quat, strict=True))), 1.0, abs_tol=1e-4):
                raise ModelVerificationError(f"{path.name}: TCP orientation mismatch for {link_name}")
    usd = (model_root / "usd/payloads/base.usda").read_text(encoding="utf-8")
    for link_name in ("left_gripper_tcp_link", "right_gripper_tcp_link"):
        block = search(rf'def Xform "{link_name}"\s*\{{(?P<body>.*?)\n\s*\}}', usd, S)
        if block is None:
            raise ModelVerificationError(f"USD missing {link_name}")
        translate = search(r"xformOp:translate\s*=\s*\(([^)]+)\)", block.group("body"))
        orientation = search(r"xformOp:orient\s*=\s*\(([^)]+)\)", block.group("body"))
        if translate is None or orientation is None or not _close(_numbers(translate.group(1), 3), expected_xyz):
            raise ModelVerificationError(f"USD TCP position mismatch for {link_name}")
        quat = _numbers(orientation.group(1), 4)
        if not isclose(abs(sum(left * right for left, right in zip(quat, expected_quat, strict=True))), 1.0, abs_tol=1e-4):
            raise ModelVerificationError(f"USD TCP orientation mismatch for {link_name}")
