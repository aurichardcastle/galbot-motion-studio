import sys


def test_core_imports_without_galbot_sdk() -> None:
    import galbot_motion_studio.contracts.core  # noqa: F401
    import galbot_motion_studio.model.manifest  # noqa: F401
    import galbot_motion_studio.ports.command  # noqa: F401

    assert "galbot_sdk" not in sys.modules


def test_runnable_demo_imports_without_galbot_sdk() -> None:
    import galbot_motion_studio.cli  # noqa: F401
    import galbot_motion_studio.pipeline  # noqa: F401
    import galbot_motion_studio.safety.supervisor  # noqa: F401

    assert "galbot_sdk" not in sys.modules
