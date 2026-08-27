import mujoco
import numpy as np

from galbot_motion_studio.model.loader import load_verified_fixed_base_model


def test_fixed_base_model_loads_with_expected_geometry_inventory() -> None:
    model = load_verified_fixed_base_model()
    assert (model.nq, model.nv, model.nu, model.ngeom) == (33, 33, 23, 268)
    disabled_collision_geoms = (model.geom_contype == 0) & (model.geom_conaffinity == 1)
    assert int(disabled_collision_geoms.sum()) == 184


def test_torso_local_y_is_gravity_aligned() -> None:
    """The benchmark's "height" coordinate must not drift to torso-forward Z."""
    model = load_verified_fixed_base_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_base_link")
    assert torso >= 0
    torso_rotation = np.array(data.xmat[torso]).reshape(3, 3)
    np.testing.assert_allclose(torso_rotation[:, 1], (0.0, 0.0, 1.0), atol=2e-3)
