"""Display overlay tests.

The overlay's whole job is that a stopped robot never looks like a tracking one.
These tests assert on what is actually drawn -- the text handed to OpenCV and the
pixels that change -- rather than on internal state, because the failure that
cost the operator hours was a canvas that looked fine while the robot was frozen.
"""

import numpy as np
import pytest

from galbot_motion_studio.adapters.mujoco_preview import MujocoPreviewSink
from galbot_motion_studio.display import (
    UPPER_BODY_JOINTS,
    CalibrationStatus,
    StudioDisplay,
)
from galbot_motion_studio.model.loader import load_verified_fixed_base_model
from galbot_motion_studio.safety.clearance import HOME_QPOS
from galbot_motion_studio.synthetic import synthetic_observation
from galbot_motion_studio.vision.freshness import ControlGroup

LEFT_ARM = str(ControlGroup.LEFT_ARM)
RIGHT_ARM = str(ControlGroup.RIGHT_ARM)
HEAD = str(ControlGroup.HEAD)


class RecordingCv2:
    """Real cv2, with every drawn string recorded.

    Asserting on pixels alone cannot tell "LEFT ARM HELD" from any other smear
    of light pixels, and the point of this overlay is the words.
    """

    def __init__(self, real):
        self._real = real
        self.texts: list[str] = []
        #: text -> the colours it was drawn in this session. A state that is only
        #: ever asserted by its words can be reintroduced in the wrong colour.
        self.colours: dict[str, list[tuple]] = {}

    def putText(self, image, text, origin, font, scale, color, thickness, *args, **kwargs):
        self.texts.append(text)
        self.colours.setdefault(text, []).append(tuple(color))
        return self._real.putText(
            image, text, origin, font, scale, color, thickness, *args, **kwargs
        )

    def __getattr__(self, name):
        return getattr(self._real, name)

    def said(self, fragment: str) -> bool:
        return any(fragment in text for text in self.texts)


@pytest.fixture(scope="module")
def model():
    return load_verified_fixed_base_model()


@pytest.fixture
def studio(model):
    display = StudioDisplay(model, width=640, height=480)
    display._cv2 = RecordingCv2(display._cv2)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    try:
        yield display, sink, display._cv2
    finally:
        display.close()


@pytest.fixture
def camera():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def observation():
    return synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)


def joint_pose(offset: float = 0.0) -> dict[str, float]:
    return {name: offset for name, _label, _group in UPPER_BODY_JOINTS}


# --- per-limb hold state -------------------------------------------------


def test_a_held_limb_is_drawn_differently_from_a_tracked_one(
    studio, camera, observation
) -> None:
    """The core promise: a frozen left arm must not look like a tracked one."""
    display, sink, _ = studio
    tracked = display.render(
        camera, sink.data, status="ALLOW", observation=observation, held_groups=()
    )
    held = display.render(
        camera,
        sink.data,
        status="ALLOW",
        observation=observation,
        held_groups=frozenset({LEFT_ARM}),
    )
    assert not np.array_equal(tracked[:, :640], held[:, :640])


def test_closing_the_preview_window_requests_a_clean_stop() -> None:
    """A window-manager close must work as reliably as q/Esc at the demo."""

    class ClosedWindowCv2:
        WND_PROP_VISIBLE = 5

        def imshow(self, _window, _canvas) -> None:
            return None

        def waitKey(self, _delay) -> int:
            return -1

        def getWindowProperty(self, _window, _property) -> float:
            return 0.0

    display = object.__new__(StudioDisplay)
    display._cv2 = ClosedWindowCv2()
    display._window = "test preview"
    display._window_opened = False

    assert not display.show_canvas(np.zeros((1, 1, 3), dtype=np.uint8))


def test_a_held_limb_is_named_on_the_camera_and_in_the_chip_row(
    studio, camera, observation
) -> None:
    display, sink, cv2_spy = studio
    display.render(
        camera,
        sink.data,
        status="ALLOW",
        observation=observation,
        held_groups=frozenset({LEFT_ARM}),
    )
    assert cv2_spy.said("LEFT ARM HELD"), cv2_spy.texts
    assert cv2_spy.said("FROZEN: LEFT ARM"), cv2_spy.texts
    assert not cv2_spy.said("RIGHT ARM HELD"), cv2_spy.texts


def test_allow_never_reads_as_clean_while_a_limb_is_frozen(
    studio, camera, observation
) -> None:
    """The exact regression: a green ALLOW beside a silently frozen limb.

    A 1486-frame session showed ALLOW while a limb was held and the operator had
    no way to see it.  The headline must carry the hold, and must not be green.
    """
    display, sink, cv2_spy = studio
    canvas = display.render(
        camera,
        sink.data,
        status="ALLOW",
        observation=observation,
        held_groups=frozenset({LEFT_ARM}),
    )
    assert cv2_spy.said("ALLOW - 1 LIMB HELD"), cv2_spy.texts
    assert cv2_spy.said("FROZEN: LEFT ARM"), cv2_spy.texts
    clean = display.render(camera, sink.data, status="ALLOW", observation=observation)
    # A clean frame says ALLOW and nothing else; the hold words must be gone, and
    # the exit hint stays on screen either way.  The two are separate elements in
    # the app bar now, so they are asserted separately rather than as one
    # concatenated headline string.
    assert cv2_spy.said("ALLOW"), cv2_spy.texts
    assert cv2_spy.said("q/Esc exits"), cv2_spy.texts
    clean_texts = [t for t in cv2_spy.texts if t.startswith("ALLOW")]
    assert clean_texts[-1] == "ALLOW", clean_texts
    # Different headline colour, not merely different words.
    assert not np.array_equal(canvas[16:40, 16:600], clean[16:40, 16:600])


def test_allow_never_reads_as_clean_while_the_head_is_saturated(
    studio, camera, observation
) -> None:
    display, sink, cv2_spy = studio
    display.render(
        camera,
        sink.data,
        status="ALLOW",
        observation=observation,
        saturated_groups=frozenset({HEAD}),
        reasons=("head: HEAD_SOFT_LIMIT",),
    )
    assert cv2_spy.said("ALLOW - HEAD SATURATED"), cv2_spy.texts
    assert cv2_spy.said("LIMITED: HEAD"), cv2_spy.texts
    assert cv2_spy.said("HEAD_SOFT_LIMIT"), cv2_spy.texts


def test_every_control_group_can_be_held_and_is_pluralised(
    studio, camera, observation
) -> None:
    display, sink, cv2_spy = studio
    display.render(
        camera,
        sink.data,
        status="HOLD",
        observation=observation,
        held_groups=frozenset({HEAD, LEFT_ARM, RIGHT_ARM}),
    )
    assert cv2_spy.said("3 LIMBS HELD"), cv2_spy.texts
    assert cv2_spy.said("FROZEN: HEAD, LEFT ARM, RIGHT ARM"), cv2_spy.texts


# --- rejection reason ----------------------------------------------------


def test_the_specific_rejection_reason_is_on_the_canvas(
    studio, camera, observation
) -> None:
    """IDENTITY_NOT_STABLE in a terminal behind the window is no message at all."""
    display, sink, cv2_spy = studio
    display.render(
        camera,
        sink.data,
        status="HOLD",
        observation=observation,
        held_groups=frozenset({HEAD, LEFT_ARM, RIGHT_ARM}),
        reasons=("observation: IDENTITY_NOT_STABLE",),
    )
    assert cv2_spy.said("IDENTITY_NOT_STABLE"), cv2_spy.texts
    assert cv2_spy.said("WHY:"), cv2_spy.texts


def test_extra_reasons_are_counted_rather_than_dropped(
    studio, camera, observation
) -> None:
    display, sink, cv2_spy = studio
    display.render(
        camera,
        sink.data,
        status="HOLD",
        observation=observation,
        held_groups=frozenset({LEFT_ARM}),
        reasons=("observation: LOW_CONFIDENCE", "clearance: rejected", "third"),
    )
    assert cv2_spy.said("LOW_CONFIDENCE"), cv2_spy.texts
    assert cv2_spy.said("(+2 more)"), cv2_spy.texts


def test_a_hold_with_no_reason_string_still_explains_itself(
    studio, camera, observation
) -> None:
    display, sink, cv2_spy = studio
    display.render(
        camera,
        sink.data,
        status="HOLD",
        observation=observation,
        held_groups=frozenset({LEFT_ARM}),
        reasons=(),
    )
    assert cv2_spy.said("WHY:"), cv2_spy.texts


def test_no_reason_banner_when_nothing_is_wrong(studio, camera, observation) -> None:
    display, sink, cv2_spy = studio
    display.render(camera, sink.data, status="ALLOW", observation=observation)
    assert not cv2_spy.said("WHY:"), cv2_spy.texts


# --- per-joint activity readout ------------------------------------------


def test_every_commanded_joint_appears_in_the_readout(studio, camera) -> None:
    """The operator asked to see what is being tracked; nothing may be missing."""
    display, sink, cv2_spy = studio
    display.render(camera, sink.data, status="ALLOW", joints=joint_pose())
    for _name, label, _group in UPPER_BODY_JOINTS:
        assert cv2_spy.said(label), f"{label} missing from readout"
    assert cv2_spy.said("JOINT ACTIVITY")


def test_the_readout_lists_exactly_the_joints_the_session_commands() -> None:
    """Pinned to `JOINT_ORDER`, not to a count.

    This used to assert `len(names) == 18`, which is a number, not an invariant --
    and the readout was in fact missing `leg_joint4`, the torso, which every clip
    commands on every frame. A magic count agrees with itself forever; the clip's
    own joint order does not.
    """
    from galbot_motion_studio.cli import JOINT_ORDER

    names = {name for name, _label, _group in UPPER_BODY_JOINTS}
    assert names == set(JOINT_ORDER), (
        f"not shown: {sorted(set(JOINT_ORDER) - names)}; "
        f"shown but not commanded: {sorted(names - set(JOINT_ORDER))}"
    )
    assert {"head_joint1", "head_joint2"} <= names
    assert {f"left_arm_joint{i}" for i in range(1, 8)} <= names
    assert {f"right_arm_joint{i}" for i in range(1, 8)} <= names
    assert {"left_gripper_joint", "right_gripper_joint"} <= names
    assert len(names) == len(UPPER_BODY_JOINTS), "a joint is listed twice"


def test_a_moving_joint_registers_activity_and_a_still_one_does_not(
    studio, camera
) -> None:
    display, sink, _ = studio
    pose = joint_pose()
    display.render(camera, sink.data, status="ALLOW", joints=pose)
    pose["left_arm_joint1"] = 0.05
    display.render(camera, sink.data, status="ALLOW", joints=pose)
    assert display._joint_activity["left_arm_joint1"] > 0.01
    assert display._joint_activity["right_arm_joint1"] == pytest.approx(0.0)


def _drive(display, camera, sink, frames, *, dead=(), held=frozenset(), step=0.01):
    """Render `frames` solves in which every joint moves except `dead`."""
    from galbot_motion_studio.display import UPPER_BODY_JOINTS

    pose = joint_pose()
    for index in range(frames):
        for name, _label, _group in UPPER_BODY_JOINTS:
            if name not in dead:
                pose[name] = index * step
        display.render(
            camera, sink.data, status="ALLOW", joints=dict(pose), held_groups=held
        )
    return pose


def test_a_dead_joint_is_called_out_while_its_own_limb_keeps_moving(
    studio, camera
) -> None:
    """The state this red exists for: one joint stops, its siblings do not."""
    from galbot_motion_studio.display import FAIL_BGR, LEFT_ARM

    display, sink, cv2_spy = studio
    _drive(display, camera, sink, 60, dead={"left_arm_joint4"})

    assert display._joint_still_frames["left_arm_joint4"] >= 50
    assert display._group_still_frames[LEFT_ARM] == 0, (
        "the limb itself was moving, so its stillness counter must be zero"
    )
    assert FAIL_BGR in cv2_spy.colours.get("L arm 4", []), (
        f"the dead joint was not drawn in red: {cv2_spy.colours.get('L arm 4')}"
    )
    assert FAIL_BGR not in cv2_spy.colours.get("L arm 3", []), (
        "a healthy sibling was called dead"
    )


def test_an_operator_standing_still_is_not_a_dead_robot(studio, camera) -> None:
    """The false positive that reddened the whole card on a healthy session.

    Measured before this rule: 9 of the 19 joints tripped red on the witness
    capture, red appeared in 23% of composite frames, and ALL nineteen went red
    about two seconds after the operator stepped out of shot -- because the
    counter advanced on every REPAINT of an unchanged pose, and needed no
    evidence that anything else was moving.
    """
    from galbot_motion_studio.display import FAIL_BGR

    display, sink, cv2_spy = studio
    pose = joint_pose(0.2)
    for _ in range(120):
        display.render(camera, sink.data, status="ALLOW", joints=dict(pose))

    reddened = sorted(
        text for text, colours in cv2_spy.colours.items() if FAIL_BGR in colours
    )
    assert not reddened, f"a motionless session called these dead: {reddened}"


def test_a_repaint_of_an_unchanged_pose_is_not_counted_as_a_still_solve(
    studio, camera
) -> None:
    """The counters mean SOLVES; `render` runs at camera cadence, 2-3x faster.

    Between two solves the window repaints with the same joint readback. Counting
    those made every constant expressed in "frames" mean a third of what it says,
    so a joint tripped the dead threshold in a third of the documented time.
    """
    display, sink, _ = studio
    _drive(display, camera, sink, 12, dead={"left_arm_joint4"})
    before = dict(display._joint_still_frames)
    groups_before = dict(display._group_still_frames)

    pose = joint_pose(0.11)
    display.render(camera, sink.data, status="ALLOW", joints=dict(pose))  # a new pose
    settled = dict(display._joint_still_frames)
    for _ in range(30):                       # 30 repaints of that same pose
        display.render(camera, sink.data, status="ALLOW", joints=dict(pose))

    assert display._joint_still_frames == settled, (
        "repaints of an unchanged pose advanced the still-counters: "
        f"{settled} -> {display._joint_still_frames}"
    )
    assert before and groups_before  # the drive really did populate them


def test_a_resting_limb_is_not_dead_just_because_the_other_arm_is_busy(
    studio, camera
) -> None:
    """Corroboration: stillness only accuses a joint whose OWN group is moving.

    Gesturing with one arm while the other hangs at your side is the most ordinary
    thing an operator does, and it reddened all seven joints of the resting arm.
    """
    from galbot_motion_studio.display import FAIL_BGR, UPPER_BODY_JOINTS

    display, sink, cv2_spy = studio
    resting = {name for name, _l, _g in UPPER_BODY_JOINTS if not name.startswith("right_arm")}
    _drive(display, camera, sink, 120, dead=resting)

    left_labels = {label for name, label, _g in UPPER_BODY_JOINTS if name.startswith("left_arm")}
    reddened = sorted(
        label for label in left_labels if FAIL_BGR in cv2_spy.colours.get(label, [])
    )
    assert not reddened, f"the resting arm was called dead: {reddened}"
    # ...and the busy arm is still being watched, so the guard is not just "off".
    assert display._group_still_frames["right_arm"] == 0


def test_a_held_limb_is_never_called_dead_and_its_bars_fall_at_once(
    studio, camera
) -> None:
    """A joint the supervisor is freezing says nothing about whether it works."""
    from galbot_motion_studio.display import FAIL_BGR, LEFT_ARM

    display, sink, cv2_spy = studio
    _drive(display, camera, sink, 20)                       # everything moving
    moving = display._joint_activity["left_arm_joint4"]
    assert moving > 0.0
    _drive(display, camera, sink, 60, dead={"left_arm_joint4"}, held=frozenset({LEFT_ARM}))

    assert display._joint_activity["left_arm_joint4"] == 0.0, (
        "a held joint drew a long bar while the app bar called the limb frozen"
    )
    assert FAIL_BGR not in cv2_spy.colours.get("L arm 4", []), (
        "a held joint was called dead"
    )


def test_activity_decays_to_zero_while_the_robot_is_held(studio, camera) -> None:
    """A HOLD frame carries no readback; the bars must fall, not freeze or blank."""
    display, sink, _ = studio
    pose = joint_pose()
    display.render(camera, sink.data, status="ALLOW", joints=pose)
    pose["head_joint1"] = 0.4
    display.render(camera, sink.data, status="ALLOW", joints=pose)
    peak = display._joint_activity["head_joint1"]
    for _ in range(30):
        display.render(camera, sink.data, status="HOLD", joints=None)
    assert display._joint_activity["head_joint1"] < peak * 0.1


def test_a_nan_readback_does_not_poison_the_readout(studio, camera) -> None:
    display, sink, _ = studio
    display.render(
        camera,
        sink.data,
        status="ALLOW",
        joints={"left_arm_joint1": float("nan"), "head_joint1": float("inf")},
    )
    assert display.overlay_errors == 0
    assert display._joint_activity["left_arm_joint1"] == pytest.approx(0.0)


# --- calibration feedback ------------------------------------------------


def test_calibration_names_the_missing_body_parts_in_english(studio, camera) -> None:
    """"pose_15" in a terminal is not something an operator can act on."""
    display, sink, cv2_spy = studio
    display.render(
        camera,
        sink.data,
        status="CALIBRATING",
        calibration=CalibrationStatus(
            frames=15,
            missing_landmarks=("pose_15", "pose_16"),
            required_landmarks=9,
            identity="ACQUIRING",
            confidence=0.01,
            required_confidence=0.5,
        ),
    )
    assert cv2_spy.said("left wrist"), cv2_spy.texts
    assert cv2_spy.said("right wrist"), cv2_spy.texts
    assert not cv2_spy.said("pose_15"), cv2_spy.texts


def test_calibration_shows_progress_and_every_blocking_condition(
    studio, camera
) -> None:
    display, sink, cv2_spy = studio
    display.render(
        camera,
        sink.data,
        status="CALIBRATING",
        calibration=CalibrationStatus(
            frames=94,
            missing_landmarks=(),
            required_landmarks=9,
            identity="ACQUIRING",
            identity_ok=False,
            confidence=0.01,
            required_confidence=0.5,
        ),
    )
    assert cv2_spy.said("frame 94"), cv2_spy.texts
    assert cv2_spy.said("landmarks 9/9"), cv2_spy.texts
    assert cv2_spy.said("CONFIDENCE 0.01 / 0.50"), cv2_spy.texts
    assert cv2_spy.said("IDENTITY ACQUIRING"), cv2_spy.texts


def test_the_calibration_hint_is_drawn_without_degrading_the_overlay(
    studio, camera
) -> None:
    """Regression: ``_draw_calibration`` read a ``hint`` field that did not exist.

    The AttributeError was swallowed by ``_safe``, so calibration silently lost
    its panel tail and the window advertised OVERLAY DEGRADED every frame.
    """
    display, sink, cv2_spy = studio
    display.render(
        camera,
        sink.data,
        status="CALIBRATING",
        calibration=CalibrationStatus(
            frames=15,
            missing_landmarks=("pose_15",),
            required_landmarks=9,
            identity="ACQUIRING",
            confidence=0.01,
            required_confidence=0.5,
            hint="STEP BACK - can't see: left wrist",
        ),
    )
    assert display.overlay_errors == 0
    assert cv2_spy.said("STEP BACK"), cv2_spy.texts
    assert not cv2_spy.said("OVERLAY DEGRADED"), cv2_spy.texts


def test_a_satisfied_calibration_check_is_marked_ok(studio, camera) -> None:
    display, sink, cv2_spy = studio
    display.render(
        camera,
        sink.data,
        status="CALIBRATING",
        calibration=CalibrationStatus(
            frames=3,
            missing_landmarks=(),
            required_landmarks=9,
            identity="STABLE",
            identity_ok=True,
            confidence=0.9,
            required_confidence=0.5,
        ),
    )
    assert cv2_spy.said("[OK] ALL REQUIRED BODY PARTS VISIBLE"), cv2_spy.texts
    assert cv2_spy.said("[OK] CONFIDENCE"), cv2_spy.texts


# --- robustness: an overlay bug must never end the session ---------------


def test_a_failing_overlay_element_degrades_the_window_not_the_session(
    studio, camera, observation
) -> None:
    """A previous session died in a traceback inside the display. Never again."""
    display, sink, cv2_spy = studio

    def explode(*args, **kwargs):
        raise RuntimeError("synthetic overlay bug")

    display._draw_joint_activity = explode
    canvas = display.render(
        camera,
        sink.data,
        status="ALLOW",
        observation=observation,
        joints=joint_pose(),
    )
    assert canvas.shape == (480, 1280, 3)
    assert display.overlay_errors == 1
    assert cv2_spy.said("OVERLAY DEGRADED"), cv2_spy.texts


def test_render_survives_completely_absent_data(studio, camera) -> None:
    display, sink, _ = studio
    canvas = display.render(
        camera,
        sink.data,
        status="HOLD",
        observation=None,
        held_groups=None,
        reasons=None,
        joints=None,
        calibration=None,
    )
    assert canvas.shape == (480, 1280, 3)
    assert display.overlay_errors == 0


def test_an_unknown_control_group_name_does_not_break_the_overlay(
    studio, camera, observation
) -> None:
    """held_groups is data from another module; a rename must not crash the window."""
    display, sink, cv2_spy = studio
    canvas = display.render(
        camera,
        sink.data,
        status="HOLD",
        observation=observation,
            held_groups=frozenset({"unknown_group", LEFT_ARM}),
    )
    assert canvas.shape == (480, 1280, 3)
    assert display.overlay_errors == 0
    assert cv2_spy.said("UNKNOWN_GROUP"), cv2_spy.texts


def test_an_observation_missing_every_landmark_still_renders(studio, camera) -> None:
    display, sink, _ = studio
    empty = synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    stripped = empty.model_copy(update={"landmarks": ()})
    canvas = display.render(
        camera,
        sink.data,
        status="HOLD",
        observation=stripped,
        held_groups=frozenset({LEFT_ARM, HEAD}),
    )
    assert canvas.shape == (480, 1280, 3)
    assert display.overlay_errors == 0


# --- the pre-existing composite guarantees -------------------------------


def test_composite_render_contains_camera_skeleton_and_robot_panel(model) -> None:
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    display = StudioDisplay(model, width=320, height=240)
    camera = np.zeros((240, 320, 3), dtype=np.uint8)
    observation = synthetic_observation(
        1, timestamp_ns=1_000_000_000, motion_fraction=0.5
    )
    try:
        canvas = display.render(
            camera,
            sink.data,
            status="ALLOW",
            telemetry="30 FPS | 20 ms",
            observation=observation,
        )
    finally:
        display.close()

    assert canvas.shape == (240, 640, 3)
    # Sliced at the REAL panel boundary, not at half the canvas. The two panels
    # are deliberately unequal (ROBOT_PANEL_SHARE), so a hardcoded midpoint would
    # be testing an assumption the layout no longer makes -- and would keep
    # passing if the split silently drifted.
    boundary = display._width
    assert 0 < boundary < canvas.shape[1]
    assert np.count_nonzero(canvas[:, :boundary]) > 0, "camera panel is empty"
    assert np.count_nonzero(canvas[:, boundary:]) > 0, "twin panel is empty"
    assert display._width + display._robot_width == canvas.shape[1]


@pytest.mark.parametrize("mirrored", (False, True))
def test_the_two_panels_always_agree_about_which_side_is_which(model, mirrored) -> None:
    """The guarantee, which mirroring is only one way of meeting.

    A correct mapping with only ONE panel mirrored reads as a left/right swap
    (reported 2026-08-26), so the rule is mirror BOTH or NEITHER. Asserted as the
    property -- the operator's left and the robot's left land on the same half of
    their own panels -- rather than as "the camera is flipped", so the default can
    change without the guarantee quietly changing with it.

    The default is NEITHER: flipping the render also flips the chassis, and the
    GALBOT wordmark on the base then reads backwards in every frame.
    """
    import mujoco

    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    display = StudioDisplay(model, width=320, height=240, mirror_camera=mirrored)
    # A marker on the operator's own left, which an unmirrored camera puts on the
    # RIGHT of the image (you are looking at them, not at a mirror).
    camera = np.zeros((240, 320, 3), dtype=np.uint8)
    camera[:, -20:] = (0, 0, 255)
    left_arm_geoms = [
        geom
        for geom in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or "")
        .startswith("left_arm")
    ]
    try:
        canvas = display.render(camera, sink.data, status="ALLOW")
        box_x, box_y, box_w, box_h = display._camera_box
        renderer = mujoco.Renderer(model, height=display._height, width=display._robot_width)
        try:
            renderer.enable_segmentation_rendering()
            renderer.update_scene(sink.data, camera=display._robot_camera)
            segmentation = renderer.render()[:, :, 0]
        finally:
            renderer.close()
    finally:
        display.close()
    # This half of the guarantee takes the production flip as given -- it mirrors
    # the reference render itself -- and asserts only that the two panels AGREE.
    # That the flip actually happens is a separate claim, and it is carried by
    # `test_the_twin_panel_really_is_flipped_when_the_camera_is`, without which
    # this test passes with `flip(robot_bgr, 1)` deleted (verified).
    if mirrored:
        segmentation = segmentation[:, ::-1]
    columns = np.nonzero(np.isin(segmentation, left_arm_geoms).any(axis=0))[0]
    assert columns.size, "the robot's left arm is not visible at all"
    robot_left_on_left = float(columns.mean()) < display._robot_width / 2

    row = box_y + box_h // 2
    strip = max(2, box_w // 32)
    marker_on_left = bool(np.all(canvas[row, box_x:box_x + strip][:, 2] > 200))
    marker_on_right = bool(
        np.all(canvas[row, box_x + box_w - strip:box_x + box_w][:, 2] > 200)
    )
    assert marker_on_left != marker_on_right, "the operator marker is on neither side"
    operator_left_on_left = marker_on_left
    assert operator_left_on_left == robot_left_on_left, (
        f"mirror_camera={mirrored}: the operator's left is on the "
        f"{'left' if operator_left_on_left else 'right'} of their panel but the "
        f"robot's left arm is on the {'left' if robot_left_on_left else 'right'} "
        "of its panel -- the panels disagree"
    )


def test_the_twin_panel_really_is_flipped_when_the_camera_is(model) -> None:
    """The half the agreement test cannot see: does the robot render flip at all?

    `test_the_two_panels_always_agree_about_which_side_is_which` mirrors its own
    reference render, so it compares a flip the TEST performed and never observes
    `display.py`'s. Deleting that line left it green. This one is differential and
    needs no reference: render the same asymmetric pose with the flag off and on,
    and the twin half of the canvas must be the mirror image of itself -- and must
    NOT be identical, which is what deleting the flip would make it.
    """
    import mujoco

    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    # The rest pose is nearly left-right symmetric, so a flipped panel would be
    # indistinguishable from an upright one. Throw one arm out to the side.
    for name, angle in (("left_arm_joint2", 1.2), ("left_arm_joint4", -1.1)):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint >= 0, name
        sink.data.qpos[int(model.jnt_qposadr[joint])] = angle
    mujoco.mj_forward(model, sink.data)
    camera = np.zeros((240, 320, 3), dtype=np.uint8)

    panels = {}
    segmentation = None
    for mirrored in (False, True):
        display = StudioDisplay(model, width=320, height=240, mirror_camera=mirrored)
        try:
            canvas = display.render(camera, sink.data, status="ALLOW")
            panels[mirrored] = canvas[:, display._width:].copy()
            if segmentation is None:
                renderer = mujoco.Renderer(
                    model, height=display._height, width=display._robot_width
                )
                try:
                    renderer.enable_segmentation_rendering()
                    renderer.update_scene(sink.data, camera=display._robot_camera)
                    segmentation = renderer.render()[:, :, 0]
                finally:
                    renderer.close()
        finally:
            display.close()

    # Compare the ROBOT's own bounding box, not the whole panel: the HUD is drawn
    # after the flip and stays put, so a whole-panel mirror comparison can never
    # match and says nothing either way.
    robot_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) > 0]
    mask = np.isin(segmentation, robot_geoms)
    rows = np.nonzero(mask.any(axis=1))[0]
    columns = np.nonzero(mask.any(axis=0))[0]
    assert rows.size and columns.size, "the robot is not visible in the twin panel"
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(columns.min()), int(columns.max()) + 1
    width = panels[False].shape[1]

    upright = panels[False][r0:r1, c0:c1].astype(int)
    same_place = panels[True][r0:r1, c0:c1].astype(int)
    mirrored_place = panels[True][r0:r1, width - c1:width - c0][:, ::-1].astype(int)

    moved = float(np.abs(upright - same_place).mean())
    matches_mirror = float(np.abs(upright - mirrored_place).mean())
    assert matches_mirror < moved, (
        "the twin panel is not mirrored by --mirror-camera: the robot's box "
        f"differs from its mirrored counterpart by {matches_mirror:.2f} and from "
        f"itself in place by {moved:.2f} -- if the flip were applied these would "
        "be the other way round"
    )
    assert moved > 0.5, (
        "the robot occupies the same pixels with and without --mirror-camera, so "
        "the render is not being flipped at all"
    )


def test_a_small_window_drops_the_joint_readout_rather_than_drawing_mush(
    model,
) -> None:
    """Legibility over completeness: unreadable rows are worse than none."""
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    display = StudioDisplay(model, width=320, height=240)
    display._cv2 = RecordingCv2(display._cv2)
    camera = np.zeros((240, 320, 3), dtype=np.uint8)
    try:
        display.render(camera, sink.data, status="ALLOW", joints=joint_pose())
    finally:
        display.close()
    assert not display._cv2.said("JOINT ACTIVITY")
    assert display.overlay_errors == 0
