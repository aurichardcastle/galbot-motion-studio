"""Layout invariants for the HUD, at panel shapes the operator actually sees.

The overlay draws into a two-panel canvas whose left panel holds the camera image
top-anchored, with a HUD column beneath it.  Whether that column exists at all
depends on the camera's aspect against the panel's, so the layout has two modes,
and the second one only appears at panel shapes the unit tests happened not to
use.  These pin both.

The specific failure this file was written for: the safety card stacked itself
under the joint-activity card by reading only that card's BOTTOM edge, while
taking its own x from the HUD column.  When the joint card fell back to the
twin's side of the canvas -- which is exactly what it does when the camera fills
its panel -- the safety card was drawn at the left column's x and the fallback's
y, on top of the operator.
"""

from __future__ import annotations

import numpy as np
import pytest

from galbot_motion_studio.adapters.mujoco_preview import MujocoPreviewSink
from galbot_motion_studio.display import UPPER_BODY_JOINTS, StudioDisplay
from galbot_motion_studio.model.loader import load_verified_fixed_base_model
from galbot_motion_studio.safety.clearance import HOME_QPOS
from galbot_motion_studio.synthetic import synthetic_observation


@pytest.fixture(scope="module")
def model():
    return load_verified_fixed_base_model()


def joints() -> dict[str, float]:
    return {name: 0.1 for name, _label, _group in UPPER_BODY_JOINTS}


def render(model, panel: tuple[int, int], camera_shape: tuple[int, int]):
    """Render one frame and return (display, canvas)."""
    width, height = panel
    display = StudioDisplay(model, width=width, height=height)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.full((camera_shape[0], camera_shape[1], 3), 200, dtype=np.uint8)
    observation = synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    try:
        canvas = display.render(
            camera,
            sink.data,
            status="ALLOW",
            telemetry="12.0 FPS | process 60 ms | dropped 2",
            observation=observation,
            joints=joints(),
            clearance_m=0.0071,
        )
        return display, canvas
    finally:
        display.close()


@pytest.mark.parametrize(
    "panel,camera_shape",
    (
        ((860, 1080), (1080, 1920)),   # the real fullscreen shape: a HUD column exists
        ((640, 480), (480, 640)),      # camera fills its panel: no column, fallback
        ((320, 240), (240, 320)),      # too small for the readout at all
        ((480, 900), (1080, 1920)),    # narrow and tall
        ((900, 400), (1080, 1920)),    # wide and short
    ),
)
def test_the_hud_never_draws_outside_its_canvas(model, panel, camera_shape) -> None:
    """Every panel shape renders without an overlay error and at the right size."""
    display, canvas = render(model, panel, camera_shape)
    assert canvas.shape == (panel[1], panel[0] * 2, 3)
    assert display.overlay_errors == 0, display._overlay_error_note


def _recorded_cards(display):
    """Wrap `_card` so a test can see every card rect that was actually drawn."""
    drawn: list[tuple[str, tuple[int, int, int, int]]] = []
    original = display._card

    def record(canvas, rect, title, **kwargs):
        drawn.append((title, rect))
        return original(canvas, rect, title, **kwargs)

    display._card = record
    return drawn


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


@pytest.mark.parametrize(
    "panel,camera_shape",
    (
        ((860, 1080), (1080, 1920)),   # real fullscreen: a HUD column exists
        ((640, 480), (480, 640)),      # camera fills its panel: no column
        ((480, 900), (1080, 1920)),
        ((900, 400), (1080, 1920)),
        ((700, 700), (1000, 1000)),    # near-square camera
    ),
)
def test_no_card_overlaps_another_card_or_the_camera_image(model, panel, camera_shape) -> None:
    """The invariant the layout cursor exists to make true by construction.

    Cards used to hand each other position through three different instance
    attributes, one of which was written in only one of two layouts -- which put
    the SAFETY card at the HUD column's x and the fallback's y, i.e. on top of
    the operator. A column either has room for a card or it does not; either way
    two cards cannot claim the same pixels.
    """
    width, height = panel
    display = StudioDisplay(model, width=width, height=height)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.full((camera_shape[0], camera_shape[1], 3), 200, dtype=np.uint8)
    observation = synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    drawn = _recorded_cards(display)
    try:
        display.render(
            camera, sink.data, status="ALLOW",
            telemetry="12.0 FPS | process 60 ms | dropped 2",
            observation=observation, joints=joints(), clearance_m=0.0071,
        )
        camera_x, camera_y, camera_w, camera_h = display._camera_box
    finally:
        display.close()
    assert display.overlay_errors == 0, display._overlay_error_note
    for index, (title_a, rect_a) in enumerate(drawn):
        for title_b, rect_b in drawn[index + 1:]:
            assert not _overlap(rect_a, rect_b), f"{title_a} {rect_a} overlaps {title_b} {rect_b}"
        image = (camera_x, camera_y, camera_x + camera_w, camera_y + camera_h)
        assert not _overlap(rect_a, image), f"{title_a} {rect_a} is drawn on the camera image"


def test_the_safety_readout_survives_the_fallback_panel_size(model) -> None:
    """640x480 is FALLBACK_PANEL_SIZE -- every install without pyobjc.

    The tracking and joint cards each grew their own twin-side fallback; the
    safety card did not, so the one readout the runbook calls "the honest one to
    dwell on" simply vanished there. A shared column fixes all three at once.
    """
    display = StudioDisplay(model, width=640, height=480)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.full((480, 640, 3), 200, dtype=np.uint8)
    observation = synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    drawn = _recorded_cards(display)
    try:
        display.render(camera, sink.data, status="ALLOW", observation=observation,
                       joints=joints(), clearance_m=0.0071)
    finally:
        display.close()
    titles = [title for title, _rect in drawn]
    assert "SAFETY" in titles, titles
    assert "TRACKING" in titles, titles


def test_a_card_that_fails_does_not_displace_the_cards_below_it(model) -> None:
    """`_safe` swallows a drawing failure; the rest of the frame must still be sane."""
    display = StudioDisplay(model, width=860, height=1080)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    observation = synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    try:
        drawn = _recorded_cards(display)
        display.render(camera, sink.data, status="ALLOW", observation=observation,
                       joints=joints(), clearance_m=0.0071)
        assert [t for t, _ in drawn] == ["TRACKING", "SAFETY", "JOINT ACTIVITY"]

        def explode(*_args, **_kwargs):
            raise RuntimeError("synthetic joint-card bug")

        display._draw_joint_activity = explode
        drawn.clear()
        display.render(camera, sink.data, status="ALLOW", observation=observation,
                       joints=joints(), clearance_m=0.0071)
        assert [t for t, _ in drawn] == ["TRACKING", "SAFETY"]
        for index, (title_a, rect_a) in enumerate(drawn):
            for title_b, rect_b in drawn[index + 1:]:
                assert not _overlap(rect_a, rect_b), f"{title_a} overlaps {title_b}"
        assert display.overlay_errors == 1
    finally:
        display.close()


def test_a_camera_frame_that_changes_size_mid_session_still_renders(model) -> None:
    """A camera can renegotiate resolution; the reused panel buffer must cope."""
    display = StudioDisplay(model, width=640, height=480)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    observation = synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    try:
        for shape in ((480, 640), (1080, 1920), (240, 320), (720, 1280)):
            camera = np.full((shape[0], shape[1], 3), 200, dtype=np.uint8)
            canvas = display.render(camera, sink.data, status="ALLOW",
                                    observation=observation, joints=joints())
            assert canvas.shape == (480, 1280, 3)
        assert display.overlay_errors == 0, display._overlay_error_note
    finally:
        display.close()


def test_a_returned_canvas_is_not_overwritten_by_the_next_render(model) -> None:
    """`render` hands the caller a frame; cli.py holds it across a video write."""
    display = StudioDisplay(model, width=320, height=240)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.full((240, 320, 3), 200, dtype=np.uint8)
    observation = synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    try:
        first = display.render(camera, sink.data, status="ALLOW", observation=observation)
        snapshot = first.copy()
        display.render(camera, sink.data, status="HOLD", observation=observation,
                       held_groups=frozenset({"left_arm"}))
        display.render(camera, sink.data, status="ALLOW", observation=observation)
        assert np.array_equal(first, snapshot), (
            "a later render mutated a canvas already handed to the caller"
        )
    finally:
        display.close()


def test_hostile_numbers_do_not_break_the_overlay(model) -> None:
    """NaN and inf reach this module from a real pipeline; they must not raise."""
    display = StudioDisplay(model, width=860, height=1080)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    observation = synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    poisoned = joints()
    poisoned["left_arm_joint1"] = float("nan")
    poisoned["right_arm_joint1"] = float("inf")
    try:
        for clearance in (None, float("nan"), float("inf"), -0.01, 0.0, 1e9):
            display.render(camera, sink.data, status="ALLOW", observation=observation,
                           joints=poisoned, clearance_m=clearance)
        assert display.overlay_errors == 0, display._overlay_error_note
    finally:
        display.close()


# --- what SURVIVES to the canvas, not what was handed to putText -------------
#
# The existing suite records the strings passed to `cv2.putText`. That is the
# right test for "did we decide to say it" and the wrong test for "can the
# operator read it": an element drawn and then painted over by a later opaque
# element passes `said(...)` with zero surviving pixels. Two real defects hid
# behind exactly that gap -- the telemetry chips erasing the FROZEN pill, and the
# calibration card erasing the WHY banner -- so these assert on ink.


def _ink(canvas, rect, colour, tolerance: int = 60) -> int:
    """Count pixels in `rect` close to `colour` (BGR)."""
    x0, y0, x1, y1 = rect
    region = canvas[y0:y1, x0:x1].astype(int)
    target = np.asarray(colour, dtype=int)
    return int((np.abs(region - target).sum(axis=2) < tolerance).sum())


@pytest.mark.parametrize(
    "panel_width,held",
    (
        # Wide canvas, two held groups: the hold pills stay on the app bar's FIRST
        # row, beside the status pill, which is the layout the chip width budget
        # exists for.
        (860, frozenset({"left_arm", "head"})),
        # Narrow canvas, everything held: the pills no longer fit beside the
        # status pill and take a second row. Different code path, same guarantee.
        (430, frozenset({"head", "torso", "left_arm", "right_arm"})),
    ),
)
def test_the_hold_words_survive_a_full_telemetry_row(model, panel_width, held) -> None:
    """The chips must yield to the pills, not paint over them.

    DIFFERENTIAL, on purpose. A first version of this test counted amber pixels
    in the app-bar band and asserted a floor -- and passed with the hold pills
    deleted, because the status pill is amber on a partial frame and so is a
    sub-standoff clearance chip. A pixel test that has never been seen to fail is
    a string test with extra steps.

    So render twice, once with the hold pills suppressed, and require the
    difference. Delete the pills and the two renders coincide and this fails.
    """
    from galbot_motion_studio.display import WARN_BGR

    def render_bar(suppress_holds: bool) -> int:
        display = StudioDisplay(model, width=panel_width, height=1080)
        sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
        camera = np.full((1080, 1920, 3), 200, dtype=np.uint8)
        observation = synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)
        if suppress_holds:
            original = display._pill

            def without_holds(canvas, origin, text, **kwargs):
                if text.startswith(("FROZEN", "LIMITED")):
                    return origin[0]
                return original(canvas, origin, text, **kwargs)

            display._pill = without_holds
        try:
            canvas = display.render(
                camera, sink.data, status="ALLOW",
                telemetry="14.2 FPS | process 61 ms | dropped 3",
                observation=observation, joints=joints(), clearance_m=0.006,
                held_groups=held,
                held_grippers=frozenset({"left_gripper_joint"}),
                saturated_groups=frozenset({"head"}),
            )
            band = 2 * display._bar_unit
        finally:
            display.close()
        return _ink(canvas, (0, 0, canvas.shape[1], band), WARN_BGR)

    with_holds = render_bar(False)
    without_holds = render_bar(True)
    assert with_holds - without_holds > 600, (
        f"the hold pills contributed only {with_holds - without_holds} amber pixels "
        f"({with_holds} vs {without_holds}): they were drawn and then painted over "
        "by the telemetry chips"
    )


def _app_bar_rects(display):
    """Record every pill rect and every telemetry-chip rect drawn in the app bar.

    Both go through `_round_rect`, so attribute them by re-entrancy: anything
    drawn while `_pill` is on the stack belongs to a pill, everything else in the
    bar is a chip.
    """
    pills: list[tuple[int, int, int, int]] = []
    chips: list[tuple[int, int, int, int]] = []
    inside_pill = []
    original_pill = display._pill
    original_rect = display._round_rect

    def pill(canvas, origin, text, **kwargs):
        inside_pill.append(text)
        try:
            return original_pill(canvas, origin, text, **kwargs)
        finally:
            inside_pill.pop()

    def round_rect(canvas, rect, fill, border=None, radius=6):
        (pills if inside_pill else chips).append(tuple(rect))
        return original_rect(canvas, rect, fill, border, radius)

    display._pill = pill
    display._round_rect = round_rect
    return pills, chips


@pytest.mark.parametrize("panel_width", (300, 380, 460, 540, 620, 700, 860, 1100))
def test_no_telemetry_chip_ever_overlaps_a_hold_pill(model, panel_width) -> None:
    """The structural form of the guarantee, swept across widths.

    Counting pixels only catches the collision at whichever width happens to
    produce one; the chips are laid out right-to-left and the pills left-to-right,
    so whether they meet depends entirely on the canvas width. Assert instead that
    the two sets of rectangles are disjoint, at every width, which is the property
    the width budget actually provides.
    """
    display = StudioDisplay(model, width=panel_width, height=1080)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    observation = synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    pills, chips = _app_bar_rects(display)
    try:
        display.render(
            camera, sink.data, status="ALLOW",
            telemetry="14.2 FPS | process 61 ms | dropped 3",
            observation=observation, joints=joints(), clearance_m=0.006,
            held_groups=frozenset({"left_arm", "head"}),
            held_grippers=frozenset({"left_gripper_joint"}),
            saturated_groups=frozenset({"head"}),
        )
        bar = display._app_bar_height
    finally:
        display.close()
    bar_pills = [r for r in pills if r[1] < bar]
    bar_chips = [r for r in chips if r[1] < bar]
    assert bar_pills, "no pills were drawn in the app bar at all"
    for chip in bar_chips:
        for pill_rect in bar_pills:
            assert not _overlap(chip, pill_rect), (
                f"telemetry chip {chip} overlaps pill {pill_rect} at panel "
                f"width {panel_width}: the chip would erase a safety word"
            )
    for rect in bar_pills + bar_chips:
        assert rect[0] >= 0 and rect[2] <= display._width + display._robot_width, (
            f"{rect} is drawn off the canvas at panel width {panel_width}"
        )


def test_the_reason_banner_survives_alongside_the_calibration_card(model) -> None:
    """Both are drawn at the top of the camera panel; neither may erase the other.

    Asserted on the banner's OWN accent bar, at the rect the banner reports it
    used -- not on amber anywhere in the panel, which a `FROZEN:` pill in the app
    bar would have satisfied on its own.
    """
    from galbot_motion_studio.display import WARN_BGR, CalibrationStatus

    display = StudioDisplay(model, width=860, height=1080)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    observation = synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    drawn = _recorded_cards(display)
    try:
        canvas = display.render(
            camera, sink.data, status="CALIBRATING", observation=observation,
            reasons=("observation: IDENTITY_NOT_STABLE",),
            held_groups=frozenset({"left_arm"}),
            calibration=CalibrationStatus(
                frames=94, missing_landmarks=("pose_15",), required_landmarks=9,
                identity="ACQUIRING", identity_ok=False, confidence=0.31,
                required_confidence=0.5, hint="STEP BACK - can't see: left wrist",
            ),
        )
        # Where `_draw_reason` puts its 4 px leading accent bar. The calibration
        # card no longer shares this space at all -- it is a card in the column,
        # so it cannot cover the banner and it no longer covers the operator.
        top = display._app_bar_height + 12
        titles = [t for t, _r in drawn]
        assert "CALIBRATION" in titles, titles
    finally:
        display.close()
    accent = _ink(canvas, (14, top + 3, 19, top + 31), WARN_BGR, tolerance=90)
    assert accent > 60, (
        f"only {accent} px of the WHY banner's accent bar survive at y={top}: the "
        "calibration card was drawn over it"
    )


def test_a_transient_overlay_failure_does_not_latch_the_degraded_badge(model) -> None:
    """`overlay_errors` is a session total; the BADGE is about this frame."""
    display = StudioDisplay(model, width=860, height=1080)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    observation = synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    healthy = display._draw_joint_activity
    try:
        def explode(*_a, **_k):
            raise RuntimeError("one bad frame")

        display._draw_joint_activity = explode
        display.render(camera, sink.data, status="ALLOW", observation=observation,
                       joints=joints())
        assert display._overlay_error_note is not None
        assert display.overlay_errors == 1

        display._draw_joint_activity = healthy
        display.render(camera, sink.data, status="ALLOW", observation=observation,
                       joints=joints())
        assert display._overlay_error_note is None, (
            "the degraded badge latched: it would claim a degraded overlay for the "
            "rest of the session after one transient failure"
        )
        assert display.overlay_errors == 1, "the session total must not be reset"
    finally:
        display.close()


@pytest.mark.parametrize("bad_camera", (
    np.zeros((480, 640), dtype=np.uint8),            # greyscale, 2-D
    np.zeros((480, 640, 4), dtype=np.uint8),         # BGRA
    [[0, 0, 0]],                                     # not an array at all
))
def test_a_malformed_camera_frame_degrades_the_window_not_the_session(model, bad_camera) -> None:
    """`cv2.VideoCapture.read()` output reaches this module unvalidated."""
    display = StudioDisplay(model, width=640, height=480)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    try:
        canvas = display.render(bad_camera, sink.data, status="ALLOW")
        assert canvas.shape == (480, 1280, 3)
        assert display.overlay_errors >= 1
        assert "camera frame" in (display._overlay_error_note or "")
    finally:
        display.close()


def test_a_float32_joint_readback_is_not_reported_as_a_dead_joint(model) -> None:
    """np.float64 subclasses float; np.float32 does not."""
    display = StudioDisplay(model, width=860, height=1080)
    try:
        for step in range(60):
            display._update_activity(
                {name: np.float32(0.01 * step) for name, _l, _g in UPPER_BODY_JOINTS}
            )
        name = UPPER_BODY_JOINTS[2][0]
        assert display._joint_previous[name] is not None
        assert display._joint_activity[name] > 0.0
        assert display._joint_still_frames[name] == 0, (
            "a float32 readback was treated as absent, so a moving joint would be "
            "called out in red as dead"
        )
    finally:
        display.close()


def test_the_clearance_colour_is_one_rule_for_the_bar_and_the_card(model) -> None:
    """The chip and the card must never disagree about the same number."""
    from galbot_motion_studio.display import (
        CLEARANCE_FLOOR_DISPLAY_M,
        CLEARANCE_STANDOFF_DISPLAY_M,
        FAIL_BGR,
        GOOD_BGR,
        WARN_BGR,
        _clearance_colour,
    )

    assert _clearance_colour(CLEARANCE_FLOOR_DISPLAY_M - 1e-6) == FAIL_BGR
    assert _clearance_colour(CLEARANCE_FLOOR_DISPLAY_M) == WARN_BGR
    assert _clearance_colour(CLEARANCE_STANDOFF_DISPLAY_M - 1e-6) == WARN_BGR
    assert _clearance_colour(CLEARANCE_STANDOFF_DISPLAY_M) == GOOD_BGR


# --- regressions found by adversarial review, pinned ------------------------


@pytest.mark.parametrize("width", (320, 400, 480, 640, 860, 1020, 1200))
@pytest.mark.parametrize("height", (300, 480, 700, 900, 1080))
def test_nothing_overlaps_anything_across_the_panel_sweep(model, width, height) -> None:
    """The overlap detector, as a standing test rather than a one-off audit.

    A review swept 294 panel shapes and found 102 illegal overlaps -- the reason
    banner erased by a card at projector resolutions, and cards colliding with
    each other -- none of which any single-size test could see. Sweep it here so
    the next layout change has to survive the same net.
    """
    display = StudioDisplay(model, width=width, height=height)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    observation = synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    from galbot_motion_studio.display import CalibrationStatus

    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    original_card = display._card
    original_reason = display._draw_reason

    def card(canvas, rect, title, **kwargs):
        boxes.append((title, tuple(rect)))
        return original_card(canvas, rect, title, **kwargs)

    def reason(canvas, status, held, reasons):
        before = display._app_bar_height + 12
        result = original_reason(canvas, status, held, reasons)
        if reasons:
            boxes.append(("WHY", (14, before, display._width - 14, before + 34)))
        return result

    original_pill = display._pill

    def pill(canvas, origin, text, **kwargs):
        end = original_pill(canvas, origin, text, **kwargs)
        boxes.append((f"pill:{text[:18]}", (origin[0], origin[1], end, origin[1] + 26)))
        return end

    display._card = card
    display._draw_reason = reason
    display._pill = pill
    try:
        display.render(
            camera, sink.data, status="CALIBRATING", observation=observation,
            joints=joints(), clearance_m=0.006,
            reasons=("observation: IDENTITY_NOT_STABLE",),
            held_groups=frozenset({"left_arm"}),
            calibration=CalibrationStatus(
                frames=94, missing_landmarks=("pose_15",), required_landmarks=9,
                identity="ACQUIRING", identity_ok=False, confidence=0.31,
                required_confidence=0.5, hint="STEP BACK",
            ),
        )
        canvas_width = display._width + display._robot_width
        display_bar_height = display._app_bar_height
    finally:
        display.close()
    assert display.overlay_errors == 0, display._overlay_error_note
    for name, rect in boxes:
        assert rect[2] <= canvas_width and rect[3] <= height, (
            f"{name} {rect} is drawn off a {canvas_width}x{height} canvas"
        )
    # Pills sit beside each other in the app bar by design, so compare only the
    # things that own exclusive space: cards, the reason banner, and any pill that
    # is NOT in the app bar (the OVERLAY DEGRADED badge).
    bar = display_bar_height
    exclusive = [
        (name, rect) for name, rect in boxes
        if not name.startswith("pill:") or rect[1] >= bar
    ]
    for index, (name_a, rect_a) in enumerate(exclusive):
        for name_b, rect_b in exclusive[index + 1:]:
            assert not _overlap(rect_a, rect_b), (
                f"{name_a} {rect_a} overlaps {name_b} {rect_b} at {width}x{height}"
            )


def test_a_panel_too_narrow_to_split_is_rejected_at_construction(model) -> None:
    """Better a clear error than a zero-width renderer and a crash inside render."""
    from galbot_motion_studio.display import MIN_PANEL_WIDTH

    with pytest.raises(ValueError, match="panel width"):
        StudioDisplay(model, width=1, height=240)
    display = StudioDisplay(model, width=MIN_PANEL_WIDTH, height=240)
    try:
        assert display._width >= 2 and display._robot_width >= 2
    finally:
        display.close()


@pytest.mark.parametrize("dtype", (np.float32, np.uint16, np.int8))
def test_a_camera_frame_this_module_cannot_draw_on_is_treated_as_malformed(
    model, dtype
) -> None:
    """Every cv2 text/shape call needs CV_8U.

    A float32 frame used to pass the shape checks, reach the overlay, and fail
    inside `putText` -- which killed the whole app bar, because it is one `_safe`
    element. That took the status pill, the FROZEN pill AND the OVERLAY DEGRADED
    badge with it, and leaked a non-uint8 canvas to the video writer.
    """
    display = StudioDisplay(model, width=860, height=1080)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.zeros((1080, 1920, 3), dtype=dtype)
    recorded: list[str] = []
    real_cv2 = display._cv2

    class Spy:
        """Records every drawn string. Delegates to the REAL cv2, not to
        `display._cv2` -- which is this object, and recursed."""

        def __getattr__(self, name):
            return getattr(real_cv2, name)

        def putText(self, image, text, *args, **kwargs):
            recorded.append(text)
            return real_cv2.putText(image, text, *args, **kwargs)

    display._cv2 = Spy()
    try:
        canvas = display.render(
            camera, sink.data, status="ALLOW", clearance_m=0.006,
            held_groups=frozenset({"left_arm"}),
        )
    finally:
        display.close()
    assert canvas.dtype == np.uint8, "a non-uint8 canvas would break the video writer"
    assert display.overlay_errors == 1
    assert "camera frame" in (display._overlay_error_note or "")
    assert any("LIMB" in text for text in recorded), recorded
    assert any("FROZEN" in text for text in recorded), recorded


def test_a_non_finite_clearance_never_reads_as_healthy(model) -> None:
    from galbot_motion_studio.display import FAIL_BGR, _clearance_colour

    assert _clearance_colour(float("nan")) == FAIL_BGR
    assert _clearance_colour(float("-inf")) == FAIL_BGR


def test_the_mirror_flag_is_wired_from_the_parser_to_the_render(model) -> None:
    """A flag nobody can reach is not a flag.

    The selfie view is the escape hatch for an operator who finds the unmirrored
    view hard to self-correct against, so it has to work end to end: parser →
    `studio.mirror_camera` → both panels actually flipping.
    """
    from galbot_motion_studio.cli import build_parser

    parser = build_parser()
    base = ["preview", "--video", "clip.mp4", "--output", "out.json"]
    assert parser.parse_args(base).mirror_camera is False
    assert parser.parse_args(base + ["--mirror-camera"]).mirror_camera is True

    display = StudioDisplay(model, width=320, height=240)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.zeros((240, 320, 3), dtype=np.uint8)
    camera[:, :20] = (0, 0, 255)
    try:
        assert display.mirror_camera is False
        plain = display.render(camera, sink.data, status="ALLOW")
        box_x, box_y, box_w, box_h = display._camera_box
        display.mirror_camera = True          # what cli.py assigns
        assert display.mirror_camera is True
        mirrored = display.render(camera, sink.data, status="ALLOW")
    finally:
        display.close()
    # The marker is on the LEFT of the source image, so unmirrored it stays on the
    # left of the panel and mirrored it crosses to the right.
    row = box_y + box_h // 2
    strip = max(2, box_w // 32)
    left_plain = plain[row, box_x:box_x + strip][:, 2].mean()
    left_mirrored = mirrored[row, box_x:box_x + strip][:, 2].mean()
    assert left_plain > 200, f"unmirrored, the marker should stay left; got {left_plain}"
    assert left_mirrored < 60, (
        f"mirrored, the marker should cross to the right; got {left_mirrored} -- "
        "the flag did not reach the render"
    )


@pytest.mark.parametrize("panel_width", (320, 400, 480, 560, 700, 860))
def test_a_card_is_never_drawn_in_a_column_too_narrow_for_it(model, panel_width) -> None:
    """A card either gets the width it needs, moves column, or is not drawn.

    The joint readout used to reserve room in the HUD column and only THEN
    discover the column was too narrow for two legible columns of joints -- so it
    silently ate ~300 px of the stack and lost itself, while the twin column sat
    free and wide enough. The width test has to happen before the reservation.
    """
    from galbot_motion_studio.display import CARD_MIN_WIDTH, JOINT_COLUMN_MIN_WIDTH

    display = StudioDisplay(model, width=panel_width, height=1080)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    observation = synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    drawn = _recorded_cards(display)
    try:
        display.render(camera, sink.data, status="ALLOW", observation=observation,
                       joints=joints(), clearance_m=0.0071)
    finally:
        display.close()
    needed = {"JOINT ACTIVITY": 2 * JOINT_COLUMN_MIN_WIDTH + 28}
    for title, rect in drawn:
        width = rect[2] - rect[0]
        assert width >= needed.get(title, CARD_MIN_WIDTH), (
            f"{title} was drawn {width} px wide, below the "
            f"{needed.get(title, CARD_MIN_WIDTH)} px its contents need"
        )


def test_the_fitted_panel_floor_agrees_with_the_constructor_floor(monkeypatch) -> None:
    """Two width floors that disagree turn a freak screen into a traceback.

    `_fitted_panel_size` used to fall back only below 2 px while `StudioDisplay`
    raises below `MIN_PANEL_WIDTH`, so a fitted width of 2-7 -- reachable on a
    screen aspect under ~0.015 -- would have been handed straight to the
    constructor and raised out of `preview`. Unreachable on the demo machine;
    still a guard that has to hold, because the fallback exists precisely for
    screens this code cannot measure sensibly.
    """
    import galbot_motion_studio.display as display_module
    from galbot_motion_studio.display import (
        FALLBACK_PANEL_SIZE,
        MIN_PANEL_WIDTH,
        _fitted_panel_size,
    )

    # Aspect 0.01: half-canvas would be 5 px, under the constructor's floor.
    monkeypatch.setattr(display_module, "_screen_content_size", lambda: (10, 1000))
    assert _fitted_panel_size() == FALLBACK_PANEL_SIZE

    # And the fallback itself must clear the floor it is protecting.
    assert FALLBACK_PANEL_SIZE[0] >= MIN_PANEL_WIDTH

    # A normal 16:10 screen is unaffected.
    monkeypatch.setattr(display_module, "_screen_content_size", lambda: (1728, 1080))
    width, height = _fitted_panel_size()
    assert width >= MIN_PANEL_WIDTH and (2 * width, height) == (1728, 1080)


def test_mirror_camera_without_a_surface_to_draw_on_is_refused(tmp_path) -> None:
    """A presentation flag with nothing to present on must not fail silently.

    `--mirror-camera` only changes what is drawn, so without --display,
    --fullscreen or --preview-video it did nothing and said nothing -- and it is
    reached for exactly when something already looks wrong on screen.
    """
    from galbot_motion_studio.cli import main

    base = [
        "preview",
        "--video", str(tmp_path / "clip.mp4"),
        "--output", str(tmp_path / "out.json"),
        "--mirror-camera",
    ]
    with pytest.raises(SystemExit) as refused:
        main(base)
    assert "--mirror-camera requires" in str(refused.value)

    # With a surface the flag is accepted, and the run gets far enough to try to
    # open the (nonexistent) clip -- which is proof that validation moved past
    # --mirror-camera rather than that the guard was simply deleted.
    from galbot_motion_studio.adapters.recorded_video import RecordedVideoError

    with pytest.raises(RecordedVideoError):
        main(base + ["--preview-video", str(tmp_path / "composite.mp4")])


@pytest.mark.parametrize(
    "panel,expected_columns",
    ((640, 3), (700, 2), (860, 2), (1020, 2)),
)
def test_the_joint_readout_survives_every_panel_size_it_can_be_read_at(
    model, panel, expected_columns
) -> None:
    """Nineteen joints, always all of them, in as few columns as will fit.

    Two columns is the design; three is the fallback that keeps the card at a
    480 px panel, where two columns of ten rows are eleven pixels too tall. The
    fallback must NOT creep up into the demo sizes -- three narrow columns at
    1080 px would be a worse card that no test would otherwise notice.
    """
    from galbot_motion_studio.display import UPPER_BODY_JOINTS

    display = StudioDisplay(model, width=panel, height=int(panel * 0.75) if panel == 640 else 1080)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    height = display._height
    camera = np.full((height, panel, 3), 200, dtype=np.uint8)
    labels = {label for _name, label, _group in UPPER_BODY_JOINTS}
    placed: list[tuple[str, int]] = []
    real = display._cv2

    class Spy:
        def __getattr__(self, name):
            return getattr(real, name)

        def putText(self, canvas, text, origin, *args, **kwargs):
            if text in labels:
                placed.append((text, origin[0]))
            return real.putText(canvas, text, origin, *args, **kwargs)

    display._cv2 = Spy()
    try:
        display.render(
            camera, sink.data, status="ALLOW",
            joints={name: 0.1 for name, _l, _g in UPPER_BODY_JOINTS},
        )
    finally:
        display._cv2 = real
        display.close()

    drawn = {label for label, _x in placed}
    assert drawn == labels, f"missing from the readout at {panel} px: {sorted(labels - drawn)}"
    assert len({x for _label, x in placed}) == expected_columns, (
        f"expected {expected_columns} columns at panel width {panel}, got "
        f"{sorted({x for _label, x in placed})}"
    )


@pytest.mark.parametrize("panel", (860, 700, 640, 480, 1020))
def test_no_card_writes_below_its_own_bottom_edge(model, panel) -> None:
    """A card's content must stay inside the card.

    The SAFETY card drew its tick labels and its state note on a baseline ten
    pixels BELOW its own rectangle, at every panel size, on every frame -- so
    `floor`, `standoff` and `supervisor allowing` floated in the twelve-pixel
    gutter and read as a caption for the card underneath. Nothing caught it,
    because every existing layout test asserts that card RECTANGLES do not
    overlap, and the text was outside every rectangle.
    """
    display = StudioDisplay(model, width=panel, height=1080)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    observation = synthetic_observation(1, timestamp_ns=1_000_000_000, motion_fraction=0.5)
    cards = _recorded_cards(display)
    texts: list[tuple[str, int, int, int, int]] = []
    real = display._cv2

    class Spy:
        def __getattr__(self, name):
            return getattr(real, name)

        def putText(self, canvas, text, origin, font, scale, *args, **kwargs):
            (tw, th), base = real.getTextSize(text, font, scale, 1)
            x, y = origin
            texts.append((text, x, y - th, x + tw, y + base))
            return real.putText(canvas, text, origin, font, scale, *args, **kwargs)

    display._cv2 = Spy()
    try:
        display.render(
            camera, sink.data, status="ALLOW", observation=observation,
            joints=joints(), clearance_m=0.0071,
        )
    finally:
        display._cv2 = real
        display.close()

    # Cards are stacked with a 12 px gap, so text that STARTS in that gap belongs
    # to the card above it -- which is the whole point: the original defect drew
    # the text entirely outside the card, so an "is it inside?" test would have
    # skipped it as belonging to nothing.
    gutter = 12
    for title, (cx0, cy0, cx1, cy1) in cards:
        for text, tx0, ty0, tx1, ty1 in texts:
            horizontally_inside = tx0 >= cx0 and tx1 <= cx1
            attributable = cy0 <= ty0 < cy1 + gutter
            if horizontally_inside and attributable:
                assert ty1 <= cy1, (
                    f"{title!r} drew {text!r} down to y={ty1}, below its own bottom "
                    f"edge y={cy1} -- it lands in the gutter, or on the next card"
                )


@pytest.mark.parametrize(
    "clearance_m",
    (0.0005, 0.00499, 0.005, 0.0060, 0.00649, 0.0065, 0.0071, 0.02),
)
def test_the_clearance_chip_is_drawn_in_the_colour_the_rule_gives(
    model, clearance_m
) -> None:
    """Asserted on the colour that reached the canvas, not on the helper alone.

    The helper had a test; the app-bar chip did not. Reintroduce a threshold
    inside `_draw_chrome` and the helper's test still passes while the chip and
    the SAFETY card disagree about the same number on screen -- which is the exact
    defect the "one rule" comment was written for.
    """
    from galbot_motion_studio.display import _clearance_colour

    display = StudioDisplay(model, width=860, height=1080)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    seen: dict[str, tuple] = {}
    real = display._cv2

    class Spy:
        def __getattr__(self, name):
            return getattr(real, name)

        def putText(self, canvas, text, origin, font, scale, colour, *args, **kwargs):
            if text.startswith("clearance "):
                seen[text] = tuple(colour)
            return real.putText(canvas, text, origin, font, scale, colour, *args, **kwargs)

    display._cv2 = Spy()
    try:
        display.render(camera, sink.data, status="ALLOW", clearance_m=clearance_m)
    finally:
        display._cv2 = real
        display.close()

    expected_text = f"clearance {clearance_m * 1000:.1f} mm"
    assert expected_text in seen, f"the chip was not drawn at all: {sorted(seen)}"
    assert seen[expected_text] == _clearance_colour(clearance_m), (
        f"the app-bar chip drew {expected_text!r} in {seen[expected_text]}, but the "
        f"one rule says {_clearance_colour(clearance_m)}"
    )


def test_a_latched_reason_tells_the_operator_what_to_do_about_it(model) -> None:
    """A raw enum on the banner is the failure the banner exists to fix.

    `TORSO_YAW_RECALIBRATION_REQUIRED` freezes both arms, both grippers and the
    torso. It is recoverable in-run now, but only by doing a specific thing --
    facing the camera and holding still for fifteen consecutive frames -- and an
    operator reading the bare enum has no way to know that. Four of five realistic
    perturbations put a demo into this state, so the banner has to carry the
    action, not the code.
    """
    from galbot_motion_studio.display import LEFT_ARM, RIGHT_ARM, TORSO, StudioDisplay

    display = StudioDisplay(model, width=860, height=1080)
    sink = MujocoPreviewSink(model=model, initial_pose=HOME_QPOS)
    camera = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    said: list[str] = []
    real = display._cv2

    class Spy:
        def __getattr__(self, name):
            return getattr(real, name)

        def putText(self, canvas, text, *args, **kwargs):
            said.append(text)
            return real.putText(canvas, text, *args, **kwargs)

    display._cv2 = Spy()
    try:
        display.render(
            camera, sink.data, status="ALLOW",
            held_groups=(LEFT_ARM, RIGHT_ARM, TORSO),
            reasons=("left_arm: TORSO_YAW_RECALIBRATION_REQUIRED",
                     "right_arm: TORSO_YAW_RECALIBRATION_REQUIRED",
                     "torso: TORSO_YAW_RECALIBRATION_REQUIRED"),
        )
    finally:
        display._cv2 = real
        display.close()

    banner = next((text for text in said if text.startswith("WHY:")), None)
    assert banner is not None, f"no WHY banner was drawn: {said[:8]}"
    assert "HOLD STILL" in banner.upper(), (
        f"the banner named the state but not the remedy: {banner!r}"
    )
