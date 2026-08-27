"""Optional OpenCV camera + MuJoCo preview window for the local studio.

Everything in this module is presentation.  It reads the pipeline's verdicts and
draws them; it never produces, gates, or modifies a command.

The overlay exists because a silent window is indistinguishable from a broken
one.  A real 1486-frame session spent seconds at a time holding on
``observation: IDENTITY_NOT_STABLE`` while the window showed a normally-coloured
skeleton and a status line that read ALLOW, and the operator had no way to see
that the robot had stopped following them, let alone why.  So: a held limb is
drawn differently from a tracked one, the rejection reason is on the canvas, the
per-joint readout says which joints are actually moving, and calibration says
what it is still waiting for.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from typing import Any, ClassVar, Iterable, Mapping

import mujoco
import numpy as np

from galbot_motion_studio.contracts.human import HumanObservation
from galbot_motion_studio.vision.freshness import ControlGroup


#: Control-group names as they arrive in ``PipelineResult.held_groups`` (plain
#: strings, produced by ``str()`` on a ``ControlGroup`` StrEnum member).  Taken
#: from the enum so a rename there cannot silently un-highlight a limb here.
HEAD = str(ControlGroup.HEAD)
LEFT_ARM = str(ControlGroup.LEFT_ARM)
RIGHT_ARM = str(ControlGroup.RIGHT_ARM)
TORSO = str(ControlGroup.TORSO)
LEFT_GRIPPER = "left_gripper_joint"
RIGHT_GRIPPER = "right_gripper_joint"

GROUP_LABELS = {
    HEAD: "HEAD",
    TORSO: "TUMMY YAW",
    LEFT_ARM: "LEFT ARM",
    RIGHT_ARM: "RIGHT ARM",
    LEFT_GRIPPER: "LEFT GRIPPER",
    RIGHT_GRIPPER: "RIGHT GRIPPER",
}

#: Drawn skeleton segments and the control group each one belongs to.  ``None``
#: is the shoulder span: both arms require both shoulders, so it is only dimmed
#: when both arms are held rather than blamed on either one.
POSE_CONNECTIONS = (
    ("pose_11", "pose_12", None),
    ("pose_11", "pose_13", LEFT_ARM),
    ("pose_13", "pose_15", LEFT_ARM),
    ("pose_12", "pose_14", RIGHT_ARM),
    ("pose_14", "pose_16", RIGHT_ARM),
)

#: Nose and both eyes, the three landmarks ``retarget/face_pose.py`` reads.  The
#: head had no skeleton at all before, so a held head was invisible.
FACE_CONNECTIONS = (
    ("face_33", "face_1", HEAD),
    ("face_1", "face_263", HEAD),
)

POSE_POINT_GROUPS = {
    "pose_11": None,
    "pose_12": None,
    "pose_13": LEFT_ARM,
    "pose_15": LEFT_ARM,
    "pose_14": RIGHT_ARM,
    "pose_16": RIGHT_ARM,
    "face_1": HEAD,
    "face_33": HEAD,
    "face_263": HEAD,
}

#: Where a "HELD" label is parked for each limb: on the elbow for an arm, on the
#: nose for the head.  A limb whose anchor landmark is missing gets no inline
#: label; the chip row at the bottom of the camera panel still names it.
LIMB_LABEL_ANCHORS = {HEAD: "face_1", LEFT_ARM: "pose_13", RIGHT_ARM: "pose_14"}

HAND_CONNECTION_INDEXES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

HAND_GROUPS = (
    ("left_hand", LEFT_GRIPPER, (255, 120, 0)),
    ("right_hand", RIGHT_GRIPPER, (255, 0, 180)),
)

#: Human-readable landmark names, so a prompt can say "left wrist" rather than
#: "pose_15".  ``cli.py`` imports this rather than keeping a second copy.
LANDMARK_LABELS = {
    "face_1": "nose",
    "face_33": "left eye",
    "face_263": "right eye",
    "pose_11": "left shoulder",
    "pose_12": "right shoulder",
    "pose_13": "left elbow",
    "pose_14": "right elbow",
    "pose_15": "left wrist",
    "pose_16": "right wrist",
}

#: Every joint this studio commands, in the order the readout lists them, with
#: the control group that drives each one. This must stay equal to the clip's own
#: `joint_order` -- it listed 18 of the 19 for a while, and the missing one was
#: the torso: commanded on 463 of 463 frames of witness-live-9 over a 0.72 rad
#: range, and invisible on the card whose entire job is to show a joint that has
#: stopped moving. `tests/unit/test_display.py` pins the two lists together.
UPPER_BODY_JOINTS: tuple[tuple[str, str, str], ...] = (
    ("leg_joint4", "torso", TORSO),
    ("head_joint1", "head 1", HEAD),
    ("head_joint2", "head 2", HEAD),
    ("left_arm_joint1", "L arm 1", LEFT_ARM),
    ("left_arm_joint2", "L arm 2", LEFT_ARM),
    ("left_arm_joint3", "L arm 3", LEFT_ARM),
    ("left_arm_joint4", "L arm 4", LEFT_ARM),
    ("left_arm_joint5", "L arm 5", LEFT_ARM),
    ("left_arm_joint6", "L arm 6", LEFT_ARM),
    ("left_arm_joint7", "L arm 7", LEFT_ARM),
    ("left_gripper_joint", "L grip", LEFT_GRIPPER),
    ("right_arm_joint1", "R arm 1", RIGHT_ARM),
    ("right_arm_joint2", "R arm 2", RIGHT_ARM),
    ("right_arm_joint3", "R arm 3", RIGHT_ARM),
    ("right_arm_joint4", "R arm 4", RIGHT_ARM),
    ("right_arm_joint5", "R arm 5", RIGHT_ARM),
    ("right_arm_joint6", "R arm 6", RIGHT_ARM),
    ("right_arm_joint7", "R arm 7", RIGHT_ARM),
    ("right_gripper_joint", "R grip", RIGHT_GRIPPER),
)

# --- design system -----------------------------------------------------------
#
# The window is what a room full of people look at, so it is laid out rather than
# accumulated.  Two rules the previous overlay broke, both reported:
#
#   1. Every pixel is either the camera, the twin, or DESIGNED chrome.  The camera
#      is 16:9 inside a screen-shaped panel, so ~55% of the left panel is not
#      camera -- that is a HUD column by design, not "the black bit at the bottom".
#   2. Chrome over live video is a translucent scrim, never opaque furniture: it
#      must not displace the image, because the skeleton is drawn in the image's
#      own coordinates and any displacement slides it off the operator.
#
# BGR throughout, because OpenCV.
SURFACE_BG_BGR = (22, 18, 15)       # page   #0F1216
SURFACE_CARD_BGR = (33, 27, 23)     # card   #171B21
SURFACE_RAISED_BGR = (45, 38, 32)   # header #20262D
SURFACE_LINE_BGR = (61, 52, 44)     # border #2C343D
TEXT_PRIMARY_BGR = (240, 234, 230)  # #E6EAF0
TEXT_MUTED_BGR = (165, 149, 139)    # #8B95A5
TEXT_FAINT_BGR = (110, 99, 92)      # #5C636E
ACCENT_BGR = (200, 190, 90)         # teal    #5ABEC8
GOOD_BGR = (94, 197, 34)            # green   #22C55E
WARN_BGR = (11, 158, 245)           # amber   #F59E0B
FAIL_BGR = (68, 68, 239)            # red     #EF4444

#: Type scale.  Three sizes carry the HUD -- a scale with five is one nobody can
#: scan -- plus two deliberate exceptions named below, because an unnamed literal
#: is how a scale becomes five sizes by accident.
TYPE_TITLE = 0.62
TYPE_BODY = 0.46
TYPE_SMALL = 0.38
#: Labels drawn ON the camera image rather than on HUD chrome. Slightly smaller
#: than TYPE_SMALL and always outlined, because they compete with whatever the
#: room looks like instead of with a known surface.
TYPE_ONBODY = 0.45
#: The floor/standoff ticks under the clearance track. Smaller than anything else
#: on purpose: they are a scale, not a readout, and must not compete with the
#: value they annotate.
TYPE_TICK = 0.30
#: Joint rows when the column is too short for TYPE_SMALL to breathe.
TYPE_DENSE = 0.32

#: Skeleton colours, drawn ON the camera image rather than on HUD chrome, so they
#: stay saturated: they have to read over whatever the room happens to look like.
TRACKED_LINE_BGR = (0, 255, 255)
TRACKED_POINT_BGR = (0, 80, 255)
HELD_BGR = (0, 180, 255)
HELD_LINE_BGR = (120, 120, 120)

#: A joint activity bar reaches full scale at this much movement in one rendered
#: frame.  Measured, not assumed: over the 463 recorded frames of
#: `artifacts/witness-live-9/live.json`, per-frame |dq| for joints that were
#: actually moving is p50 0.0195, p90 0.0802, p95 and max both 0.1000 rad -- the
#: last being the SIM per-command step ceiling itself (`SIM_MAX_JOINT_STEP_RAD`,
#: safety/profiles.py), which binds.  At 0.05 the median moving joint reads 39%
#: and a joint at the governor's limit reads 100%.
#:
#: 0.015 was calibrated against a ~20 fps loop the pipeline no longer runs at, and
#: pinned every arm bar at 100% on a real capture, which tells an operator
#: nothing.  Note this is the SIM ceiling; the hardware profile's step limit is
#: 0.05, so on hardware this constant would be the whole cap rather than half of
#: it.  Purely a display scale: it gates nothing.
ACTIVITY_FULL_SCALE_RAD = 0.05
#: Peak-hold decay, so a quick gesture stays legible for a few frames instead of
#: flickering for one.
ACTIVITY_DECAY = 0.86
#: Below this a joint is drawn as not moving at all.
ACTIVITY_DEAD_RAD = 2e-4
#: Consecutive still SOLVES before a tracked (not held) joint is called out in
#: red, and also how far back a sibling joint's movement counts as corroboration.
#: ~5 s at the measured 8-10 solves/s.
#:
#: This counts solves, not repaints. `render()` runs at CAMERA cadence -- 2-3x the
#: solve rate -- and every repaint between two solves is handed the same joint
#: readback, so counting renders made a healthy joint "still" three times faster
#: than the constant claims. `_update_activity` now ignores a pose it has already
#: seen.
ACTIVITY_STUCK_FRAMES = 50

#: Below this row height the labels stop being readable, so the whole readout is
#: dropped rather than drawn as mush.
JOINT_PANEL_MIN_ROW = 11
#: A joint column needs room for a label, a bar and a signed value.  Below this
#: the two columns start colliding, and a colliding readout is worse than an
#: absent one -- the same rule as JOINT_PANEL_MIN_ROW, in the other axis.
JOINT_COLUMN_MIN_WIDTH = 164
#: Narrower than this and a card's label/value pair collides with itself, so the
#: column refuses the card rather than drawing mush.
CARD_MIN_WIDTH = 260

#: Display-only mirrors of the SIM clearance thresholds, so the HUD can draw the
#: floor and the standoff on the same track as the measured value.  Deliberately
#: local constants: the display must never be able to change what the supervisor
#: enforces, and importing the policy here would put a presentation module on
#: that path.  Kept in step with `safety/profiles.py` and `pipeline.py`.
CLEARANCE_FLOOR_DISPLAY_M = 0.005
CLEARANCE_STANDOFF_DISPLAY_M = 0.0065
#: Full scale of the clearance track.  4x the floor, so ordinary margins sit in
#: the middle rather than pinned at either end.
CLEARANCE_DISPLAY_FULL_SCALE_M = 0.020


def _clearance_colour(clearance_m: float) -> tuple[int, int, int]:
    """One rule for the clearance colour, used by the app bar AND the card.

    They used to disagree: the chip turned green at 6.0 mm while the card stayed
    amber until 6.5 mm, so between those two values the same number was drawn in
    two different colours in the same frame. Neither had a red state at all, so a
    clearance BELOW the floor -- the line the card itself draws in red -- still
    read as a warning rather than a failure.
    """
    if not isfinite(clearance_m) or clearance_m < CLEARANCE_FLOOR_DISPLAY_M:
        # NaN compares false against everything, so without this branch a
        # non-finite clearance would fall through to green -- the one colour it
        # must never be. Both HUD callers happen to reject non-finite values
        # before they get here (the card has its own "no supervised target this
        # frame" line), so this is a property of the FUNCTION rather than a live
        # guard: it is what makes the helper safe to call from anywhere, and it is
        # asserted directly rather than through a render.
        return FAIL_BGR
    if clearance_m < CLEARANCE_STANDOFF_DISPLAY_M:
        return WARN_BGR
    return GOOD_BGR

#: Panel size used when the screen cannot be measured (non-macOS, no pyobjc).
#: The historical default, so nothing changes where nothing can be known.
FALLBACK_PANEL_SIZE = (640, 480)
#: Ceiling on panel height, and so on render cost.  Re-measured 2026-08-27 on the
#: shipped asymmetric HUD (700 + 1020 px panels, a 1720x1080 canvas, one canvas
#: pixel per screen point on the 2x Retina panel): **p50 15.60 ms/frame**, min
#: 15.05, mean 15.59, 40 reps.  The same screen in device pixels (3440x2160) costs
#: **p50 27.43 ms** -- a **1.76x** ratio, which is the whole reason for this cap.
#: An earlier version of this comment quoted 19.8 ms for "864x1080 panels on the
#: current HUD"; the current HUD has no such configuration -- that was an
#: equal-panel measurement relabelled, and it is replaced above.  Render at
#: points, never at device pixels, and stop growing there -- a larger screen gets
#: the same canvas scaled up rather than a slower frame.
MAX_PANEL_HEIGHT = 1080
#: Share of the canvas width given to the twin.  The two panels are NOT equal.
#:
#: The twin is the wide subject: across the 463 recorded poses below it spans
#: 1.34 m laterally at the median and 1.87 m at its widest, while a
#: portrait panel's horizontal field of view is its vertical one scaled by the
#: aspect -- so the narrower the panel, the further back the camera has to sit and
#: the smaller the robot gets.
#:
#: Measured over ALL 463 recorded poses of `artifacts/witness-live-9/live.json`
#: (field `observed_joints_rad`), each rendered through segmentation with a 2x
#: OVERSCAN ON BOTH AXES so a pixel outside the panel is measurable rather than
#: saturating at the edge.  Overscan WIDENS THE LENS (fovy scaled by the same
#: factor as the image) instead of moving the camera back: moving it back keeps
#: the robot's size but flattens its perspective, and a hand reaching toward the
#: camera then stops crossing the edge -- which understated the clipping rate by
#: 18 points when it was done that way.
#:
#: Two different things are worth counting and they are NOT the same number --
#: "off the left or right edge", which is the hands, and "off any edge", which at
#: close distances is the wheeled base leaving the bottom of frame on every single
#: pose.  Robot height is the unclipped bounding box, in the overscan render:
#:
#:     panel  860, distance 2.20   L/R 61.8%   any 100.0%   robot 743 px  <- HEAD
#:     panel 1020, distance 2.90   L/R  3.2%   any   3.2%   robot 540 px  <- chosen
#:
#: So the shipped twin is 27% SMALLER, not bigger: an earlier draft of this
#: comment claimed the opposite by comparing on-screen extents, which for every
#: other row in the table are CLIPPED at the panel edge.  Smaller and whole beats
#: bigger and cut off -- but the trade is real and it is this one.
#:
#: The camera panel loses that width and does not miss it: the operator's own
#: image is a 16:9 letterbox in a portrait panel either way, and the space under
#: it is HUD, which gets no worse for being narrower.  The TOTAL canvas width is
#: unchanged, so a two-panel canvas is still `2 * width` and every caller and
#: recording that assumed that still holds.
ROBOT_PANEL_SHARE = 0.593
#: Smallest panel the split can produce two non-empty halves from. At width 1 the
#: twin panel rounded to zero, `mujoco.Renderer` accepted it, and `cv2.cvtColor`
#: then raised out of `render()` past every degradation guard.
MIN_PANEL_WIDTH = 8
#: The window is opened at this fraction of the free screen area.  The canvas is
#: now screen-shaped, so opening it at 1:1 would swallow the whole desktop.
WINDOW_OPEN_FRACTION = 0.72


def _screen_content_size() -> tuple[float, float] | None:
    """Size in POINTS of the content view an OpenCV window gets when fullscreen.

    A fullscreen window takes the whole screen except, on a notched display, the
    strip beside the notch: the safe-area inset.  That inset under-reports the
    strip slightly (32 pt against a measured 37), which leaves the canvas 0.5%
    narrow -- an 8-point sliver down one edge, measured fill 0.995, versus the
    864-device-pixel dead band this replaces.  ``visibleFrame`` estimates the
    strip more closely but moves with the Dock, and the canvas size is baked into
    every recording, so the deterministic inset wins.  Returns ``None`` off macOS
    or without pyobjc, where the caller keeps the historical fixed panel size.
    """
    try:
        screen = import_module("AppKit").NSScreen.mainScreen()
        size = screen.frame().size
        width = float(size.width)
        height = float(size.height) - float(screen.safeAreaInsets().top)
    except Exception:  # noqa: BLE001 - optional platform capability
        return None
    if not (isfinite(width) and isfinite(height)) or width <= 0 or height <= 0:
        return None
    return width, height


def _fitted_panel_size() -> tuple[int, int]:
    """Panel size whose two-panel canvas has the same aspect as the screen.

    This is the whole fullscreen fix.  The Cocoa backend fit-scales the canvas
    into the content view and anchors it BOTTOM-LEFT, so every bit of aspect
    mismatch collapses into a single dead band at the top: the 2.667-aspect
    1280x480 canvas filled 0.600 of a fullscreen window and left 864 device
    pixels of black above itself.  Give the canvas the content view's aspect and
    the fit-scale becomes a fill-scale; the bottom-anchoring then has no slack to
    put anywhere.  ``WINDOW_FREERATIO``/``WINDOW_KEEPRATIO`` cannot do this --
    ``WND_PROP_ASPECT_RATIO`` reads back -1.0 (not implemented) on this Cocoa
    build, so those flags are inert.
    """
    measured = _screen_content_size()
    if measured is None:
        return FALLBACK_PANEL_SIZE
    screen_width, screen_height = measured
    # Screen-shaped, so a FULLSCREEN window fills rather than anchoring the canvas
    # bottom-left with a dead band above -- a 16:9 canvas did exactly that
    # (reported 2026-08-27, ~700 px of black over the whole top of the display).
    # The camera's own letterbox is dealt with in _fit_camera by anchoring the
    # image to the TOP of its panel, so the leftover is one block at the bottom
    # that the HUD already writes into, rather than black bands above AND below.
    height = int(min(screen_height, MAX_PANEL_HEIGHT))
    height -= height % 2
    # canvas is two panels wide, so canvas aspect == screen aspect means
    # 2 * panel_width / panel_height == screen_width / screen_height.
    width = int(round(screen_width * height / screen_height / 2.0))
    # MIN_PANEL_WIDTH, not 2: the constructor REJECTS anything narrower, so a
    # lower floor here would hand `StudioDisplay` a width it raises on and turn a
    # freak screen geometry into a traceback out of `preview`.
    if width < MIN_PANEL_WIDTH or height < 2:
        return FALLBACK_PANEL_SIZE
    return width, height


def _window_open_size(width: int, height: int) -> tuple[int, int]:
    """Windowed size for a screen-shaped canvas: shrunk to fit, aspect kept."""
    try:
        frame = import_module("AppKit").NSScreen.mainScreen().visibleFrame().size
        limit_width = float(frame.width) * WINDOW_OPEN_FRACTION
        limit_height = float(frame.height) * WINDOW_OPEN_FRACTION
    except Exception:  # noqa: BLE001 - optional platform capability
        return width, height
    scale = min(1.0, limit_width / max(1, width), limit_height / max(1, height))
    return max(1, int(width * scale)), max(1, int(height * scale))


@dataclass
class _Column:
    """A stack of cards that knows where the next one goes.

    The cards used to hand each other position through three different
    instance attributes -- one carrying a full placement, one a delta, one a
    height-plus-gap that was written in only ONE of two layouts, with the other
    layout duplicating a neighbour's height formula by hand.  That is three
    protocols for one job, and the duplicated formula is the same class of bug
    that once put a card on top of the operator.

    A card asks a column for room; the column either gives it a rect and moves
    on, or says no and the card tries the next column.  "A card that was not
    drawn does not shift the one below it" is then true by construction rather
    than by remembering to reset three attributes.
    """

    x: int
    top: int
    width: int
    floor: int
    min_width: int = 0

    def place(self, height: int, *, gap: int = 12) -> tuple[int, int, int] | None:
        """Reserve ``height`` and return ``(left, top, width)``, or ``None``."""
        if self.width < self.min_width or self.top + height > self.floor:
            return None
        rect = (self.x, self.top, self.width)
        self.top += height + gap
        return rect


@dataclass(frozen=True)
class CalibrationStatus:
    """What calibration is still waiting for, in operator terms.

    Calibration used to report itself only to a terminal behind the window
    ("calibrating... frame 15: confidence=0.01<0.5"), and once took 94 frames
    with nothing on screen to act on.  Every field here answers "what do I do
    differently?".
    """

    frames: int = 0
    missing_landmarks: tuple[str, ...] = ()
    required_landmarks: int = 0
    identity: str = ""
    identity_ok: bool = False
    confidence: float = 0.0
    required_confidence: float = 0.0
    #: The single next action, in plain words ("STEP BACK - can't see: left
    #: wrist").  The caller already computes one for the terminal; this puts the
    #: same sentence under the checklist where the operator is actually looking.
    hint: str = ""

    @property
    def visible_landmarks(self) -> int:
        return max(0, self.required_landmarks - len(self.missing_landmarks))

    @property
    def confidence_ok(self) -> bool:
        return self.confidence >= self.required_confidence


class StudioDisplay:
    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        #: Per-panel size.  ``None`` (the default) fits the two-panel canvas to
        #: the screen's aspect so a fullscreen window has no dead band; an
        #: explicit size is honoured as given.
        width: int | None = None,
        height: int | None = None,
        #: Mirror BOTH panels, giving the operator a selfie view of themselves.
        #: Off by default -- see the note in `render`.
        mirror_camera: bool = False,
        #: Matches the runtime gate the pipeline applies, so the readout says
        #: what the mapper will actually do rather than something adjacent.
        min_confidence: float = 0.35,
    ) -> None:
        self._min_confidence = float(min_confidence)
        self._cv2 = import_module("cv2")
        if width is None or height is None:
            fitted_width, fitted_height = _fitted_panel_size()
            width = fitted_width if width is None else width
            height = fitted_height if height is None else height
        # `width` is the NOMINAL half-canvas. The real split is asymmetric (see
        # ROBOT_PANEL_SHARE); their sum is exactly `2 * width`, so the canvas
        # contract is unchanged.
        if width < MIN_PANEL_WIDTH:
            raise ValueError(
                f"panel width must be at least {MIN_PANEL_WIDTH} px; got {width}. "
                "Below that the split leaves a zero-width panel and the renderer "
                "produces an empty image."
            )
        # No clamping: the MIN_PANEL_WIDTH guard above already makes both panels
        # comfortably positive (at the floor of 8 the split is 9 and 7), and a
        # clamp that can never fire is a claim that the guard might not hold.
        robot_width = round(2 * width * ROBOT_PANEL_SHARE)
        camera_width = 2 * width - robot_width
        self._renderer = mujoco.Renderer(model, height=height, width=robot_width)
        self._robot_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self._robot_camera)
        # Match the operator's front-facing webcam view. The previous default
        # looked steeply down at the robot from 45 degrees and made an overhead
        # reach read visually as a small forward-arm nudge.
        # Raised from z=1.0: the panel is 16:9, so centring on the chassis spent
        # height on the wheeled base while the arms -- the thing being watched --
        # sat in the top half.
        # Framing is set by the HORIZONTAL fit, not the vertical one, and it was
        # got wrong twice by reasoning about height.  Same harness and same 463
        # poses as ROBOT_PANEL_SHARE above; percentages are frames with a robot
        # pixel off the LEFT or RIGHT edge, then off ANY edge:
        #
        #     panel  860, lookat 1.00, distance 2.20   L/R 61.8%   any 100.0%
        #     panel 1020, lookat 1.00, distance 2.90   L/R  3.2%   any   3.2%
        #
        # The first row is what HEAD commits. The 100% column is the wheeled base
        # leaving the bottom of frame on every pose at that distance -- also true
        # before, also unmentioned until it was measured. At the shipped setting
        # nothing leaves the frame on 96.8% of poses, and the base never does.
        # An intermediate "tightening" (lookat 1.05, distance 2.05) made the hands
        # worse and was written up as keeping "~48 px of horizontal margin" on the
        # strength of a measurement of the VERTICAL axis; it is gone.
        self._robot_camera.lookat[:] = (0.05, 0.0, 1.00)
        # Pulled back from 2.2: at that distance the twin filled the panel and
        # arm excursions ran off the edge, so the movement being demonstrated was
        # partly out of shot (reported 2026-08-26).
        self._robot_camera.distance = 2.90
        self._robot_camera.azimuth = 180.0
        self._robot_camera.elevation = 0.0
        #: The CAMERA panel's width. Most drawing happens in it, and the twin
        #: panel starts at exactly this x, so `self._width` remains the right
        #: origin for anything drawn on the twin's side.
        self._width = camera_width
        self._robot_width = robot_width
        self._height = height
        #: Where the webcam image actually landed inside the camera panel
        #: (x, y, w, h).  The panel is not the camera's aspect, so the image is
        #: letterboxed into it and the skeleton has to follow it there.
        self._camera_box = (0, 0, camera_width, height)
        self._mirror_camera = mirror_camera
        #: Open the preview window fullscreen.  Set by the CLI's --fullscreen.
        self.fullscreen = False
        self._window = "Galbot Motion Studio — SIMULATION ONLY"
        #: Name of the capture device, drawn in the banner.  An operator cannot
        #: otherwise tell a built-in webcam from a Continuity iPhone that macOS
        #: handed over, and the terminal line scrolls away (2026-08-26).
        self.camera_label = ""
        self._window_opened = False
        self._font = self._cv2.FONT_HERSHEY_SIMPLEX
        # Per-joint motion state for the activity readout. Peak-hold with decay,
        # plus a still-frame counter so a joint that never moves can be called
        # out rather than merely drawn short.
        #: Base row height of the app bar. `self._height` is fixed at construction,
        #: so this is too -- it was being recomputed twice a frame and once more in
        #: a test that hardcoded the formula.
        self._bar_unit = max(44, min(64, height // 16))
        #: The app bar's height this frame. It grows to two rows when the hold
        #: words do not fit beside the status pill.
        self._app_bar_height = self._bar_unit
        #: The card stacks, rebuilt each frame by `_place`.
        self._columns: tuple[_Column, _Column] | None = None
        #: Reused INTERNAL scratch.  A repaint runs at camera cadence, so clearing
        #: an 860x1080 panel that is about to be overwritten anyway, and rebuilding
        #: a solid fill image for every gradient band, is pure cost on the
        #: operator's latency path.  Neither buffer is ever returned to a caller --
        #: see the canvas comment in `render`.
        self._scrim_fill_cache: np.ndarray | None = None
        self._camera_panel: np.ndarray | None = None
        self._joint_activity: dict[str, float] = {}
        self._joint_previous: dict[str, float] = {}
        self._joint_still_frames: dict[str, int] = {}
        #: Solves since ANY joint of a control group moved. A joint is only called
        #: dead while its own group is demonstrably moving without it.
        self._group_still_frames: dict[str, int] = {}
        #: An overlay bug must degrade the window, never end the session: a
        #: previous run died in a traceback mid-session. Each element is drawn
        #: inside :meth:`_safe`, which counts failures here instead of raising.
        self.overlay_errors = 0
        self._overlay_error_note: str | None = None

    def render(
        self,
        camera_bgr: Any,
        data: mujoco.MjData,
        *,
        status: str,
        telemetry: str = "",
        observation: HumanObservation | None = None,
        held_groups: Iterable[str] | None = None,
        held_grippers: Iterable[str] | None = None,
        saturated_groups: Iterable[str] | None = None,
        reasons: Iterable[str] | None = None,
        joints: Mapping[str, float] | None = None,
        calibration: CalibrationStatus | None = None,
        clearance_m: float | None = None,
    ) -> np.ndarray:
        self._begin_frame()
        self._renderer.update_scene(data, camera=self._robot_camera)
        robot_rgb = self._renderer.render()
        # cvtColor, not `[:, :, ::-1]`: the reversed view is negative-stride, so
        # everything downstream copies it byte by byte.  Measured on this machine
        # at the shipped 1020x1080 twin panel, 200 reps each, interleaved:
        # materialising the negative-stride view costs **p50 3.78 ms** (min 3.53,
        # mean 3.80); cvtColor costs **p50 0.36 ms** (min 0.26, mean 0.37), and
        # the two outputs are array-equal.  An earlier version of this comment
        # gave the minima while calling them costs, one of them beside a p50 it
        # could not be.  This is on the repaint path, which runs at camera
        # cadence.
        robot_bgr = self._cv2.cvtColor(robot_rgb, self._cv2.COLOR_RGB2BGR)
        try:
            camera = self._fit_camera(camera_bgr)
        except Exception as error:  # noqa: BLE001 - a bad frame must not end the session
            # `_safe` promises an overlay bug degrades the window rather than
            # ending the session, and the frame path was the one place that
            # promise did not hold: a 2-D greyscale, 4-channel BGRA, or non-array
            # camera frame raised straight out of `render`.  Nothing validates the
            # shape of what `cv2.VideoCapture.read()` returns, so show the twin
            # and say the camera panel is gone.
            self.overlay_errors += 1
            self._overlay_error_note = f"camera frame: {type(error).__name__}"
            print(f"camera frame unusable, showing the twin only: {error!r}")
            camera = np.empty((self._height, self._width, 3), dtype=robot_bgr.dtype)
            camera[:] = SURFACE_BG_BGR
            self._camera_box = (0, 0, self._width, self._height)
        if self._mirror_camera:
            # Mirror BOTH or NEITHER -- never one. The guarantee that matters is
            # that the two panels agree about which side is which; a correct
            # mapping with only the camera mirrored reads as a left/right swap
            # (reported 2026-08-26).
            #
            # Measured with segmentation rendering at the shipped framing: with
            # neither panel flipped, the robot's LEFT arm sits at x=660 of 1020
            # (right half) and an unmirrored operator's left hand is likewise on
            # the right of their panel -- they agree. With both flipped, both move
            # to the left half -- they also agree.
            #
            # Neither is the default because flipping the render also flips the
            # chassis, and the GALBOT wordmark on the base then reads "TOBJAG" in
            # every frame of the window and of the retained composite.mp4. The
            # selfie view is worth having for solo operating and is one flag away
            # (`--mirror-camera`); it is not worth a reversed brand mark in a
            # demo. Presentation only either way: no joint value, command or
            # decision changes.
            camera = self._cv2.flip(camera, 1)
            robot_bgr = self._cv2.flip(robot_bgr, 1)
        held = frozenset(
            str(name) for name in (*tuple(held_groups or ()), *tuple(held_grippers or ()))
        )
        saturated = frozenset(str(name) for name in (saturated_groups or ()))
        if observation is not None:
            self._safe(
                "skeleton",
                self._draw_pose,
                camera,
                observation,
                mirrored=self._mirror_camera,
                held=held,
            )
        # A FRESH canvas every frame, deliberately.  Reusing one buffer here saved
        # about a millisecond and made `render()` return an array the next call
        # overwrites -- so two canvases held at once were silently the same
        # pixels.  Two display tests caught it immediately, and the video writer
        # and the preview-frame manifest both hold a canvas.  A returned frame is
        # the caller's; the internal scratch below is ours to reuse.
        canvas = np.concatenate((camera, robot_bgr), axis=1)

        # A green ALLOW next to a silently frozen limb is the exact thing that
        # cost the operator hours, so a partial frame never reads as a clean one.
        partial = status == "ALLOW" and bool(held or saturated)
        # `clean` is the one bit the app bar needs beyond the colour: only a frame
        # with nothing held and nothing saturated gets the solid green pill. Passed
        # as a boolean rather than re-derived from the pill's own text, which is
        # what it used to be -- a string comparison standing in for the guarantee
        # this overlay exists to make.
        clean = status == "ALLOW" and not partial
        if status == "FAULT":
            state_color = FAIL_BGR
        elif clean:
            state_color = GOOD_BGR
        elif status == "CALIBRATING":
            state_color = ACCENT_BGR
        else:
            state_color = WARN_BGR
        self._safe(
            "hud",
            self._draw_chrome,
            canvas,
            state_color=state_color,
            clean=clean,
            status=status,
            telemetry=telemetry,
            held=held,
            saturated=saturated,
            clearance_m=clearance_m,
        )
        # Calibration first, so it takes the top of the card column: while it is
        # up it IS the message. It used to be an opaque banner across the top of
        # the camera panel, which buried the operator's nose, both eyes and both
        # shoulders -- while telling them to "centre head and chest in frame".
        if calibration is not None:
            self._safe("calibration", self._draw_calibration, canvas, calibration)
        self._safe("reason", self._draw_reason, canvas, status, held | saturated, reasons)
        self._safe(
            "tracking card",
            self._draw_tracking_card,
            canvas,
            observation,
            self._min_confidence,
            held,
            saturated,
            None if calibration is None else calibration.required_confidence,
        )
        # SAFETY before JOINT ACTIVITY, deliberately. The joint readout is the
        # flexible card -- it shrinks its row height to fit whatever is left -- so
        # it goes last and takes the remainder. Placed the other way round it ate
        # the whole column and the clearance readout vanished at 640x480, which is
        # every install without pyobjc. It is also the better hierarchy: the
        # supervisor's margin outranks a per-joint readout.
        self._safe("safety card", self._draw_safety_card, canvas, clearance_m, status, held)
        self._safe("joint activity", self._draw_joint_activity, canvas, joints, held)
        if self._overlay_error_note is not None:
            # Above the twin's caption, not in the app bar: the chip row is laid
            # out right-to-left across that same band and would sit on top of it.
            self._safe(
                "degraded note",
                self._pill,
                canvas,
                (self._width + 16, max(0, self._height - 58)),
                f"OVERLAY DEGRADED ({self._overlay_error_note})",
                fg=FAIL_BGR,
                bg=SURFACE_CARD_BGR,
            )
        return canvas

    def _draw_chrome(
        self,
        canvas: np.ndarray,
        *,
        state_color: tuple[int, int, int],
        clean: bool,
        status: str,
        telemetry: str,
        held: frozenset[str],
        saturated: frozenset[str],
        clearance_m: float | None = None,
    ) -> None:
        """The app bar and the two panel captions.

        Drawn as translucent scrims over the live panels rather than as bars that
        push them down: the skeleton is registered to the camera image's own
        rectangle, so moving that rectangle would slide the drawn skeleton off the
        operator.
        """
        canvas_width = canvas.shape[1]
        bar = self._bar_unit

        # The twin renders against MuJoCo's own sky, which leaves a large flat
        # field above its head.  Fading that into the page colour turns empty sky
        # into a deliberate gradient and makes the robot itself the brightest
        # thing on its side of the screen.
        #
        # Drawn FIRST. A long "FROZEN: ..." pill legitimately runs past the panel
        # divider, and this scrim was fading out the half of it that crossed over.
        self._scrim(
            canvas, (self._width, 0, canvas_width, int(self._height * 0.30)),
            from_top=True,
        )

        # The hold words are the load-bearing part of this bar, so the layout is
        # built around them: everything else yields, and if they still do not fit
        # they get a row of their own rather than being clipped or painted over.
        suffixes = []
        if held:
            suffixes.append(f"{len(held)} LIMB{'S' if len(held) != 1 else ''} HELD")
        if saturated:
            names = ", ".join(
                GROUP_LABELS.get(group, group.upper()) for group in sorted(saturated)
            )
            suffixes.append(f"{names} SATURATED")
        suffix = "" if not suffixes else f" - {' / '.join(suffixes)}"
        status_text = f"{status}{suffix}"
        hold_pills = [
            (label, ", ".join(
                GROUP_LABELS.get(group, group.upper()) for group in sorted(names)
            ))
            for label, names in (("FROZEN", held), ("LIMITED", saturated))
            if names
        ]

        def pill_width(text: str, scale: float) -> int:
            (tw, _), _ = self._cv2.getTextSize(text, self._font, scale, 1)
            return tw + 16

        title = "GALBOT MOTION STUDIO"
        title_width = pill_width(title, TYPE_BODY) + 14
        needed = 18 + pill_width(status_text, TYPE_BODY) + 10
        show_title = title_width + needed + 120 < canvas_width
        # The wordmark is a nicety and drops first. The SIMULATION ONLY badge is a
        # claim about what the robot is, and it is the first thing an engineer
        # looks for -- so it is judged on its OWN width, never on the wordmark's,
        # and it survives every size at which anything is drawn at all. It used to
        # require `show_title`, so a narrow window dropped the safety badge and
        # kept nothing in its place; in fullscreen there is no title bar to fall
        # back on.
        show_sim_badge = (
            18 + pill_width("SIMULATION ONLY", TYPE_SMALL) + needed + 10 < canvas_width
        )
        hold_width = sum(pill_width(f"{a}: {b}", TYPE_BODY) + 10 for a, b in hold_pills)
        second_row = bool(hold_pills) and (
            18 + (title_width if show_title else 0)
            + (pill_width("SIMULATION ONLY", TYPE_SMALL) + 10 if show_sim_badge else 0)
            + pill_width(status_text, TYPE_BODY) + 10 + hold_width + 140 > canvas_width
        )
        bar_rows = 2 if second_row else 1
        bar_height = bar * bar_rows - (bar // 3 if second_row else 0)
        # Published so the WHY banner and the calibration card stack below the bar
        # whichever height it turned out to be, rather than under a one-row
        # assumption that a second row silently invalidates.
        self._app_bar_height = bar_height

        self._scrim(canvas, (0, 0, canvas_width, bar_height + 26), from_top=True)
        self._cv2.line(
            canvas, (0, bar_height), (canvas_width, bar_height), SURFACE_LINE_BGR, 1,
            self._cv2.LINE_AA,
        )
        baseline = bar - 14
        cursor = 18
        if show_title:
            self._cv2.putText(
                canvas, title, (cursor, baseline + 4), self._font,
                TYPE_BODY, TEXT_PRIMARY_BGR, 1, self._cv2.LINE_AA,
            )
            cursor += title_width
        if show_sim_badge:
            cursor = self._pill(
                canvas, (cursor, baseline - 10), "SIMULATION ONLY",
                fg=ACCENT_BGR, bg=SURFACE_CARD_BGR,
            ) + 10
        # The status pill and the hold pills are set at TYPE_BODY, not TYPE_SMALL.
        # The overlay this replaced drew the same words at scale 0.7 thickness 2 --
        # 19 px tall, readable across a room -- and a first pass at this design
        # shrank them to 10 px with a tenth of the ink. These strings are the
        # module's entire reason for existing; the telemetry chips can be small,
        # these cannot.
        cursor = self._pill(
            canvas, (cursor, baseline - 13),
            self._clip(status_text, max(40, canvas_width - cursor - 28), TYPE_BODY, 1),
            fg=(18, 18, 18) if clean else state_color,
            bg=state_color if clean else SURFACE_CARD_BGR,
            scale=TYPE_BODY,
        )
        pill_floor = cursor + 24
        hold_cursor, hold_baseline = (18, baseline + bar - bar // 3) if second_row else (
            cursor + 10, baseline - 13
        )
        for label, pretty in hold_pills:
            room = canvas_width - hold_cursor - 28
            if room < 60:
                # No room left even for a stub. Better to say nothing than to draw
                # a pill off the edge of the canvas, where "LIMI..." tells nobody
                # which limbs are affected.
                break
            hold_cursor = self._pill(
                canvas, (hold_cursor, hold_baseline),
                self._clip(f"{label}: {pretty}", room, TYPE_BODY, 1),
                fg=WARN_BGR, bg=SURFACE_CARD_BGR, scale=TYPE_BODY,
            ) + 10
        if not second_row:
            pill_floor = hold_cursor + 24

        # Right side: the numbers a reviewer asks about, as separate chips rather
        # than one run-on line.  Laid out right-to-left so the rightmost chip is
        # always flush with the edge whatever the widths turn out to be, and
        # STOPPING where the pills ended.  The chips are drawn second and their
        # backgrounds are opaque, so without that budget they simply erase
        # whatever the left side wrote -- and the left side is where the hold
        # words live.  Measured before the guard: at six held groups five chip
        # rectangles overlapped the pills and the hold words were gone from the
        # canvas, while `putText` had still been handed them, so the tests saw
        # nothing wrong.  Telemetry is the thing that yields.
        chips: list[tuple[str, tuple[int, int, int]]] = [("q/Esc exits", TEXT_MUTED_BGR)]
        if self.camera_label:
            chips.insert(0, (self.camera_label, TEXT_PRIMARY_BGR))
        if clearance_m is not None and isfinite(clearance_m):
            chips.insert(0, (f"clearance {clearance_m * 1000:.1f} mm",
                             _clearance_colour(clearance_m)))
        for part in reversed([p.strip() for p in telemetry.split("|") if p.strip()]):
            chips.insert(0, (part, TEXT_PRIMARY_BGR))
        x = canvas_width - 18
        for text, colour in reversed(chips):
            text = self._clip(text, canvas_width // 4, TYPE_SMALL, 1)
            (tw, th), _ = self._cv2.getTextSize(text, self._font, TYPE_SMALL, 1)
            if x - (tw + 20) < pill_floor:
                # Out of room. Drop this chip and every wider one behind it rather
                # than overwrite a safety word.
                break
            x -= tw + 20
            # Opaque, not translucent: these chips sit over the twin's bright sky
            # as often as over the dark camera frame, and a chip legible on one
            # and not the other is a chip nobody trusts.
            self._round_rect(
                canvas, (x, baseline - 13, x + tw + 20, baseline - 13 + th + 14),
                SURFACE_BG_BGR, SURFACE_LINE_BGR, radius=5,
            )
            self._cv2.putText(
                canvas, text, (x + 10, baseline + 3), self._font, TYPE_SMALL,
                colour, 1, self._cv2.LINE_AA,
            )
            x -= 8

        # Panel captions, on their own scrims so they read over any content.
        caption = 34
        self._scrim(
            canvas, (self._width, self._height - caption - 26, canvas_width, self._height),
            from_top=False,
        )
        self._cv2.putText(
            canvas, "ROBOT DIGITAL TWIN", (self._width + 18, self._height - 13),
            self._font, TYPE_SMALL, TEXT_PRIMARY_BGR, 1, self._cv2.LINE_AA,
        )
        camera_x, camera_y, camera_w, camera_h = self._camera_box
        # Clamped below the app bar: a very wide, short source (a 3840x600 capture,
        # say) fits to a shallow image whose bottom edge lands inside the bar, and
        # this scrim -- drawn last -- would fade the hold words sitting on it.
        label_y = min(self._height - 12, camera_y + camera_h - 12)
        if label_y - 34 < bar_height + 4:
            return
        self._scrim(
            canvas,
            (camera_x, max(0, label_y - 34), camera_x + camera_w, min(self._height, label_y + 10)),
            from_top=False,
        )
        self._cv2.putText(
            canvas, "HUMAN TRACKING", (camera_x + 18, label_y), self._font,
            TYPE_SMALL, TEXT_PRIMARY_BGR, 1, self._cv2.LINE_AA,
        )

    def _fit_camera(self, camera_bgr: np.ndarray) -> np.ndarray:
        """Put the webcam frame in the camera panel without distorting anyone.

        The panel is not the camera's aspect -- 1920x1080 into 640x480 is a
        horizontal squeeze to 0.75, and the operator was visibly narrowed in
        every recorded frame.  Fit the image and letterbox it instead, and record
        where it landed so the skeleton is drawn on the operator rather than
        beside them.
        """
        panel_width, panel_height = self._width, self._height
        if getattr(camera_bgr, "dtype", None) != np.uint8:
            # Every OpenCV text and shape call this module makes requires CV_8U.
            # A float32 or uint16 frame passes the shape checks, reaches the
            # overlay, and then fails inside `putText` -- which took the status
            # pill, the FROZEN pill and even the OVERLAY DEGRADED badge with it,
            # because the app bar is a single `_safe` element. Reject it here,
            # where the caller already handles a malformed frame.
            raise TypeError(
                f"camera frame must be 8-bit; got {getattr(camera_bgr, 'dtype', type(camera_bgr))}"
            )
        source_height, source_width = camera_bgr.shape[:2]
        if source_width <= 0 or source_height <= 0:
            self._camera_box = (0, 0, panel_width, panel_height)
            panel = np.empty((panel_height, panel_width, 3), dtype=np.uint8)
            panel[:] = SURFACE_BG_BGR
            return panel
        scale = min(panel_width / source_width, panel_height / source_height)
        inner_width = max(1, min(panel_width, int(round(source_width * scale))))
        inner_height = max(1, min(panel_height, int(round(source_height * scale))))
        fitted = self._cv2.resize(camera_bgr, (inner_width, inner_height))
        if (inner_width, inner_height) == (panel_width, panel_height):
            self._camera_box = (0, 0, panel_width, panel_height)
            return fitted
        # The page colour, not black.  A 16:9 image in a screen-shaped panel always
        # leaves room underneath; painting it as a surface is what turns "the black
        # bit at the bottom" into the HUD column the readouts live in.
        #
        # The buffer is reused and only the parts the image does NOT cover are
        # repainted: the image region is fully overwritten below, so clearing it
        # first was 4.6 ms a frame of writing bytes that were about to be replaced.
        panel = self._camera_panel
        if (
            panel is None
            or panel.shape[:2] != (panel_height, panel_width)
            or panel.dtype != camera_bgr.dtype
        ):
            panel = np.empty((panel_height, panel_width, 3), dtype=camera_bgr.dtype)
            self._camera_panel = panel
        left = (panel_width - inner_width) // 2
        # Top-anchored, not centred: centring split the leftover into a band above
        # AND below the operator, which reads as two black bars. Anchoring to the
        # top collects it into one region at the bottom, which is where the shape
        # and hold labels are drawn anyway.
        top = 0
        if inner_height < panel_height:
            panel[inner_height:, :] = SURFACE_BG_BGR
        if left > 0:
            panel[:inner_height, :left] = SURFACE_BG_BGR
        right = left + inner_width
        if right < panel_width:
            panel[:inner_height, right:] = SURFACE_BG_BGR
        panel[top:top + inner_height, left:right] = fitted
        self._camera_box = (left, top, inner_width, inner_height)
        return panel

    def show_canvas(self, canvas: np.ndarray) -> bool:
        if not self._window_opened and hasattr(self._cv2, "namedWindow"):
            # AUTOSIZE (the imshow default) pins the image at its native size, so
            # the macOS fullscreen button grows the window and letterboxes the
            # canvas instead of filling it.  WINDOW_NORMAL lets OpenCV scale the
            # frame to whatever size the operator gives the window; no ratio flag
            # is passed with it because WND_PROP_ASPECT_RATIO reads back -1.0 on
            # this Cocoa build, so FREERATIO and KEEPRATIO both do nothing.  What
            # actually fills the window is the canvas aspect (_fitted_panel_size).
            try:
                self._cv2.namedWindow(self._window, self._cv2.WINDOW_NORMAL)
                h, w = canvas.shape[:2]
                self._cv2.resizeWindow(self._window, *_window_open_size(int(w), int(h)))
                if self.fullscreen:
                    self._cv2.setWindowProperty(
                        self._window,
                        self._cv2.WND_PROP_FULLSCREEN,
                        self._cv2.WINDOW_FULLSCREEN,
                    )
            except Exception:  # noqa: BLE001 - optional OpenCV UI capability
                pass
        self._cv2.imshow(self._window, canvas)
        self._window_opened = True
        if (self._cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            return False
        # Closing the native window is an intentional operator action, not a
        # transient no-key event.  Without this check OpenCV returns -1 from
        # waitKey, the loop recreates the window, and the only intuitive stop
        # gesture on stage silently fails.  Some headless OpenCV backends do
        # not implement the property query; in that case preserve the prior
        # q/Esc behaviour rather than turning a display capability gap into a
        # control failure.
        try:
            return self._cv2.getWindowProperty(
                self._window, self._cv2.WND_PROP_VISIBLE
            ) >= 1.0
        except Exception:  # noqa: BLE001 - optional OpenCV UI capability
            return True

    @property
    def mirror_camera(self) -> bool:
        """Whether both panels are mirrored.  Set by the CLI's --mirror-camera."""
        return self._mirror_camera

    @mirror_camera.setter
    def mirror_camera(self, value: bool) -> None:
        self._mirror_camera = bool(value)

    def _begin_frame(self) -> None:
        """Reset the per-frame overlay-health flag.

        `_overlay_error_note` used to latch for the whole session: one transient
        bad frame stamped OVERLAY DEGRADED on every frame after it, for ever, even
        once the cause was gone.  A HUD whose thesis is "never lie about state"
        must not lie about its own.  `overlay_errors` still accumulates -- that is
        a session total and should -- but the BADGE now reflects this frame.
        """
        self._overlay_error_note = None
        # A one-row bar is the default. `_draw_chrome` raises it when the hold
        # words need a second row; if that draw fails inside `_safe`, the cards
        # below must not stack against a stale offset from a previous frame.
        self._app_bar_height = self._bar_unit
        self._columns = None

    def _safe(self, element: str, draw: Any, *args: Any, **kwargs: Any) -> None:
        """Draw one overlay element, or lose that element and keep the session.

        The live loop must survive a presentation bug.  A failure here is
        recorded and surfaced on the canvas rather than raised into the runner.
        """
        try:
            draw(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - presentation must not be fatal
            self.overlay_errors += 1
            if self._overlay_error_note is None:
                self._overlay_error_note = f"{element}: {type(error).__name__}"
                # Printed once per distinct element per frame, not once per
                # session: a repeating failure is worth seeing repeat.
                print(f"overlay element failed and was skipped ({element}): {error!r}")

    def _draw_pose(
        self,
        camera: np.ndarray,
        observation: HumanObservation,
        *,
        mirrored: bool,
        held: frozenset[str] = frozenset(),
    ) -> None:
        panel_width = camera.shape[1]
        # Landmarks are normalised to the camera FRAME, which is letterboxed
        # inside the panel; mapping them to the panel instead would slide the
        # skeleton off the operator by the size of the bars.
        left, top, width, height = self._camera_box
        by_name = {landmark.name: landmark for landmark in observation.landmarks}

        def point(name: str) -> tuple[int, int] | None:
            """Panel position of a landmark the camera actually SAW.

            MediaPipe reports a full skeleton even for parts outside the image,
            inferring coordinates well beyond [0, 1] -- and it is confident about
            them. Drawn literally, an operator close to the lens got a shoulder
            line stretching the full width of the panel to two points that are not
            on their body, plus HELD labels anchored to those points, which is
            what made the overlay look broken rather than merely busy.

            Off-frame and low-confidence landmarks are therefore not drawn at all.
            A missing limb is honest; a limb drawn in the wrong place is not.
            """
            landmark = by_name.get(name)
            if landmark is None:
                return None
            x, y = landmark.normalized_xyz[0], landmark.normalized_xyz[1]
            if not (-0.02 <= x <= 1.02 and -0.02 <= y <= 1.02):
                return None
            if min(landmark.visibility, landmark.presence) < self._min_confidence:
                return None
            return (
                left + int(round(((1.0 - x) if mirrored else x) * width)),
                top + int(round(y * height)),
            )

        def frozen(group: str | None) -> bool:
            # The shared shoulder span belongs to neither arm alone, so it only
            # goes grey once nothing is driving either arm.
            if group is None:
                return LEFT_ARM in held and RIGHT_ARM in held
            return group in held

        for start, end, group in POSE_CONNECTIONS + FACE_CONNECTIONS:
            first, second = point(start), point(end)
            if first is None or second is None:
                continue
            stopped = frozen(group)
            self._cv2.line(
                camera,
                first,
                second,
                HELD_LINE_BGR if stopped else TRACKED_LINE_BGR,
                2 if stopped else 4,
            )
        for name, group in POSE_POINT_GROUPS.items():
            spot = point(name)
            if spot is None:
                continue
            if frozen(group):
                # Hollow, grey, and thin: unmistakably not the filled orange dot
                # of a joint that is actually driving the robot.
                self._cv2.circle(camera, spot, 6, HELD_LINE_BGR, 1)
            else:
                self._cv2.circle(camera, spot, 6, TRACKED_POINT_BGR, -1)
        for prefix, group, color in HAND_GROUPS:
            stopped = frozen(group)
            hand_color = HELD_LINE_BGR if stopped else color
            hand_points = {
                index: spot
                for index in range(21)
                if (spot := point(f"{prefix}_{index}")) is not None
            }
            for start, end in HAND_CONNECTION_INDEXES:
                if start in hand_points and end in hand_points:
                    self._cv2.line(camera, hand_points[start], hand_points[end], hand_color, 2)
            if not stopped:
                for spot in hand_points.values():
                    self._cv2.circle(camera, spot, 2, hand_color, -1)
        for group in (HEAD, LEFT_ARM, RIGHT_ARM):
            if group not in held:
                continue
            anchor = point(LIMB_LABEL_ANCHORS[group])
            if anchor is None:
                # The limb the caption names is not on screen. The TRACKING card
                # still says it is held; a caption floating on the panel edge,
                # pointing at nothing, does not.
                continue
            self._label(
                camera,
                f"{GROUP_LABELS[group]} HELD",
                (
                    max(left + 10, min(anchor[0] + 10, left + width - 150)),
                    # Clear of the panel caption along the bottom edge, which the
                    # label used to sit on top of.
                    max(top + 26, min(anchor[1] - 10, top + height - 34)),
                ),
                scale=TYPE_ONBODY,
                color=HELD_BGR,
            )

    #: Reason codes the operator can act on, and the action. A raw enum on the
    #: banner is the same failure the banner exists to fix: the operator reads
    #: `TORSO_YAW_RECALIBRATION_REQUIRED`, sees both arms and the torso frozen,
    #: and has no way to know what to DO about it. It is recoverable in-run now
    #: (`pipeline._recover_torso_yaw`) -- fifteen consecutive face-on, continuous
    #: frames -- so the banner says what earns those frames rather than naming the
    #: enum and stopping.
    REASON_REMEDIES: ClassVar[dict[str, str]] = {
        "TORSO_YAW_RECALIBRATION_REQUIRED": "FACE THE CAMERA and hold still to resume",
        "TORSO_YAW_DISCONTINUITY": "square up to the camera and hold still",
        "TORSO_YAW_CONTINUITY_GAP": "square up to the camera and hold still",
        "IDENTITY_NOT_STABLE": "one person in frame, hold still",
        "FRAME_NOT_LIVE": "the camera image stopped changing",
    }

    def _draw_reason(
        self,
        canvas: np.ndarray,
        status: str,
        held: frozenset[str],
        reasons: Iterable[str] | None,
    ) -> None:
        """Put the rejection reason where the operator is already looking.

        ``IDENTITY_NOT_STABLE`` scrolling past in a terminal behind the window
        is the same as no message at all.
        """
        texts = [str(reason) for reason in (reasons or ()) if str(reason)]
        if not texts and not held:
            return
        if not texts:
            texts = ["input not trustworthy for the held limb(s)"]
        headline = texts[0]
        # The remedy, not just the code. Matched as a substring because reasons
        # arrive as "left_arm: TORSO_YAW_RECALIBRATION_REQUIRED".
        remedy = next(
            (
                action
                for code, action in self.REASON_REMEDIES.items()
                if any(code in text for text in texts)
            ),
            None,
        )
        if len(texts) > 1:
            headline = f"{headline}  (+{len(texts) - 1} more)"
        if remedy is not None:
            headline = f"{headline}  -->  {remedy}"
        accent = FAIL_BGR if status == "FAULT" else WARN_BGR
        top = self._app_bar_height + 12
        height = 34
        if top + height > self._height - 8 or self._width < CARD_MIN_WIDTH:
            # Every `_place`d card has a floor check; this banner is positioned
            # directly and had none, so on a very short panel it was drawn off the
            # bottom of the canvas.
            return
        rect = (14, top, self._width - 14, top + height)
        self._round_rect(canvas, rect, SURFACE_CARD_BGR, SURFACE_LINE_BGR, radius=6)
        # A colour bar on the leading edge, so the severity is readable before the
        # sentence is.
        self._cv2.rectangle(canvas, (14, top + 3), (18, top + height - 3), accent, -1)
        self._cv2.putText(
            canvas,
            self._clip(f"WHY: {headline}", self._width - 60, TYPE_BODY, 1),
            (30, top + height - 12),
            self._font,
            TYPE_BODY,
            accent,
            1,
            self._cv2.LINE_AA,
        )

    #: The landmarks the swivel mapper needs per side: this shoulder, the
    #: opposite shoulder (which fixes where psi is measured FROM), this elbow and
    #: this wrist. Deliberately not the hips -- psi never needs them.
    _ARM_CHAIN: ClassVar[dict[str, tuple[tuple[str, str], ...]]] = {
        "LEFT": (("pose_11", "shoulder"), ("pose_12", "opp shoulder"),
                 ("pose_13", "elbow"), ("pose_15", "wrist")),
        "RIGHT": (("pose_12", "shoulder"), ("pose_11", "opp shoulder"),
                  ("pose_14", "elbow"), ("pose_16", "wrist")),
        # The head needs the face, and the face is the first thing lost when the
        # camera is tilted down to bring the hips in. Measured on a real session:
        # the arms tracked on 87.1% of frames while the head tracked on 31.1%,
        # because the operator's head was simply outside the frame. Both arms and
        # the head were usable together on only 21.5%.
        "HEAD": (("face_1", "nose"), ("face_33", "eye"), ("face_263", "eye")),
    }

    #: Which control group each tracking row reports on, so a HELD limb reads as
    #: held rather than as "tracking" beside a motionless arm.
    _ARM_CHAIN_GROUPS: ClassVar[dict[str, str]] = {
        "LEFT": LEFT_ARM,
        "RIGHT": RIGHT_ARM,
        "HEAD": HEAD,
    }

    def _place(self, height: int) -> tuple[int, int, int] | None:
        """Room for one fixed-height card. See :meth:`_place_flexible`."""
        result = self._place_flexible((height,))
        return None if result is None else result[0]

    def _build_columns(self) -> None:
        """The two card stacks for this frame.

        The camera is top-anchored in its panel, so everything below it is the
        HUD column.  When the image fills the panel -- a square-ish camera, or the
        small windows the unit tests use -- there is no column at all, and every
        card falls back to the twin's side together rather than each inventing its
        own rule.  Previously two cards had a fallback and one did not, so the
        SAFETY readout simply vanished at `FALLBACK_PANEL_SIZE`, which is every
        install without pyobjc.
        """
        camera_x, camera_y, camera_w, camera_h = self._camera_box
        self._columns = (
            _Column(
                x=camera_x + 12,
                top=camera_y + camera_h + 12,
                width=max(0, camera_w - 24),
                floor=self._height - 8,
                min_width=CARD_MIN_WIDTH,
            ),
            _Column(
                x=self._width + 12,
                top=self._app_bar_height + 12,
                width=self._robot_width - 24,
                floor=self._height - 8,
                min_width=CARD_MIN_WIDTH,
            ),
        )

    def _place_flexible(
        self, heights: tuple[int, ...], *, min_width: int = 0
    ) -> tuple[tuple[int, int, int], int] | None:
        """Place a card that can shrink, trying every size in ONE column first.

        Column-major, deliberately.  Size-major would move a card that could have
        shrunk to fit the HUD column over to the twin's side at full size -- which
        is what the joint readout did during calibration, jumping across the
        canvas while there was still room where it belonged.  A card stays where
        it belongs and gets smaller; it only crosses over when its own column
        cannot hold it at any legible size.
        """
        if self._columns is None:
            self._build_columns()
        assert self._columns is not None
        for column in self._columns:
            if column.width < min_width:
                # Checked BEFORE placing. A card that reserved room and then
                # declined to draw -- because the column was too narrow for its
                # own contents -- silently ate ~300 px of the column and lost the
                # readout even though the other column was free and wide enough.
                continue
            for height in heights:
                rect = column.place(height)
                if rect is not None:
                    return rect, height
        return None

    def _draw_tracking_card(
        self,
        canvas: np.ndarray,
        observation: HumanObservation | None,
        min_confidence: float,
        held: frozenset[str],
        saturated: frozenset[str],
        trusted_confidence: float | None = None,
    ) -> None:
        """Per-limb: is the SHAPE of this limb drivable from what the camera sees?

        A limb can be un-held and still shape-blind: the wrist mapper only needs a
        wrist, so an unseen elbow costs nothing it would report.  The swivel mapper
        needs the elbow, and without it the arm silently falls back to wrist-only.
        Three capture sessions were spent before anyone noticed a landmark had been
        missing throughout, so this states it while there is still time to move.

        ``trusted_confidence`` is the threshold the CALIBRATION gate is applying
        right now, when one is up. Without it this card judged every landmark at
        the runtime ``min_confidence`` (0.35) while calibration was blocking at
        0.50, so at 0.40 the card read three solid greens and "tracking" while the
        card directly above it said BODY PARTS FOUND BUT NOT TRUSTED. Between the
        two thresholds a limb is now "weak", in the same amber as a hold: not
        blind, not yet trusted.
        """
        rows = len(self._ARM_CHAIN)
        card_height = 44 + rows * 26 + 8
        placement = self._place(card_height)
        if placement is None:
            return
        left, top, width = placement
        content = self._card(canvas, (left, top, left + width, top + card_height), "TRACKING")
        by_name = (
            {}
            if observation is None
            else {landmark.name: landmark for landmark in observation.landmarks}
        )
        for index, (side, chain) in enumerate(self._ARM_CHAIN.items()):
            y = content + index * 26
            missing = []
            weak = []
            for name, label in chain:
                landmark = by_name.get(name)
                if landmark is None or min(
                    landmark.visibility, landmark.presence
                ) < min_confidence:
                    if label not in missing:
                        missing.append(label)
                elif (
                    trusted_confidence is not None
                    and min(landmark.visibility, landmark.presence) < trusted_confidence
                    and label not in weak
                ):
                    weak.append(label)
            group = self._ARM_CHAIN_GROUPS.get(side)
            frozen = group is not None and group in held
            if observation is None:
                state, colour = "no observation", TEXT_FAINT_BGR
            elif frozen:
                state, colour = "HELD", WARN_BGR
            elif missing:
                state, colour = "no " + ", ".join(missing), WARN_BGR
            elif weak:
                state, colour = "weak " + ", ".join(weak), WARN_BGR
            else:
                state, colour = "tracking", GOOD_BGR
            what = "SHAPE" if side != "HEAD" else "TRACK"
            self._cv2.circle(canvas, (left + 20, y + 1), 4, colour, -1)
            self._cv2.putText(
                canvas, f"{side} {what}", (left + 34, y + 6), self._font,
                TYPE_SMALL, TEXT_PRIMARY_BGR, 1, self._cv2.LINE_AA,
            )
            text = self._clip(state, width - 150, TYPE_SMALL, 1)
            (tw, _), _ = self._cv2.getTextSize(text, self._font, TYPE_SMALL, 1)
            self._cv2.putText(
                canvas, text, (left + width - tw - 16, y + 6), self._font,
                TYPE_SMALL, colour, 1, self._cv2.LINE_AA,
            )

    def _draw_joint_activity(
        self,
        canvas: np.ndarray,
        joints: Mapping[str, float] | None,
        held: frozenset[str],
    ) -> None:
        """Per-joint "is this actually moving?" readout.

        Answers the question the operator kept having to guess at: not whether the
        robot is being commanded, but which joints of it are responding.  Two
        columns, because nineteen single-file rows is a column of noise nobody
        scans -- and because two columns fit the HUD width at a legible size
        instead of a tiny one.
        """
        self._update_activity(joints, held)
        # Two columns wherever two fit; three only when they do not. Nineteen
        # joints over two columns need ten rows, and at a 480 px panel that is
        # eleven pixels more than the column has -- which dropped the whole
        # readout, on the one card whose job is to show a joint that has stopped
        # moving. Three shorter columns keep it, at the same legible row height.
        result = None
        for columns in (2, 3):
            per_column = (len(UPPER_BODY_JOINTS) + columns - 1) // columns
            # Ask for the tallest legible card first; if it does not fit anywhere,
            # drop the readout rather than draw rows nobody can read.
            rows = tuple(range(26, JOINT_PANEL_MIN_ROW - 1, -2))
            result = self._place_flexible(
                tuple(44 + per_column * r + 10 for r in rows),
                min_width=columns * JOINT_COLUMN_MIN_WIDTH + 28,
            )
            if result is not None:
                break
        if result is None:
            return
        (left, top, width), card_height = result
        row = (card_height - 54) // per_column
        content = self._card(
            canvas, (left, top, left + width, top + card_height), "JOINT ACTIVITY"
        )
        column_width = (width - 28) // columns
        # The dense fallback: below an 18 px row TYPE_SMALL's glyphs touch.
        scale = TYPE_SMALL if row >= 18 else TYPE_DENSE
        bar_width = max(28, column_width - 132)
        for index, (name, label, group) in enumerate(UPPER_BODY_JOINTS):
            column, position = divmod(index, per_column)
            x = left + 14 + column * column_width
            y = content + position * row
            activity = self._joint_activity.get(name, 0.0)
            stopped = group in held
            still = self._joint_still_frames.get(name, 0)
            # Corroboration: has this joint's OWN control group moved recently?
            # Without this the red state fired on any long stillness -- including
            # the operator simply standing still, which reddened the whole card.
            group_moving = (
                self._group_still_frames.get(group, ACTIVITY_STUCK_FRAMES)
                < ACTIVITY_STUCK_FRAMES
            )
            if stopped:
                colour = WARN_BGR
            elif (
                activity < ACTIVITY_DEAD_RAD
                and still >= ACTIVITY_STUCK_FRAMES
                and group_moving
            ):
                # Not held, not moving, and its own limb IS moving: this is the
                # "arm_joint2 is dead" case, and it should be impossible to miss.
                colour = FAIL_BGR
            elif activity < ACTIVITY_DEAD_RAD:
                colour = TEXT_FAINT_BGR
            else:
                colour = GOOD_BGR
            # The joint's NAME carries the alarm colour too. Colouring only the
            # number halved the alarm's ink and left the one word that says WHICH
            # joint is dead in ordinary grey -- and red is now also used for the
            # clearance track's floor tick, so "there is red on screen" had
            # stopped meaning "a joint is dead".
            label_colour = colour if colour in (FAIL_BGR, WARN_BGR) else TEXT_MUTED_BGR
            self._cv2.putText(
                canvas, label, (x, y + 4), self._font, scale, label_colour, 1,
                self._cv2.LINE_AA,
            )
            bar_left = x + 56
            track_top = y - 6
            track_bottom = y + 2
            self._round_rect(
                canvas,
                (bar_left, track_top, bar_left + bar_width, track_bottom),
                SURFACE_RAISED_BGR, None, radius=3,
            )
            filled = int(bar_width * min(1.0, activity / ACTIVITY_FULL_SCALE_RAD))
            if filled > 2:
                self._round_rect(
                    canvas,
                    (bar_left, track_top, bar_left + filled, track_bottom),
                    colour, None, radius=3,
                )
            value = self._joint_previous.get(name)
            self._cv2.putText(
                canvas,
                "  --  " if value is None else f"{value:+.2f}",
                (bar_left + bar_width + 8, y + 4),
                self._font, scale, colour, 1, self._cv2.LINE_AA,
            )

    def _draw_safety_card(
        self,
        canvas: np.ndarray,
        clearance_m: float | None,
        status: str,
        held: frozenset[str],
    ) -> None:
        """The supervisor's live margin, drawn against the floor it must clear.

        A number alone does not say whether 5.6 mm is comfortable or one frame
        from a HOLD.  The floor and the standoff are drawn as marks on the same
        track, so the answer is the picture rather than arithmetic the viewer has
        to do while watching a demo.
        """
        # 114, not 92. The tick labels and the state note sit on a baseline at
        # `content + 58` = `top + 102`, and at 92 they were drawn TEN PIXELS BELOW
        # the card -- floating in the gutter and reading as a caption for the card
        # underneath, on every frame and at every panel size. Measured, not
        # guessed: baseline 102 plus the descender is 105, so 114 leaves the same
        # 9 px of bottom padding the other cards have.
        card_height = 114
        placement = self._place(card_height)
        if placement is None:
            return
        left, top, width = placement
        content = self._card(
            canvas, (left, top, left + width, top + card_height), "SAFETY", accent=ACCENT_BGR
        )
        if clearance_m is None or not isfinite(clearance_m):
            self._cv2.putText(
                canvas, "no supervised target this frame", (left + 14, content + 6),
                self._font, TYPE_SMALL, TEXT_FAINT_BGR, 1, self._cv2.LINE_AA,
            )
            return
        millimetres = clearance_m * 1000.0
        colour = _clearance_colour(clearance_m)
        self._cv2.putText(
            canvas, "self-clearance", (left + 14, content + 6), self._font,
            TYPE_SMALL, TEXT_MUTED_BGR, 1, self._cv2.LINE_AA,
        )
        value = f"{millimetres:.1f} mm"
        (vw, _), _ = self._cv2.getTextSize(value, self._font, TYPE_TITLE, 1)
        self._cv2.putText(
            canvas, value, (left + width - vw - 16, content + 10), self._font,
            TYPE_TITLE, colour, 1, self._cv2.LINE_AA,
        )
        track_left, track_right = left + 14, left + width - 16
        track_y = content + 30
        self._round_rect(
            canvas, (track_left, track_y, track_right, track_y + 10),
            SURFACE_RAISED_BGR, None, radius=5,
        )
        span = max(1e-6, CLEARANCE_DISPLAY_FULL_SCALE_M)
        def mark(value_m: float) -> int:
            fraction = max(0.0, min(1.0, value_m / span))
            return int(track_left + fraction * (track_right - track_left))
        filled = mark(clearance_m)
        if filled > track_left + 2:
            self._round_rect(
                canvas, (track_left, track_y, filled, track_y + 10), colour, None, radius=5
            )
        for threshold, label, tone in (
            (CLEARANCE_FLOOR_DISPLAY_M, "floor", FAIL_BGR),
            (CLEARANCE_STANDOFF_DISPLAY_M, "standoff", TEXT_MUTED_BGR),
        ):
            x = mark(threshold)
            self._cv2.line(canvas, (x, track_y - 4), (x, track_y + 14), tone, 1, self._cv2.LINE_AA)
            self._cv2.putText(
                canvas, label, (x - 12, track_y + 28), self._font, TYPE_TICK, tone, 1,
                self._cv2.LINE_AA,
            )
        note = "supervisor allowing" if status == "ALLOW" and not held else (
            "supervisor holding" if status != "ALLOW" else "partial - some limbs held"
        )
        (nw, _), _ = self._cv2.getTextSize(note, self._font, TYPE_SMALL, 1)
        self._cv2.putText(
            canvas, note, (track_right - nw, track_y + 28), self._font, TYPE_SMALL,
            TEXT_MUTED_BGR, 1, self._cv2.LINE_AA,
        )

    def _update_activity(
        self, joints: Mapping[str, float] | None, held: frozenset[str] = frozenset()
    ) -> None:
        """Peak-hold joint movement, and the evidence behind the red DEAD state.

        Bars decay every render, so a quick gesture stays legible and a frozen
        robot's bars fall to zero instead of going blank.

        The still-counters are stricter, because they drive a red "this joint has
        died" call the operator is meant to trust, and it was firing on healthy
        sessions: 9 of 19 joints on the witness capture, red in 23% of composite
        frames, and ALL nineteen about two seconds after the operator stepped out
        of shot. Three rules, each removing one class of false positive:

        * **A repaint is not new evidence.** Between solves `render` is handed the
          same readback; if no joint moved at all, this is that same pose again,
          so nothing is counted. The bars still decay.
        * **A HELD joint is commanded not to move.** Silence from a joint the
          supervisor is deliberately freezing says nothing about whether it works,
          so its counter is left alone and its bar is zeroed -- an amber "held"
          row with a long bar claimed the limb was moving at the governor limit
          while the app bar said it was frozen.
        * **Stillness only counts against a joint whose own control group is
          moving.** A whole arm at rest is an operator at rest. A joint still for
          50 solves while its siblings move is the "arm_joint2 is dead" case this
          state exists for. Single-joint groups (the torso, each gripper) have no
          corroboration available and so are never called dead -- said out loud
          because it is a real gap, not an oversight.
        """
        readings: dict[str, float | None] = {}
        for name, _label, _group in UPPER_BODY_JOINTS:
            value = None if joints is None else joints.get(name)
            # Duck-typed, not `isinstance(value, (int, float))`: `np.float64`
            # subclasses `float` but `np.float32` does not, so a readback that
            # arrived as float32 was treated as absent -- every bar empty, every
            # value "--", and after 50 frames the readout would call a perfectly
            # healthy robot DEAD in red. `isfinite` still rejects NaN and inf.
            try:
                value = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                value = None
            readings[name] = None if value is None or not isfinite(value) else value

        deltas: dict[str, float] = {}
        for name, _label, _group in UPPER_BODY_JOINTS:
            value = readings[name]
            previous = self._joint_previous.get(name)
            deltas[name] = (
                0.0 if value is None or previous is None else abs(value - previous)
            )
        new_pose = any(delta >= ACTIVITY_DEAD_RAD for delta in deltas.values())

        moved_groups = {
            group
            for name, _label, group in UPPER_BODY_JOINTS
            if deltas[name] >= ACTIVITY_DEAD_RAD
        }
        for group in {group for _name, _label, group in UPPER_BODY_JOINTS}:
            if group in moved_groups:
                self._group_still_frames[group] = 0
            elif new_pose:
                self._group_still_frames[group] = (
                    self._group_still_frames.get(group, 0) + 1
                )

        for name, _label, group in UPPER_BODY_JOINTS:
            decayed = self._joint_activity.get(name, 0.0) * ACTIVITY_DECAY
            value = readings[name]
            if value is not None:
                self._joint_previous[name] = value
            if group in held:
                self._joint_activity[name] = 0.0
                self._joint_still_frames[name] = 0
                continue
            self._joint_activity[name] = max(deltas[name], decayed)
            if not new_pose:
                continue
            self._joint_still_frames[name] = (
                0
                if deltas[name] >= ACTIVITY_DEAD_RAD
                else self._joint_still_frames.get(name, 0) + 1
            )

    def _draw_calibration(self, canvas: np.ndarray, status: CalibrationStatus) -> None:
        """A checklist of what calibration is still waiting for."""
        lines: list[tuple[bool, str]] = []
        if status.missing_landmarks:
            parts = ", ".join(
                sorted(LANDMARK_LABELS.get(name, name) for name in status.missing_landmarks)
            )
            # No "- step back" here. The hint below is computed from where the
            # operator actually IS in frame (`cli.py::_framing_advice`) and says
            # "AIM CAMERA DOWN" as often as not -- so the card gave two different
            # instructions about the same landmark, two lines apart.
            lines.append((False, f"CAN'T SEE: {parts}"))
        elif not status.confidence_ok:
            # Present-but-worthless is the confusing case: MediaPipe still emits a
            # full landmark set for a view it cannot actually read, so the names
            # are all there at ~0.00 confidence. Reporting that as OK told an
            # operator "ALL REQUIRED BODY PARTS VISIBLE" while the camera was
            # pointed at the ceiling and the skeleton was drawn in a corner of the
            # room (observed 2026-08-26). Say what to DO instead.
            lines.append(
                (
                    False,
                    "BODY PARTS FOUND BUT NOT TRUSTED - centre head and chest in frame",
                )
            )
        else:
            lines.append((True, "ALL REQUIRED BODY PARTS VISIBLE"))
        lines.append(
            (
                status.confidence_ok,
                f"CONFIDENCE {status.confidence:.2f} / {status.required_confidence:.2f}"
                + ("" if status.confidence_ok else " - more light, face the camera"),
            )
        )
        lines.append(
            (
                status.identity_ok,
                f"IDENTITY {status.identity or 'UNKNOWN'}"
                + ("" if status.identity_ok else " - hold still, one person in frame"),
            )
        )
        row = 24
        # The hint is the one line the operator has to act on, so it lives INSIDE
        # the card with the checklist it explains. Drawn below the card it landed
        # on raw camera pixels and became unreadable over hair, a bright window,
        # or the skeleton itself.
        lines_shown = len(lines) + bool(status.hint)
        card_height = 44 + lines_shown * row + 10
        placement = self._place(card_height)
        if placement is None:
            return
        left, top, width = placement
        rect = (left, top, left + width, top + card_height)
        content = self._card(canvas, rect, "CALIBRATION", accent=ACCENT_BGR)
        header = (
            f"CALIBRATING  frame {status.frames}  "
            f"landmarks {status.visible_landmarks}/{status.required_landmarks}"
        )
        (hw, _), _ = self._cv2.getTextSize(header, self._font, TYPE_SMALL, 1)
        self._cv2.putText(
            canvas, header, (left + width - hw - 14, top + 20), self._font,
            TYPE_SMALL, TEXT_MUTED_BGR, 1, self._cv2.LINE_AA,
        )
        for index, (ok, text) in enumerate(lines):
            y = content + index * row
            self._cv2.circle(canvas, (left + 14, y - 4), 4, GOOD_BGR if ok else FAIL_BGR, -1)
            self._cv2.putText(
                canvas,
                self._clip(("[OK] " if ok else "[--] ") + text, width - 66, TYPE_BODY, 1),
                (left + 28, y),
                self._font,
                TYPE_BODY,
                # A blocked check is drawn in the hazard colour, not in grey
                # dimmer than the checks that PASSED -- which is what it became
                # when the palette went in.
                TEXT_PRIMARY_BGR if ok else FAIL_BGR,
                1,
                self._cv2.LINE_AA,
            )
        if status.hint:
            y = content + len(lines) * row + 4
            self._round_rect(
                canvas, (left + 14, y - 17, left + width - 14, y + 7),
                SURFACE_RAISED_BGR, None, radius=5,
            )
            self._cv2.putText(
                canvas,
                self._clip(status.hint, width - 66, TYPE_BODY, 1),
                (left + 26, y),
                self._font, TYPE_BODY, ACCENT_BGR, 1, self._cv2.LINE_AA,
            )

    # --- drawing primitives --------------------------------------------------

    def _round_rect(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        fill: tuple[int, int, int] | None,
        border: tuple[int, int, int] | None = None,
        radius: int = 6,
    ) -> None:
        """A filled, optionally outlined, rounded rectangle.

        Rounded corners are the cheapest signal that a region is a deliberate
        surface rather than whatever was left over.
        """
        x0, y0, x1, y1 = rect
        x0, y0 = max(0, x0), max(0, y0)
        x1 = min(canvas.shape[1], x1)
        y1 = min(canvas.shape[0], y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        radius = max(0, min(radius, (x1 - x0) // 2 - 1, (y1 - y0) // 2 - 1))
        if fill is not None:
            self._cv2.rectangle(canvas, (x0 + radius, y0), (x1 - radius, y1), fill, -1)
            self._cv2.rectangle(canvas, (x0, y0 + radius), (x1, y1 - radius), fill, -1)
            for cx, cy in (
                (x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                (x0 + radius, y1 - radius), (x1 - radius, y1 - radius),
            ):
                # LINE_AA on the FILL, not only the border. Without it every
                # rounded surface in the HUD -- cards, pills, and the nineteen
                # activity bars -- had stair-stepped corners, which is what makes
                # a layout read as "drawn by a script" on a projector.
                self._cv2.circle(canvas, (cx, cy), radius, fill, -1, self._cv2.LINE_AA)
        if border is not None:
            aa = self._cv2.LINE_AA
            for start, end in (
                ((x0 + radius, y0), (x1 - radius, y0)),
                ((x0 + radius, y1), (x1 - radius, y1)),
                ((x0, y0 + radius), (x0, y1 - radius)),
                ((x1, y0 + radius), (x1, y1 - radius)),
            ):
                self._cv2.line(canvas, start, end, border, 1, aa)
            for cx, cy, angle in (
                (x0 + radius, y0 + radius, 180), (x1 - radius, y0 + radius, 270),
                (x1 - radius, y1 - radius, 0), (x0 + radius, y1 - radius, 90),
            ):
                self._cv2.ellipse(
                    canvas, (cx, cy), (radius, radius), angle, 0, 90, border, 1, aa
                )

    def _scrim(
        self, canvas: np.ndarray, rect: tuple[int, int, int, int], *, from_top: bool
    ) -> None:
        """Fade a strip of live video toward the page colour.

        A translucent gradient, not an opaque bar: the operator can still see
        themselves through it, and -- because it composites in place -- it does
        not move the image the skeleton is registered to.

        Blended as a small number of constant-alpha bands through
        ``cv2.addWeighted`` rather than one per-row float expression.  Measured at
        the real panel size, the four scrims this HUD draws cost 7.2-8.0 ms a
        frame in the float form against 0.31-0.56 ms banded -- 13-26x, and most of
        a repaint either way.

        ``bands = max(4, min(20, height // 5))``, so the fade is one alpha step
        per 5 px until it hits the 20-band cap, and coarser after that: the
        largest scrim here is the twin's 324 px sky fade, which gets 20 bands at
        16 px a step.  Over a gradient that spans only ~18 levels of the page
        colour that is still below the eye's contrast threshold, and it is why the
        cap is affordable.
        """
        x0, y0, x1, y1 = rect
        x0, y0 = max(0, x0), max(0, y0)
        x1 = min(canvas.shape[1], x1)
        y1 = min(canvas.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return
        height = y1 - y0
        fill = self._scrim_fill(canvas.shape, canvas.dtype)
        bands = max(4, min(20, height // 5))
        for index in range(bands):
            band_top = y0 + index * height // bands
            band_bottom = y0 + (index + 1) * height // bands
            if band_bottom <= band_top:
                continue
            position = (index + 0.5) / bands
            alpha = 0.92 * ((1.0 - position) if from_top else position)
            band = canvas[band_top:band_bottom, x0:x1]
            self._cv2.addWeighted(
                band, 1.0 - alpha,
                fill[: band_bottom - band_top, : x1 - x0], alpha, 0.0, dst=band,
            )

    def _scrim_fill(self, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """A page-coloured image the size of the whole canvas, allocated once.

        ``addWeighted`` needs a real array to blend against, and building one per
        band per frame was the whole point of the cost.  Canvas-sized rather than
        band-sized so one buffer serves every scrim whatever its rect: ~5.6 MB at
        1720x1080, once per session, against ~90 allocations a second.
        """
        cached = self._scrim_fill_cache
        if (
            cached is None
            or cached.shape[0] < shape[0]
            or cached.shape[1] < shape[1]
            or cached.dtype != dtype
        ):
            # dtype as well as size.  `addWeighted` refuses to mix types, so a
            # camera delivering anything but uint8 made this one call raise -- and
            # since the whole app bar is a single `_safe` element, that took the
            # status pill, the FROZEN pill, the clearance chip and both panel
            # captions with it.  The most safety-relevant element in the module
            # was the one that died.
            cached = np.empty((shape[0], shape[1], 3), dtype=dtype)
            cached[:] = SURFACE_BG_BGR
            self._scrim_fill_cache = cached
        return cached

    def _pill(
        self,
        canvas: np.ndarray,
        origin: tuple[int, int],
        text: str,
        *,
        fg: tuple[int, int, int],
        bg: tuple[int, int, int],
        scale: float = TYPE_SMALL,
    ) -> int:
        """A small rounded label chip.  Returns the x just past its right edge."""
        (tw, th), _ = self._cv2.getTextSize(text, self._font, scale, 1)
        x, y = origin
        pad_x, pad_y = 8, 5
        self._round_rect(
            canvas, (x, y, x + tw + 2 * pad_x, y + th + 2 * pad_y), bg, None, radius=4
        )
        self._cv2.putText(
            canvas, text, (x + pad_x, y + th + pad_y - 1), self._font, scale, fg, 1,
            self._cv2.LINE_AA,
        )
        return x + tw + 2 * pad_x

    def _card(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        title: str,
        *,
        accent: tuple[int, int, int] = TEXT_FAINT_BGR,
    ) -> int:
        """A titled surface.  Returns the y where its content may start."""
        x0, y0 = rect[0], rect[1]
        self._round_rect(canvas, rect, SURFACE_CARD_BGR, SURFACE_LINE_BGR, radius=7)
        self._cv2.putText(
            canvas, title, (x0 + 14, y0 + 20), self._font, TYPE_SMALL,
            TEXT_MUTED_BGR, 1, self._cv2.LINE_AA,
        )
        self._cv2.line(
            canvas, (x0 + 14, y0 + 29), (x0 + 14 + 22, y0 + 29), accent, 2,
            self._cv2.LINE_AA,
        )
        return y0 + 44

    _OUTLINE_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    def _label(
        self,
        canvas: np.ndarray,
        text: str,
        origin: tuple[int, int],
        *,
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        """Text with a dark outline, so it survives a bright or busy camera frame.

        The outline is drawn as offset copies at the SAME thickness, not as one
        fatter pass underneath.  A Hershey glyph's advance widens with thickness
        -- measured, "LEFT SHAPE: tracking" at scale 0.5 is 143 px at thickness 1
        and 152 px at thickness 3 -- so a thickness+2 underlay starts aligned and
        drifts a whole glyph right by the end of the string, leaving a stray dark
        character trailing every label.  It was on screen in every capture up to
        22 Aug.
        """
        x, y = origin
        for dx, dy in self._OUTLINE_OFFSETS:
            self._cv2.putText(
                canvas,
                text,
                (x + dx, y + dy),
                self._font,
                scale,
                (0, 0, 0),
                thickness,
                self._cv2.LINE_AA,
            )
        self._cv2.putText(
            canvas, text, origin, self._font, scale, color, thickness, self._cv2.LINE_AA
        )

    def _clip(self, text: str, max_px: int, scale: float, thickness: int) -> str:
        (measured, _), _ = self._cv2.getTextSize(text, self._font, scale, thickness)
        if measured <= max_px or measured <= 0:
            return text
        keep = max(4, int(len(text) * max_px / measured) - 3)
        return text[:keep] + "..."

    def close(self) -> None:
        self._renderer.close()
        if self._window_opened:
            # The operator may have already closed the native window.  That is a
            # normal request to end preview, not a reason to discard an otherwise
            # complete capture during its final publication transaction.
            try:
                self._cv2.destroyWindow(self._window)
            except self._cv2.error:
                pass
