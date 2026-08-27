"""Which way to move the camera, from where the body actually lands in frame.

`cli._framing_advice` replaced a hardcoded "step back" hint on the strength of a
specific observed defect: the operator was told to step back while the real
problem was a camera tilted so far down that their head was out of shot. Four
user-facing instructions came in with it and none of them had a test, so the
branch that fires is whichever the author last reasoned about.

MediaPipe reports normalized coordinates that can fall OUTSIDE [0, 1] when it
infers a joint beyond the image edge, which is what makes "you are below the
view" distinguishable from "you are too far away" at all.
"""

from __future__ import annotations

import pytest

from galbot_motion_studio.cli import _framing_advice


class _Landmark:
    def __init__(self, name: str, x: float, y: float) -> None:
        self.name = name
        self.normalized_xyz = (x, y, 0.0)


class _Observation:
    def __init__(self, landmarks) -> None:
        self.landmarks = tuple(landmarks)


def _body(centre_x: float, centre_y: float, *, prefix: str = "pose_") -> _Observation:
    """Eight pose landmarks scattered tightly around one centroid."""
    offsets = ((-0.02, -0.02), (0.02, -0.02), (-0.02, 0.02), (0.02, 0.02),
               (-0.01, 0.0), (0.01, 0.0), (0.0, -0.01), (0.0, 0.01))
    return _Observation(
        _Landmark(f"{prefix}{index}", centre_x + dx, centre_y + dy)
        for index, (dx, dy) in enumerate(offsets)
    )


@pytest.mark.parametrize(
    "centre_x,centre_y,expected",
    (
        (0.50, 1.30, "TILT CAMERA DOWN - you are below the view"),
        (0.50, -0.30, "TILT CAMERA UP - you are above the view"),
        (1.30, 0.50, "MOVE LEFT / PAN CAMERA RIGHT"),
        (-0.30, 0.50, "MOVE RIGHT / PAN CAMERA LEFT"),
        (0.50, 0.50, None),      # centred: distance is the real issue, not framing
        (0.50, 1.00, None),      # at the very bottom edge, still inside
        (0.00, 0.00, None),      # at the very top-left corner, still inside
    ),
)
def test_the_advice_names_the_edge_the_operator_has_fallen_off(
    centre_x, centre_y, expected
) -> None:
    assert _framing_advice(_body(centre_x, centre_y)) == expected


def test_vertical_advice_wins_over_horizontal_when_both_are_out() -> None:
    """A camera aimed at the ceiling is one adjustment, not two.

    Deliberate ordering: fixing the tilt usually fixes the pan as a side effect,
    and two instructions at once is how an operator freezes.
    """
    assert _framing_advice(_body(1.30, 1.30)) == "TILT CAMERA DOWN - you are below the view"


def test_advice_is_silent_when_there_is_nothing_to_measure() -> None:
    """No pose landmarks at all, and face landmarks must not be mistaken for them."""
    assert _framing_advice(_Observation(())) is None
    assert _framing_advice(_body(0.5, 5.0, prefix="face_")) is None


def test_a_landmark_without_usable_coordinates_is_skipped_not_fatal() -> None:
    """The hint is drawn during calibration; it may never raise out of the loop."""
    observation = _Observation(
        (
            _Landmark("pose_0", 0.5, 1.4),
            _Observation(()),                     # no `.name`-shaped attributes
        )
    )
    observation.landmarks = (
        _Landmark("pose_0", 0.5, 1.4),
        _Landmark("pose_1", 0.5, 1.4),
    )
    short = _Landmark("pose_2", 0.0, 0.0)
    short.normalized_xyz = (0.5,)                 # truncated tuple
    empty = _Landmark("pose_3", 0.0, 0.0)
    empty.normalized_xyz = None                   # absent entirely
    observation.landmarks = (*observation.landmarks, short, empty)
    assert _framing_advice(observation) == "TILT CAMERA DOWN - you are below the view"
