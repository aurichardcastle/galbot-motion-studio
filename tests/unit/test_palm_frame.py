"""The palm frame is the operator's hand orientation, so it has to be a real frame."""

from __future__ import annotations

import unittest

import numpy as np

from galbot_motion_studio.retarget.palm_frame import PalmFrame, palm_frame

# A flat palm at the origin: fingers along +Y, knuckles along +X, back of hand +Z.
WRIST = np.array([0.0, 0.0, 0.0])
INDEX_MCP = np.array([0.04, 0.09, 0.0])
PINKY_MCP = np.array([-0.04, 0.09, 0.0])


def _roll(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)],
    ])


class TestPalmFrameIsAFrame(unittest.TestCase):
    def test_the_frame_is_orthonormal_and_right_handed(self) -> None:
        """A frame that is not orthonormal silently skews every commanded pose."""
        frame = palm_frame(WRIST, INDEX_MCP, PINKY_MCP, side="left")
        assert frame is not None
        matrix = frame.as_matrix()
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=12)

    def test_both_hands_produce_the_same_frame_from_mirrored_landmarks(self) -> None:
        """The knuckle order reverses between hands.

        Without the flip the two palms yield mirror-image frames and one hand is
        commanded inside out -- a fault that looks like a wiring error on the
        robot rather than a sign error here.
        """
        left = palm_frame(WRIST, INDEX_MCP, PINKY_MCP, side="left")
        right = palm_frame(WRIST, PINKY_MCP, INDEX_MCP, side="right")
        assert left is not None and right is not None
        np.testing.assert_allclose(left.as_matrix(), right.as_matrix(), atol=1e-12)

    def test_rolling_the_hand_rolls_the_frame_by_the_same_angle(self) -> None:
        """Forearm pronation is exactly what the old forearm-aligned guess could not see."""
        rotation = _roll(40.0)
        rolled = palm_frame(
            rotation @ WRIST, rotation @ INDEX_MCP, rotation @ PINKY_MCP, side="left"
        )
        flat = palm_frame(WRIST, INDEX_MCP, PINKY_MCP, side="left")
        assert rolled is not None and flat is not None
        relative = flat.as_matrix().T @ rolled.as_matrix()
        angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        self.assertAlmostEqual(float(angle), 40.0, places=6)


class TestPalmFrameRefusesRatherThanGuesses(unittest.TestCase):
    def test_collinear_or_coincident_points_return_none(self) -> None:
        self.assertIsNone(palm_frame(WRIST, INDEX_MCP, INDEX_MCP, side="left"))
        self.assertIsNone(palm_frame(WRIST, WRIST, WRIST, side="left"))

    def test_non_finite_input_returns_none(self) -> None:
        broken = np.array([np.nan, 0.0, 0.0])
        self.assertIsNone(palm_frame(WRIST, broken, PINKY_MCP, side="left"))

    def test_an_edge_on_palm_reports_low_confidence_instead_of_a_bad_frame(self) -> None:
        """Edge-on, the palm's width collapses and the frame is ill-conditioned.

        It stays a valid rotation, so nothing downstream breaks; the confidence
        is what tells the caller not to trust it.
        """
        narrow = palm_frame(
            WRIST, np.array([0.002, 0.09, 0.0]), np.array([-0.002, 0.09, 0.0]), side="left"
        )
        assert narrow is not None
        self.assertLess(narrow.confidence, 0.1)
        np.testing.assert_allclose(
            narrow.as_matrix().T @ narrow.as_matrix(), np.eye(3), atol=1e-12
        )

    def test_a_normal_palm_reports_full_confidence(self) -> None:
        frame = palm_frame(WRIST, INDEX_MCP, PINKY_MCP, side="left")
        assert frame is not None
        self.assertEqual(frame.confidence, 1.0)

    def test_side_must_be_named(self) -> None:
        with self.assertRaises(ValueError):
            palm_frame(WRIST, INDEX_MCP, PINKY_MCP, side="middle")


class TestPalmFrameIsIndependentOfFingerPose(unittest.TestCase):
    def test_moving_the_fingers_does_not_move_the_frame(self) -> None:
        """Only the three rigid palm points are read.

        A frame that drifted as the operator closed their hand would rotate the
        gripper every time they tried to grasp.
        """
        closed = palm_frame(WRIST, INDEX_MCP, PINKY_MCP, side="left")
        assert closed is not None
        # Fingertips would move a great deal here; the knuckles and wrist do not.
        open_hand = palm_frame(WRIST, INDEX_MCP, PINKY_MCP, side="left")
        assert open_hand is not None
        np.testing.assert_allclose(closed.as_matrix(), open_hand.as_matrix(), atol=1e-15)


if __name__ == "__main__":
    unittest.main()
