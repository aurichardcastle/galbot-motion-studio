"""Tests for the fault-injecting G1 test double.

Run with the project interpreter::

    .venv/bin/python \
        -m unittest discover -s side-projects/galbot-teleop/tests -v

pytest is not installed in that venv, so these are plain ``unittest`` cases.
They run unchanged under pytest if it ever appears.

The tests are organised one class per injectable fault, plus determinism,
configuration scoping, and a set of "honesty" checks that fail if anybody later
replaces a measured constant with a convenient one.
"""

from __future__ import annotations

import contextlib
import io
import json
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from galbot_motion_studio.testing.base import (  # noqa: E402
    GROUP_JOINTS,
    JOINT_LIMITS,
    MEASURED_POSE_2026_08_07_1806,
    ControlStatus,
    MissingTeardownError,
    RobotAdapter,
    ToleranceUnknownError,
    UninitializedGetterError,
)
from galbot_motion_studio.testing.fault_injection import (  # noqa: E402
    ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD,
    ACCEPTANCE_TOLERANCE_UPPER_BOUND_RAD,
    MEASURED_BOW_COMMANDED_RAD,
    MEASURED_DOWN_COMMANDED_RAD,
    MEASURED_TIMEOUT_PARTIAL_REACHED_COUNT,
    MEASURED_TIMEOUT_PARTIAL_TOTAL_COUNT,
    MISSING_TEARDOWN_EXIT_CODE,
    FakeG1Robot,
    FaultConfig,
    LatchCause,
    UnknownBandPolicy,
    judge_reach,
    swallow_sigint,
)

# Halfway into the band that hardware never probed.
MID_BAND_RAD = (ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD + ACCEPTANCE_TOLERANCE_UPPER_BOUND_RAD) / 2

HEAD_HOME = [
    MEASURED_POSE_2026_08_07_1806["head_joint1"],
    MEASURED_POSE_2026_08_07_1806["head_joint2"],
]
LEG_HOME = [MEASURED_POSE_2026_08_07_1806[n] for n in GROUP_JOINTS["leg"]]
LEFT_ARM_HOME = [MEASURED_POSE_2026_08_07_1806[n] for n in GROUP_JOINTS["left_arm"]]


class RobotTestCase(unittest.TestCase):
    """Gives every test its own state file, and tears the robot down properly."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state_path = Path(self._tmp.name) / "g1_state.json"

    def make_robot(self, **kwargs) -> FakeG1Robot:
        kwargs.setdefault("state_path", self.state_path)
        kwargs.setdefault("seed", 1234)
        return FakeG1Robot(**kwargs)

    def teardown_robot(self, robot: FakeG1Robot) -> None:
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()

    def initialised_robot(self, **kwargs) -> FakeG1Robot:
        robot = self.make_robot(**kwargs)
        self.assertTrue(robot.init())
        return robot


# ==========================================================================
class TestProtocolShape(RobotTestCase):
    def test_fake_satisfies_the_protocol(self) -> None:
        robot = self.make_robot()
        self.assertIsInstance(robot, RobotAdapter)

    def test_required_status_members_exist(self) -> None:
        for name in ("SUCCESS", "TIMEOUT", "FAULT", "INVALID_INPUT"):
            self.assertIn(name, ControlStatus.__members__)

    def test_getters_and_group_names(self) -> None:
        robot = self.initialised_robot()
        self.assertIn("head", robot.get_joint_group_names())
        self.assertEqual(len(robot.get_joint_positions(["head"])["head"]), 2)
        self.assertEqual(len(robot.get_joint_positions(["leg"])["leg"]), 5)
        self.assertEqual(len(robot.get_joint_positions(["left_arm"])["left_arm"]), 7)
        self.teardown_robot(robot)

    def test_unknown_group_returns_empty_not_error(self) -> None:
        """MEASURED finding 1b: the documented-but-wrong plural "legs" returned
        0 joints on hardware, with no exception."""
        robot = self.initialised_robot()
        self.assertEqual(robot.get_joint_positions(["legs"]), {"legs": []})
        self.teardown_robot(robot)


# ==========================================================================
class TestFault1And2_SuccessWithoutReaching(RobotTestCase):
    """Faults 1 and 2. Failure-matrix row R3."""

    def test_replays_the_two_measured_head_commands(self) -> None:
        """The load-bearing regression test: both 7 Aug data points, exactly.

        head_joint2 sat at -0.0002. Commanding start+0.50 (= +0.4998, which is
        0.0062 rad past the URDF upper limit) returned SUCCESS with a readback
        of +0.4936. Commanding start-0.25 (= -0.2502, 0.0359 rad past the lower
        limit) returned TIMEOUT.
        """
        robot = self.initialised_robot(config=FaultConfig(clamp_to_limit=True))
        self.assertAlmostEqual(robot.get_joint_positions(["head"])["head"][1], -0.0002)

        bow = robot.set_joint_positions([0.0, MEASURED_BOW_COMMANDED_RAD], ["head"])
        self.assertIs(bow, ControlStatus.SUCCESS)
        reached = robot.get_joint_positions(["head"])["head"][1]
        # The robot stopped at the URDF limit, which printed as +0.4936 on hardware.
        self.assertEqual(reached, JOINT_LIMITS["head_joint2"].upper)
        self.assertEqual(round(reached, 4), 0.4936)
        # ...and SUCCESS was returned for a target it did not reach.
        self.assertNotAlmostEqual(reached, MEASURED_BOW_COMMANDED_RAD, places=5)

        down = robot.set_joint_positions([0.0, MEASURED_DOWN_COMMANDED_RAD], ["head"])
        self.assertIs(down, ControlStatus.TIMEOUT)
        self.assertEqual(
            robot.get_joint_positions(["head"])["head"][1],
            JOINT_LIMITS["head_joint2"].lower,
        )
        self.teardown_robot(robot)

    def test_success_returned_while_short_of_target(self) -> None:
        """Fault 1 without needing a limit violation to cause it."""
        robot = self.initialised_robot(
            config=FaultConfig(stop_short_rad=ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD)
        )
        target = 0.30
        status = robot.set_joint_positions([target, 0.0], ["head"])
        reached = robot.get_joint_positions(["head"])["head"][0]

        self.assertIs(status, ControlStatus.SUCCESS)
        self.assertLess(reached, target)
        self.assertAlmostEqual(target - reached, ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD)
        self.teardown_robot(robot)

    def test_clamped_target_is_judged_against_the_original_command(self) -> None:
        """Fault 2: the comparison is against what you asked for, not what ran."""
        robot = self.initialised_robot(config=FaultConfig(clamp_to_limit=True))
        far_past_limit = JOINT_LIMITS["head_joint2"].upper + 1.0
        status = robot.set_joint_positions([0.0, far_past_limit], ["head"])
        self.assertIs(status, ControlStatus.TIMEOUT)
        # It still executed the clamped value -- silently.
        self.assertEqual(
            robot.get_joint_positions(["head"])["head"][1], JOINT_LIMITS["head_joint2"].upper
        )
        self.teardown_robot(robot)

    def test_in_range_command_under_clamp_fault_is_untouched(self) -> None:
        robot = self.initialised_robot(config=FaultConfig(clamp_to_limit=True))
        status = robot.set_joint_positions([0.1, 0.2], ["head"])
        self.assertIs(status, ControlStatus.SUCCESS)
        self.assertEqual(robot.get_joint_positions(["head"])["head"], [0.1, 0.2])
        self.teardown_robot(robot)

    # -- the unmeasured band ------------------------------------------------
    def test_unmeasured_band_refuses_to_guess_by_default(self) -> None:
        robot = self.initialised_robot(config=FaultConfig(stop_short_rad=MID_BAND_RAD))
        with self.assertRaises(ToleranceUnknownError) as ctx:
            robot.set_joint_positions([0.5, 0.0], ["head"])
        self.assertIn("UNMEASURED", str(ctx.exception))
        self.teardown_robot(robot)

    def test_unmeasured_band_policies(self) -> None:
        cases = {
            UnknownBandPolicy.PESSIMISTIC: ControlStatus.TIMEOUT,
            UnknownBandPolicy.OPTIMISTIC: ControlStatus.SUCCESS,
        }
        for policy, expected in cases.items():
            with self.subTest(policy=policy):
                path = Path(self._tmp.name) / f"state_{policy.value}.json"
                robot = FakeG1Robot(
                    state_path=path,
                    seed=1,
                    config=FaultConfig(stop_short_rad=MID_BAND_RAD, unknown_band_policy=policy),
                )
                self.assertTrue(robot.init())
                self.assertIs(robot.set_joint_positions([0.5, 0.0], ["head"]), expected)
                self.teardown_robot(robot)

    def test_random_band_policy_is_seeded_and_produces_both_answers(self) -> None:
        path = Path(self._tmp.name) / "band_random.json"
        robot = FakeG1Robot(
            state_path=path,
            seed=7,
            config=FaultConfig(
                stop_short_rad=MID_BAND_RAD, unknown_band_policy=UnknownBandPolicy.RANDOM
            ),
        )
        self.assertTrue(robot.init())
        seen = {robot.set_joint_positions([0.3 + 0.01 * i, 0.0], ["head"]) for i in range(20)}
        self.assertEqual(seen, {ControlStatus.SUCCESS, ControlStatus.TIMEOUT})
        self.teardown_robot(robot)

    def test_judge_reach_bounds_are_inclusive_on_the_measured_side(self) -> None:
        self.assertIs(judge_reach(ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD), ControlStatus.SUCCESS)
        self.assertIs(judge_reach(ACCEPTANCE_TOLERANCE_UPPER_BOUND_RAD), ControlStatus.TIMEOUT)
        self.assertIs(judge_reach(0.0), ControlStatus.SUCCESS)
        self.assertIs(judge_reach(10.0), ControlStatus.TIMEOUT)
        with self.assertRaises(ToleranceUnknownError):
            judge_reach(MID_BAND_RAD)


# ==========================================================================
class TestFault3_TimeoutWithPartialMotion(RobotTestCase):
    """Fault 3. Failure-matrix row R4. TIMEOUT is not "nothing happened"."""

    def test_six_of_seven_joints_reach_while_status_is_timeout(self) -> None:
        robot = self.initialised_robot(config=FaultConfig(timeout_partial=True))
        target = [v + 0.30 for v in LEFT_ARM_HOME]
        status = robot.set_joint_positions(target, ["left_arm"])
        reached = robot.get_joint_positions(["left_arm"])["left_arm"]

        self.assertIs(status, ControlStatus.TIMEOUT)
        arrived = [i for i, (r, t) in enumerate(zip(reached, target)) if abs(r - t) < 1e-9]
        self.assertEqual(len(arrived), MEASURED_TIMEOUT_PARTIAL_REACHED_COUNT)
        self.assertEqual(len(reached), MEASURED_TIMEOUT_PARTIAL_TOTAL_COUNT)
        # MEASURED: it was left_arm_joint3 -- index 2 -- that stalled.
        self.assertNotIn(2, arrived)
        self.teardown_robot(robot)

    def test_stalled_joint_travelled_part_of_the_way(self) -> None:
        """MEASURED: -0.5625 -> -0.2625 stalled at -0.3477, 71.6% of the way."""
        robot = self.initialised_robot(config=FaultConfig(timeout_partial=True))
        start = robot.get_joint_positions(["left_arm"])["left_arm"][2]
        target = start + 0.30
        commanded = list(LEFT_ARM_HOME)
        commanded[2] = target
        robot.set_joint_positions(commanded, ["left_arm"])
        reached = robot.get_joint_positions(["left_arm"])["left_arm"][2]

        self.assertGreater(reached, start)          # it moved
        self.assertLess(reached, target)            # but not all the way
        self.assertAlmostEqual((reached - start) / 0.30, 0.716, places=6)
        self.teardown_robot(robot)

    def test_unreached_indices_are_configurable(self) -> None:
        robot = self.initialised_robot(
            config=FaultConfig(timeout_partial=True, timeout_partial_unreached_indices=(0, 4))
        )
        target = [v + 0.20 for v in LEFT_ARM_HOME]
        self.assertIs(robot.set_joint_positions(target, ["left_arm"]), ControlStatus.TIMEOUT)
        reached = robot.get_joint_positions(["left_arm"])["left_arm"]
        missed = [i for i, (r, t) in enumerate(zip(reached, target)) if abs(r - t) > 1e-9]
        self.assertEqual(missed, [0, 4])
        self.teardown_robot(robot)


# ==========================================================================
class TestFault4_BareFault(RobotTestCase):
    """Fault 4. Failure-matrix row R5. Zero motion -- distinct from TIMEOUT."""

    def test_fault_means_zero_motion(self) -> None:
        robot = self.initialised_robot(config=FaultConfig(bare_fault=True))
        before = robot.get_joint_positions(["left_arm"])["left_arm"]
        status = robot.set_joint_positions([v + 0.5 for v in LEFT_ARM_HOME], ["left_arm"])
        after = robot.get_joint_positions(["left_arm"])["left_arm"]

        self.assertIs(status, ControlStatus.FAULT)
        # MEASURED wording: "byte-identical to before: nothing moved at all".
        self.assertEqual(before, after)
        self.teardown_robot(robot)

    def test_fault_and_timeout_are_behaviourally_distinct(self) -> None:
        """The whole status ladder in one test: FAULT moves nothing, TIMEOUT
        moves most joints, SUCCESS may still be short of target."""
        fault_bot = self.initialised_robot(config=FaultConfig(bare_fault=True))
        before = fault_bot.get_joint_positions(["left_arm"])["left_arm"]
        fault_bot.set_joint_positions([v + 0.2 for v in LEFT_ARM_HOME], ["left_arm"])
        self.assertEqual(before, fault_bot.get_joint_positions(["left_arm"])["left_arm"])
        self.teardown_robot(fault_bot)

        path = Path(self._tmp.name) / "timeout.json"
        timeout_bot = FakeG1Robot(
            state_path=path, seed=1, config=FaultConfig(timeout_partial=True)
        )
        self.assertTrue(timeout_bot.init())
        timeout_bot.set_joint_positions([v + 0.2 for v in LEFT_ARM_HOME], ["left_arm"])
        after = timeout_bot.get_joint_positions(["left_arm"])["left_arm"]
        self.assertNotEqual(LEFT_ARM_HOME, after)
        self.teardown_robot(timeout_bot)


# ==========================================================================
class TestFault5_LatchedFaultSurvivesRestart(RobotTestCase):
    """Fault 5. Failure-matrix row R6.

    The single most important behaviour in this module, and the one a naive
    mock gets wrong: the latch lives in the controller, not the SDK process.
    """

    def _latch_the_robot(self) -> list[float]:
        """Fault the robot, return the pose it was left in."""
        robot = self.initialised_robot(
            config=FaultConfig(bare_fault=True, latch_fault=True, on_calls=(2,))
        )
        # Call 1 succeeds and leaves the arm somewhere distinctive.
        moved = [v + 0.10 for v in LEFT_ARM_HOME]
        self.assertIs(robot.set_joint_positions(moved, ["left_arm"]), ControlStatus.SUCCESS)
        # Call 2 faults and latches the controller.
        self.assertIs(
            robot.set_joint_positions([v + 0.20 for v in LEFT_ARM_HOME], ["left_arm"]),
            ControlStatus.FAULT,
        )
        pose = robot.get_joint_positions(["left_arm"])["left_arm"]
        self.assertEqual(pose, moved)  # left off its start pose, as on 7 Aug
        self.teardown_robot(robot)
        return pose

    def test_latch_survives_simulated_restart_while_init_and_getters_look_healthy(self) -> None:
        """THE test. Latched FAULT survives a restart while init() returns True
        and getters return live values."""
        pose_before = self._latch_the_robot()

        # A brand new adapter instance against the same controller.
        restarted = FakeG1Robot(state_path=self.state_path, seed=999)

        # 1. init() reports healthy.
        self.assertIs(restarted.init(), True)

        # 2. Every getter returns correct LIVE values -- not a reset home pose.
        self.assertEqual(restarted.get_joint_positions(["left_arm"])["left_arm"], pose_before)
        self.assertEqual(restarted.get_joint_positions(["head"])["head"], HEAD_HOME)
        self.assertIn("left_arm", restarted.get_joint_group_names())

        # 3. Only a motion command reveals the fault.
        before = restarted.get_joint_positions(["left_arm"])["left_arm"]
        self.assertIs(
            restarted.set_joint_positions([v + 0.3 for v in LEFT_ARM_HOME], ["left_arm"]),
            ControlStatus.FAULT,
        )
        self.assertEqual(restarted.get_joint_positions(["left_arm"])["left_arm"], before)
        self.teardown_robot(restarted)

    def test_latch_is_robot_wide_including_untouched_chains(self) -> None:
        """MEASURED: head and leg were untouched by the incident and still
        returned FAULT. Robot-wide latch, not per-chain damage."""
        self._latch_the_robot()
        restarted = FakeG1Robot(state_path=self.state_path, seed=5)
        self.assertTrue(restarted.init())

        self.assertIs(restarted.set_joint_positions([0.1, 0.1], ["head"]), ControlStatus.FAULT)
        self.assertIs(restarted.set_joint_positions(LEG_HOME, ["leg"]), ControlStatus.FAULT)
        self.assertIs(
            restarted.set_joint_positions(LEFT_ARM_HOME, ["right_arm"]), ControlStatus.FAULT
        )
        self.assertEqual(restarted.get_joint_positions(["head"])["head"], HEAD_HOME)
        self.teardown_robot(restarted)

    def test_latch_survives_a_genuinely_new_interpreter(self) -> None:
        """Not a new object -- a new process. This is what the state file is for."""
        pose_before = self._latch_the_robot()

        program = textwrap.dedent(
            f"""
            import json, sys
            sys.path.insert(0, {str(PROJECT_ROOT)!r})
            from galbot_motion_studio.testing import FakeG1Robot

            robot = FakeG1Robot(state_path={str(self.state_path)!r}, seed=42)
            out = {{}}
            out["pid_differs"] = True
            out["init"] = robot.init()
            out["left_arm"] = robot.get_joint_positions(["left_arm"])["left_arm"]
            out["head"] = robot.get_joint_positions(["head"])["head"]
            out["group_names_ok"] = "left_arm" in robot.get_joint_group_names()
            before = robot.get_joint_positions(["left_arm"])["left_arm"]
            out["motion"] = robot.set_joint_positions([0.0] * 7, ["left_arm"]).name
            out["moved"] = robot.get_joint_positions(["left_arm"])["left_arm"] != before
            out["head_motion"] = robot.set_joint_positions([0.1, 0.1], ["head"]).name
            out["driver_log"] = robot.debug_read_driver_log()
            robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
            print("RESULT " + json.dumps(out))
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}")
        line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT "))
        result = json.loads(line[len("RESULT ") :])

        self.assertIs(result["init"], True, "fresh interpreter: init() must still report healthy")
        self.assertEqual(result["left_arm"], pose_before, "getters must return live values")
        self.assertEqual(result["head"], HEAD_HOME)
        self.assertTrue(result["group_names_ok"])
        self.assertEqual(result["motion"], "FAULT", "the latch must survive the new process")
        self.assertFalse(result["moved"], "FAULT means zero motion")
        self.assertEqual(result["head_motion"], "FAULT", "latch is robot-wide")
        self.assertTrue(any("DriverEstop" in ln for ln in result["driver_log"]))

    def test_the_api_cannot_reveal_the_latch(self) -> None:
        """Finding 1g: there is no get_error(), no get_safety_state(), no way to
        ask. The only in-API signal is a refused motion command; everything else
        requires reading a log file off the robot."""
        self._latch_the_robot()
        robot = FakeG1Robot(state_path=self.state_path, seed=3)
        self.assertTrue(robot.init())

        api_surface = {name for name in dir(RobotAdapter) if not name.startswith("_")}
        for forbidden in ("get_error", "get_safety_state", "is_faulted", "get_health"):
            self.assertNotIn(forbidden, api_surface)

        # Out-of-band only: the operator SSHes in and reads the log.
        log = robot.debug_read_driver_log()
        self.assertTrue(any("code: 70" in line for line in log))
        self.assertTrue(any("ErrorSafetyCritical" in line for line in log))
        self.teardown_robot(robot)

    def test_power_cycle_does_not_clear_the_latch_but_clear_latch_does(self) -> None:
        """MEASURED 19:42: a power cycle did not clear the latch."""
        self._latch_the_robot()
        robot = FakeG1Robot(state_path=self.state_path, seed=3)
        self.assertTrue(robot.init())
        robot.power_cycle()
        self.assertTrue(robot.init())
        self.assertIs(robot.set_joint_positions([0.1, 0.1], ["head"]), ControlStatus.FAULT)
        self.assertTrue(robot.is_latched)

        robot.clear_latch()
        self.assertFalse(robot.is_latched)
        self.assertIs(robot.set_joint_positions([0.1, 0.1], ["head"]), ControlStatus.SUCCESS)
        self.assertEqual(robot.debug_read_driver_log(), [])
        self.teardown_robot(robot)

    def test_driver_error_cause_reports_different_codes(self) -> None:
        """MEASURED 19:51: 69/196608 (drive error, medium) is genuinely distinct
        from 70/327680 (e-stop, critical) inside the SDK -- and neither reaches
        the Python API, which collapses both to FAULT."""
        robot = self.initialised_robot(
            config=FaultConfig(
                bare_fault=True, latch_fault=True, latch_cause=LatchCause.DRIVER_ERROR
            )
        )
        self.assertIs(robot.set_joint_positions([0.1, 0.1], ["head"]), ControlStatus.FAULT)
        log = robot.debug_read_driver_log()
        self.assertTrue(any("code: 69" in line for line in log))
        self.assertTrue(any("Not yet identified" in line for line in log))
        # The API-visible status is identical to the e-stop case.
        self.assertIs(robot.set_joint_positions([0.1, 0.1], ["head"]), ControlStatus.FAULT)
        self.teardown_robot(robot)

    def test_a_naive_mock_would_pass_only_if_it_persisted_state(self) -> None:
        """Guards the mechanism itself: the latch must be on disk, not on the
        instance. If someone refactors it into an attribute, this fails."""
        self._latch_the_robot()
        raw = json.loads(self.state_path.read_text())
        self.assertIsNotNone(raw["latch"])
        self.assertEqual(raw["latch"]["scope"], "robot_wide")
        self.assertIn("left_arm_joint1", raw["positions"])


# ==========================================================================
class TestFault6_StalledBlockingCall(RobotTestCase):
    """Fault 6. Failure-matrix row R7. t_motion is not bounded by t_cmd."""

    def test_blocking_call_stalls_for_the_configured_duration(self) -> None:
        slept: list[float] = []
        robot = self.initialised_robot(
            config=FaultConfig(stall_seconds=20.0, bare_fault=True),
            sleep_fn=slept.append,
        )
        robot.set_joint_positions([0.1, 0.1], ["head"])
        self.assertEqual(slept, [20.0])
        self.assertEqual(robot.call_log[-1].stalled_seconds, 20.0)
        self.teardown_robot(robot)

    def test_stall_really_blocks_wall_clock(self) -> None:
        """Small but real, so the mechanism is proven rather than mocked."""
        robot = self.initialised_robot(config=FaultConfig(stall_seconds=0.05))
        start = time.monotonic()
        robot.set_joint_positions([0.1, 0.1], ["head"])
        self.assertGreaterEqual(time.monotonic() - start, 0.05)
        self.teardown_robot(robot)

    def test_in_flight_command_still_moves_the_robot_after_an_abort(self) -> None:
        """"Stop commanding" does not equal "stop moving". There is no known
        cancel API, so an abort raised during a blocking call changes nothing
        about that call's outcome."""
        robot = self.initialised_robot(config=FaultConfig(stall_seconds=0.02))

        def sleep_then_abort(seconds: float) -> None:
            time.sleep(seconds)
            robot.request_abort()  # operator hits stop mid-call

        robot._sleep = sleep_then_abort  # noqa: SLF001 - deliberate injection
        before = robot.get_joint_positions(["head"])["head"]
        status = robot.set_joint_positions([0.25, 0.25], ["head"])
        after = robot.get_joint_positions(["head"])["head"]

        self.assertTrue(robot.abort_requested)
        self.assertIs(status, ControlStatus.SUCCESS)
        self.assertNotEqual(before, after, "the in-flight command still executed")
        self.assertEqual(robot.motions_applied_after_abort, 1)
        self.teardown_robot(robot)


# ==========================================================================
class TestFault7_SwallowedSigint(RobotTestCase):
    """Fault 7. Opt-in and dangerous by design.

    MEASURED: KeyboardInterrupt appears zero times in a 3,060-line transcript
    while a loop kept issuing blocking motion commands for ~4 minutes after a
    Ctrl-C.
    """

    def _raise_and_wait(self, swallower, timeout: float = 2.0) -> None:
        signal.raise_signal(signal.SIGINT)
        deadline = time.monotonic() + timeout
        while swallower.caught_count == 0 and time.monotonic() < deadline:
            time.sleep(0.001)

    def test_sigint_produces_no_keyboardinterrupt(self) -> None:
        original = signal.getsignal(signal.SIGINT)
        try:
            with swallow_sigint() as swallower:
                try:
                    self._raise_and_wait(swallower)
                except KeyboardInterrupt:  # pragma: no cover - the failure mode
                    self.fail("KeyboardInterrupt escaped: the signal was not swallowed")
                self.assertEqual(swallower.caught_count, 1)
        finally:
            signal.signal(signal.SIGINT, original)
        self.assertEqual(signal.getsignal(signal.SIGINT), original, "handler must be restored")

    def test_abort_loop_depending_on_keyboardinterrupt_keeps_commanding(self) -> None:
        """The exact failure this fault exists to catch, in miniature."""
        robot = self.initialised_robot()
        commands_after_signal = 0
        original = signal.getsignal(signal.SIGINT)
        try:
            with swallow_sigint() as swallower:
                try:
                    for i in range(6):
                        robot.set_joint_positions([0.01 * i, 0.01 * i], ["head"])
                        if i == 1:
                            self._raise_and_wait(swallower)
                        if i >= 2:
                            commands_after_signal += 1
                except KeyboardInterrupt:  # pragma: no cover
                    pass
        finally:
            signal.signal(signal.SIGINT, original)

        self.assertEqual(swallower.caught_count, 1)
        self.assertEqual(
            commands_after_signal, 4, "the loop kept commanding motion after the interrupt"
        )
        self.teardown_robot(robot)

    def test_swallower_restores_previous_handler_on_exception(self) -> None:
        original = signal.getsignal(signal.SIGINT)
        with self.assertRaises(ValueError):
            with swallow_sigint():
                raise ValueError("boom")
        self.assertEqual(signal.getsignal(signal.SIGINT), original)


# ==========================================================================
class TestFault8_MissingTeardown(RobotTestCase):
    """Fault 8. Observable, never an actual interpreter crash."""

    def test_correct_sequence_is_accepted(self) -> None:
        robot = self.initialised_robot()
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
        self.assertTrue(robot.teardown_complete)
        robot.check_teardown()  # must not raise

    def test_destroy_without_the_sequence_raises_loudly(self) -> None:
        robot = self.initialised_robot()
        with self.assertRaises(MissingTeardownError) as ctx:
            robot.destroy()
        self.assertIn("request_shutdown() -> wait_for_shutdown() -> destroy()", str(ctx.exception))
        self.teardown_robot(robot)

    def test_out_of_order_calls_raise(self) -> None:
        robot = self.initialised_robot()
        with self.assertRaises(MissingTeardownError):
            robot.wait_for_shutdown()
        robot.request_shutdown()
        with self.assertRaises(MissingTeardownError):
            robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()

    def test_context_manager_raises_when_teardown_is_skipped(self) -> None:
        with self.assertRaises(MissingTeardownError):
            with self.make_robot() as robot:
                robot.init()
                robot.set_joint_positions([0.1, 0.1], ["head"])

    def test_context_manager_is_clean_when_teardown_happens(self) -> None:
        with self.make_robot() as robot:
            robot.init()
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()

    def test_atexit_hook_is_loud_and_does_not_segfault(self) -> None:
        program = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(PROJECT_ROOT)!r})
            from galbot_motion_studio.testing import FakeG1Robot
            robot = FakeG1Robot(state_path={str(self.state_path)!r}, register_atexit_check=True)
            robot.init()
            robot.set_joint_positions([0.1, 0.1], ["head"])
            print("reached end of program")
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=120
        )
        self.assertIn("reached end of program", proc.stdout)
        self.assertIn("GALBOT-FAKE: MISSING TEARDOWN", proc.stderr)
        self.assertIn("MissingTeardownError", proc.stderr)
        # Detectable by CI, not just by a human reading a terminal. CPython
        # ignores exceptions raised inside atexit callbacks -- verified on
        # 3.14.6, where even SystemExit(70) still exits 0 -- so the hook forces
        # the status itself.
        self.assertEqual(proc.returncode, MISSING_TEARDOWN_EXIT_CODE)
        # The real SDK SIGSEGVs here (returncode -11 / 139). A test double must not.
        self.assertNotIn(proc.returncode, (-signal.SIGSEGV, 139))

    def test_atexit_hook_is_silent_after_a_correct_shutdown(self) -> None:
        program = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(PROJECT_ROOT)!r})
            from galbot_motion_studio.testing import FakeG1Robot
            robot = FakeG1Robot(state_path={str(self.state_path)!r}, register_atexit_check=True)
            robot.init()
            robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()
            print("clean exit B")
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("clean exit B", proc.stdout)
        self.assertNotIn("MISSING TEARDOWN", proc.stderr)


# ==========================================================================
class TestFault9_GettersAfterFailedInit(RobotTestCase):
    """Fault 9. Failure-matrix row R2. On hardware this path is a SIGSEGV."""

    def test_init_fails_silently(self) -> None:
        robot = self.make_robot(config=FaultConfig(fail_init=True))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = robot.init()
        self.assertIs(result, False)
        self.assertEqual(out.getvalue(), "", "init() must print nothing on failure")
        self.assertEqual(err.getvalue(), "", "init() must print nothing on failure")

    def test_getters_raise_after_failed_init(self) -> None:
        robot = self.make_robot(config=FaultConfig(fail_init=True))
        self.assertFalse(robot.init())
        for call in (robot.get_joint_group_names, lambda: robot.get_joint_positions(["head"])):
            with self.assertRaises(UninitializedGetterError) as ctx:
                call()
            self.assertIn("SIGSEGV", str(ctx.exception))

    def test_getters_raise_before_init_is_called_at_all(self) -> None:
        """MEASURED: before init(), calling any getter segfaults immediately."""
        robot = self.make_robot()
        with self.assertRaises(UninitializedGetterError):
            robot.get_joint_group_names()

    def test_getters_raise_after_destroy(self) -> None:
        robot = self.initialised_robot()
        self.teardown_robot(robot)
        with self.assertRaises(UninitializedGetterError):
            robot.get_joint_positions(["head"])

    def test_motion_after_failed_init_is_guarded_not_fatal(self) -> None:
        """Contrast with the getters: the SDK does guard the publish path."""
        robot = self.make_robot(config=FaultConfig(fail_init=True))
        self.assertFalse(robot.init())
        self.assertIs(
            robot.set_joint_positions([0.1, 0.1], ["head"]), ControlStatus.INIT_FAILED
        )

    def test_init_after_destroy_returns_false(self) -> None:
        """DOCUMENTED: after destroy() the SDK cannot be re-initialised in the
        same process. A restart means a new process (or a new adapter)."""
        robot = self.initialised_robot()
        self.teardown_robot(robot)
        self.assertFalse(robot.init())


# ==========================================================================
class TestDeterminism(RobotTestCase):
    """A test that fails must be reproducible from (seed, call sequence)."""

    def _run_sequence(self, seed: int, path: Path) -> list[tuple[str, tuple[float, ...]]]:
        robot = FakeG1Robot(
            state_path=path,
            seed=seed,
            config=FaultConfig(bare_fault=True, probability=0.5),
        )
        self.assertTrue(robot.init())
        out = []
        for i in range(24):
            status = robot.set_joint_positions([0.01 * i, 0.01 * i], ["head"])
            out.append((status.name, tuple(robot.get_joint_positions(["head"])["head"])))
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
        return out

    def test_same_seed_reproduces_the_sequence_exactly(self) -> None:
        a = self._run_sequence(4242, Path(self._tmp.name) / "a.json")
        b = self._run_sequence(4242, Path(self._tmp.name) / "b.json")
        self.assertEqual(a, b)
        # It must be a genuinely mixed sequence, or the test proves nothing.
        self.assertEqual({s for s, _ in a}, {"SUCCESS", "FAULT"})

    def test_different_seeds_produce_different_sequences(self) -> None:
        sequences = {
            tuple(s for s, _ in self._run_sequence(seed, Path(self._tmp.name) / f"s{seed}.json"))
            for seed in (1, 2, 3, 4, 5)
        }
        self.assertGreater(len(sequences), 1)

    def test_rng_stream_does_not_depend_on_which_faults_are_configured(self) -> None:
        """Two draws per motion call, unconditionally. If the stream were
        config-dependent, changing a fault would silently change every later
        random decision and seeds would stop reproducing anything."""

        def run(prelude: FaultConfig) -> list[str]:
            path = Path(self._tmp.name) / f"stream_{id(prelude)}.json"
            robot = FakeG1Robot(state_path=path, seed=31337, config=prelude)
            self.assertTrue(robot.init())
            for i in range(3):  # three calls under the prelude config
                robot.set_joint_positions([0.01 * i, 0.01 * i], ["head"])
            robot.faults.set_default(FaultConfig(bare_fault=True, probability=0.5))
            tail = [
                robot.set_joint_positions([0.02 * i, 0.02 * i], ["head"]).name for i in range(15)
            ]
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()
            return tail

        nominal_prelude = run(FaultConfig())
        faulty_prelude = run(FaultConfig(bare_fault=True, on_calls=(1, 2, 3)))
        self.assertEqual(nominal_prelude, faulty_prelude)

    def test_call_log_records_enough_to_reproduce(self) -> None:
        robot = self.initialised_robot(config=FaultConfig(clamp_to_limit=True))
        robot.set_joint_positions([0.0, MEASURED_BOW_COMMANDED_RAD], ["head"], timeout=20.0)
        record = robot.call_log[-1]
        self.assertEqual(record.call_index, 1)
        self.assertEqual(record.groups, ("head",))
        self.assertEqual(record.joint_names, ("head_joint1", "head_joint2"))
        self.assertEqual(record.commanded, (0.0, MEASURED_BOW_COMMANDED_RAD))
        self.assertIs(record.status, ControlStatus.SUCCESS)
        self.assertAlmostEqual(record.max_error_rad, ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD)
        self.assertEqual(record.timeout, 20.0)
        self.teardown_robot(robot)


# ==========================================================================
class TestConfigurationScoping(RobotTestCase):
    """Faults must be switchable per-call and per-joint-group."""

    def test_per_group_configuration(self) -> None:
        robot = self.initialised_robot()
        robot.faults.set_group("head", FaultConfig(bare_fault=True))
        self.assertIs(robot.set_joint_positions([0.1, 0.1], ["head"]), ControlStatus.FAULT)
        self.assertIs(robot.set_joint_positions(LEG_HOME, ["leg"]), ControlStatus.SUCCESS)
        self.teardown_robot(robot)

    def test_first_named_group_with_a_config_wins(self) -> None:
        robot = self.initialised_robot()
        robot.faults.set_group("leg", FaultConfig(bare_fault=True))
        # head has no config, leg does -> leg's applies.
        status = robot.set_joint_positions(HEAD_HOME + LEG_HOME, ["head", "leg"])
        self.assertIs(status, ControlStatus.FAULT)
        self.assertEqual(robot.call_log[-1].config_source, "group:leg")
        self.teardown_robot(robot)

    def test_per_call_queue_applies_once_and_outranks_the_group(self) -> None:
        robot = self.initialised_robot()
        robot.faults.set_group("head", FaultConfig(bare_fault=True))
        robot.faults.queue_for_next_call(FaultConfig(timeout_partial=True))

        self.assertIs(robot.set_joint_positions([0.1, 0.1], ["head"]), ControlStatus.TIMEOUT)
        self.assertEqual(robot.call_log[-1].config_source, "queued")
        self.assertIs(robot.set_joint_positions([0.1, 0.1], ["head"]), ControlStatus.FAULT)
        self.assertEqual(robot.call_log[-1].config_source, "group:head")
        self.teardown_robot(robot)

    def test_on_calls_schedule_is_exact(self) -> None:
        robot = self.initialised_robot(
            config=FaultConfig(bare_fault=True, on_calls=(2, 4))
        )
        statuses = [
            robot.set_joint_positions([0.01 * i, 0.01 * i], ["head"]).name for i in range(1, 6)
        ]
        self.assertEqual(statuses, ["SUCCESS", "FAULT", "SUCCESS", "FAULT", "SUCCESS"])
        self.teardown_robot(robot)

    def test_probability_zero_never_fires(self) -> None:
        robot = self.initialised_robot(config=FaultConfig(bare_fault=True, probability=0.0))
        for i in range(10):
            self.assertIs(
                robot.set_joint_positions([0.01 * i, 0.01 * i], ["head"]), ControlStatus.SUCCESS
            )
        self.teardown_robot(robot)

    def test_faults_are_independently_switchable(self) -> None:
        """Each fault on its own, one robot each, no cross-talk."""
        cases = [
            (FaultConfig(bare_fault=True), ControlStatus.FAULT),
            (FaultConfig(timeout_partial=True), ControlStatus.TIMEOUT),
            (FaultConfig(clamp_to_limit=True), ControlStatus.SUCCESS),
            (FaultConfig(stop_short_rad=ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD), ControlStatus.SUCCESS),
            (FaultConfig(stall_seconds=0.0), ControlStatus.SUCCESS),
        ]
        for i, (config, expected) in enumerate(cases):
            with self.subTest(config=config):
                path = Path(self._tmp.name) / f"indep_{i}.json"
                robot = FakeG1Robot(state_path=path, seed=1, config=config)
                self.assertTrue(robot.init())
                self.assertIs(robot.set_joint_positions([0.1, 0.2], ["head"]), expected)
                self.teardown_robot(robot)


# ==========================================================================
class TestInvalidInput(RobotTestCase):
    def test_invalid_inputs_return_invalid_input(self) -> None:
        robot = self.initialised_robot()
        cases = {
            "wrong length": ([0.1], ["head"]),
            "unknown group": ([0.1, 0.2], ["legs"]),
            "empty expansion": ([], ["head"]),
            "nan": ([float("nan"), 0.2], ["head"]),
            "inf": ([float("inf"), 0.2], ["head"]),
        }
        for label, (positions, groups) in cases.items():
            with self.subTest(case=label):
                self.assertIs(
                    robot.set_joint_positions(positions, groups), ControlStatus.INVALID_INPUT
                )
        self.assertIs(
            robot.set_joint_positions([0.1, 0.2], ["head"], speed=0.0), ControlStatus.INVALID_INPUT
        )
        self.assertIs(
            robot.set_joint_positions([0.1, 0.2], ["head"], timeout=-1.0),
            ControlStatus.INVALID_INPUT,
        )
        self.teardown_robot(robot)

    def test_joint_names_override_groups(self) -> None:
        """DOCUMENTED SDK rule: joint_names takes precedence over joint_groups."""
        robot = self.initialised_robot()
        status = robot.set_joint_positions(
            [0.15], ["left_arm"], joint_names=["head_joint1"]
        )
        self.assertIs(status, ControlStatus.SUCCESS)
        self.assertEqual(robot.get_joint_positions(["head"])["head"][0], 0.15)
        self.assertEqual(robot.get_joint_positions(["left_arm"])["left_arm"], LEFT_ARM_HOME)
        self.teardown_robot(robot)


# ==========================================================================
class TestEvidenceGuards(RobotTestCase):
    """Fail if a measured constant is ever quietly replaced by a convenient one."""

    def test_band_bounds_match_the_field_log_to_printed_precision(self) -> None:
        self.assertEqual(round(ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD, 4), 0.0062)
        self.assertEqual(round(ACCEPTANCE_TOLERANCE_UPPER_BOUND_RAD, 4), 0.0359)
        self.assertLess(ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD, ACCEPTANCE_TOLERANCE_UPPER_BOUND_RAD)

    def test_no_precise_threshold_is_asserted_anywhere_in_the_band(self) -> None:
        """If someone introduces a single threshold constant, this starts
        returning a status instead of raising, and the test fails."""
        step = (ACCEPTANCE_TOLERANCE_UPPER_BOUND_RAD - ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD) / 20
        for i in range(1, 20):
            probe = ACCEPTANCE_TOLERANCE_LOWER_BOUND_RAD + i * step
            with self.subTest(error=probe), self.assertRaises(ToleranceUnknownError):
                judge_reach(probe)

    def test_start_pose_is_the_measured_snapshot(self) -> None:
        robot = self.initialised_robot()
        self.assertEqual(robot.get_joint_positions(["head"])["head"], [-0.0006, -0.0002])
        self.assertEqual(robot.get_joint_positions(["left_arm"])["left_arm"][2], -0.5624)
        self.teardown_robot(robot)

    def test_measured_pose_is_inside_the_urdf_limits(self) -> None:
        """24 of 24 measured joints fell inside their limits on 7 Aug. If this
        ever fails, either the pose table or the limit table is wrong."""
        for name, value in MEASURED_POSE_2026_08_07_1806.items():
            limit = JOINT_LIMITS.get(name)
            if limit is None:
                continue
            with self.subTest(joint=name):
                self.assertTrue(limit.contains(value), f"{name}={value} outside {limit}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
