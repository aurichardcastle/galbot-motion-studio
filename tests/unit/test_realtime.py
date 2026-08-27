from threading import Event
from time import sleep

import pytest

from galbot_motion_studio.realtime import LatestWinsProcessor, RealtimeWorkerError


def test_latest_wins_drops_pending_work_instead_of_growing_latency() -> None:
    release = Event()
    started = Event()

    def process(value: int) -> int:
        if value == 1:
            started.set()
            assert release.wait(timeout=1)
        return value * 10

    worker = LatestWinsProcessor(process)
    worker.start()
    worker.submit(1)
    assert started.wait(timeout=1)
    worker.submit(2)
    worker.submit(3)
    release.set()
    sleep(0.05)
    worker.close()

    outputs = []
    while (item := worker.poll_latest()) is not None:
        outputs.append(item.result)
    assert outputs[-1] == 30
    assert worker.metrics.submitted == 3
    assert worker.metrics.dropped_before_processing == 1
    assert worker.metrics.processed == 2


def test_latest_result_supersedes_an_unconsumed_old_result() -> None:
    with LatestWinsProcessor(lambda value: value) as worker:
        worker.submit("old")
        sleep(0.02)
        worker.submit("new")
        sleep(0.02)
        latest = worker.poll_latest()
    assert latest is not None and latest.result == "new"


def test_worker_exception_is_raised_on_the_caller_thread() -> None:
    def fail(_: int) -> int:
        raise ValueError("detector boundary failed")

    worker = LatestWinsProcessor(fail)
    worker.start()
    worker.submit(1)
    with pytest.raises(RealtimeWorkerError) as raised:
        worker.poll_latest(timeout_s=1.0)
    assert isinstance(raised.value.__cause__, ValueError)
    assert str(raised.value.__cause__) == "detector boundary failed"
    with pytest.raises(RealtimeWorkerError):
        worker.close()
