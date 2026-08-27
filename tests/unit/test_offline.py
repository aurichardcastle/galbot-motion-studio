from galbot_motion_studio.offline import SynchronousProcessor


def test_synchronous_processor_preserves_every_input_in_order() -> None:
    times = iter((100, 1_000_100, 1_000_100, 4_000_100, 4_000_100, 6_000_100))
    seen: list[int] = []
    processor = SynchronousProcessor(
        lambda value: seen.append(value) or value * 10,
        clock_ns=lambda: next(times),
    )

    outputs = [processor.process(value) for value in (3, 1, 2)]

    assert seen == [3, 1, 2]
    assert [output.result for output in outputs] == [30, 10, 20]
    assert [output.queue_latency_ms for output in outputs] == [0.0, 0.0, 0.0]
    assert processor.metrics.submitted == 3
    assert processor.metrics.processed == 3
    assert processor.metrics.dropped_before_processing == 0
    assert processor.metrics.mean_processing_ms == 2.0
