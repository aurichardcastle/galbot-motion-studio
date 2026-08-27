import pytest

from galbot_motion_studio.retarget.filtering import OneEuroFilter


def test_one_euro_filter_smooths_step_without_overshoot() -> None:
    filter_ = OneEuroFilter(min_cutoff_hz=1.0, beta=0.0)
    assert filter_.filter(0.0, timestamp_ns=0) == 0.0
    values = [
        filter_.filter(1.0, timestamp_ns=index * 33_333_333)
        for index in range(1, 11)
    ]
    assert 0.0 < values[0] < values[-1] < 1.0
    assert values == sorted(values)


def test_one_euro_filter_rejects_clock_regression_and_nonfinite() -> None:
    filter_ = OneEuroFilter()
    filter_.filter(0.0, timestamp_ns=10)
    with pytest.raises(ValueError, match="strictly increase"):
        filter_.filter(0.1, timestamp_ns=10)
    with pytest.raises(ValueError, match="finite"):
        filter_.filter(float("nan"), timestamp_ns=11)
