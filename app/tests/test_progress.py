import pytest

from app.progress import estimate_remaining, format_duration


@pytest.mark.parametrize("seconds,expected", [(None, "estimating…"), (0, "00:00"),
    (1.2, "00:02"), (60, "01:00"), (3661, "1:01:01")])
def test_duration(seconds, expected):
    assert format_duration(seconds) == expected


def test_eta_uses_completed_windows():
    assert estimate_remaining(0, 0, 10) is None
    assert estimate_remaining(12, 3, 10) == 28
    assert estimate_remaining(12, 10, 10) == 0
    assert estimate_remaining(12, 1, 1) == 0
