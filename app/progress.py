import math


def format_duration(seconds):
    if seconds is None:
        return "estimating…"
    seconds = max(0, math.ceil(seconds))
    hours, remaining = divmod(seconds, 3600)
    minutes, seconds = divmod(remaining, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def estimate_remaining(elapsed, completed_windows, total_windows):
    if completed_windows <= 0:
        return None
    return max(0.0, elapsed * (total_windows - completed_windows) / completed_windows)
