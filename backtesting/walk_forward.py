"""Walk-forward optimisation (rolling train/test windows)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def build_windows(dates: list[str], train_size: int, test_size: int) -> list[WalkForwardWindow]:
    """Build rolling windows over sorted ISO date strings.

    Example:
      dates=100, train=60, test=20 -> windows at [0:60]->[60:80], [20:80]->[80:100]
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be > 0")
    if len(dates) < train_size + test_size:
        return []

    windows: list[WalkForwardWindow] = []
    step = test_size
    i = 0
    while i + train_size + test_size <= len(dates):
        train = dates[i : i + train_size]
        test = dates[i + train_size : i + train_size + test_size]
        windows.append(
            WalkForwardWindow(
                train_start=train[0],
                train_end=train[-1],
                test_start=test[0],
                test_end=test[-1],
            )
        )
        i += step
    return windows
