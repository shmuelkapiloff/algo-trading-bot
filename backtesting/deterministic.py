"""Deterministic context manager for reproducible backtests.

Usage
-----
    from backtesting.deterministic import deterministic_context

    with deterministic_context(seed=42):
        result = engine.run(bars, strategy)

    # Re-running with the same seed produces identical results.

How it works
------------
Seeds Python's built-in ``random`` module and NumPy's global RNG. The
FillSimulator accepts its own seed, but this context manager ensures that
any ad-hoc calls to ``random.random()`` elsewhere are also reproducible.
"""

from __future__ import annotations

import contextlib
import random
from typing import Generator


@contextlib.contextmanager
def deterministic_context(seed: int = 0) -> Generator[None, None, None]:
    """Seed random + (optional) numpy for the duration of the block.

    NumPy is seeded only if it is importable; it is not a hard dependency of
    the backtesting engine.
    """
    # Save current state
    py_state = random.getstate()
    np_state = None
    try:
        import numpy as np  # type: ignore
        np_state = np.random.get_state()
        np.random.seed(seed)
    except ImportError:
        pass

    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        if np_state is not None:
            try:
                import numpy as np  # type: ignore
                np.random.set_state(np_state)
            except ImportError:
                pass
