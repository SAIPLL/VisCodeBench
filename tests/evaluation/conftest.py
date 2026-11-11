"""
Shared pytest fixtures for the evaluation tests.

``log_pair`` gives a function that logs a figure into a fresh directory, so a test
can build a reference and a prediction log directory and score them end to end.
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before any plt import

import matplotlib.pyplot as plt
import pytest

from t2v_eval.logging import activate_t2v_logging


@pytest.fixture
def log_figure(tmp_path):
    """Return ``log(name, draw)``: run ``draw(ax)`` with T2V logging into ``tmp_path/name``."""
    def log(name, draw):
        log_dir = tmp_path / name
        log_dir.mkdir()
        activate_t2v_logging(str(log_dir))
        fig, ax = plt.subplots()
        draw(ax)
        plt.close(fig)
        return str(log_dir)
    yield log
    plt.close("all")
