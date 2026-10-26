import pytest

from bmlab import parallel_fit
from bmlab.controllers import EvaluationController


@pytest.fixture(autouse=True)
def serial_evaluation_by_default(monkeypatch):
    """
    Evaluations in tests run in this process, unless a test chooses
    otherwise. Starting worker processes for the tiny test fixtures
    would multiply the run time of the suite. Tests of the parallel
    evaluation set `_fit_execution` explicitly.
    """
    monkeypatch.setattr(
        EvaluationController, '_fit_execution',
        parallel_fit.serial_execution())
