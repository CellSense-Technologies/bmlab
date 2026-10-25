"""
Identity of evaluation runs.

Every call of `EvaluationController.evaluate` is a run. All runs on one
session are numbered in the order they start. Only the newest run is
current. An older run is obsolete from the moment a newer run starts,
whatever the reason of its being still alive is: it was cancelled
but has not noticed yet, or its abort flag was reset by a restart.

A run has to write its results inside `EvaluationRun.writing()`. This
raises `RunSuperseded` for an obsolete run, so it never writes. Starting
a new run takes the same lock. A write in progress therefore ends
before the new run can begin, and no write of an old run lands after it.

The numbering belongs to the session, not to the controller. It covers
a second `evaluate` call on the same controller, and a new controller
(BMicro creates one per evaluation thread) on the same session.

The module imports neither the session nor the controllers.
"""
import contextlib
import threading
import weakref


class RunSuperseded(Exception):
    """
    A newer evaluation run has started. `evaluate` catches this and
    returns, as it does for a cancelled run. It is internal.
    """


class EvaluationRun(object):
    """
    One evaluation run. Created by `EvaluationRuns.begin`.

    Attributes
    ----------
    run_id: int
        Number of the run on its session, starting with 1
    outcome: str
        'running' while the run is active, afterwards one of
        'completed', 'cancelled' (the abort input was set),
        'superseded' (a newer run started), 'failed' (an exception
        reached the caller), 'not-started' (a model or the
        repetition was missing, nothing was calculated).
    stats: dict or None
        Statistics of a parallel run, set when the run has ended (None
        for a serial run, or if the run has not ended). The keys are
        the ones of `bmlab.parallel_fit.OrderedBatchRunner.statistics`:
        `wait_seconds` is the time the controlling process waited for
        results, `utilisation` the share of its time the workers fitted.
    """

    def __init__(self, registry, run_id):
        self._registry = registry
        self.run_id = run_id
        self.outcome = 'running'
        self.stats = None

    def is_current(self):
        """True as long as no newer run has started."""
        return self._registry.latest_id == self.run_id

    def ensure_current(self):
        """Raise `RunSuperseded` if a newer run has started."""
        if not self.is_current():
            raise RunSuperseded(
                'Evaluation run {} was replaced by run {}.'.format(
                    self.run_id, self._registry.latest_id))

    @contextlib.contextmanager
    def writing(self):
        """
        Context of every write of this run into shared state (results,
        spectra, frequencies, derived values). Raises `RunSuperseded`
        if the run is obsolete. A new run cannot start while the
        context is active.
        """
        with self._registry.lock:
            self.ensure_current()
            yield


class EvaluationRuns(object):
    """The runs of one session."""

    def __init__(self):
        # Reentrant: guarded writes can nest
        self.lock = threading.RLock()
        self._latest_id = 0

    @property
    def latest_id(self):
        return self._latest_id

    def begin(self):
        """
        Start a run. Every earlier run is obsolete from now on. Waits
        for a write in progress of an earlier run to end.
        """
        with self.lock:
            self._latest_id += 1
            return EvaluationRun(self, self._latest_id)


_registries = weakref.WeakKeyDictionary()
_registries_lock = threading.Lock()


def evaluation_runs(session):
    """
    The `EvaluationRuns` of a session. It is kept outside the session
    object, which would otherwise try to serialize it.
    """
    with _registries_lock:
        registry = _registries.get(session)
        if registry is None:
            registry = EvaluationRuns()
            _registries[session] = registry
        return registry
