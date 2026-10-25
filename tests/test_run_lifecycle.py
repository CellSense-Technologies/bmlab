"""
Cancel, run identity and cleanup of evaluation runs.

- Run identity: results of a cancelled or obsolete run never reach a
  newer run, whether the newer run starts on the same controller or on
  a new controller of the same session, and whether or not the abort
  flag of the old run was reset.
- Cleanup: after completion, cancel and failure no worker process
  survives and the per-run snapshot is gone.
- Workers which die or hang stop the run with an error.
"""
import multiprocessing
import os
import threading
import time
from multiprocessing.pool import ThreadPool
from unittest import mock

import numpy as np
import pytest

from bmlab import parallel_fit
from bmlab.controllers import EvaluationController, \
    EvaluationSnapshot, ImageController
from bmlab.evaluation_run import EvaluationRuns, RunSuperseded, \
    evaluation_runs
from bmlab.parallel_fit import FitExecution, OrderedBatchRunner, \
    WorkerDied, WorkerFailure, WorkerTimeout
from bmlab.session import Session

import worker_faults
from evaluation_parity import Counter, SCENARIOS, assert_arrays_identical, \
    assert_outcomes_identical, prepare_session, run_evaluation, snapshot

WAIT = 60

SERIAL = parallel_fit.serial_execution()


def controller_class(execution=SERIAL):
    class Controller(EvaluationController):
        _fit_execution = execution
    return Controller


def thread_pool(workers):
    return ThreadPool(workers)


def fit_execution(pool_factory=thread_pool, **kwargs):
    return FitExecution(
        parallel=True, workers=2, batch_size=2, max_outstanding_jobs=2,
        pool_factory=pool_factory, **kwargs)


def outcome_of(session, count, max_count):
    """The outputs of a finished run in the form of run_evaluation."""
    outcome = snapshot(session)
    outcome.update(count=count.value, max_count=max_count.value,
                   abort_polled=None, error_type=None)
    return outcome


_reference = {}


def clean_run():
    """The outputs of a clean serial run of Water.h5 (computed once)."""
    if 'outcome' not in _reference:
        _reference['outcome'] = run_evaluation(
            controller_class(), **SCENARIOS['water_1peak'])
    return _reference['outcome']


def assert_equals_clean_run(session, count, max_count):
    assert_outcomes_identical(
        outcome_of(session, count, max_count), clean_run())


def assert_stored_points_equal_clean_run(session):
    """
    Whatever points a stopped run stored are exactly the points of
    a clean run (which points were stored may differ).
    """
    actual = snapshot(session)['results']
    expected = clean_run()['results']
    for key, value in actual.items():
        if key in ('time', 'intensity'):
            continue
        stored = ~np.isnan(value).all(axis=tuple(range(3, value.ndim)))
        assert_arrays_identical(value[stored], expected[key][stored], key)


def poisoned_images(get_image):
    def get_poisoned_image(snapshot, image_key):
        imgs = get_image(snapshot, image_key)
        return None if imgs is None else imgs + 1000.0
    return get_poisoned_image


class Gate(object):
    """
    Holds the extraction of one image key, once, until it is released.
    It also records every extraction (thread, key, run number) and
    whether the controller of the thread had a snapshot.

    With `poison`, the held extraction reads images which no clean run
    can read. A write of that point into the results can then
    not go unnoticed, although a clean run writes the same point.
    Only use it when no other run extracts at the same time.
    """

    def __init__(self, monkeypatch, image_key, poison=False):
        self.image_key = image_key
        self.reached = threading.Event()
        self.release_event = threading.Event()
        self.calls = []
        self._held = False
        self._lock = threading.Lock()
        original = ImageController.extract_spectra
        gate = self

        def extract_spectra(controller, image_key, frame_num=None):
            thread = threading.current_thread()
            run = controller._run
            gate.calls.append((
                thread, image_key,
                None if run is None else run.run_id,
                controller._snapshot is not None))
            with gate._lock:
                hold = image_key == gate.image_key and not gate._held
                if hold:
                    gate._held = True
            if hold:
                gate.reached.set()
                assert gate.release_event.wait(WAIT)
                if poison:
                    with mock.patch.object(
                            EvaluationSnapshot, 'get_image',
                            poisoned_images(EvaluationSnapshot.get_image)):
                        return original(controller, image_key, frame_num)
            return original(controller, image_key, frame_num)

        monkeypatch.setattr(
            EvaluationController, 'extract_spectra', extract_spectra)

    def wait_until_held(self):
        assert self.reached.wait(WAIT)

    def release(self):
        self.release_event.set()

    def keys_of(self, thread):
        return [key for t, key, _, _ in self.calls if t is thread]


BEGUN = []


@pytest.fixture(autouse=True)
def record_runs(monkeypatch):
    """Remember which thread began which run."""
    BEGUN.clear()
    begin = EvaluationRuns.begin

    def recording_begin(registry):
        run = begin(registry)
        BEGUN.append((threading.current_thread(), run))
        return run

    monkeypatch.setattr(EvaluationRuns, 'begin', recording_begin)


class RunThread(threading.Thread):
    """Runs controller.evaluate in a thread."""

    def __init__(self, controller, abort=None):
        super().__init__(daemon=True)
        self.controller = controller
        self.abort = abort
        self.count = Counter(0)
        self.max_count = Counter(0)
        self.error = None

    def run(self):
        try:
            self.controller.evaluate(
                abort=self.abort, count=self.count,
                max_count=self.max_count)
        except BaseException as error:
            self.error = error

    @property
    def run_made(self):
        """The run this thread began (controller.last_run may be newer)."""
        runs = [run for thread, run in BEGUN if thread is self]
        assert len(runs) == 1
        return runs[0]

    def finish(self):
        self.join(WAIT)
        assert not self.is_alive()
        assert self.error is None, self.error


def no_process_alive(pids):
    assert multiprocessing.active_children() == []
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


# Identity of runs ----------------------------------------------------------

def test_runs_are_numbered_and_only_the_newest_is_current():
    runs = EvaluationRuns()
    first = runs.begin()
    assert first.run_id == 1 and first.is_current()
    second = runs.begin()
    assert second.run_id == 2
    assert second.is_current() and not first.is_current()
    with pytest.raises(RunSuperseded):
        first.ensure_current()
    second.ensure_current()


def test_an_obsolete_run_cannot_write():
    runs = EvaluationRuns()
    first = runs.begin()
    with first.writing():
        pass
    second = runs.begin()
    with pytest.raises(RunSuperseded):
        with first.writing():
            raise AssertionError('The body of an obsolete run must not run')
    with second.writing():
        pass


def test_a_new_run_waits_for_a_write_in_progress():
    runs = EvaluationRuns()
    first = runs.begin()
    inside = threading.Event()
    leave = threading.Event()
    order = []

    def write():
        with first.writing():
            inside.set()
            leave.wait(WAIT)
            order.append('write done')

    writer = threading.Thread(target=write)
    writer.start()
    assert inside.wait(WAIT)

    def start():
        runs.begin()
        order.append('new run started')

    starter = threading.Thread(target=start)
    starter.start()
    time.sleep(0.2)
    assert order == []
    leave.set()
    writer.join(WAIT)
    starter.join(WAIT)
    assert order == ['write done', 'new run started']


def test_the_numbering_belongs_to_the_session():
    session = Session.get_instance()
    assert evaluation_runs(session) is evaluation_runs(session)
    first = EvaluationController()
    second = EvaluationController()
    prepare_session('Water.h5')
    first.evaluate(count=Counter(0), max_count=Counter(0))
    second.evaluate(count=Counter(0), max_count=Counter(0))
    assert second.last_run.run_id == first.last_run.run_id + 1
    assert first.last_run.outcome == 'completed'
    assert second.last_run.outcome == 'completed'


# Outcomes ------------------------------------------------------------------

def test_outcome_of_a_completed_run():
    session = prepare_session('Water.h5')
    controller = controller_class()()
    count, max_count = Counter(0), Counter(0)
    controller.evaluate(count=count, max_count=max_count)
    assert controller.last_run.outcome == 'completed'
    assert controller._run is None and controller._snapshot is None
    assert_equals_clean_run(session, count, max_count)


@pytest.mark.parametrize('execution', [
    SERIAL, fit_execution(), ], ids=['serial', 'parallel'])
def test_outcome_of_a_cancelled_run(execution):
    session = prepare_session('Water.h5')
    controller = controller_class(execution)()
    count = Counter(0)
    max_count = Counter(0)

    class Abort(object):
        @property
        def value(self):
            return count.value >= 3
    controller.evaluate(abort=Abort(), count=count, max_count=max_count)
    assert controller.last_run.outcome == 'cancelled'
    assert max_count.value == -1
    assert controller._run is None and controller._snapshot is None
    assert_stored_points_equal_clean_run(session)


def test_outcome_of_a_failed_run(mocker):
    prepare_session('Water.h5')
    controller = controller_class()()
    mocker.patch(
        'bmlab.controllers.fit_point', side_effect=ValueError('broken'))
    with pytest.raises(ValueError):
        controller.evaluate(count=Counter(0), max_count=Counter(0))
    assert controller.last_run.outcome == 'failed'
    assert controller._run is None and controller._snapshot is None


def test_outcome_of_a_run_which_cannot_start():
    session = prepare_session('Water.h5')
    session.set_current_repetition(None)
    controller = controller_class()()
    max_count = Counter(0)
    controller.evaluate(max_count=max_count)
    assert controller.last_run.outcome == 'not-started'
    assert max_count.value == -1


# Obsolete runs do not write into a newer run ---------------------------------

def start_second_run(session, controller, same_controller):
    """The second run, on the same or a new controller, in a thread."""
    if not same_controller:
        controller = controller_class()()
    return RunThread(controller)


@pytest.mark.parametrize('same_controller', [True, False],
                         ids=['same_controller', 'new_controller'])
def test_obsolete_run_stops_when_a_new_run_starts(
        monkeypatch, same_controller):
    session = prepare_session('Water.h5')
    gate = Gate(monkeypatch, '4', poison=True)
    first = RunThread(controller_class()())
    first.start()
    gate.wait_until_held()

    second = start_second_run(session, first.controller, same_controller)
    second.start()
    second.finish()
    first_calls_before = len(gate.keys_of(first))

    # The old run continues after the new one is complete
    gate.release()
    first.finish()

    assert first.run_made.outcome == 'superseded'
    assert first.max_count.value == -1
    # It did nothing after the release: no further extraction, no write
    assert len(gate.keys_of(first)) == first_calls_before
    assert second.run_made.outcome == 'completed'
    assert_equals_clean_run(session, second.count, second.max_count)


@pytest.mark.parametrize('same_controller', [True, False],
                         ids=['same_controller', 'new_controller'])
def test_cancel_and_restart_with_shared_and_reset_abort_flag(
        monkeypatch, same_controller):
    # What the application did: one abort flag for all runs. Cancel
    # sets it, the restart clears it, before the old run has seen it.
    session = prepare_session('Water.h5')
    abort = Counter(False)
    gate = Gate(monkeypatch, '4', poison=True)
    first = RunThread(controller_class()(), abort=abort)
    first.start()
    gate.wait_until_held()

    abort.value = True
    abort.value = False
    second = start_second_run(session, first.controller, same_controller)
    second.abort = abort
    second.start()
    second.finish()
    first_calls_before = len(gate.keys_of(first))

    gate.release()
    first.finish()

    assert first.run_made.outcome == 'superseded'
    # The old run did not revive. A revived run would extract
    # the remaining points of the scan once more.
    assert len(gate.keys_of(first)) == first_calls_before
    assert_equals_clean_run(session, second.count, second.max_count)


def test_runs_on_one_controller_keep_their_own_snapshot(monkeypatch):
    session = prepare_session('Water.h5')
    controller = controller_class()()
    gate = Gate(monkeypatch, '3')
    first = RunThread(controller)
    first.start()
    gate.wait_until_held()

    second_gate = Gate(monkeypatch, '6')
    second_gate.calls = gate.calls
    second = RunThread(controller)
    second.start()
    second_gate.wait_until_held()

    # The old run ends while the new run is under way
    gate.release()
    first.finish()
    second_gate.release()
    second.finish()

    run_numbers = {run for t, _, run, _ in gate.calls if t is second}
    assert len(run_numbers) == 1
    # The new run had its snapshot at every extraction,
    # also after the old run ended and reset its own.
    assert all(has_snapshot for t, _, _, has_snapshot in gate.calls
               if t is second)
    assert_equals_clean_run(session, second.count, second.max_count)


def poisoned(results):
    """Results which a clean run can never contain."""
    return [{key: value + 1000.0 for key, value in point.items()}
            for point in results]


class HeldPool(object):
    """
    A pool whose results are wrong (poisoned) and are delivered late,
    when the test releases them.
    """

    def __init__(self, workers=2):
        self.submitted = threading.Event()
        self.held = []
        self.terminated = False

    def apply_async(self, function, args, callback=None,
                    error_callback=None):
        self.held.append((callback, poisoned(function(*args))))
        self.submitted.set()

    def release(self):
        held, self.held = self.held, []
        for callback, results in held:
            callback(results)

    def close(self):
        pass

    def terminate(self):
        self.terminated = True

    def join(self):
        pass


@pytest.mark.parametrize('same_controller', [True, False],
                         ids=['same_controller', 'new_controller'])
def test_late_results_of_a_cancelled_run_do_not_reach_the_next_run(
        monkeypatch, same_controller):
    session = prepare_session('Water.h5')
    held_pool = HeldPool()
    abort = Counter(False)
    first = RunThread(
        controller_class(fit_execution(lambda workers: held_pool))(),
        abort=abort)
    first.start()
    assert held_pool.submitted.wait(WAIT)

    # Cancel, and restart at once with the same abort flag. The second
    # run is held mid-way, while the late results of the first arrive.
    abort.value = True
    abort.value = False
    gate = Gate(monkeypatch, '5')
    if same_controller:
        # The same controller, now with a healthy pool
        controller = first.controller
        controller._fit_execution = fit_execution()
    else:
        controller = controller_class(fit_execution())()
    second = RunThread(controller, abort=abort)
    second.start()
    gate.wait_until_held()

    held_pool.release()
    # Late results from a second source, the old run being long gone
    time.sleep(0.2)
    gate.release()
    second.finish()
    first.finish()

    assert first.run_made.outcome in ('superseded', 'cancelled')
    assert held_pool.terminated
    assert_equals_clean_run(session, second.count, second.max_count)


def test_results_which_are_ready_when_the_run_is_replaced_are_not_stored(
        monkeypatch):
    # The pool delivers a result at once. The old run is held in the
    # extraction of a later point. After the second run completed,
    # the old run goes on and must not store anything.
    session = prepare_session('Water.h5')
    gate = Gate(monkeypatch, '6', poison=True)

    class PoisonedInstantPool(HeldPool):
        def apply_async(self, function, args, callback=None,
                        error_callback=None):
            callback(poisoned(function(*args)))

    first = RunThread(controller_class(fit_execution(
        lambda workers: PoisonedInstantPool()))())
    first.start()
    gate.wait_until_held()

    second = RunThread(controller_class(fit_execution())())
    second.start()
    second.finish()

    gate.release()
    first.finish()

    assert first.run_made.outcome == 'superseded'
    assert_equals_clean_run(session, second.count, second.max_count)


def test_obsolete_run_does_not_reset_the_results_of_the_new_run(
        monkeypatch):
    # The old run is replaced before it prepares its result arrays.
    # It must not initialize the arrays of the new run again.
    session = prepare_session('Water.h5')
    gate = Gate(monkeypatch, '0', poison=True)
    first = RunThread(controller_class()())
    first.start()
    gate.wait_until_held()

    second = RunThread(controller_class()())
    second.start()
    second.finish()

    gate.release()
    first.finish()

    assert first.run_made.outcome == 'superseded'
    assert_equals_clean_run(session, second.count, second.max_count)


# Cleanup -------------------------------------------------------------------

class RecordingPool(ThreadPool):
    pools = []

    def __init__(self, workers):
        super().__init__(workers)
        self.calls = []
        RecordingPool.pools.append(self)

    def close(self):
        self.calls.append('close')
        super().close()

    def terminate(self):
        self.calls.append('terminate')
        super().terminate()

    def join(self):
        self.calls.append('join')
        super().join()


@pytest.fixture
def recording_pools():
    RecordingPool.pools = []
    return RecordingPool.pools


def test_cancelled_parallel_run_terminates_its_pool_and_drops_snapshot(
        recording_pools):
    session = prepare_session('Water.h5')
    controller = controller_class(fit_execution(RecordingPool))()
    count = Counter(0)

    class Abort(object):
        @property
        def value(self):
            return count.value >= 3

    controller.evaluate(abort=Abort(), count=count, max_count=Counter(0))

    assert [pool.calls for pool in recording_pools] \
        == [['terminate', 'join']]
    assert controller._snapshot is None and controller._run is None
    assert_stored_points_equal_clean_run(session)


def test_obsolete_parallel_run_terminates_its_pool_and_drops_snapshot(
        monkeypatch, recording_pools):
    prepare_session('Water.h5')
    gate = Gate(monkeypatch, '6', poison=True)
    first = RunThread(controller_class(fit_execution(RecordingPool))())
    first.start()
    gate.wait_until_held()
    second = RunThread(controller_class()())
    second.start()
    second.finish()
    gate.release()
    first.finish()

    assert [pool.calls for pool in recording_pools] \
        == [['terminate', 'join']]
    assert first.controller._snapshot is None
    assert first.controller._run is None


def test_failed_parallel_run_terminates_its_pool_and_drops_snapshot(
        mocker, recording_pools):
    prepare_session('Water.h5')
    controller = controller_class(fit_execution(RecordingPool))()
    mocker.patch.object(
        EvaluationController, '_store_point',
        side_effect=IndexError('broken'))
    with pytest.raises(IndexError):
        controller.evaluate(count=Counter(0), max_count=Counter(0))
    assert [pool.calls for pool in recording_pools] \
        == [['terminate', 'join']]
    assert controller._snapshot is None and controller._run is None


# Runner: dead and hung workers ---------------------------------------------

class SilentPool(object):
    """A pool which takes jobs and never returns a result."""

    def apply_async(self, function, args, callback=None,
                    error_callback=None):
        pass

    def close(self):
        pass

    def terminate(self):
        pass

    def join(self):
        pass


class SilentPoolWithDeadWorker(SilentPool):
    def check_workers(self):
        raise WorkerDied('worker died')


def synthetic_items(count=6):
    spectra = [np.zeros(4)]
    return (((idx, None), (spectra, spectra)) for idx in range(count))


def test_runner_fails_when_no_result_comes_back():
    runner = OrderedBatchRunner(
        SilentPool(), None, batch_size=2, max_outstanding_jobs=2,
        job_timeout=0.3)
    started = time.monotonic()
    with pytest.raises(WorkerTimeout, match='no result'):
        list(runner.run(synthetic_items()))
    assert time.monotonic() - started < 10


def test_runner_does_not_time_out_while_nothing_is_outstanding():
    class SlowItems(object):
        def __iter__(self):
            # The producer is slow, no job is outstanding at the end
            time.sleep(0.5)
            return iter([])

    runner = OrderedBatchRunner(
        SilentPool(), None, batch_size=2, max_outstanding_jobs=2,
        job_timeout=0.1)
    assert list(runner.run(SlowItems())) == []


def test_runner_fails_when_the_pool_reports_a_dead_worker():
    runner = OrderedBatchRunner(
        SilentPoolWithDeadWorker(), None, batch_size=2, max_outstanding_jobs=2,
        job_timeout=None)
    with pytest.raises(WorkerDied):
        list(runner.run(synthetic_items()))


def test_runner_abort_wins_while_waiting_for_a_silent_pool():
    polled = []

    def should_abort():
        polled.append(1)
        return len(polled) > 5

    runner = OrderedBatchRunner(
        SilentPool(), None, batch_size=2, max_outstanding_jobs=2,
        should_abort=should_abort, job_timeout=60)
    assert list(runner.run(synthetic_items())) == []
    assert runner.aborted


def test_default_execution_has_a_generous_timeout():
    assert parallel_fit.default_execution().job_timeout \
        == parallel_fit.JOB_TIMEOUT_SECONDS >= 60


def test_worker_failures_are_runtime_errors():
    assert issubclass(WorkerDied, WorkerFailure)
    assert issubclass(WorkerTimeout, WorkerFailure)
    assert issubclass(WorkerFailure, RuntimeError)


# Spawned worker processes ------------------------------------------------

class SpawnedPools(object):
    """Creates spawn pools, remembers their worker processes."""

    def __init__(self, initializer=None):
        self.initializer = initializer
        self.pools = []
        self.pids = []

    def __call__(self, workers):
        pool = parallel_fit.create_spawn_pool(
            workers, initializer=self.initializer)
        self.pools.append(pool)
        self.pids.extend(process.pid for process in pool._pool)
        return pool


def spawned_controller(initializer=None, **kwargs):
    pools = SpawnedPools(initializer)
    controller = controller_class(fit_execution(pools, **kwargs))()
    return controller, pools


def test_cancel_stops_a_run_with_spawned_workers(monkeypatch):
    session = prepare_session('Water.h5')
    controller, pools = spawned_controller()
    abort = Counter(False)
    gate = Gate(monkeypatch, '6')
    run = RunThread(controller, abort=abort)
    run.start()
    gate.wait_until_held()
    assert len(pools.pids) == 2

    abort.value = True
    released = time.monotonic()
    gate.release()
    run.finish()

    # Prompt: the run does not wait for outstanding work
    assert time.monotonic() - released < 10
    assert controller.last_run.outcome == 'cancelled'
    assert run.max_count.value == -1
    assert controller._snapshot is None and controller._run is None
    no_process_alive(pools.pids)
    assert_stored_points_equal_clean_run(session)


def test_cancel_stops_a_run_whose_workers_hang():
    session = prepare_session('Water.h5')
    controller, pools = spawned_controller(worker_faults.hang_in_job)
    abort = Counter(False)
    run = RunThread(controller, abort=abort)
    run.start()
    # The workers start, take their jobs and never answer
    deadline = time.monotonic() + WAIT
    while len(pools.pids) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(1.0)
    assert run.is_alive()

    abort.value = True
    cancelled = time.monotonic()
    run.finish()

    assert time.monotonic() - cancelled < 10
    assert controller.last_run.outcome == 'cancelled'
    assert run.max_count.value == -1
    no_process_alive(pools.pids)
    assert_stored_points_equal_clean_run(session)


def test_obsolete_run_with_spawned_workers_leaves_no_worker(monkeypatch):
    session = prepare_session('Water.h5')
    controller, pools = spawned_controller()
    gate = Gate(monkeypatch, '6', poison=True)
    first = RunThread(controller)
    first.start()
    gate.wait_until_held()

    second = RunThread(controller_class()())
    second.start()
    second.finish()
    gate.release()
    first.finish()

    assert controller.last_run.outcome == 'superseded'
    no_process_alive(pools.pids)
    assert_equals_clean_run(session, second.count, second.max_count)


def test_completed_run_with_spawned_workers_leaves_no_worker():
    session = prepare_session('Water.h5')
    controller, pools = spawned_controller()
    count, max_count = Counter(0), Counter(0)
    controller.evaluate(count=count, max_count=max_count)
    assert controller.last_run.outcome == 'completed'
    no_process_alive(pools.pids)
    assert_equals_clean_run(session, count, max_count)


@pytest.mark.parametrize('initializer', [
    worker_faults.die_at_start, worker_faults.die_in_job],
    ids=['dies_at_start', 'dies_with_its_job'])
def test_a_dead_worker_fails_the_run(monkeypatch, initializer):
    prepare_session('Water.h5')
    controller, pools = spawned_controller(initializer)
    serial_runs = []
    original = EvaluationController._fit_points_serial

    def spy(*args, **kwargs):
        serial_runs.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(EvaluationController, '_fit_points_serial', spy)
    count, max_count = Counter(0), Counter(0)
    started = time.monotonic()
    with pytest.raises(WorkerDied, match='worker process'):
        controller.evaluate(count=count, max_count=max_count)

    assert time.monotonic() - started < 60
    # No silent retry of the whole run in this process
    assert serial_runs == []
    assert count.value < 10
    assert controller.last_run.outcome == 'failed'
    assert controller._snapshot is None and controller._run is None
    assert len(pools.pids) == 2
    no_process_alive(pools.pids)


def test_a_hung_worker_fails_the_run():
    prepare_session('Water.h5')
    controller, pools = spawned_controller(
        worker_faults.hang_in_job, job_timeout=1.5)
    started = time.monotonic()
    with pytest.raises(WorkerTimeout, match='no result'):
        controller.evaluate(count=Counter(0), max_count=Counter(0))
    assert time.monotonic() - started < 60
    assert controller.last_run.outcome == 'failed'
    assert controller._snapshot is None and controller._run is None
    no_process_alive(pools.pids)


def test_workers_killed_from_outside_fail_the_run(monkeypatch):
    # All workers are idle and waiting for a job, when they are killed.
    # One of them holds the lock for the job queue of the pool, which
    # then is never released. The pool cannot shut down by itself.
    prepare_session('Water.h5')
    controller, pools = spawned_controller()
    gate = Gate(monkeypatch, '6')
    run = RunThread(controller)
    run.start()
    gate.wait_until_held()
    assert len(pools.pids) == 2
    time.sleep(0.5)
    for process in list(pools.pools[0]._pool):
        process.kill()
    for process in list(pools.pools[0]._pool):
        process.join(WAIT)

    gate.release()
    run.join(WAIT)
    assert not run.is_alive()
    assert isinstance(run.error, WorkerDied)
    assert controller.last_run.outcome == 'failed'
    no_process_alive(pools.pids)
