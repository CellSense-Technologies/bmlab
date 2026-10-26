import functools
import multiprocessing
import os
import threading
import time
from multiprocessing.pool import ThreadPool
from unittest import mock

import numpy as np
import pytest

from bmlab import parallel_fit
from bmlab.controllers import EvaluationController
from bmlab.parallel_fit import FitExecution, OrderedBatchRunner, fit_batch
from bmlab.point_fit import PointFitSettings, fit_point

from evaluation_parity import BOUNDS_2_GHZ, Counter, FWHM_2_GHZ, \
    RAISING_SCENARIOS, SCENARIOS, assert_arrays_identical, \
    assert_outcomes_identical, prepare_session, run_evaluation


def controller_with(execution):
    class Controller(EvaluationController):
        _fit_execution = execution
    return Controller


def parallel_controller(**kwargs):
    return controller_with(FitExecution(parallel=True, **kwargs))


SERIAL_CONTROLLER = controller_with(parallel_fit.serial_execution())


def thread_pool(workers):
    """Workers as threads: same interface, no process start-up."""
    return ThreadPool(workers)


@functools.lru_cache(maxsize=None)
def serial_outcome(scenario_id):
    """The serial result of a scenario (computed once per test run)."""
    return run_evaluation(SERIAL_CONTROLLER, **SCENARIOS[scenario_id])


def compare_with_serial(controller_class, scenario_id):
    kwargs = SCENARIOS[scenario_id]
    expected = serial_outcome(scenario_id)
    actual = run_evaluation(controller_class, **kwargs)
    if scenario_id in RAISING_SCENARIOS:
        assert expected['error_type'] is not None
        assert actual['error_type'] == expected['error_type']
        assert actual['max_count'] == expected['max_count']
        return
    assert expected['error_type'] is None
    assert_outcomes_identical(actual, expected)


ABORT_SCENARIOS = {scenario_id for scenario_id in SCENARIOS
                   if 'abort' in scenario_id and 'aborted' not in scenario_id}


# Parity -----------------------------------------------------------------

@pytest.mark.parametrize(
    'scenario_id',
    [scenario_id for scenario_id in SCENARIOS
     if scenario_id not in ABORT_SCENARIOS])
def test_all_parity_scenarios_in_batches_with_exact_equality(scenario_id):
    """
    Every parity scenario gives exactly the serial outputs when the points
    are fitted in small batches (workers are threads here, to keep the run
    time low; processes are used in the tests below).
    """
    controller = parallel_controller(
        workers=3, batch_size=3, max_outstanding_jobs=2,
        pool_factory=thread_pool)
    compare_with_serial(controller, scenario_id)


# One scenario per peak mode, with NaN spectra, in real spawned workers
SPAWN_SCENARIOS = [
    'water_1peak',
    'water_2peak_ghz_bounds',
    'water_4peak_ghz_bounds',
    'water_1peak_nan_spectra',
    'water_4peak_nan_spectra',
    'water_2peak_first_points_invalid',
    # worker raises (ValueError), parent raises (IndexError)
    'water_2peak_no_width_bounds',
    'water_1peak_more_brillouin_than_rayleigh_regions',
]


@pytest.mark.parametrize('scenario_id', SPAWN_SCENARIOS)
def test_parity_in_spawned_worker_processes(scenario_id):
    controller = parallel_controller(
        workers=2, batch_size=3, max_outstanding_jobs=3)
    compare_with_serial(controller, scenario_id)
    assert multiprocessing.active_children() == []


def test_parity_scenarios_contain_nan_and_inf_locations():
    """Guard: the compared parallel outputs are not trivially all valid."""
    outcome = run_evaluation(
        parallel_controller(workers=2, batch_size=4,
                            max_outstanding_jobs=2,
                            pool_factory=thread_pool),
        **SCENARIOS['water_4peak_nan_spectra'])
    position = outcome['results']['brillouin_peak_position_f']
    assert np.isnan(position[4]).all()
    assert np.isfinite(position[3]).any()


# Order of completion -----------------------------------------------------

class ReversingPool(object):
    """
    Pool whose jobs complete in reverse order of submission,
    if all jobs are submitted before the first one is handed back.
    Job k only finishes after job k + 1 has finished.
    """

    def __init__(self, nr_jobs):
        self.pool = ThreadPool(nr_jobs)
        self.finished = [threading.Event() for _ in range(nr_jobs)]
        self.completion_order = []
        self.submitted = 0
        self._lock = threading.Lock()

    def apply_async(self, function, args, callback=None,
                    error_callback=None):
        job = self.submitted
        self.submitted += 1

        def run():
            result = function(*args)
            if job + 1 < len(self.finished):
                assert self.finished[job + 1].wait(60)
            with self._lock:
                self.completion_order.append(job)
            self.finished[job].set()
            return result

        return self.pool.apply_async(
            run, callback=callback, error_callback=error_callback)

    def close(self):
        self.pool.close()

    def terminate(self):
        self.pool.terminate()

    def join(self):
        self.pool.join()


def test_out_of_order_completion_gives_serial_results():
    # 10 points in jobs of 2: five jobs, all outstanding at once
    pools = []

    def factory(workers):
        pools.append(ReversingPool(nr_jobs=5))
        return pools[0]

    controller = parallel_controller(
        workers=5, batch_size=2, max_outstanding_jobs=5,
        pool_factory=factory)
    kwargs = SCENARIOS['water_2peak_first_points_invalid']

    expected = serial_outcome('water_2peak_first_points_invalid')
    actual = run_evaluation(controller, **kwargs)

    # The last job completed first, the first job last
    assert pools[0].completion_order == [4, 3, 2, 1, 0]
    assert_outcomes_identical(actual, expected)


def test_rayleigh_reference_does_not_depend_on_completion_order():
    """
    The first valid Rayleigh position is found in point 1 (points 0 and
    part of 1 are invalid). With reversed completion the last point
    arrives first; it must not become the reference.
    """
    pools = []

    def factory(workers):
        pools.append(ReversingPool(nr_jobs=5))
        return pools[0]

    controller = parallel_controller(
        workers=5, batch_size=2, max_outstanding_jobs=5,
        pool_factory=factory)
    kwargs = SCENARIOS['water_2peak_first_points_invalid']

    expected = serial_outcome('water_2peak_first_points_invalid')
    actual = run_evaluation(controller, **kwargs)
    assert pools[0].completion_order == [4, 3, 2, 1, 0]

    position = actual['results']['rayleigh_peak_position_f']
    shift = actual['results']['rayleigh_shift']
    # Point 0 has no valid Rayleigh peak, point 1 frame 1 is the first
    assert np.isnan(position[0]).all()
    # Axes: x, y, z, frame, region, peak (the scan is along x)
    assert np.isnan(position[1, 0, 0, 0]).all()
    assert np.isfinite(position[1, 0, 0, 1]).all()
    # The reference point has zero shift against itself, in its valid frame
    assert np.array_equal(shift[1, 0, 0, 1],
                          np.zeros_like(shift[1, 0, 0, 1]))
    # Later points are shifted against it
    assert np.isfinite(shift[4:, 0, 0, 1]).all()
    assert (shift[4:, 0, 0, 1] != 0).any()
    assert_arrays_identical(
        shift, expected['results']['rayleigh_shift'], 'rayleigh_shift')
    assert_arrays_identical(
        actual['results']['brillouin_shift_f'],
        expected['results']['brillouin_shift_f'], 'brillouin_shift_f')


class ShufflingPool(object):
    """Jobs finish in a pseudo-random order (random delays)."""

    def __init__(self, workers, seed=1):
        self.pool = ThreadPool(workers)
        self.random = np.random.RandomState(seed)

    def apply_async(self, function, args, callback=None,
                    error_callback=None):
        delay = float(self.random.uniform(0, 0.03))

        def run():
            threading.Event().wait(delay)
            return function(*args)
        return self.pool.apply_async(
            run, callback=callback, error_callback=error_callback)

    def close(self):
        self.pool.close()

    def terminate(self):
        self.pool.terminate()

    def join(self):
        self.pool.join()


@pytest.mark.parametrize('scenario_id', [
    'water_1peak', 'water_4peak_ghz_bounds', 'water_2peak_incomplete_scan',
    'water_2peak_missing_images', 'water_1peak_missing_first_image'])
def test_random_completion_order_gives_serial_results(scenario_id):
    controller = parallel_controller(
        workers=4, batch_size=1, max_outstanding_jobs=6,
        pool_factory=lambda workers: ShufflingPool(workers))
    compare_with_serial(controller, scenario_id)


# Bounded outstanding work ------------------------------------------------

class InstantPool(object):
    """Computes a job when it is submitted; collects statistics."""

    def __init__(self, fn):
        self.fn = fn
        self.jobs = 0

    def apply_async(self, function, args, callback=None,
                    error_callback=None):
        self.jobs += 1
        callback(self.fn(*args))

    def close(self):
        pass

    def terminate(self):
        pass

    def join(self):
        pass


class DelayedPool(InstantPool):
    """
    Jobs complete a moment after submission, newest first. Many jobs
    are in flight, and results arrive out of order.
    """

    def __init__(self, fn):
        super().__init__(fn)
        self.waiting = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._complete, daemon=True)
        self.thread.start()

    def _complete(self):
        while not self.stop.wait(0.001):
            with self.lock:
                waiting, self.waiting = self.waiting, []
            for callback, result in reversed(waiting):
                callback(result)

    def apply_async(self, function, args, callback=None,
                    error_callback=None):
        self.jobs += 1
        with self.lock:
            self.waiting.append((callback, self.fn(*args)))

    def terminate(self):
        self.stop.set()
        self.thread.join()


def _run_synthetic(nr_points, batch_size, max_jobs, pool_class=InstantPool):
    """
    Run `nr_points` through a runner and measure from the outside how
    many points were taken from the input but not handed back yet.
    """
    taken = 0
    handed_back = 0
    peak_unreturned = 0

    def items():
        nonlocal taken, peak_unreturned
        for idx in range(nr_points):
            taken += 1
            peak_unreturned = max(peak_unreturned, taken - handed_back)
            # Every 7th point has no input (like a missing image)
            yield idx, None if idx % 7 == 3 else (
                [np.full(4, idx, dtype=float)], [np.arange(4.0)])

    pool = pool_class(lambda inputs, settings: [
        {'value': spectra[0][0]} for spectra, _ in inputs])
    runner = OrderedBatchRunner(pool, None, batch_size, max_jobs)
    handed_back_idx = []
    try:
        for idx, result in runner.run(items()):
            handed_back += 1
            handed_back_idx.append(idx)
            if idx % 7 == 3:
                assert result is None
            else:
                assert result['value'] == idx
    finally:
        pool.terminate()
    assert handed_back_idx == list(range(nr_points))
    return runner, peak_unreturned, pool


TUNED = parallel_fit.default_execution()
# The tuned values for the most workers (the batch grows with the worker
# count, so these hold the most points)
TUNED_MAX = parallel_fit.default_execution(
    cpus=parallel_fit.MAX_WORKERS + parallel_fit.RESERVED_CPUS)


@pytest.mark.parametrize('pool_class', [InstantPool, DelayedPool])
@pytest.mark.parametrize(
    'batch_size, max_jobs',
    [(5, 3), (TUNED.batch_size, TUNED.max_outstanding_jobs),
     (TUNED_MAX.batch_size, TUNED_MAX.max_outstanding_jobs)],
    ids=['small', 'tuned', 'tuned_max_workers'])
def test_outstanding_work_is_bounded_independent_of_scan_size(
        pool_class, batch_size, max_jobs):
    limit = (max_jobs + 1) * batch_size
    peaks = {}
    for nr_points in (50, 500, 5000):
        runner, peak_unreturned, pool = _run_synthetic(
            nr_points, batch_size, max_jobs, pool_class)
        assert pool.jobs >= nr_points * 6 / 7 / batch_size - 1
        assert runner.peak_outstanding_jobs <= max_jobs
        assert runner.peak_outstanding_points <= limit
        # Measured independently of the runner: points taken from the
        # input and not handed back. Skipped points (every 7th, without
        # input) are included. They take no job slot and only wait
        # behind the points with input, so they add 1 per 6 points.
        assert peak_unreturned <= limit * 7 / 6 + batch_size
        peaks[nr_points] = (runner.peak_outstanding_jobs,
                            runner.peak_outstanding_points,
                            peak_unreturned)
    if pool_class is InstantPool:
        # The same peaks for a 10x larger scan: no growth with scan size
        assert peaks[500] == peaks[5000]


def test_outstanding_jobs_limit_is_applied_to_waiting_results():
    """
    The limit counts jobs until their points are handed back, so a slow
    first job holds back new submissions, instead of queueing up results.
    """
    events = {'first': threading.Event()}
    sent = []

    class SlowFirstPool(object):
        def __init__(self):
            self.pool = ThreadPool(4)

        def apply_async(self, function, args, callback=None,
                        error_callback=None):
            job = len(sent)
            sent.append(job)

            def run():
                if job == 0:
                    assert events['first'].wait(60)
                return function(*args)
            return self.pool.apply_async(
                run, callback=callback, error_callback=error_callback)

        def close(self):
            self.pool.close()

        def terminate(self):
            self.pool.terminate()

        def join(self):
            self.pool.join()

    pool = SlowFirstPool()
    runner = OrderedBatchRunner(pool, None, 1, 3)

    def inputs():
        for idx in range(40):
            yield idx, ([np.full(2, float(idx))], [np.arange(2.0)])

    def fit(inputs_of_job, settings):
        return [{'value': spectra[0][0]} for spectra, _ in inputs_of_job]

    with mock.patch.object(parallel_fit, 'fit_batch', fit):
        results = []
        consumer = runner.run(inputs())
        timer = threading.Timer(0.5, events['first'].set)
        timer.start()
        for idx, result in consumer:
            results.append(idx)
            # Jobs sent but not handed back never exceed the limit, even
            # though later jobs are complete while job 0 is still running
            assert len(runner._left) <= 3
        timer.join()
    assert results == list(range(40))
    assert runner.peak_outstanding_jobs == 3
    pool.close()
    pool.join()


def run_and_get_runner(controller_class):
    """Run the water scenario and return the runner the run used."""
    runners = []
    original = OrderedBatchRunner

    def create_runner(*args, **kwargs):
        runners.append(original(*args, **kwargs))
        return runners[-1]

    with mock.patch.object(
            parallel_fit, 'OrderedBatchRunner', create_runner):
        run_evaluation(controller_class, **SCENARIOS['water_1peak'])
    return runners[0]


def test_controller_run_holds_only_a_few_points_outstanding():
    controller_class = parallel_controller(
        workers=2, batch_size=1, max_outstanding_jobs=2,
        pool_factory=thread_pool)
    run_evaluation(controller_class, **SCENARIOS['water_1peak'])
    runner = run_and_get_runner(controller_class)
    assert runner.peak_outstanding_jobs <= 2
    # 10 points in the scan
    assert runner.peak_outstanding_points <= 3


def test_controller_run_with_tuned_values_holds_a_bounded_number_of_points():
    execution = parallel_fit.default_execution()
    controller_class = parallel_controller(
        workers=3, batch_size=execution.batch_size,
        max_outstanding_jobs=execution.max_outstanding_jobs,
        pool_factory=thread_pool)
    runner = run_and_get_runner(controller_class)
    limit = (execution.max_outstanding_jobs + 1) * execution.batch_size
    assert runner.peak_outstanding_jobs <= execution.max_outstanding_jobs
    assert runner.peak_outstanding_points <= limit


# Worker processes ----------------------------------------------------------

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def read_environment(_):
    return os.getpid(), {key: os.environ.get(key)
                         for key in THREAD_VARIABLES}


def test_workers_are_spawned_and_limit_numerical_threads():
    before = {key: os.environ.get(key) for key in THREAD_VARIABLES}

    context = multiprocessing.get_context('spawn')
    with mock.patch.object(
            multiprocessing, 'get_context',
            return_value=context) as get_context:
        pool = parallel_fit.create_spawn_pool(2)
    get_context.assert_called_once_with('spawn')

    # The environment of this process is untouched
    assert {key: os.environ.get(key) for key in THREAD_VARIABLES} == before
    try:
        answers = pool.map(read_environment, range(4))
    finally:
        pool.terminate()
        pool.join()
    for pid, variables in answers:
        assert pid != os.getpid()
        assert variables == {key: '1' for key in THREAD_VARIABLES}


def test_no_worker_process_survives_a_run():
    pids = []

    def factory(workers):
        pool = parallel_fit.create_spawn_pool(workers)
        pids.extend(process.pid for process in pool._pool)
        return pool

    controller = parallel_controller(
        workers=2, batch_size=2, max_outstanding_jobs=2,
        pool_factory=factory)
    run_evaluation(controller, **SCENARIOS['water_1peak'])

    assert len(pids) == 2
    assert multiprocessing.active_children() == []
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_pool_is_created_per_run_and_closed_on_completion():
    pools = []

    class RecordingPool(ThreadPool):
        def __init__(self, workers):
            super().__init__(workers)
            self.calls = []
            pools.append(self)

        def close(self):
            self.calls.append('close')
            super().close()

        def terminate(self):
            self.calls.append('terminate')
            super().terminate()

        def join(self):
            self.calls.append('join')
            super().join()

    controller = parallel_controller(
        workers=2, batch_size=2, max_outstanding_jobs=2,
        pool_factory=RecordingPool)
    for _ in range(2):
        run_evaluation(controller, **SCENARIOS['water_1peak'])

    assert len(pools) == 2
    for pool in pools:
        assert pool.calls == ['close', 'join']


def test_failed_run_leaves_no_worker_process():
    pids = []

    def factory(workers):
        pool = parallel_fit.create_spawn_pool(workers)
        pids.extend(process.pid for process in pool._pool)
        return pool

    controller = parallel_controller(
        workers=2, batch_size=2, max_outstanding_jobs=2,
        pool_factory=factory)
    outcome = run_evaluation(
        controller, **SCENARIOS['water_2peak_no_width_bounds'])

    assert outcome['error_type'] is not None
    assert multiprocessing.active_children() == []
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


# Contract of evaluate ------------------------------------------------------

class RecordingCounter(Counter):
    def __init__(self):
        super().__init__(0)
        self.history = []

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'value' and hasattr(self, 'history'):
            self.history.append(value)


def test_progress_advances_with_every_point():
    controller_class = parallel_controller(
        workers=2, batch_size=3, max_outstanding_jobs=2,
        pool_factory=thread_pool)
    session = prepare_session('Water.h5')
    count = RecordingCounter()
    max_count = Counter(0)

    controller_class().evaluate(count=count, max_count=max_count)

    assert max_count.value == 10
    assert count.history[-10:] == list(range(1, 11))
    assert session.evaluation_model().results[
        'brillouin_peak_position_f'].shape[0] > 0


@pytest.mark.parametrize('scenario_id', sorted(ABORT_SCENARIOS))
def test_abort_stops_the_parallel_run(scenario_id):
    kwargs = SCENARIOS[scenario_id]
    serial = serial_outcome(scenario_id)
    parallel = run_evaluation(
        parallel_controller(workers=2, batch_size=2, max_outstanding_jobs=2,
                            pool_factory=thread_pool),
        **kwargs)

    assert parallel['error_type'] is None
    assert parallel['max_count'] == serial['max_count']
    assert multiprocessing.active_children() == []
    # Points which were stored are exactly the serial results
    # (which points are stored before stopping may differ).
    for key, value in parallel['results'].items():
        if key in ('time', 'intensity'):
            continue
        stored = ~np.isnan(value).all(
            axis=tuple(range(3, value.ndim)))
        assert_arrays_identical(
            value[stored], serial['results'][key][stored], key)


def test_abort_with_worker_processes_leaves_no_worker():
    kwargs = SCENARIOS['water_1peak_abort_midway']
    outcome = run_evaluation(
        parallel_controller(workers=2, batch_size=2, max_outstanding_jobs=2),
        **kwargs)
    assert outcome['error_type'] is None
    assert outcome['max_count'] == -1
    assert multiprocessing.active_children() == []


def test_default_execution_is_parallel_with_spawned_workers():
    execution = parallel_fit.default_execution()
    assert execution.parallel
    assert execution.pool_factory is None
    assert execution.workers >= 2
    assert execution.batch_size >= 1
    assert execution.max_outstanding_jobs >= execution.workers


def test_fit_batch_matches_fit_point_per_point():
    session = prepare_session(
        'Water.h5', nr_brillouin_peaks=2, bounds_w0=BOUNDS_2_GHZ,
        bounds_fwhm=FWHM_2_GHZ)
    controller = EvaluationController()
    cm = session.calibration_model()
    points = []
    for key in ('1', '2'):
        spectra, times, _ = controller.extract_spectra(key)
        points.append((spectra, list(cm.get_frequencies_by_time(times))))
    settings = PointFitSettings.snapshot(
        2, session.peak_selection_model().get_brillouin_regions(),
        session.peak_selection_model().get_rayleigh_regions(),
        BOUNDS_2_GHZ, FWHM_2_GHZ)

    results = fit_batch(points, settings)

    assert len(results) == 2
    for point, result in zip(points, results):
        expected = fit_point(*point, settings)
        for key, value in expected.items():
            assert_arrays_identical(result[key], value, key)


def test_worker_start_up_does_not_import_scipy_signal_for_single_peak_fits():
    # A worker imports bmlab.parallel_fit and fits one point. The import
    # of scipy.signal (only needed by multi-peak fits) costs 0.2 s per
    # worker start, so it must not happen for single-peak fits.
    import subprocess
    import sys
    code = (
        'import sys, numpy as np\n'
        'from bmlab import parallel_fit\n'
        'from bmlab.point_fit import PointFitSettings\n'
        'settings = PointFitSettings.snapshot(1, [(4.0e9, 6.0e9)], [], '
        'None, None)\n'
        'f = np.linspace(4.0e9, 6.0e9, 50)\n'
        'y = 1 / (1 + ((f - 5.0e9) / 1e8) ** 2)\n'
        'parallel_fit.fit_batch([([y], [f])], settings)\n'
        'print("scipy.signal" in sys.modules)\n')
    output = subprocess.run(
        [sys.executable, '-W', 'ignore', '-c', code],
        capture_output=True, text=True, check=True).stdout
    assert output.strip() == 'False'


def test_batch_size_grows_with_the_worker_count():
    sizes = {workers: parallel_fit.batch_size_for(workers)
             for workers in range(2, parallel_fit.MAX_WORKERS + 1)}
    # Small as before for 8 workers or fewer
    assert all(sizes[w] == 2 for w in range(2, 9))
    # 4 to 8 above, not shrinking with more workers
    assert all(4 <= sizes[w] <= 8 for w in range(9, 17))
    assert list(sizes.values()) == sorted(sizes.values())
    assert sizes[16] == 8
    for workers in (4, 8, 10, 16):
        execution = parallel_fit.default_execution(cpus=workers + 2)
        assert execution.workers == workers
        assert execution.batch_size == sizes[workers]


def test_points_held_for_the_workers_have_a_fixed_upper_limit():
    # (jobs + the open batch) x batch size, for the most workers
    limit = (TUNED_MAX.max_outstanding_jobs + 1) * TUNED_MAX.batch_size
    assert limit <= 1000
    for nr_points in (100, 20000):
        runner, _, _ = _run_synthetic(
            nr_points, TUNED_MAX.batch_size, TUNED_MAX.max_outstanding_jobs)
        assert runner.peak_outstanding_points <= limit


def test_runner_statistics_count_waiting_and_busy_time():
    class SlowPool(InstantPool):
        """Every job takes 20 ms in a thread."""

        def apply_async(self, function, args, callback=None,
                        error_callback=None):
            self.jobs += 1

            def work():
                time.sleep(0.02)
                callback(self.fn(*args))
            threading.Thread(target=work, daemon=True).start()

    pool = SlowPool(lambda inputs, settings: [
        {'value': 1} for _ in inputs])
    runner = OrderedBatchRunner(pool, None, 1, 2)
    started = time.perf_counter()
    items = [(idx, ([np.zeros(2)], [np.arange(2.0)])) for idx in range(10)]
    assert len(list(runner.run(items))) == 10
    wall = time.perf_counter() - started
    stats = runner.statistics(2, started)
    assert stats['jobs'] == stats['points'] == 10
    assert stats['workers'] == 2 and stats['batch_size'] == 1
    # The producer is instant, so it waits for nearly all of the run
    assert 0.5 * wall < stats['wait_seconds'] <= wall
    assert 0.02 <= stats['first_result_seconds'] < wall
    assert 0 < stats['span_seconds'] <= wall
    # A plain list result has no busy time
    assert stats['busy_seconds'] == 0.0 and stats['utilisation'] == 0.0


def test_fit_batch_reports_its_busy_time_through_pickling():
    import pickle
    results = parallel_fit.BatchResults([{'a': 1}])
    results.busy_seconds = 0.25
    copy = pickle.loads(pickle.dumps(results))
    assert copy == [{'a': 1}] and copy.busy_seconds == 0.25
    assert parallel_fit.fit_batch([], None).busy_seconds >= 0.0


@pytest.mark.parametrize('workers', [2, 3])
def test_parallel_run_records_statistics_on_the_run(workers):
    controller_class = parallel_controller(
        workers=workers, batch_size=2, max_outstanding_jobs=4,
        pool_factory=thread_pool)
    instances = []
    original = controller_class.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        instances.append(self)

    with mock.patch.object(controller_class, '__init__', init):
        run_evaluation(controller_class, **SCENARIOS['water_1peak'])
    stats = instances[0].last_run.stats
    assert stats['workers'] == workers
    assert stats['batch_size'] == 2
    assert stats['points'] == 10
    assert stats['busy_seconds'] > 0
    assert stats['utilisation'] > 0
    assert stats['wait_seconds'] >= 0
    assert stats['first_result_seconds'] > 0


def test_serial_run_has_no_statistics():
    controller = controller_with(parallel_fit.serial_execution())()
    run_evaluation(lambda: controller, **SCENARIOS['water_1peak'])
    assert controller.last_run.stats is None
