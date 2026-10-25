"""
Automatic choice of resources: worker count, small jobs in this process,
and the fallback to serial evaluation when workers cannot be started.
"""
import logging
import multiprocessing
import os
from multiprocessing.pool import ThreadPool
from unittest import mock

import pytest

from bmlab import controllers, parallel_fit
from bmlab.controllers import EvaluationController
from bmlab.parallel_fit import FitExecution

from evaluation_parity import RAISING_SCENARIOS, SCENARIOS, \
    assert_outcomes_identical, run_evaluation


def controller_with(execution):
    class Controller(EvaluationController):
        _fit_execution = execution
    return Controller


SERIAL_CONTROLLER = controller_with(parallel_fit.serial_execution())


def thread_pool(workers):
    return ThreadPool(workers)


@pytest.fixture
def automatic_execution(monkeypatch):
    """
    The evaluation chooses its execution itself, like in the application,
    and treats every job as large, so that the tiny test fixtures use
    the parallel path.
    """
    monkeypatch.setattr(EvaluationController, '_fit_execution', None)
    monkeypatch.setattr(parallel_fit, 'SMALL_JOB_UNITS', 0)
    # The test machine may have too few CPUs for workers
    monkeypatch.setattr(parallel_fit, 'available_cpus', lambda: 12)


def compare_with_serial(scenario_id, controller_class=EvaluationController):
    kwargs = SCENARIOS[scenario_id]
    expected = run_evaluation(SERIAL_CONTROLLER, **kwargs)
    actual = run_evaluation(controller_class, **kwargs)
    if scenario_id in RAISING_SCENARIOS:
        assert expected['error_type'] is not None
        assert actual['error_type'] == expected['error_type']
        assert actual['max_count'] == expected['max_count']
        return actual
    assert expected['error_type'] is None
    assert_outcomes_identical(actual, expected)
    return actual


# Worker count -------------------------------------------------------------

@pytest.mark.parametrize('cpus', range(1, 129))
def test_workers_leave_cpu_capacity_free(cpus):
    execution = parallel_fit.select_execution(10 ** 6, 1, 4, 1, cpus=cpus)
    if not execution.parallel:
        # Too few CPUs to gain anything
        assert cpus < 4
        return
    assert 2 <= execution.workers <= cpus - 2
    assert execution.workers <= parallel_fit.MAX_WORKERS


def test_worker_count_grows_with_cpus_up_to_a_limit():
    workers = [parallel_fit.select_execution(
        10 ** 6, 1, 4, 1, cpus=cpus).workers for cpus in range(4, 129)]
    assert workers == sorted(workers)
    assert workers[0] == 2
    assert workers[-1] == parallel_fit.MAX_WORKERS


@pytest.mark.parametrize('cpus, expected', [
    (4, 2), (8, 6), (12, 10), (18, 16), (24, 22), (32, 30), (63, 61),
    (64, 61), (128, 61)])
def test_worker_count_for_typical_computers(cpus, expected):
    execution = parallel_fit.select_execution(10 ** 6, 1, 4, 1, cpus=cpus)
    assert execution.workers == expected


def test_available_cpus_is_at_least_one():
    assert parallel_fit.available_cpus() >= 1


def test_available_cpus_respects_the_cpu_affinity():
    with mock.patch.object(os, 'process_cpu_count', create=True,
                           return_value=3):
        assert parallel_fit.available_cpus() == 3


# Small jobs -----------------------------------------------------------------

def test_estimated_work_grows_with_every_dimension_of_the_job():
    base = parallel_fit.estimate_work(100, 2, 4, 1)
    assert parallel_fit.estimate_work(200, 2, 4, 1) > base
    assert parallel_fit.estimate_work(100, 3, 4, 1) > base
    assert parallel_fit.estimate_work(100, 2, 5, 1) > base
    assert parallel_fit.estimate_work(100, 2, 4, 2) > base
    assert parallel_fit.estimate_work(100, 2, 4, 4) > \
        parallel_fit.estimate_work(100, 2, 4, 2)


def test_small_jobs_are_serial_and_large_jobs_parallel():
    def select(points, peaks=1):
        return parallel_fit.select_execution(points, 1, 4, peaks, cpus=12)

    assert not select(1).parallel
    assert not select(50).parallel
    assert select(6767).parallel
    # The decision flips once, from serial to parallel
    decisions = [select(points).parallel for points in range(1, 2000)]
    assert decisions == sorted(decisions)
    assert not decisions[0] and decisions[-1]
    # Multi-peak fits are more expensive, so the same scan is
    # large earlier
    first = {peaks: next(points for points in range(1, 5000)
                         if select(points, peaks).parallel)
             for peaks in (1, 2, 4)}
    assert first[4] < first[2] < first[1]


def test_threshold_matches_the_measured_break_even():
    """
    On the organoid scan (4 regions, 1 frame) the parallel run is not
    slower than the serial run from about 100 points. The threshold lies
    above that and stays below 300 points.
    """
    first = next(points for points in range(1, 5000)
                 if parallel_fit.select_execution(
                     points, 1, 4, 1, cpus=12).parallel)
    assert 100 <= first <= 300


def test_small_job_runs_without_starting_workers(monkeypatch):
    monkeypatch.setattr(EvaluationController, '_fit_execution', None)
    monkeypatch.setattr(parallel_fit, 'available_cpus', lambda: 12)
    with mock.patch.object(
            parallel_fit, 'create_spawn_pool',
            side_effect=thread_pool) as create:
        for scenario_id in ('water_1peak', 'water_2peak_ghz_bounds',
                            '3D_1peak'):
            compare_with_serial(scenario_id)
    create.assert_not_called()


def test_large_job_starts_workers(automatic_execution):
    with mock.patch.object(
            parallel_fit, 'create_spawn_pool',
            side_effect=thread_pool) as create:
        compare_with_serial('water_1peak')
    create.assert_called_once()
    assert create.call_args.args == (
        parallel_fit.select_execution(1, 1, 1, 1, cpus=12).workers,)


def test_computer_with_too_few_cpus_runs_serially(monkeypatch,
                                                  automatic_execution):
    monkeypatch.setattr(parallel_fit, 'available_cpus', lambda: 2)
    with mock.patch.object(
            parallel_fit, 'create_spawn_pool',
            side_effect=thread_pool) as create:
        compare_with_serial('water_1peak')
    create.assert_not_called()


# Start-up failure -----------------------------------------------------------

START_FAILURES = [
    OSError('cannot create process'),
    PermissionError('not allowed'),
    RuntimeError('context already used'),
    ImportError('cannot import worker module'),
]


def failing_factory(error):
    def factory(workers):
        raise error
    return factory


FALLBACK_SCENARIOS = [
    'water_1peak',
    'water_2peak_ghz_bounds',
    'water_4peak_nan_spectra',
    'water_1peak_missing_images',
    'water_2peak_first_points_invalid',
    'water_1peak_abort_midway',
    'water_1peak_abort_immediately',
    'water_2peak_no_width_bounds',
]


@pytest.mark.parametrize('scenario_id', FALLBACK_SCENARIOS)
def test_start_failure_gives_exactly_the_serial_results(scenario_id,
                                                        caplog):
    controller = controller_with(FitExecution(
        parallel=True, workers=3, batch_size=2, max_outstanding_jobs=4,
        pool_factory=failing_factory(OSError('cannot create process'))))
    with caplog.at_level(logging.WARNING, logger='bmlab'):
        compare_with_serial(scenario_id, controller)
    # Everything, including the abort polls and the progress counter,
    # equals the serial run
    messages = [r for r in caplog.records
                if 'Could not start worker processes' in r.getMessage()]
    assert len(messages) == 1


@pytest.mark.parametrize('error', START_FAILURES,
                         ids=lambda error: type(error).__name__)
def test_start_failure_is_logged_with_its_cause(error, caplog):
    controller = controller_with(FitExecution(
        parallel=True, workers=2, pool_factory=failing_factory(error)))
    with caplog.at_level(logging.WARNING, logger='bmlab'):
        outcome = run_evaluation(controller, **SCENARIOS['water_1peak'])
    assert outcome['error_type'] is None
    assert outcome['count'] == outcome['max_count'] == 10
    records = [r for r in caplog.records
               if 'Could not start worker processes' in r.getMessage()]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.WARNING
    assert record.name == 'bmlab.controllers'
    assert type(error).__name__ in record.getMessage()
    assert str(error) in record.getMessage()
    # The traceback is attached to the log record
    assert record.exc_info[1] is error


def test_start_failure_in_the_default_execution_falls_back(
        automatic_execution, caplog):
    """A start failure also falls back with the automatic execution."""
    with mock.patch.object(
            parallel_fit, 'create_spawn_pool',
            side_effect=OSError('cannot create process')) as create:
        with caplog.at_level(logging.WARNING, logger='bmlab'):
            compare_with_serial('water_2peak_ghz_bounds')
    create.assert_called_once()
    assert 'Could not start worker processes' in caplog.text


def test_start_failure_of_the_process_pool_falls_back(
        monkeypatch, caplog):
    """The real pool creation fails, e.g. the system refuses a process."""
    def refuse(process):
        raise OSError('Resource temporarily unavailable')

    monkeypatch.setattr(
        multiprocessing.context.SpawnProcess, 'start', refuse)
    environment = dict(os.environ)
    controller = controller_with(FitExecution(parallel=True, workers=2))
    with caplog.at_level(logging.WARNING, logger='bmlab'):
        compare_with_serial('water_1peak', controller)
    assert 'Resource temporarily unavailable' in caplog.text
    # The thread limits for the workers are not left in this process
    assert dict(os.environ) == environment


def test_nothing_is_submitted_when_workers_do_not_start():
    controller = controller_with(FitExecution(
        parallel=True, workers=2,
        pool_factory=failing_factory(OSError('no process'))))
    with mock.patch.object(
            parallel_fit, 'OrderedBatchRunner') as runner:
        outcome = run_evaluation(controller, **SCENARIOS['water_1peak'])
    assert outcome['error_type'] is None
    runner.assert_not_called()


def test_start_failure_leaves_no_worker_process():
    controller = controller_with(FitExecution(
        parallel=True, workers=2,
        pool_factory=failing_factory(OSError('no process'))))
    run_evaluation(controller, **SCENARIOS['water_1peak'])
    assert multiprocessing.active_children() == []


# Failure after submission ---------------------------------------------------

class NoSerialRetry(object):
    """
    Context manager: fails the test if the controlling process fits a
    point itself or starts the serial loop, which a whole-run retry
    would do.
    """

    def __enter__(self):
        def fail(*args, **kwargs):
            raise AssertionError('whole-run serial retry')
        self.patches = [
            mock.patch.object(controllers, 'fit_point', fail),
            mock.patch.object(
                EvaluationController, '_fit_points_serial', fail)]
        for patch in self.patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in self.patches:
            patch.stop()


class FailingSubmissionPool(object):
    """
    Pool, which accepts a number of jobs and then fails on submitting
    the next one.
    """

    def __init__(self, workers, accepted_jobs, error):
        self.pool = ThreadPool(workers)
        self.accepted_jobs = accepted_jobs
        self.error = error
        self.jobs = 0
        self.calls = []

    def apply_async(self, function, args, callback=None,
                    error_callback=None):
        if self.jobs >= self.accepted_jobs:
            raise self.error
        self.jobs += 1
        return self.pool.apply_async(
            function, args, callback=callback,
            error_callback=error_callback)

    def close(self):
        self.calls.append('close')
        self.pool.close()

    def terminate(self):
        self.calls.append('terminate')
        self.pool.terminate()

    def join(self):
        self.calls.append('join')
        self.pool.join()


@pytest.mark.parametrize('accepted_jobs', [1, 2, 4])
def test_failure_after_submission_reaches_the_caller_without_retry(
        accepted_jobs, caplog):
    pools = []

    def factory(workers):
        pools.append(FailingSubmissionPool(
            workers, accepted_jobs, RuntimeError('worker connection lost')))
        return pools[-1]

    controller = controller_with(FitExecution(
        parallel=True, workers=2, batch_size=2, max_outstanding_jobs=2,
        pool_factory=factory))
    with NoSerialRetry(), caplog.at_level(logging.WARNING, logger='bmlab'):
        outcome = run_evaluation(controller, **SCENARIOS['water_1peak'])

    assert outcome['error_type'] is RuntimeError
    assert pools[0].jobs == accepted_jobs
    # The run was not restarted, and nothing was logged as a fallback
    assert 'Could not start worker processes' not in caplog.text
    assert len(pools) == 1
    # Work in flight is dropped, nothing is left behind
    assert pools[0].calls == ['terminate', 'join']
    # The points before the failure were stored once, then the run stopped
    assert outcome['count'] < 10


def test_error_in_a_worker_reaches_the_caller_without_retry():
    controller = controller_with(FitExecution(
        parallel=True, workers=2, batch_size=2, max_outstanding_jobs=2,
        pool_factory=thread_pool))
    expected = run_evaluation(
        SERIAL_CONTROLLER, **SCENARIOS['water_2peak_no_width_bounds'])
    with NoSerialRetry():
        actual = run_evaluation(
            controller, **SCENARIOS['water_2peak_no_width_bounds'])
    assert expected['error_type'] is not None
    assert actual['error_type'] == expected['error_type']


def test_error_in_a_worker_process_reaches_the_caller_without_retry():
    controller = controller_with(FitExecution(
        parallel=True, workers=2, batch_size=2, max_outstanding_jobs=2))
    expected = run_evaluation(
        SERIAL_CONTROLLER, **SCENARIOS['water_2peak_no_width_bounds'])
    with NoSerialRetry():
        actual = run_evaluation(
            controller, **SCENARIOS['water_2peak_no_width_bounds'])
    assert actual['error_type'] == expected['error_type']
    assert multiprocessing.active_children() == []


def test_failure_when_storing_a_result_reaches_the_caller_without_retry():
    """An error in the controlling process after submission is not retried."""
    controller = controller_with(FitExecution(
        parallel=True, workers=2, batch_size=2, max_outstanding_jobs=2,
        pool_factory=thread_pool))
    scenario = SCENARIOS['water_1peak_more_brillouin_than_rayleigh_regions']
    expected = run_evaluation(SERIAL_CONTROLLER, **scenario)
    with NoSerialRetry():
        actual = run_evaluation(controller, **scenario)
    assert expected['error_type'] is IndexError
    assert actual['error_type'] is IndexError
