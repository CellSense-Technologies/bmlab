"""
The workers start before the points are prepared. The time and exposure
of an image are read when its point is prepared, not for all images at
the start, so the first jobs go out early also on large scans. A run
which does not start leaves no worker process, and a failed start still
gives the serial results.
"""
import multiprocessing
from multiprocessing.pool import ThreadPool
from unittest import mock

import pytest

from bmlab import parallel_fit
from bmlab.controllers import EvaluationController, EvaluationSnapshot
from bmlab.parallel_fit import FitExecution

from evaluation_parity import SCENARIOS, run_evaluation


class RecordingPool(ThreadPool):
    """A pool in threads, which records what happens to it."""

    def __init__(self, workers, events):
        super().__init__(workers)
        self.events = events
        self.jobs = 0

    def apply_async(self, *args, **kwargs):
        self.jobs += 1
        self.events.append('submit')
        return super().apply_async(*args, **kwargs)

    def terminate(self):
        self.events.append('terminate')
        super().terminate()

    def join(self):
        self.events.append('join')
        super().join()


def controller_with(events, pools=None, factory=None):
    def pool_factory(workers):
        events.append('create_pool')
        pool = RecordingPool(workers, events)
        if pools is not None:
            pools.append(pool)
        return pool

    class Controller(EvaluationController):
        _fit_execution = FitExecution(
            parallel=True, workers=2, batch_size=2, max_outstanding_jobs=4,
            pool_factory=factory or pool_factory)
    return Controller


@pytest.fixture
def events(monkeypatch):
    """
    Records the pool events and every read of an image time from the
    file ('time <key>'), not the reads served from the snapshot.
    """
    events = []
    original = EvaluationSnapshot.get_time

    def get_time(snapshot, image_key):
        if image_key not in snapshot._times:
            events.append('time ' + image_key)
        return original(snapshot, image_key)

    monkeypatch.setattr(EvaluationSnapshot, 'get_time', get_time)
    return events


def pool_events(events):
    return [event for event in events if not event.startswith('time ')]


def test_workers_start_before_the_points_are_prepared(events):
    outcome = run_evaluation(
        controller_with(events), **SCENARIOS['water_1peak'])
    assert outcome['error_type'] is None
    # Only the first image is read to choose the execution
    assert events[:2] == ['time 0', 'create_pool']
    assert 'submit' in events


def test_times_are_read_on_demand_and_once(events):
    outcome = run_evaluation(
        controller_with(events), **SCENARIOS['water_1peak'])
    assert outcome['error_type'] is None
    reads = [event for event in events if event.startswith('time ')]
    # Each image time is read once for the whole run
    assert len(reads) == len(set(reads)) == outcome['max_count']
    # The first job goes out before the time of the last image is read
    assert events.index('submit') < events.index(reads[-1])


def test_failed_metadata_read_ends_the_workers(events, monkeypatch):
    pools = []
    original = EvaluationSnapshot.get_time

    def get_time(snapshot, image_key):
        if image_key == '5':
            raise OSError('metadata cannot be read')
        return original(snapshot, image_key)

    monkeypatch.setattr(EvaluationSnapshot, 'get_time', get_time)
    outcome = run_evaluation(
        controller_with(events, pools), **SCENARIOS['water_1peak'])
    assert outcome['error_type'] is OSError
    assert len(pools) == 1
    assert pool_events(events)[0] == 'create_pool'
    assert pool_events(events)[-2:] == ['terminate', 'join']


@pytest.mark.parametrize('scenario_id', ['water_1peak_missing_first_image'])
def test_run_which_does_not_start_starts_no_workers(events, scenario_id):
    pools = []
    outcome = run_evaluation(
        controller_with(events, pools), **SCENARIOS[scenario_id])
    assert outcome['max_count'] == -1
    assert pools == []
    assert pool_events(events) == []
    assert multiprocessing.active_children() == []


def test_run_without_a_file_starts_no_workers(events):
    from bmlab.session import Session
    Session.get_instance().clear()
    pools = []
    controller = controller_with(events, pools)()
    controller.evaluate()
    assert controller.last_run.outcome == 'not-started'
    assert pools == [] and pool_events(events) == []


def test_failed_start_runs_serially(events):
    def refuse(workers):
        events.append('create_pool')
        raise OSError('cannot create process')

    controller = controller_with(events, factory=refuse)
    with mock.patch.object(parallel_fit, 'OrderedBatchRunner') as runner:
        outcome = run_evaluation(controller, **SCENARIOS['water_1peak'])
    assert outcome['error_type'] is None
    assert outcome['count'] == outcome['max_count'] == 10
    runner.assert_not_called()
    assert pool_events(events) == ['create_pool']
