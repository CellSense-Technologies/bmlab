"""
Batched fits of measurement points in worker processes.

The controlling process keeps everything that touches files, the session
or the result arrays. The worker processes only receive plain arrays and a
`PointFitSettings` snapshot and run `bmlab.point_fit.fit_point`.

This module offers:

- `FitExecution`: the (internal) settings of a run: serial or parallel,
  number of workers, batch size and the limit of outstanding jobs.
- `select_execution`: chooses those settings automatically for a job
  (serial for small jobs, worker count that leaves CPU capacity free).
- `create_spawn_pool`: a worker pool created with an explicit spawn
  context, whose workers are limited to one numerical-library thread.
  It notices workers which die, and it always ends with no worker alive.
- `fit_batch`: the function the workers run.
- `OrderedBatchRunner`: sends points to a pool in bounded batches and
  hands the results back in the original order of the points,
  independent of the order in which the workers finish. It stops the run
  with a `WorkerFailure`, if a worker process dies or no job completes
  for a long time.

The module imports neither the session nor the controllers, so a spawned
worker imports it cheaply.
"""
import contextlib
import logging
import multiprocessing
import multiprocessing.pool
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

from bmlab.point_fit import fit_point

logger = logging.getLogger(__name__)

# Environment variables limiting the threads of numerical libraries.
# Workers get them so that N workers do not start N x (number of cores)
# threads. They are only set for starting the workers, the controlling
# process keeps its own environment.
WORKER_THREAD_ENVIRONMENT = {
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'BLIS_NUM_THREADS': '1',
}

_environment_lock = threading.Lock()

# A run fails, if no result is handed back for this long while jobs are
# outstanding (a hung worker). A job takes a fraction of a second. The
# limit is generous, because the first job also has to wait for the
# workers to import their libraries, which is slow on some computers.
JOB_TIMEOUT_SECONDS = 120.0


class WorkerFailure(RuntimeError):
    """
    The worker processes cannot finish the run. The message tells the
    user what happened. The run is stopped and the exception reaches
    the caller of `evaluate` like any other error. The run is not
    repeated in this process: part of its work was done already.
    """


class WorkerDied(WorkerFailure):
    """A worker process ended while the run was not finished."""


class WorkerTimeout(WorkerFailure):
    """No worker returned a result for too long."""


@dataclass(frozen=True)
class FitExecution:
    """
    How the fits of a run are executed. This is an internal setting of
    the evaluation, there is no user setting for it.

    Attributes
    ----------
    parallel: bool
        Fit in worker processes (True) or in the calling process (False)
    workers: int
        Number of worker processes
    batch_size: int
        Number of points per job
    max_outstanding_jobs: int
        Maximum number of jobs which are sent to the workers, but whose
        results have not been handed back in order yet.
    pool_factory: callable or None
        pool_factory(workers) returns a pool with the interface of
        `multiprocessing.pool.Pool` (apply_async, close, join, terminate).
        None creates a spawn pool. It is an internal hook to run the
        fits with another kind of pool.
        A pool may also have a method `check_workers()`, which raises
        `WorkerFailure` if a worker has died. The spawn pool has it.
    job_timeout: float
        Seconds without any result while jobs are outstanding, after
        which the run fails with `WorkerTimeout`.
    """
    parallel: bool = True
    workers: int = 2
    batch_size: int = 8
    max_outstanding_jobs: int = 4
    pool_factory: object = None
    job_timeout: float = JOB_TIMEOUT_SECONDS

    def create_pool(self):
        """
        Start the workers with `pool_factory`, or with a spawn pool if
        there is none. This is the only place where the evaluation
        starts processes. It is not a public interface.
        """
        if self.pool_factory is not None:
            return self.pool_factory(self.workers)
        return create_spawn_pool(self.workers)


# Tuned values.

# All CPUs except `RESERVED_CPUS` are used. Every worker needs memory
# for its own copy of the modules and the data of its jobs.
# Upper limit of worker processes, only for very large computers: before
# Python 3.12 a pool on Windows cannot wait for more than 63 objects, two
# of which it needs itself (see concurrent.futures.process).
MAX_WORKERS = 61
# CPUs not used for fitting. The controlling process and the rest of the
# system (the GUI, other programs) need some capacity.
RESERVED_CPUS = 2
# Small batches balance the load best. One fit is short, so larger
# batches save no time with few workers.
BATCH_SIZE = 2
# With many workers the controlling process is the limit. Every job costs
# it thread wake-ups and GIL hand-overs, so more points go into a job.
# The batch grows with the worker count, up to this
# size. The points held for the workers stay bounded:
# at most MAX_WORKERS x OUTSTANDING_JOBS_PER_WORKER x MAX_BATCH_SIZE.
MANY_WORKERS = 8
MAX_BATCH_SIZE = 8
# Jobs sent but not handed back, per worker.
OUTSTANDING_JOBS_PER_WORKER = 2

# Fit cost model, used to tell small jobs from large ones. The cost of a
# job is estimated in units: one single-peak fit of one region of one
# frame is one unit. Fitting 2 or 4 peaks per region costs about
# 4 or 10 times as much as 1 peak.
SMALL_JOB_UNITS = 700
# Jobs below this many units run in this process: for them, starting
# the workers costs more than the parallel fits save. The threshold has
# a safety margin for slow process starts.


def available_cpus():
    """The number of CPUs this process may use."""
    count_cpus = getattr(os, 'process_cpu_count', None)
    if count_cpus is not None:
        return count_cpus() or 1
    try:
        return len(os.sched_getaffinity(0)) or 1
    except AttributeError:
        return os.cpu_count() or 1


def default_workers(cpus=None):
    """
    Number of worker processes for the given number of CPUs, leaving
    `RESERVED_CPUS` free. Less than 2 means: do not use workers.
    """
    if cpus is None:
        cpus = available_cpus()
    return min(MAX_WORKERS, cpus - RESERVED_CPUS)


def estimate_work(nr_points, nr_frames, nr_regions, nr_peaks):
    """
    Cost of the fits of a job in units (see `SMALL_JOB_UNITS`).

    Parameters
    ----------
    nr_points: int
        Number of measurement points
    nr_frames: int
        Number of spectra (frames) per point
    nr_regions: int
        Number of Brillouin and Rayleigh regions
    nr_peaks: int
        Number of Brillouin peaks fitted per region
    """
    peak_cost = 1 + 3 * (max(1, nr_peaks) - 1)
    return nr_points * nr_frames * nr_regions * peak_cost


def batch_size_for(workers):
    """
    Number of points per job for a number of workers: `BATCH_SIZE` for
    `MANY_WORKERS` workers or fewer, above that half the number of
    workers, from 4 up to `MAX_BATCH_SIZE`.
    """
    if workers <= MANY_WORKERS:
        return BATCH_SIZE
    return max(2 * BATCH_SIZE, min(MAX_BATCH_SIZE, workers // 2))


def default_execution(cpus=None):
    """
    The tuned settings for parallel execution. Use `select_execution` to
    choose between parallel and serial execution for a job.
    """
    workers = max(2, default_workers(cpus))
    return FitExecution(
        parallel=True, workers=workers, batch_size=batch_size_for(workers),
        max_outstanding_jobs=OUTSTANDING_JOBS_PER_WORKER * workers)


def select_execution(nr_points, nr_frames, nr_regions, nr_peaks, cpus=None):
    """
    Choose how to execute the fits of a job: in this process for small
    jobs and on computers with too few CPUs, otherwise in worker
    processes. There is no user setting for this.
    """
    if default_workers(cpus) < 2:
        return serial_execution()
    if estimate_work(
            nr_points, nr_frames, nr_regions, nr_peaks) < SMALL_JOB_UNITS:
        return serial_execution()
    return default_execution(cpus)


def serial_execution():
    return FitExecution(parallel=False)


def _limit_threads():
    os.environ.update(WORKER_THREAD_ENVIRONMENT)


@contextlib.contextmanager
def _worker_environment():
    """
    Set the thread limits in the environment while workers are started.
    Workers inherit the environment before they import numpy,
    so the limits apply from the start. Afterwards the environment
    of the controlling process is restored.
    """
    with _environment_lock:
        previous = {key: os.environ.get(key)
                    for key in WORKER_THREAD_ENVIRONMENT}
        os.environ.update(WORKER_THREAD_ENVIRONMENT)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class SpawnPool(multiprocessing.pool.Pool):
    """
    A pool of spawned worker processes for one run.

    Compared with `multiprocessing.pool.Pool`:
    - `check_workers` reports a dead worker. The plain pool replaces
      a dead worker without telling anybody and never completes the job
      the worker had. A run would wait for it forever.
    - `terminate` and `join` do not wait for a pool which cannot shut
      down, and always end with every worker process killed.
      A worker which died while waiting for a job can leave the pool
      with a lock which is never released. The plain `terminate` then
      blocks.

    This uses the attribute `_pool` (the worker processes) of the base
    class, which is not public.
    """

    # Seconds the pool may take to shut down before it is given up
    # (a healthy pool needs a few milliseconds)
    SHUTDOWN_SECONDS = 2.0
    # Seconds to wait for a killed worker to end
    KILL_SECONDS = 10.0

    def __init__(self, workers, context, initializer=None, initargs=()):
        super().__init__(
            processes=workers,
            initializer=initializer or _limit_threads,
            initargs=initargs, context=context)
        # The workers of the first generation. A replacement is only
        # started after one of them has ended.
        self._first_workers = list(self._pool)
        self._wedged = False

    def check_workers(self):
        """Raise `WorkerDied` if a worker process has ended."""
        for worker in self._first_workers:
            exitcode = worker.exitcode
            if exitcode is not None:
                raise WorkerDied(
                    'A worker process of the evaluation ended '
                    'unexpectedly (process {}, exit code {}). '
                    'The evaluation was stopped. '
                    'Start it again.'.format(worker.pid, exitcode))

    def terminate(self):
        stopper = threading.Thread(
            target=multiprocessing.pool.Pool.terminate, args=(self,),
            daemon=True)
        stopper.start()
        stopper.join(self.SHUTDOWN_SECONDS)
        if stopper.is_alive():
            self._wedged = True
            logger.warning(
                'The worker pool did not shut down. '
                'Its worker processes are killed.')
        self._kill_workers()

    def join(self):
        if self._wedged:
            # The helper threads of the pool may never end,
            # the processes are all there is to wait for.
            return
        super().join()

    def _kill_workers(self):
        workers = {id(worker): worker
                   for worker in self._first_workers + list(self._pool)}
        for worker in workers.values():
            try:
                if worker.exitcode is None:
                    worker.kill()
            except (OSError, ValueError):
                pass
        for worker in workers.values():
            try:
                worker.join(self.KILL_SECONDS)
            except (OSError, ValueError, AssertionError):
                pass


def create_spawn_pool(workers, initializer=None, initargs=()):
    """
    Create a pool of worker processes started with the spawn method.
    All workers are started before this function returns.
    `initializer` (default: limit the numerical threads) is for tests.
    """
    context = multiprocessing.get_context('spawn')
    with _worker_environment():
        return SpawnPool(workers, context, initializer, initargs)


class BatchResults(list):
    """
    The results of a batch, a list. `busy_seconds` is the time the
    worker needed for the batch (for the statistics of the run).
    """
    busy_seconds = 0.0


def fit_batch(points, settings):
    """
    Fit a batch of points. This runs in a worker process.

    Parameters
    ----------
    points: list of (spectra, frequencies)
        The inputs of `fit_point` for every point of the batch
    settings: PointFitSettings

    Returns
    -------
    results: BatchResults (a list of dict)
        The results of `fit_point`, one per point, in the order of the
        points
    """
    start = time.perf_counter()
    results = BatchResults(
        fit_point(spectra, frequencies, settings)
        for spectra, frequencies in points)
    results.busy_seconds = time.perf_counter() - start
    return results


class OrderedBatchRunner(object):
    """
    Fit points in batches in a pool, hand the results back in order.

    Points are collected into batches of `batch_size`. A full batch
    is sent as one job to the pool. At most `max_outstanding_jobs` jobs
    are sent but not yet handed back. When the limit is reached, the
    runner stops taking new points until the oldest job is complete,
    so the number of points held for the workers is bounded,
    independent of the size of the scan. Results are handed back in the
    order of the points, so everything derived from them is independent
    of the order in which the workers finish.

    Points without input (e.g. missing images) are handed back in their
    place with the result None.

    `should_abort` is polled in the producer loop and while waiting for
    results. It may also raise an exception, which stops the run.
    While waiting, the runner asks the pool whether a worker has died
    (`pool.check_workers()`, if the pool has it), and fails with
    `WorkerTimeout` if no result is handed back for `job_timeout`
    seconds although jobs are outstanding.
    """

    POLL_SECONDS = 0.05
    # Minimum time between two checks of the workers
    HEALTH_SECONDS = 0.2

    def __init__(self, pool, settings, batch_size, max_outstanding_jobs,
                 should_abort=None, job_timeout=None,
                 clock=time.monotonic):
        self.pool = pool
        self.settings = settings
        self.batch_size = max(1, int(batch_size))
        self.max_outstanding_jobs = max(1, int(max_outstanding_jobs))
        self.should_abort = should_abort
        self.job_timeout = job_timeout
        self._clock = clock
        self._progress_at = clock()
        self._health_checked_at = None
        self.aborted = False
        # Highest numbers of jobs and points held for the workers
        # (sent or in the open batch, not yet handed back)
        self.peak_outstanding_jobs = 0
        self.peak_outstanding_points = 0
        # Statistics (see `statistics`). They use their own clock,
        # because the resolution of time.monotonic is 16 ms on Windows.
        self._stats_clock = time.perf_counter
        self.wait_seconds = 0.0
        self.busy_seconds = 0.0
        self.jobs_sent = 0
        self.points_sent = 0
        self._first_sent_at = None
        self._first_completion_at = None
        self._last_completion_at = None

        self._next_job = 0
        # Points in the order they were taken: [payload, job, index in job]
        self._order = deque()
        # Open batch, not sent yet
        self._open_inputs = []
        # Sent jobs not handed back completely: job -> points left
        self._left = {}
        self._done = {}
        self._points_held = 0
        self._completed = queue.Queue()

    def run(self, items):
        """
        Parameters
        ----------
        items: iterable of (payload, point_input)
            point_input is (spectra, frequencies) or None.
            The payload is handed back with the result.

        Yields
        ------
        (payload, result): result is the dict of `fit_point`, or None
            for points without input. Stops early, if abort is requested
            (see `aborted`).
        """
        iterator = iter(items)
        while True:
            if self._abort_requested():
                return
            self._check_workers()
            try:
                payload, point_input = next(iterator)
            except StopIteration:
                break
            if point_input is None:
                self._order.append([payload, None, 0])
            else:
                self._order.append(
                    [payload, self._next_job, len(self._open_inputs)])
                self._open_inputs.append(point_input)
                self._points_held += 1
                self._update_peaks()
                if len(self._open_inputs) >= self.batch_size:
                    yield from self._send_open_batch()
                    if self.aborted:
                        return
            yield from self._hand_back_ready()

        yield from self._send_open_batch()
        while self._order and not self.aborted:
            yield from self._hand_back_ready()
            if self._order and not self._wait_for_completion():
                return

    def statistics(self, workers, pool_started_at=None):
        """
        Statistics of the run, cheap to collect. Call after `run`.

        workers: number of worker processes of the pool
        pool_started_at: `time.perf_counter()` before the pool was
            created, for `first_result_seconds`

        Returns a dictionary:
        workers, batch_size, jobs, points
        wait_seconds: time this process waited for results, because
            nothing was ready to hand back and no job slot was free
            (the time it had nothing else to do)
        busy_seconds: time the workers spent fitting, all workers added
        span_seconds: time from sending the first job to receiving the
            last result
        utilisation: busy_seconds / (workers x span_seconds). Near 1,
            the workers are the limit. Low with a small wait time means
            this process cannot feed the workers fast enough.
        first_result_seconds: time from pool_started_at to the first
            result (None without pool_started_at)
        """
        span = 0.0
        if self._first_sent_at is not None \
                and self._last_completion_at is not None:
            span = max(0.0, self._last_completion_at - self._first_sent_at)
        utilisation = 0.0
        if span > 0 and workers:
            utilisation = self.busy_seconds / (workers * span)
        first_result = None
        if pool_started_at is not None \
                and self._first_completion_at is not None:
            first_result = self._first_completion_at - pool_started_at
        return {
            'workers': workers,
            'batch_size': self.batch_size,
            'jobs': self.jobs_sent,
            'points': self.points_sent,
            'wait_seconds': self.wait_seconds,
            'busy_seconds': self.busy_seconds,
            'span_seconds': span,
            'utilisation': utilisation,
            'first_result_seconds': first_result,
        }

    # Internals ---------------------------------------------------------

    def _abort_requested(self):
        if self.should_abort is not None and self.should_abort():
            self.aborted = True
        return self.aborted

    def _update_peaks(self):
        self.peak_outstanding_jobs = max(
            self.peak_outstanding_jobs, len(self._left))
        self.peak_outstanding_points = max(
            self.peak_outstanding_points, self._points_held)

    def _send_open_batch(self):
        """Send the open batch once a job slot is free (generator)."""
        if not self._open_inputs:
            return
        # Handing back results frees job slots
        while len(self._left) >= self.max_outstanding_jobs:
            yield from self._hand_back_ready()
            if len(self._left) >= self.max_outstanding_jobs \
                    and not self._wait_for_completion():
                return

        if not self._left:
            # The workers had nothing to do, the wait starts now
            self._progress_at = self._clock()
        job = self._next_job
        self._next_job += 1
        inputs = self._open_inputs
        self._open_inputs = []
        self._left[job] = len(inputs)
        self._update_peaks()
        self.jobs_sent += 1
        self.points_sent += len(inputs)
        if self._first_sent_at is None:
            self._first_sent_at = self._stats_clock()
        self.pool.apply_async(
            fit_batch, (inputs, self.settings),
            callback=lambda results, job=job:
            self._completed.put((job, True, results)),
            error_callback=lambda error, job=job:
            self._completed.put((job, False, error)))

    def _wait_for_completion(self):
        """
        Wait for at least one job to complete. Returns False, if an abort
        was requested while waiting.
        """
        start = self._stats_clock()
        try:
            return self._wait_for_completion_loop()
        finally:
            self.wait_seconds += self._stats_clock() - start

    def _wait_for_completion_loop(self):
        while True:
            if self._collect():
                return True
            try:
                self._store_completion(
                    self._completed.get(timeout=self.POLL_SECONDS))
                return True
            except queue.Empty:
                pass
            if self._abort_requested():
                return False
            self._check_workers()
            self._check_stalled()

    def _check_workers(self):
        """Raise `WorkerFailure` if a worker of the pool has died."""
        check = getattr(self.pool, 'check_workers', None)
        if check is None:
            return
        now = self._clock()
        if self._health_checked_at is not None \
                and now - self._health_checked_at < self.HEALTH_SECONDS:
            return
        self._health_checked_at = now
        check()

    def _check_stalled(self):
        """
        Raise `WorkerTimeout` if jobs are outstanding, and no result was
        handed back for `job_timeout` seconds.
        """
        if not self.job_timeout or not self._left:
            return
        waited = self._clock() - self._progress_at
        if waited > self.job_timeout:
            raise WorkerTimeout(
                'The worker processes of the evaluation returned no '
                'result for {:.0f} seconds. The evaluation was '
                'stopped. Start it again.'.format(waited))

    def _collect(self):
        """Take over all completed jobs, without waiting."""
        found = False
        while True:
            try:
                self._store_completion(self._completed.get_nowait())
            except queue.Empty:
                return found
            found = True

    def _store_completion(self, completion):
        job, success, value = completion
        if not success:
            raise value
        self._done[job] = value
        self.busy_seconds += getattr(value, 'busy_seconds', 0.0)
        self._last_completion_at = self._stats_clock()
        if self._first_completion_at is None:
            self._first_completion_at = self._last_completion_at

    def _hand_back_ready(self):
        """Hand back all points at the front, which are complete."""
        self._collect()
        while self._order:
            payload, job, index = self._order[0]
            if job is None:
                result = None
            elif job in self._done:
                result = self._done[job][index]
            else:
                return
            self._order.popleft()
            if job is not None:
                self._progress_at = self._clock()
                self._points_held -= 1
                self._left[job] -= 1
                if not self._left[job]:
                    del self._left[job]
                    del self._done[job]
            yield payload, result
