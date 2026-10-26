"""
Faults for worker processes, for the tests of the run lifecycle.

The functions are initializers of a spawned pool, see
`bmlab.parallel_fit.create_spawn_pool`. They run in the worker process
when it starts. They must be importable by name from the worker.
"""
import os
import time

from bmlab import parallel_fit


def die_at_start():
    """The worker ends before it can take a job."""
    os._exit(3)


def die_in_job():
    """The worker ends when it gets its first job."""
    def fit_batch(points, settings):
        os._exit(7)
    parallel_fit.fit_batch = fit_batch


def hang_in_job():
    """The worker never finishes its first job."""
    def fit_batch(points, settings):
        while True:
            time.sleep(0.1)
    parallel_fit.fit_batch = fit_batch
