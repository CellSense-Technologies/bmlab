"""
The derived values are calculated for the newly stored points while an
evaluation runs, and for all points once at the end. Both ways give the
same bits, and the work of a run does not grow with the square of the
number of points.
"""
import pathlib

import numpy as np
import pytest

from bmlab import controllers
from bmlab.controllers import calculate_derived_values
from bmlab.session import Session


def evaluation_model(dim_x, nr_peaks=2, nr_regions=2):
    session = Session.get_instance()
    session.set_file(
        pathlib.Path(__file__).parent / 'data' / 'Water.h5')
    session.set_current_repetition('0')
    evm = session.evaluation_model()
    evm.initialize_results_arrays({
        'dim_x': dim_x,
        'dim_y': 1,
        'dim_z': 1,
        'nr_images': 2,
        'nr_brillouin_regions': nr_regions,
        'nr_brillouin_peaks': nr_peaks,
        'nr_rayleigh_regions': nr_regions,
    })
    return evm


def random_point_values(evm, rng, ind_x):
    """Results of one point, with NaN at random places."""
    for key in ('brillouin_peak_position_f', 'rayleigh_peak_position_f'):
        values = rng.normal(5.0, 1.0, evm.results[key][ind_x, 0, 0].shape)
        values[rng.random(values.shape) < 0.2] = np.nan
        evm.results[key][ind_x, 0, 0] = values


def same_bits(a, b):
    return a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize('nr_peaks, nr_regions', [(1, 1), (2, 2), (4, 3)])
def test_incremental_values_equal_full_values_during_the_run(
        nr_peaks, nr_regions):
    # Reference: today's way, all points at every tenth step
    rng = np.random.default_rng(1)
    evm = evaluation_model(300, nr_peaks, nr_regions)
    expected = []
    stored = []
    seen = []
    for idx in range(300):
        random_point_values(evm, rng, idx)
        stored.append((idx, 0, 0))
        if idx % 10:
            continue
        calculate_derived_values()
        expected.append(evm.results['brillouin_shift_f'].copy())
        seen.append(idx)

    rng = np.random.default_rng(1)
    evm = evaluation_model(300, nr_peaks, nr_regions)
    stored = []
    actual = []
    for idx in range(300):
        random_point_values(evm, rng, idx)
        stored.append((idx, 0, 0))
        if idx % 10:
            continue
        calculate_derived_values(stored)
        stored = []
        actual.append(evm.results['brillouin_shift_f'].copy())

    assert len(actual) == len(expected) == 30
    for before, after in zip(expected, actual):
        assert same_bits(before, after)

    # and once at the end for all points
    calculate_derived_values()
    final = evm.results['brillouin_shift_f'].copy()
    assert not np.isnan(final).all()
    assert same_bits(final, evm.results['brillouin_shift_f'])


def test_calculation_for_points_keeps_the_values_of_other_points():
    rng = np.random.default_rng(2)
    evm = evaluation_model(20)
    for idx in range(20):
        random_point_values(evm, rng, idx)
    calculate_derived_values()
    full = evm.results['brillouin_shift_f'].copy()
    evm.results['brillouin_shift_f'][:] = 123.0
    calculate_derived_values([(3, 0, 0), (7, 0, 0)])
    result = evm.results['brillouin_shift_f']
    assert same_bits(result[3], full[3]) and same_bits(result[7], full[7])
    assert (result[[0, 1, 2, 4, 5, 6, 8]] == 123.0).all()
    calculate_derived_values([])
    assert (result[[0, 1, 2, 4, 5, 6, 8]] == 123.0).all()


def test_work_of_the_derived_values_does_not_grow_with_the_square(
        monkeypatch):
    # A scan with 20,000 points: every point is calculated once during
    # the run (tenth step) and the whole scan once at the end.
    nr_points = 20000
    evm = evaluation_model(nr_points)
    evm.results['brillouin_peak_position_f'][:] = np.linspace(
        4, 6, nr_points).reshape(-1, 1, 1, 1, 1, 1)
    evm.results['rayleigh_peak_position_f'][:] = 0.5
    rows = []
    original = controllers._brillouin_shift

    def counting(brillouin, rayleigh):
        rows.append(brillouin.shape[0])
        return original(brillouin, rayleigh)
    monkeypatch.setattr(controllers, '_brillouin_shift', counting)

    stored = []
    for idx in range(nr_points):
        stored.append((idx, 0, 0))
        if not idx % 10:
            calculate_derived_values(stored)
            stored = []
    during_run = sum(rows)
    assert during_run == nr_points - len(stored)

    calculate_derived_values()
    assert rows[-1] == nr_points
    assert sum(rows) <= 2 * nr_points
