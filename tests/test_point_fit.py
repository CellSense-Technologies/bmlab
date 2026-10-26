import copy
import multiprocessing
import pickle

import numpy as np
import pytest

from bmlab.controllers import EvaluationController
from bmlab.point_fit import PointFitSettings, fit_point

from evaluation_parity import BOUNDS_2_DEFAULT, BOUNDS_2_GHZ, \
    BOUNDS_4_GHZ, BRILLOUIN_REGIONS, FWHM_2_GHZ, FWHM_4_GHZ, FWHM_OPEN, \
    RAISING_SCENARIOS, RAYLEIGH_REGIONS, SCENARIOS, \
    assert_arrays_identical, assert_outcomes_identical, \
    prepare_session, run_evaluation
from legacy_evaluation import LegacyEvaluationController


@pytest.mark.parametrize('scenario_id', list(SCENARIOS.keys()))
def test_evaluate_matches_pre_change_behaviour(scenario_id):
    """
    The serial evaluation gives exactly the same outputs as the
    implementation before the point-fit function was extracted.
    Both run on fresh, equivalent sessions.
    """
    kwargs = SCENARIOS[scenario_id]
    expected = run_evaluation(LegacyEvaluationController, **kwargs)
    actual = run_evaluation(EvaluationController, **kwargs)

    if scenario_id in RAISING_SCENARIOS:
        # The failing point is not written partially any more,
        # but the evaluation fails with the same error.
        assert expected['error_type'] is not None
        assert actual['error_type'] == expected['error_type']
        assert actual['max_count'] == expected['max_count']
        return

    assert expected['error_type'] is None
    assert_outcomes_identical(actual, expected)


def test_scenarios_cover_peak_modes_and_special_results():
    """
    Guard against scenarios silently turning trivial: the compared
    outputs contain valid fits, invalid fits and multi-peak results.
    """
    def run(scenario_id):
        return run_evaluation(EvaluationController, **SCENARIOS[scenario_id])

    for scenario_id, nr_peaks_stored in (('water_1peak', 1),
                                         ('water_2peak_ghz_bounds', 3),
                                         ('water_4peak_ghz_bounds', 5)):
        position = run(scenario_id)['results']['brillouin_peak_position_f']
        assert position.shape[-1] == nr_peaks_stored
        assert np.isfinite(position[..., 0]).all()
        if nr_peaks_stored > 1:
            assert np.isfinite(position[..., 1:]).any()

    invalid = run('water_1peak_regions_outside_axis')['results']
    invalid_position = invalid['brillouin_peak_position_f']
    # The region outside of the frequency axis is sorted last
    assert np.isfinite(invalid_position[..., 0, :]).all()
    assert np.isnan(invalid_position[..., 1, :]).all()

    incomplete = run('water_2peak_incomplete_scan')['results']
    assert np.isnan(incomplete['rayleigh_peak_position_f'][6:]).all()
    assert np.isfinite(incomplete['rayleigh_peak_position_f'][:6]).all()


def _point_inputs(session, image_key):
    """Spectra and frequencies of one point as given to fit_point."""
    spectra, times, _ = EvaluationController().extract_spectra(image_key)
    frequencies = list(
        session.calibration_model().get_frequencies_by_time(times))
    return spectra, frequencies


def _legacy_point_results(session, image_key, controller_class):
    """Results of a single point after running the evaluation."""
    controller_class().evaluate()
    resolution = session.get_payload_resolution()
    ind = EvaluationController.get_indices_from_key(resolution, image_key)
    evm = session.evaluation_model()
    return {key: evm.results[key][ind] for key in (
        'brillouin_peak_position_f', 'brillouin_peak_fwhm_f',
        'brillouin_peak_intensity', 'brillouin_peak_offset',
        'rayleigh_peak_position_f', 'rayleigh_peak_fwhm_f',
        'rayleigh_peak_intensity', 'rayleigh_peak_offset')}


POINT_MODES = {
    'one_peak': dict(nr_brillouin_peaks=1),
    'two_peaks': dict(nr_brillouin_peaks=2, bounds_w0=BOUNDS_2_GHZ,
                      bounds_fwhm=FWHM_2_GHZ),
    'four_peaks': dict(nr_brillouin_peaks=4, bounds_w0=BOUNDS_4_GHZ,
                       bounds_fwhm=FWHM_4_GHZ),
}


def _settings_of(session, nr_brillouin_peaks, bounds_w0=None,
                 bounds_fwhm=None):
    return PointFitSettings.snapshot(
        nr_brillouin_peaks,
        session.peak_selection_model().get_brillouin_regions(),
        session.peak_selection_model().get_rayleigh_regions(),
        bounds_w0, bounds_fwhm)


@pytest.mark.parametrize('mode', list(POINT_MODES.keys()))
def test_fit_point_matches_pre_change_results(mode):
    kwargs = POINT_MODES[mode]
    session = prepare_session('Water.h5', **kwargs)
    spectra, frequencies = _point_inputs(session, '3')
    settings = _settings_of(session, **kwargs)

    results = fit_point(spectra, frequencies, settings)

    session = prepare_session('Water.h5', **kwargs)
    expected = _legacy_point_results(
        session, '3', LegacyEvaluationController)
    assert results.keys() == expected.keys()
    for key, value in expected.items():
        assert_arrays_identical(results[key], value, key)


@pytest.mark.parametrize('mode', list(POINT_MODES.keys()))
def test_fit_point_inputs_survive_pickling(mode):
    kwargs = POINT_MODES[mode]
    session = prepare_session('Water.h5', **kwargs)
    spectra, frequencies = _point_inputs(session, '5')
    settings = _settings_of(session, **kwargs)

    expected = fit_point(spectra, frequencies, settings)

    spectra2, frequencies2, settings2 = pickle.loads(
        pickle.dumps((spectra, frequencies, settings)))
    assert settings2 == settings
    results = fit_point(spectra2, frequencies2, settings2)

    assert results.keys() == expected.keys()
    for key, value in expected.items():
        assert_arrays_identical(results[key], value, key)


def _fit_in_worker(arguments):
    return fit_point(*arguments)


def test_fit_point_runs_in_spawned_process():
    kwargs = POINT_MODES['two_peaks']
    session = prepare_session('Water.h5', **kwargs)
    spectra, frequencies = _point_inputs(session, '2')
    settings = _settings_of(session, **kwargs)
    expected = fit_point(spectra, frequencies, settings)

    context = multiprocessing.get_context('spawn')
    with context.Pool(1) as pool:
        results = pool.map(_fit_in_worker,
                           [(spectra, frequencies, settings)])[0]

    assert results.keys() == expected.keys()
    for key, value in expected.items():
        assert_arrays_identical(results[key], value, key)


def test_fit_point_result_layout():
    kwargs = POINT_MODES['four_peaks']
    session = prepare_session('Water.h5', **kwargs)
    spectra, frequencies = _point_inputs(session, '1')

    results = fit_point(spectra, frequencies,
                        _settings_of(session, **kwargs))

    nr_frames = len(spectra)
    for key, value in results.items():
        assert value.dtype == np.float64
        if key.startswith('brillouin'):
            assert value.shape == (nr_frames, 2, 5)
        else:
            assert value.shape == (nr_frames, 2, 1)


def test_fit_point_invalid_fits_are_nan():
    spectra = [np.full(200, np.nan), np.full(200, np.nan)]
    frequencies = [np.linspace(-2e9, 18e9, 200)] * 2
    settings = PointFitSettings.snapshot(
        2, BRILLOUIN_REGIONS, RAYLEIGH_REGIONS,
        BOUNDS_2_DEFAULT, FWHM_OPEN)

    results = fit_point(spectra, frequencies, settings)

    for key, value in results.items():
        assert np.isnan(value).all(), key


def test_settings_snapshot_is_independent_of_later_changes():
    bounds = [['min', '5'], ['5.5', 'inf']]
    regions = [(4.0e9, 6.0e9)]
    settings = PointFitSettings.snapshot(2, regions, regions, bounds, bounds)
    reference = copy.deepcopy(settings)

    bounds[0][1] = '99'
    regions.append((1.0, 2.0))

    assert settings == reference
    assert pickle.loads(pickle.dumps(settings)) == reference
