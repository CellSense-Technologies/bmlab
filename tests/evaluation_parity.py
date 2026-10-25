"""
Helpers for the evaluation parity tests.

They set up a fresh session for a fixture file, run an evaluation
controller on it (optionally with perturbed input data) and capture
everything the evaluation leaves behind, so that two controllers can
be compared exactly.
"""
import contextlib
import pathlib
from unittest import mock

import numpy as np

from bmlab.controllers import CalibrationController, \
    EvaluationController, EvaluationSnapshot, ExtractionController, \
    PeakSelectionController
from bmlab.models import Orientation
from bmlab.models.setup import AVAILABLE_SETUPS
from bmlab.session import Session

DATA_DIR = pathlib.Path(__file__).parent / 'data'

BRILLOUIN_REGIONS = [(4.0e9, 6.0e9), (9.0e9, 11.0e9)]
RAYLEIGH_REGIONS = [(-2.0e9, 2.0e9), (13.0e9, 17.0e9)]


class Counter(object):
    """Mimics the multiprocessing.Value used for progress reporting."""

    def __init__(self, value=0):
        self.value = value


class AbortAfter(object):
    """Abort flag which turns true after a number of reads."""

    def __init__(self, reads):
        self.reads = reads
        self.polled = 0

    @property
    def value(self):
        self.polled += 1
        return self.polled > self.reads


def prepare_session(file_name, nr_brillouin_peaks=1,
                    bounds_w0=None, bounds_fwhm=None,
                    brillouin_regions=None, rayleigh_regions=None,
                    repetition=None):
    """
    Load a fixture into the (singleton) session with fresh models and
    configure it like Controller.evaluate does, without evaluating.
    """
    if brillouin_regions is None:
        brillouin_regions = BRILLOUIN_REGIONS
    if rayleigh_regions is None:
        rayleigh_regions = RAYLEIGH_REGIONS

    session = Session.get_instance()
    session.set_file(DATA_DIR / file_name)
    if repetition is None:
        repetition = session.file.repetition_keys()[0]
    session.set_current_repetition(repetition)
    session.set_setup(AVAILABLE_SETUPS[0])
    session.orientation = Orientation(rotation=1, reflection={
        'vertically': False, 'horizontally': False})

    ec = ExtractionController()
    cc = CalibrationController()
    psc = PeakSelectionController()
    evc = EvaluationController()

    ec.find_points_all()
    for calib_key in session.get_calib_keys():
        cc.find_peaks(calib_key)
        cc.calibrate(calib_key)

    for region in brillouin_regions:
        psc.add_brillouin_region_frequency(region)
    for region in rayleigh_regions:
        psc.add_rayleigh_region_frequency(region)

    evc.set_nr_brillouin_peaks(nr_brillouin_peaks)
    evc.set_bounds(bounds_w0)
    evc.set_bounds_fwhm(bounds_fwhm)
    return session


def snapshot(session):
    """Copy everything the evaluation stored in the evaluation model."""
    evm = session.evaluation_model()
    return {
        'results': {key: np.array(value, copy=True)
                    for key, value in evm.results.items()},
        'spectra': {key: [np.array(s, copy=True) for s in value]
                    for key, value in evm.spectra.items()},
        'frequencies': {key: [np.array(f, copy=True) for f in value]
                        for key, value in evm.frequencies.items()},
    }


def run_evaluation(controller_class, file_name, abort_reads=None,
                   perturb=None, **prepare_kwargs):
    """
    Run an evaluation in a fresh session and return a snapshot of
    its outputs, the progress counters, and a raised exception (if any).

    perturb: optional callable receiving the session. It returns a
        context manager, which is active while the controller
        is created and evaluates (e.g. patches of input data access).
    """
    session = prepare_session(file_name, **prepare_kwargs)
    count = Counter(0)
    max_count = Counter(0)
    abort = None if abort_reads is None else AbortAfter(abort_reads)
    error = None

    with contextlib.ExitStack() as stack:
        if perturb is not None:
            stack.enter_context(perturb(session))
        controller = controller_class()
        try:
            controller.evaluate(abort=abort, count=count,
                                max_count=max_count)
        except Exception as e:
            error = e

    outcome = snapshot(session)
    outcome['count'] = count.value
    outcome['max_count'] = max_count.value
    outcome['abort_polled'] = None if abort is None else abort.polled
    outcome['error_type'] = None if error is None else type(error)
    return outcome


def assert_arrays_identical(actual, expected, label=''):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, label
    assert actual.dtype == expected.dtype, label
    assert np.array_equal(actual, expected, equal_nan=True), label
    if actual.dtype.kind == 'f':
        # NaN locations and the sign of infinities must match as well
        assert np.array_equal(np.isnan(actual), np.isnan(expected)), label
        assert np.array_equal(np.isposinf(actual),
                              np.isposinf(expected)), label
        assert np.array_equal(np.isneginf(actual),
                              np.isneginf(expected)), label


def assert_outcomes_identical(actual, expected):
    assert actual['error_type'] == expected['error_type']
    assert actual['count'] == expected['count']
    assert actual['max_count'] == expected['max_count']
    assert actual['abort_polled'] == expected['abort_polled']

    assert actual['results'].keys() == expected['results'].keys()
    for key in expected['results']:
        assert_arrays_identical(
            actual['results'][key], expected['results'][key],
            'results[{}]'.format(key))

    for name in ('spectra', 'frequencies'):
        assert actual[name].keys() == expected[name].keys(), name
        for image_key in expected[name]:
            assert len(actual[name][image_key]) \
                == len(expected[name][image_key]), name
            for idx, (a, e) in enumerate(zip(actual[name][image_key],
                                             expected[name][image_key])):
                assert_arrays_identical(
                    a, e, '{}[{}][{}]'.format(name, image_key, idx))


# Perturbations of the input data ---------------------------------------

def _wrap_images(session, modify):
    """
    Modify the images read by the evaluation. The previous code reads
    them through the session, the current code through the per-run
    EvaluationSnapshot, so both paths are patched.
    """
    original = session.get_payload_image
    original_snapshot = EvaluationSnapshot.get_image

    def get_payload_image(image_key, frame_num=None):
        imgs = original(image_key, frame_num)
        return modify(image_key, imgs)

    def get_snapshot_image(snapshot, image_key):
        imgs = original_snapshot(snapshot, image_key)
        return modify(image_key, imgs)

    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch.object(
        session, 'get_payload_image', side_effect=get_payload_image))
    stack.enter_context(mock.patch.object(
        EvaluationSnapshot, 'get_image', get_snapshot_image))
    return stack


def missing_images(keys):
    """Images of the given keys are not available (incomplete scan)."""
    keys = set(keys)

    def perturb(session):
        return _wrap_images(
            session,
            lambda key, imgs: None if key in keys else imgs)
    return perturb


def nan_images(frames_by_key):
    """
    Set whole frames to NaN, or parts of them.
    frames_by_key: {image_key: {frame: (row_slice, col_slice)}}
    """
    def modify(key, imgs):
        if imgs is not None and key in frames_by_key:
            imgs = np.array(imgs, dtype=float, copy=True)
            for frame, (rows, cols) in frames_by_key[key].items():
                imgs[frame, rows, cols] = np.nan
        return imgs

    def perturb(session):
        return _wrap_images(session, modify)
    return perturb


def fewer_frames(keys, nr_frames=1):
    """The given points contain fewer frames than the first point."""
    keys = set(keys)

    def perturb(session):
        return _wrap_images(
            session,
            lambda key, imgs: imgs[:nr_frames]
            if imgs is not None and key in keys else imgs)
    return perturb


def missing_frequency_axes(times_after):
    """No frequency axis is available for points later than a time."""
    def perturb(session):
        cm = session.calibration_model()
        original = cm.get_frequencies_by_time

        def get_frequencies_by_time(time):
            if np.min(time) > times_after:
                return None
            return original(time)

        return mock.patch.object(
            cm, 'get_frequencies_by_time',
            side_effect=get_frequencies_by_time)
    return perturb


def no_frequency_axes(session):
    cm = session.calibration_model()
    return mock.patch.object(
        cm, 'get_frequencies_by_time', return_value=None)


# Scenarios -------------------------------------------------------------

FWHM_OPEN = [['0', 'inf'], ['0', 'inf']]
BOUNDS_2_DEFAULT = [['min', 'max'], ['min', 'max']]
BOUNDS_2_GHZ = [['min', '5'], ['5.5', 'inf']]
FWHM_2_GHZ = [['0', '1'], ['0.1', 'inf']]
BOUNDS_2_ANTI_STOKES = [['-5.5', '-4'], ['4', '5.5']]
BOUNDS_4_DEFAULT = [['min', 'max']] * 4
BOUNDS_4_GHZ = [['min', '5'], ['5.5', 'Inf'], ['-inf', 'max'], ['min', 'max']]
FWHM_4_GHZ = [['0', 'inf'], ['0', '0.5'], ['0.1', 'inf'], ['min', 'max']]

OUTSIDE_AXIS_REGIONS = [(30.0e9, 31.0e9), (9.0e9, 11.0e9)]

_FULL = (slice(None), slice(None))

# Each scenario: id -> kwargs for run_evaluation.
SCENARIOS = {
    # Peak modes and bounds
    'water_1peak': dict(file_name='Water.h5'),
    'water_1peak_single_regions': dict(
        file_name='Water.h5',
        brillouin_regions=[(4.0e9, 6.0e9)],
        rayleigh_regions=[(-2.0e9, 2.0e9)]),
    'water_1peak_fewer_brillouin_than_rayleigh_regions': dict(
        file_name='Water.h5',
        brillouin_regions=[(4.0e9, 6.0e9)]),
    'water_1peak_more_brillouin_than_rayleigh_regions': dict(
        file_name='Water.h5',
        brillouin_regions=[(4.0e9, 6.0e9), (9.0e9, 11.0e9),
                           (11.5e9, 12.5e9)]),
    'water_2peak_default_bounds': dict(
        file_name='Water.h5', nr_brillouin_peaks=2,
        bounds_w0=BOUNDS_2_DEFAULT, bounds_fwhm=FWHM_OPEN),
    'water_2peak_ghz_bounds': dict(
        file_name='Water.h5', nr_brillouin_peaks=2,
        bounds_w0=BOUNDS_2_GHZ, bounds_fwhm=FWHM_2_GHZ),
    'water_2peak_anti_stokes_bounds': dict(
        file_name='Water.h5', nr_brillouin_peaks=2,
        bounds_w0=BOUNDS_2_ANTI_STOKES, bounds_fwhm=FWHM_OPEN),
    'water_4peak_default_bounds': dict(
        file_name='Water.h5', nr_brillouin_peaks=4,
        bounds_w0=BOUNDS_4_DEFAULT, bounds_fwhm=[['0', 'inf']] * 4),
    'water_4peak_ghz_bounds': dict(
        file_name='Water.h5', nr_brillouin_peaks=4,
        bounds_w0=BOUNDS_4_GHZ, bounds_fwhm=FWHM_4_GHZ),
    # Invalid fits
    'water_1peak_regions_outside_axis': dict(
        file_name='Water.h5', brillouin_regions=OUTSIDE_AXIS_REGIONS),
    'water_2peak_regions_outside_axis': dict(
        file_name='Water.h5', nr_brillouin_peaks=2,
        brillouin_regions=OUTSIDE_AXIS_REGIONS,
        bounds_w0=BOUNDS_2_DEFAULT, bounds_fwhm=FWHM_OPEN),
    'water_1peak_nan_spectra': dict(
        file_name='Water.h5',
        perturb=nan_images({
            '3': {0: _FULL},
            '5': {1: (slice(150, 250), slice(None))},
            '7': {0: (slice(None), slice(0, 200)), 1: _FULL}})),
    'water_2peak_nan_spectra': dict(
        file_name='Water.h5', nr_brillouin_peaks=2,
        bounds_w0=BOUNDS_2_GHZ, bounds_fwhm=FWHM_2_GHZ,
        perturb=nan_images({
            '3': {0: _FULL},
            '5': {1: (slice(150, 250), slice(None))},
            '7': {0: (slice(None), slice(0, 200)), 1: _FULL}})),
    'water_4peak_nan_spectra': dict(
        file_name='Water.h5', nr_brillouin_peaks=4,
        bounds_w0=BOUNDS_4_GHZ, bounds_fwhm=FWHM_4_GHZ,
        perturb=nan_images({'4': {0: _FULL, 1: _FULL}})),
    # The first valid Rayleigh position is taken from a later point
    'water_2peak_first_points_invalid': dict(
        file_name='Water.h5', nr_brillouin_peaks=2,
        bounds_w0=BOUNDS_2_DEFAULT, bounds_fwhm=FWHM_OPEN,
        perturb=nan_images({'0': {0: _FULL, 1: _FULL}, '1': {0: _FULL}})),
    # Missing spectra, frequency axes, and incomplete scans
    'water_1peak_missing_images': dict(
        file_name='Water.h5', perturb=missing_images({'2', '5'})),
    'water_2peak_missing_images': dict(
        file_name='Water.h5', nr_brillouin_peaks=2,
        bounds_w0=BOUNDS_2_DEFAULT, bounds_fwhm=FWHM_OPEN,
        perturb=missing_images({'2', '5'})),
    'water_1peak_missing_first_image': dict(
        file_name='Water.h5', perturb=missing_images({'0'})),
    'water_2peak_incomplete_scan': dict(
        file_name='Water.h5', nr_brillouin_peaks=2,
        bounds_w0=BOUNDS_2_DEFAULT, bounds_fwhm=FWHM_OPEN,
        perturb=missing_images({'6', '7', '8', '9'})),
    'water_1peak_point_with_fewer_frames': dict(
        file_name='Water.h5', perturb=fewer_frames({'4'})),
    'water_2peak_point_with_fewer_frames': dict(
        file_name='Water.h5', nr_brillouin_peaks=2,
        bounds_w0=BOUNDS_2_DEFAULT, bounds_fwhm=FWHM_OPEN,
        perturb=fewer_frames({'4'})),
    'water_1peak_missing_later_frequency_axes': dict(
        file_name='Water.h5', perturb=missing_frequency_axes(45.0)),
    'water_2peak_missing_later_frequency_axes': dict(
        file_name='Water.h5', nr_brillouin_peaks=2,
        bounds_w0=BOUNDS_2_DEFAULT, bounds_fwhm=FWHM_OPEN,
        perturb=missing_frequency_axes(45.0)),
    'water_1peak_no_frequency_axes': dict(
        file_name='Water.h5', perturb=no_frequency_axes),
    # Abort
    'water_1peak_abort_never': dict(
        file_name='Water.h5', abort_reads=100),
    'water_1peak_abort_midway': dict(
        file_name='Water.h5', abort_reads=4),
    'water_2peak_abort_midway': dict(
        file_name='Water.h5', abort_reads=4, nr_brillouin_peaks=2,
        bounds_w0=BOUNDS_2_DEFAULT, bounds_fwhm=FWHM_OPEN),
    'water_1peak_abort_immediately': dict(
        file_name='Water.h5', abort_reads=0),
    # Fixtures without usable image data and an aborted measurement
    '0D_1peak': dict(file_name='0D.h5'),
    '1D-x_1peak': dict(file_name='1D-x.h5'),
    '2D-xy_1peak': dict(file_name='2D-xy.h5'),
    '2D-xy_2peak': dict(
        file_name='2D-xy.h5', nr_brillouin_peaks=2,
        bounds_w0=BOUNDS_2_DEFAULT, bounds_fwhm=FWHM_OPEN),
    '3D_1peak': dict(file_name='3D.h5'),
    'binning_1peak': dict(file_name='Binning.h5'),
    'aborted_repetition0': dict(
        file_name='aborted_repetition0.h5', repetition='0'),
    'aborted_repetition1': dict(
        file_name='aborted_repetition0.h5', repetition='1'),
    # Invalid configurations: the evaluation raises
    'water_2peak_one_rayleigh_region': dict(
        file_name='Water.h5', nr_brillouin_peaks=2,
        bounds_w0=BOUNDS_2_DEFAULT, bounds_fwhm=FWHM_OPEN,
        rayleigh_regions=[(-2.0e9, 2.0e9)]),
    'water_2peak_no_width_bounds': dict(
        file_name='Water.h5', nr_brillouin_peaks=2,
        bounds_w0=BOUNDS_2_DEFAULT, bounds_fwhm=None),
    'water_2peak_single_regions_no_fsr': dict(
        file_name='Water.h5', nr_brillouin_peaks=2,
        bounds_w0=BOUNDS_2_DEFAULT, bounds_fwhm=FWHM_OPEN,
        brillouin_regions=[(4.0e9, 6.0e9)],
        rayleigh_regions=[(-2.0e9, 2.0e9)]),
}

# Scenarios whose evaluation raises. The exception type must stay the same,
# the partially written results of the failing point may differ.
RAISING_SCENARIOS = {
    'water_1peak_more_brillouin_than_rayleigh_regions',
    'water_2peak_one_rayleigh_region',
    'water_2peak_no_width_bounds',
    'water_2peak_single_regions_no_fsr',
}
