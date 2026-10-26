import contextlib
import logging
import threading
import time

import numpy as np
from scipy.signal import medfilt2d, find_peaks
from skimage.measure import label
from skimage.morphology import closing, disk

from math import floor

from bmlab import Session
from bmlab import parallel_fit
from bmlab.evaluation_run import RunSuperseded, evaluation_runs
from bmlab.fits import fit_vipa, VIPA, fit_lorentz_region
from bmlab import point_fit
from bmlab.point_fit import PointFitSettings, fit_point
from bmlab.image import extract_lines_along_arc, find_max_in_radius
from bmlab.export import FluorescenceExport, \
    FluorescenceCombinedExport, BrillouinExport

import warnings

logger = logging.getLogger(__name__)


class ExtractionController(object):

    def __init__(self):
        self.session = Session.get_instance()
        return

    def add_point(self, calib_key, point):
        time = self.session.get_calibration_time(calib_key)
        if time is None:
            return
        em = self.session.extraction_model()
        em.add_point(calib_key, time, *point)

    def set_point(self, calib_key, index, point):
        time = self.session.get_calibration_time(calib_key)
        if time is None:
            return
        em = self.session.extraction_model()
        em.set_point(calib_key, index, time, *point)

    def optimize_points(self, calib_key, radius=10):
        em = self.session.extraction_model()

        imgs = self.session.get_calibration_image(calib_key)
        if imgs is None:
            return
        img = np.nanmean(imgs, axis=0)
        img = medfilt2d(img)

        points = em.get_points(calib_key)
        time = em.get_time(calib_key)
        em.clear_points(calib_key)

        for p in points:
            new_point = find_max_in_radius(img, p, radius)
            # Warning: x-axis in imshow is 1-axis in img, y-axis is 0-axis
            em.add_point(
                calib_key, time, new_point[0], new_point[1])

    def find_points_all(self):
        calib_keys = self.session.get_calib_keys()

        if not calib_keys:
            return

        for calib_key in calib_keys:
            self.find_points(calib_key)

    def find_points(self, calib_key, min_height=10,
                    min_area=20, max_distance=50):
        em = self.session.extraction_model()

        imgs = self.session.get_calibration_image(calib_key)
        if imgs is None:
            return
        img = np.nanmean(imgs, axis=0)
        time = self.session.get_calibration_time(calib_key)
        if time is None:
            return

        # Account for binning for default values
        disc_size = 10
        binning = self.session.get_calibration_binning_factor(calib_key)
        if binning:
            max_distance = max_distance / binning
            disc_size = disc_size / binning
            min_area = min_area / (binning**2)

        img = medfilt2d(img)
        # This is the background level
        threshold = np.median(img)

        # Try to find a signal dependent estimate
        # for the minimal peak height.
        # Discard all values smaller than the background noise
        # plus a minimal peak height
        img_peaks = img[img > threshold + min_height]
        # Calculate the signal dependent peak threshold
        height = (np.nanmean(img_peaks) + threshold) / 2
        img_closed = closing(img > height, disk(disc_size))

        # Find all peaks higher than the min_height
        image_label, num = label(img_closed, return_num=True)

        all_peaks = []
        for region in range(1, num + 1):
            # Mask of everything but the peak
            mask = (image_label != region)
            # Set everything but the peak to zero
            tmp = img.copy()
            tmp[mask] = 0

            # Discard all peaks with an area that is too small
            if np.sum(np.logical_not(mask)) >= min_area:
                # Find indices of peak maximum
                ind = np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)

                all_peaks.append(ind)

        # Filter found peaks
        p0 = (0, img.shape[1])
        p1 = (img.shape[0], 0)
        peaks = filter(
            lambda peak:
            self.distance_point_to_line(peak, p0, p1) < max_distance,
            all_peaks)

        # Add found peaks to model
        em.set_points(calib_key, time, list(peaks))

    def distance_point_to_line(self, point, line0, line1):
        return abs(
            (line1[1] - line0[1]) * (line0[0] - point[0]) -
            (line0[1] - point[1]) * (line1[0] - line0[0]))\
               / np.sqrt((line1[1] - line0[1])**2 + (line1[0] - line0[0])**2)


class EvaluationSnapshot(object):
    """
    Measurement metadata of the current repetition, read once for
    a single evaluation run.

    The snapshot holds its own repetition wrapper, which is created for
    this run only and never handed out. It therefore shares no state with
    the wrappers and position arrays that the GUI obtains through
    ``Session.get_payload_positions``. The snapshot belongs to the
    evaluation run that created it and is dropped when that run ends.
    """

    def __init__(self, session):
        """
        Only the image keys and the resolution are read here. The time
        and exposure of an image are read when they are first needed and
        then kept. Reading them for all images at the start would delay
        the first fits on large scans by seconds.
        """
        repetition = session.current_repetition()
        if repetition is None:
            raise ValueError('No repetition selected')
        self._session = session
        self._payload = repetition.payload
        self.image_keys = self._payload.image_keys(sort_by_time=True)
        self.resolution = self._payload.resolution
        self._times = {}
        self._exposures = {}

    def get_time(self, image_key):
        if image_key not in self._times:
            self._times[image_key] = self._payload.get_time(image_key)
        return self._times[image_key]

    def get_exposure(self, image_key):
        if image_key not in self._exposures:
            self._exposures[image_key] = self._payload.get_exposure(image_key)
        return self._exposures[image_key]

    def get_image(self, image_key):
        imgs = self._payload.get_image(image_key)
        if imgs is None:
            return None
        # The orientation can be changed by the user, so we apply
        # the current one like Session.get_payload_image does.
        return self._session.orientation.apply(imgs)


class _PerThread(object):
    """
    An attribute whose value belongs to one controller and one thread
    (default None). A run keeps its state in such attributes, so two
    runs which share a controller, one in each thread, do not
    see or reset each other's state.
    """

    def __set_name__(self, owner, name):
        self.name = name

    @staticmethod
    def _local(instance):
        local = instance.__dict__.get('_per_thread')
        if local is None:
            local = instance.__dict__.setdefault(
                '_per_thread', threading.local())
        return local

    def __get__(self, instance, owner=None):
        if instance is None:
            return None
        return getattr(self._local(instance), self.name, None)

    def __set__(self, instance, value):
        setattr(self._local(instance), self.name, value)


class ImageController(object):

    # Metadata snapshot of the evaluation run in progress, if any.
    # Only EvaluationController.evaluate sets it.
    _snapshot = _PerThread()
    # The evaluation run in progress (bmlab.evaluation_run), if any.
    # Only EvaluationController.evaluate sets it.
    _run = _PerThread()

    def _writing(self):
        """
        Context for writes into shared state. Inside an evaluation run
        it raises RunSuperseded if a newer run has started.
        """
        run = self._run
        return contextlib.nullcontext() if run is None else run.writing()

    def __init__(self, model, get_image, get_time, get_exposure):
        self.session = Session.get_instance()
        self.model = model
        self.get_image = get_image
        self.get_time = get_time
        self.get_exposure = get_exposure

    def extract_spectra(self, image_key, frame_num=None):
        em = self.session.extraction_model()
        if not em:
            return None, None, None
        get_time, get_exposure, get_image = \
            self.get_time, self.get_exposure, self.get_image
        if self._snapshot is not None:
            get_time = self._snapshot.get_time
            get_exposure = self._snapshot.get_exposure
            get_image = self._snapshot.get_image
        time = get_time(image_key)
        arc = em.get_arc_by_time(time)
        if arc.size == 0:
            return None, None, None

        imgs = get_image(image_key)
        if imgs is None:
            return None, None, None
        if frame_num is not None:
            imgs = imgs[frame_num:frame_num+1]

        # Extract values from *all* frames in the current calibration
        spectra = []
        for img in imgs:
            values_by_img = extract_lines_along_arc(
                img,
                arc
            )
            spectra.append(values_by_img)

        exposure = get_exposure(image_key)
        times = exposure * np.arange(len(imgs)) + time

        intensities = np.nanmean(imgs, axis=(1, 2))

        # We only set the spectra if we extracted all
        if frame_num is None:
            with self._writing():
                self.model().set_spectra(image_key, spectra)
        return spectra, times, intensities


class CalibrationController(ImageController):

    def __init__(self, *args, **kwargs):
        session = Session.get_instance()
        super(CalibrationController, self).__init__(
            model=session.calibration_model,
            get_image=session.get_calibration_image,
            get_time=session.get_calibration_time,
            get_exposure=session.get_calibration_exposure
        )
        return

    def find_peaks(self, calib_key, min_prominence=15,
                   num_brillouin_samples=2, min_height=15):
        spectra, _, _ = self.extract_spectra(calib_key)
        if spectra is None:
            return
        spectrum = np.mean(spectra, axis=0)
        # This is the background value
        base = np.nanmedian(spectrum)
        peaks, properties = find_peaks(
            spectrum, prominence=min_prominence, width=True,
            height=min_height+base)

        # Number of peaks we are searching for
        # (2 Rayleigh + 2 times number calibration samples)
        num_peaks = 2 + 2 * num_brillouin_samples

        # Check if we found enough peak candidates
        if len(peaks) < num_peaks:
            # If we didn't find enough peaks, we try again
            # without a minimum height
            peaks, properties = find_peaks(
                spectrum, prominence=min_prominence, width=True)
            # If there a still too few, we give up
            if len(peaks) < num_peaks:
                return

        # We need to identify the position between the
        # Stokes and Anti-Stokes Brillouin peaks

        # In case there are just enough peaks,
        # we use the position in the middle:
        if len(peaks) == num_peaks:
            idx = int(num_peaks / 2)
            center = np.mean(peaks[idx - 1:idx + 1])
        # Otherwise we use the center of mass as the middle
        else:
            # Set everything below the background value to zero,
            # so it does not affect the center calculation
            spectrum[spectrum < base] = 0
            # Calculate the center of mass
            center = np.nansum(spectrum * range(1, len(spectrum) + 1))\
                / np.nansum(spectrum)

            # Check that we have enough peaks on both sides of the center
            num_peaks_right = len(peaks[peaks > center])
            num_peaks_left = len(peaks[peaks <= center])

            # If not enough peaks on the right, shift center to the left
            if num_peaks_right < (num_brillouin_samples + 1):
                center = np.mean(
                    peaks[-(num_brillouin_samples + 2):-num_brillouin_samples]
                )
            # If not enough peaks on the left, shift center to the right
            elif num_peaks_left < (num_brillouin_samples + 1):
                center = np.mean(
                    peaks[num_brillouin_samples:num_brillouin_samples + 2]
                )

        num_peaks_left = len(peaks[peaks <= center])

        indices_brillouin = range(
            num_peaks_left - num_brillouin_samples,
            num_peaks_left + num_brillouin_samples
        )
        indices_rayleigh = [
            num_peaks_left - num_brillouin_samples - 1,
            num_peaks_left + num_brillouin_samples
        ]

        def peak_to_region(i):
            r = (
                        peaks[i]
                        + properties['widths'][i] * np.array((-4, 4))
                ).astype(int)
            r[r > len(spectrum)] = len(spectrum)
            return tuple(r)

        regions_brillouin = list(map(peak_to_region, indices_brillouin))
        # Merge the Brillouin regions if necessary
        if num_brillouin_samples > 1:
            regions_brillouin = [
                (regions_brillouin[0][0],
                 regions_brillouin[num_brillouin_samples - 1][1]),
                (regions_brillouin[num_brillouin_samples][0],
                 regions_brillouin[-1][1]),
            ]

        regions_rayleigh = map(peak_to_region, indices_rayleigh)

        cm = self.session.calibration_model()
        # Add Brillouin regions
        for i, region in enumerate(regions_brillouin):
            # We use "set_brillouin_region" here so overlapping
            # regions don't get merged
            cm.set_brillouin_region(calib_key, i, region)
        # Add Rayleigh regions
        for i, region in enumerate(regions_rayleigh):
            # We use "set_brillouin_region" here so overlapping
            # regions don't get merged
            cm.set_rayleigh_region(calib_key, i, region)

    def calibrate(self, calib_key, count=None, max_count=None):

        setup = self.session.setup
        em = self.session.extraction_model()
        cm = self.session.calibration_model()

        if not calib_key\
                or not setup\
                or not em\
                or not cm:
            if max_count is not None:
                max_count.value = -1
            return

        spectra, _, _ = self.extract_spectra(calib_key)
        time = self.session.get_calibration_time(calib_key)

        if spectra is None or len(spectra) == 0:
            if max_count is not None:
                max_count.value = -1
            return

        self.fit_rayleigh_regions(calib_key)
        self.fit_brillouin_regions(calib_key)

        vipa_params = []
        frequencies = []

        if max_count is not None:
            max_count.value += len(spectra)

        for frame_num, spectrum in enumerate(spectra):
            peaks = cm.get_sorted_peaks(calib_key, frame_num)

            params = fit_vipa(peaks, setup)
            if params is None:
                continue
            vipa_params.append(params)
            xdata = np.arange(len(spectrum))

            frequencies.append(VIPA(xdata, params) - setup.f0)
            if count is not None:
                count.value += 1

        cm.set_vipa_params(calib_key, vipa_params)
        cm.set_frequencies(calib_key, time, frequencies)

        evm = self.session.evaluation_model()
        if evm is not None:
            evm.invalidate_results()

    def clear_calibration(self, calib_key):
        cm = self.session.calibration_model()
        if not cm:
            return

        cm.clear_brillouin_fits(calib_key)
        cm.clear_rayleigh_fits(calib_key)
        cm.clear_frequencies(calib_key)
        cm.clear_vipa_params(calib_key)

        evm = self.session.evaluation_model()
        if evm is not None:
            evm.invalidate_results()

    def fit_rayleigh_regions(self, calib_key):
        cm = self.session.calibration_model()
        spectra = cm.get_spectra(calib_key)
        regions = cm.get_rayleigh_regions(calib_key)

        cm.clear_rayleigh_fits(calib_key)
        for frame_num, spectrum in enumerate(spectra):
            for region_key, region in enumerate(regions):
                xdata = np.arange(len(spectrum))
                w0, fwhm, intensity, offset = \
                    fit_lorentz_region(region, xdata, spectrum)
                cm.add_rayleigh_fit(calib_key, region_key, frame_num,
                                    w0, fwhm, intensity, offset)

    def fit_brillouin_regions(self, calib_key):
        cm = self.session.calibration_model()
        spectra = cm.get_spectra(calib_key)
        regions = cm.get_brillouin_regions(calib_key)
        setup = self.session.setup
        if not setup:
            return

        cm.clear_brillouin_fits(calib_key)
        for frame_num, spectrum in enumerate(spectra):
            for region_key, region in enumerate(regions):
                xdata = np.arange(len(spectrum))
                w0s, fwhms, intensities, offset = \
                    fit_lorentz_region(
                        region,
                        xdata,
                        spectrum,
                        setup.calibration.num_brillouin_samples
                    )
                cm.add_brillouin_fit(calib_key, region_key, frame_num,
                                     w0s, fwhms, intensities, offset)

    def expected_frequencies(self, calib_key=None, current_frame=None):
        cm = self.session.calibration_model()

        if calib_key not in cm.vipa_params or \
                current_frame > len(cm.vipa_params[calib_key]) - 1:
            return None

        return self.session.setup.calibration.shifts \
            + self.session.setup.calibration.orders \
            * cm.vipa_params[calib_key][current_frame][3]


class PeakSelectionController(object):

    def __init__(self):
        self.session = Session.get_instance()
        return

    def add_brillouin_region_frequency(self, region_frequency):
        psm = self.session.peak_selection_model()
        psm.add_brillouin_region(region_frequency)

    def add_rayleigh_region_frequency(self, region_frequency):
        psm = self.session.peak_selection_model()
        psm.add_rayleigh_region(region_frequency)


class EvaluationController(ImageController):

    def __init__(self, *args, **kwargs):
        session = Session.get_instance()
        super(EvaluationController, self).__init__(
            model=session.evaluation_model,
            get_image=session.get_payload_image,
            get_time=session.get_payload_time,
            get_exposure=session.get_payload_exposure
        )
        return

    def set_nr_brillouin_peaks(self, nr_brillouin_peaks):
        evm = self.session.evaluation_model()
        if not evm:
            return
        evm.setNrBrillouinPeaks(nr_brillouin_peaks)

    def set_bounds(self, bounds):
        evm = self.session.evaluation_model()
        if not evm:
            return
        evm.bounds_w0 = bounds

    def set_bounds_fwhm(self, bounds):
        evm = self.session.evaluation_model()
        if not evm:
            return
        evm.bounds_fwhm = bounds

    # The run the last call of evaluate made (bmlab.evaluation_run).
    # `last_run.outcome` tells how it ended, `last_run.run_id` is its
    # number on the session.
    last_run = None

    def evaluate(self, abort=None, count=None, max_count=None):
        """
        Evaluate the current repetition.

        Every call is a run with an identity (see bmlab.evaluation_run).
        A run which was replaced by a newer run, on this or another
        controller of the session, stops without writing anything more
        and returns like a cancelled run (max_count -1). Setting the
        abort input stops a run, too. Use `last_run.outcome` to tell
        the cases apart.
        """
        run = evaluation_runs(self.session).begin()
        self._run = run
        self.last_run = run
        try:
            # Measurement metadata is read once and used for the whole
            # run. It is private to this run and dropped on every way
            # out.
            try:
                self._snapshot = EvaluationSnapshot(self.session)
            except ValueError:
                self._snapshot = None
                run.outcome = 'not-started'
                if max_count is not None:
                    max_count.value = -1
                return
            run.outcome = self._evaluate_run(abort, count, max_count)
        except RunSuperseded:
            run.outcome = 'superseded'
            if max_count is not None:
                max_count.value = -1
        except BaseException:
            run.outcome = 'failed'
            raise
        finally:
            self._snapshot = None
            self._run = None

    def _evaluate_run(self, abort, count, max_count):
        em = self.session.extraction_model()
        if not em:
            if max_count is not None:
                max_count.value = -1
            return 'not-started'

        cm = self.session.calibration_model()
        if not cm:
            if max_count is not None:
                max_count.value = -1
            return 'not-started'

        pm = self.session.peak_selection_model()
        if not pm:
            if max_count is not None:
                max_count.value = -1
            return 'not-started'

        evm = self.session.evaluation_model()
        if not evm:
            if max_count is not None:
                max_count.value = -1
            return 'not-started'

        image_keys = self._snapshot.image_keys

        if max_count is not None:
            max_count.value = len(image_keys)

        brillouin_regions = pm.get_brillouin_regions()
        rayleigh_regions = pm.get_rayleigh_regions()

        resolution = self._snapshot.resolution

        # Get first spectrum to find number of images
        spectra, _, _ = self.extract_spectra('0')

        if not spectra:
            if max_count is not None:
                max_count.value = -1
            return 'not-started'

        # We create a variable for this value here,
        # so changing nr_brillouin_peaks during evaluation
        # does not create issues
        nr_brillouin_peaks = evm.nr_brillouin_peaks
        # The settings of the fits are also snapshotted once per run.
        fit_settings = PointFitSettings.snapshot(
            nr_brillouin_peaks, brillouin_regions, rayleigh_regions,
            evm.bounds_w0, evm.bounds_fwhm)
        execution = self._fit_execution
        if execution is None:
            # Automatic: small jobs run in this process
            execution = parallel_fit.select_execution(
                len(image_keys), len(spectra),
                len(brillouin_regions) + len(rayleigh_regions),
                nr_brillouin_peaks)

        # The workers start before the result arrays are initialised.
        # The time and exposure of a point are read just before its
        # extraction, so the first points go out as soon as possible.
        pool = None
        pool_started_at = time.perf_counter()
        if execution.parallel:
            pool = self._start_pool(execution)
            if pool is None:
                execution = parallel_fit.serial_execution()
        try:
            with self._writing():
                evm.initialize_results_arrays({
                    # measurement points in x direction
                    'dim_x': resolution[0],
                    # measurement points in y direction
                    'dim_y': resolution[1],
                    # measurement points in z direction
                    'dim_z': resolution[2],
                    # number of images per measurement point
                    'nr_images': len(spectra),
                    # number of Brillouin regions
                    'nr_brillouin_regions': len(brillouin_regions),
                    # number of peaks to fit per region
                    'nr_brillouin_peaks': nr_brillouin_peaks,
                    # number of Rayleigh regions
                    'nr_rayleigh_regions': len(rayleigh_regions),
                })
        except BaseException:
            # No work was sent. No worker process may stay.
            self._stop_pool(pool)
            raise

        if pool is not None:
            # The call owns the pool from here on and ends it
            aborted = self._fit_points_parallel(
                execution, pool, pool_started_at, image_keys, resolution,
                cm, evm,
                fit_settings, len(spectra), len(rayleigh_regions),
                abort, count)
        else:
            aborted = self._fit_points_serial(
                image_keys, resolution, cm, evm, fit_settings,
                len(spectra), len(rayleigh_regions), abort, count)
        if aborted:
            with self._writing():
                calculate_derived_values()
            if max_count is not None:
                max_count.value = -1
            return 'cancelled'

        with self._writing():
            calculate_derived_values()

        return 'completed'

    def _start_pool(self, execution):
        """
        Start the workers. Returns the pool, or None if the workers
        cannot start. Nothing was submitted, extracted or stored yet,
        so the run then goes on in this process and gives exactly the
        serial results. (Failures after work was submitted are not
        handled this way, see `_fit_points_parallel`.)
        """
        try:
            return execution.create_pool()
        except Exception as error:
            logger.warning(
                'Could not start worker processes (%s: %s). '
                'Evaluating in this process instead.',
                type(error).__name__, error, exc_info=True)
            return None

    @staticmethod
    def _stop_pool(pool):
        """Stop a pool whose work is not needed. Leaves no worker."""
        if pool is None:
            return
        try:
            pool.terminate()
        finally:
            pool.join()

    # Internal setting of how the fits are executed (None: defaults),
    # see bmlab.parallel_fit.FitExecution. Not a user setting.
    _fit_execution = None

    def _prepare_point(self, cm, evm, image_key, ind):
        """
        Extract the spectra of a point in the controlling process and
        store everything which does not need a fit.

        Returns (spectra, frequencies), or None if there is nothing to fit.
        """
        ind_x, ind_y, ind_z = ind
        spectra, times, intensities = self.extract_spectra(image_key)
        if spectra is None:
            return None
        with self._writing():
            evm.results['time'][ind_x, ind_y, ind_z, :, 0, 0] = times
            evm.results['intensity'][ind_x, ind_y, ind_z, :, 0, 0] = \
                intensities

        frequencies = cm.get_frequencies_by_time(times)
        # If we don't have frequency axis, we cannot evaluate on it
        if frequencies is None:
            return None
        frequencies = list(frequencies)
        with self._writing():
            evm.set_frequencies(image_key, frequencies)
        return spectra, frequencies

    @staticmethod
    def _store_point(evm, ind, nr_frames, point_results,
                     rayleigh_peak_initial):
        """
        Write the fit results of a point and calculate the shift of the
        Rayleigh peaks. Points must be stored in measurement order.

        Returns the (possibly new) reference Rayleigh peak positions.
        """
        ind_x, ind_y, ind_z = ind
        for key, values in point_results.items():
            evm.results[key][ind_x, ind_y, ind_z, :nr_frames] = values

        # Calculate the shift of the Rayleigh peaks,
        # in order to follow the peaks in case of a drift
        rayleigh_peak_current = evm.results['rayleigh_peak_position_f'][
                ind_x, ind_y, ind_z, :, :, :]
        # If we haven't found a valid Rayleigh peak position,
        # but the current one is valid, use it
        if not np.isnan(rayleigh_peak_current).all()\
                and np.isnan(rayleigh_peak_initial).all():
            rayleigh_peak_initial = rayleigh_peak_current
        shift = rayleigh_peak_current - rayleigh_peak_initial
        evm.results['rayleigh_shift'][ind_x, ind_y, ind_z, :, :, :] = shift
        return rayleigh_peak_initial

    def _store_and_calculate(self, evm, ind, idx, nr_frames, point_results,
                             rayleigh_peak_initial, stored):
        """
        Store the results of a point (see `_store_point`). Every ten steps
        calculate the derived values of the points `stored` since the
        last time and empty `stored`.

        Returns the (possibly new) reference Rayleigh peak positions.
        """
        with self._writing():
            rayleigh_peak_initial = self._store_point(
                evm, ind, nr_frames, point_results, rayleigh_peak_initial)
            stored.append(ind)
            if not (idx % 10):
                calculate_derived_values(stored)
                stored.clear()
        return rayleigh_peak_initial

    def _fit_points_serial(self, image_keys, resolution, cm, evm,
                           fit_settings, nr_frames, nr_rayleigh_regions,
                           abort, count):
        """
        Fit all points one after the other in this process.
        Returns True if the evaluation was aborted.
        """
        # Initialize the Rayleigh shift
        # used for compensating drifts
        rayleigh_peak_initial =\
            np.nan * np.ones((nr_frames, nr_rayleigh_regions, 1))
        # Points stored since the derived values were calculated
        stored = []
        # Loop over all measurement positions
        for idx, image_key in enumerate(image_keys):

            # A run which a newer run has replaced does nothing more
            if self._run is not None:
                self._run.ensure_current()

            if count is not None:
                count.value += 1

            # Calculate the indices for the given key
            ind = self.get_indices_from_key(resolution, image_key)

            if (abort is not None) and abort.value:
                return True
            prepared = self._prepare_point(cm, evm, image_key, ind)
            if prepared is None:
                continue
            spectra, frequencies = prepared

            # All fits of a point (single-peak fits, bounds depending
            # on the Rayleigh peaks, multi-peak fits) are done together.
            point_results = fit_point(spectra, frequencies, fit_settings)
            rayleigh_peak_initial = self._store_and_calculate(
                evm, ind, idx, len(spectra), point_results,
                rayleigh_peak_initial, stored)
        return False

    def _fit_points_parallel(self, execution, pool, pool_started_at,
                             image_keys, resolution, cm, evm, fit_settings,
                             nr_frames, nr_rayleigh_regions, abort, count):
        """
        Fit points in batches in worker processes. This process
        extracts the spectra and stores the results in measurement order
        as soon as the points up to them are complete. Workers only get
        arrays and the fit settings.
        The pool belongs to this call. It is closed or terminated before
        the call returns.
        Returns True if the evaluation was aborted.
        """
        rayleigh_peak_initial =\
            np.nan * np.ones((nr_frames, nr_rayleigh_regions, 1))
        # Points stored since the derived values were calculated
        stored = []

        def points():
            for idx, image_key in enumerate(image_keys):
                ind = self.get_indices_from_key(resolution, image_key)
                prepared = self._prepare_point(cm, evm, image_key, ind)
                if prepared is None:
                    yield (idx, ind, 0), None
                else:
                    yield (idx, ind, len(prepared[0])), prepared

        def should_abort():
            # Raises if a newer run has replaced this one
            if self._run is not None:
                self._run.ensure_current()
            return (abort is not None) and abort.value

        finished = False
        runner = None
        try:
            runner = parallel_fit.OrderedBatchRunner(
                pool, fit_settings, execution.batch_size,
                execution.max_outstanding_jobs, should_abort,
                execution.job_timeout)
            for (idx, ind, nr_spectra), point_results in runner.run(points()):
                if count is not None:
                    count.value += 1
                if point_results is None:
                    continue
                rayleigh_peak_initial = self._store_and_calculate(
                    evm, ind, idx, nr_spectra, point_results,
                    rayleigh_peak_initial, stored)
            finished = not runner.aborted
        finally:
            if runner is not None and self._run is not None:
                self._run.stats = runner.statistics(
                    execution.workers, pool_started_at)
            # The pool is per run. After normal completion the workers
            # have no work left and exit. Otherwise (abort, failure)
            # unfinished work is dropped. No worker survives the run.
            try:
                if finished:
                    pool.close()
                else:
                    pool.terminate()
            finally:
                pool.join()
        return runner.aborted

    @staticmethod
    def fit_spectra(spectra, frequencies, region, nr_peaks=1,
                    bounds_w0=None, bounds_fwhm=None):
        return point_fit.fit_spectra(
            spectra, frequencies, region, nr_peaks,
            bounds_w0, bounds_fwhm)

    def create_bounds(self, brillouin_regions, rayleigh_peaks):
        """
        This function converts the bounds settings into
        a bounds object for the fitting function,
        see bmlab.point_fit.create_bounds.
        """
        evm = self.session.evaluation_model()
        return point_fit.create_bounds(
            evm.bounds_w0, brillouin_regions, rayleigh_peaks)

    def create_bounds_fwhm(self, brillouin_regions, rayleigh_peaks):
        """
        This function converts the fwhm bounds settings into
        a fwhm bounds object for the fitting function,
        see bmlab.point_fit.create_bounds_fwhm.
        """
        evm = self.session.evaluation_model()
        return point_fit.create_bounds_fwhm(
            evm.bounds_fwhm, brillouin_regions, rayleigh_peaks)

    def get_data(self, parameter_key, brillouin_peak_index=0):
        """
        This function returns the evaluated data,
        its positions, dimensionality and labels
        Parameters
        ----------
        parameter_key: str
            The key of the parameter requested.
            See bmlab.model.evaluation_model.get_parameter_keys()
        brillouin_peak_index: int
            The index of the Brillouin peak to show in case
            we did a multi-peak fit
            0: the single-peak fit
            1:nr_brillouin_peaks: the multi-peak fits
            nr_brillouin_peaks+1: all multi-peak fits average
            nr_brillouin_peaks+2: all multi-peak fits weighted average

        Returns
        -------
        data: np.ndarray
            The data to show. This is always a 3-dimensional array.
        positions: list
            This is a list of length 3 containing ndarrays with
            the spatial positions of the data points.
        dimensionality: int
            Whether it's a 0, 1, 2, or 3D measurement
        labels: list
            The labels of the positions
        """
        resolution = self.session.get_payload_resolution()

        dimensionality = sum(np.array(resolution) > 1)

        # Get the positions and squeeze them
        pos = self.session.get_payload_positions()

        positions = list(pos.values())
        labels = list(map(lambda axis_label:
                          r'$' + axis_label + '$ [$\\mu$m]', ['x', 'y', 'z']))

        evm = self.session.evaluation_model()
        data = evm.results[parameter_key]

        # Ensure that we always get the expected shape
        # (even if the array was not initialized yet)
        if data.size == 0:
            data = np.empty(resolution)
            data[:] = np.nan

        # Slice the appropriate Brillouin peak if necessary and possible
        if data.ndim >= 6:
            nr_peaks_stored = data.shape[5]
            if nr_peaks_stored > 1\
                    and brillouin_peak_index < nr_peaks_stored + 2:
                if brillouin_peak_index < nr_peaks_stored:
                    sliced = data[:, :, :, :, :, brillouin_peak_index]
                # Average all multi-peak fits
                if brillouin_peak_index == nr_peaks_stored:
                    sliced = data[:, :, :, :, :, 1:]
                # Weighted average of all multi-peak fits
                if brillouin_peak_index == nr_peaks_stored + 1:
                    weight =\
                        evm.results[
                            'brillouin_peak_intensity'][:, :, :, :, :, 1:]\
                        * evm.results['brillouin_peak_fwhm_f'][
                          :, :, :, :, :, 1:]
                    # Nansum returns 0 if all entries are NaN, dividing by this
                    # hence gives an invalid value error
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            action='ignore',
                            message='invalid value encountered in divide'
                        )
                        sliced = \
                            np.nansum(data[:, :, :, :, :, 1:] *
                                      weight, axis=5)\
                            / np.nansum(weight, axis=5)
            else:
                sliced = data[:, :, :, :, :, 0]
        else:
            sliced = data

        # Average all non-spatial dimensions.
        # Do not show warning which occurs when a slice contains only NaNs.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore',
                message='Mean of empty slice'
            )
            data = np.nanmean(
                sliced,
                axis=tuple(range(3, sliced.ndim))
            )

        # Scale the date in case of GHz
        data = evm.parameters[parameter_key]['scaling'] * data

        return data, positions, dimensionality, labels

    def get_fits(self, image_key):
        resolution = self.session.get_payload_resolution()
        indices = self.get_indices_from_key(resolution, image_key)

        evm = self.session.evaluation_model()
        if not evm:
            return
        return evm.get_fits(*indices)

    @staticmethod
    def get_key_from_indices(resolution, ind_x, ind_y, ind_z):
        if len(resolution) != 3:
            raise ValueError('resolution has wrong dimension')
        if ind_x >= resolution[0]:
            raise IndexError('x index out of range')
        if ind_y >= resolution[1]:
            raise IndexError('y index out of range')
        if ind_z >= resolution[2]:
            raise IndexError('z index out of range')
        return str(int(ind_z * (resolution[0] * resolution[1])
                   + ind_y * resolution[0] + ind_x))

    @staticmethod
    def get_indices_from_key(resolution, key):
        key = int(key)
        ind_z = floor(key / (resolution[0] * resolution[1]))
        ind_y = floor(
            (key - ind_z * (resolution[0] * resolution[1])) / resolution[0])
        ind_x = (key % (resolution[0] * resolution[1])) % resolution[0]
        # ind_y = (key - ind_x) % resolution[0]
        if ind_x >= resolution[0]\
                or ind_y >= resolution[1]\
                or ind_z >= resolution[2]:
            raise ValueError('Invalid key')
        return ind_x, ind_y, ind_z


def _brillouin_shift(brillouin_peak_position_f, rayleigh_peak_position_f):
    """
    The Brillouin shift in GHz of the given points. The values of a point
    only depend on the peak positions of this point, so the arrays can
    hold all points (x, y, z, frames, regions, peaks) or a selection of
    points (n, frames, regions, peaks). Returns an array without the
    peaks of the Rayleigh regions.
    """
    shape_brillouin = brillouin_peak_position_f.shape
    shape_rayleigh = rayleigh_peak_position_f.shape

    # We calculate every possible combination of
    # Brillouin peak and Rayleigh peak position difference
    # and then use the smallest absolute value.
    # That saves us from sorting Rayleigh peaks to Brillouin peaks,
    # because a Brillouin peak always belongs to the Rayleigh peak nearest.
    brillouin_shift_f = np.nan * np.ones(
        (*shape_brillouin, shape_rayleigh[-2]))
    for idx in range(shape_rayleigh[-2]):
        brillouin_shift_f[..., idx] = abs(
            brillouin_peak_position_f -
            np.tile(
                rayleigh_peak_position_f[..., [idx], :],
                (*([1] * (len(shape_brillouin) - 1)), shape_brillouin[-1])
            )
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore',
            message='All-NaN slice encountered'
        )
        return np.nanmin(brillouin_shift_f, -1)


def calculate_derived_values(points=None):
    """
    We calculate the Brillouin shift in GHz here.

    Parameters
    ----------
    points: list of (ind_x, ind_y, ind_z) or None
        Calculate the values of these points only and keep the values of
        all other points. This costs time proportional to the number of
        points given, not to the size of the scan. The values are the
        same as the ones a calculation for all points gives.
        None calculates all points.
    """
    session = Session.get_instance()
    evm = session.evaluation_model()
    if not evm:
        return

    if evm.results['brillouin_peak_position_f'].size == 0:
        return

    if evm.results['rayleigh_peak_position_f'].size == 0:
        return

    brillouin = evm.results['brillouin_peak_position_f']
    rayleigh = evm.results['rayleigh_peak_position_f']

    shift = evm.results.get('brillouin_shift_f')
    if points is None or shift is None \
            or shift.shape != brillouin.shape:
        evm.results['brillouin_shift_f'] = _brillouin_shift(
            brillouin, rayleigh)
        return

    if not len(points):
        return
    ind_x, ind_y, ind_z = np.asarray(points, dtype=int).T
    shift[ind_x, ind_y, ind_z] = _brillouin_shift(
        brillouin[ind_x, ind_y, ind_z], rayleigh[ind_x, ind_y, ind_z])


class Controller(object):

    def __init__(self):
        self.session = Session.get_instance()
        return

    def evaluate(self, filepath, setup, orientation,
                 brillouin_regions, rayleigh_regions,
                 repetitions=None, nr_brillouin_peaks=1,
                 multi_peak_bounds=None,
                 multi_peak_bounds_fwhm=None):
        # Load data file
        self.session.set_file(filepath)

        # Evaluate all repetitions if not requested differently
        if repetitions is None:
            repetitions = self.session.file.repetition_keys()

        for repetition in repetitions:
            # Select repetition
            self.session.set_current_repetition(repetition)
            self.session.set_setup(setup)

            # Set orientation
            self.session.orientation = orientation

            ec = ExtractionController()
            cc = CalibrationController()
            psc = PeakSelectionController()
            evc = EvaluationController()

            # First add all extraction points because this
            # can influence the extraction for other calibrations
            ec.find_points_all()

            # Then do the calibration
            for calib_key in self.session.get_calib_keys():
                cc.find_peaks(calib_key)

                cc.calibrate(calib_key)

            for region in brillouin_regions:
                psc.add_brillouin_region_frequency(region)

            for region in rayleigh_regions:
                psc.add_rayleigh_region_frequency(region)

            evc.set_nr_brillouin_peaks(nr_brillouin_peaks)
            evc.set_bounds(multi_peak_bounds)
            evc.set_bounds_fwhm(multi_peak_bounds_fwhm)

            evc.evaluate()

        return self.session


class ExportController(object):

    def __init__(self):
        return

    @staticmethod
    def get_configuration():
        return {
            'fluorescence': {
                'export': True,
            },
            'fluorescenceCombined': {
                'export': True,
            },
            'brillouin': {
                'export': True,
                'parameters': ['brillouin_shift_f'],
                'brillouin_shift_f': {
                    'cax': ('min', 'max'),
                }
            },
        }

    def export(self, configuration=None):
        if not configuration:
            configuration = self.get_configuration()

        FluorescenceExport().export(configuration)
        FluorescenceCombinedExport().export(configuration)

        # BrillouinExport needs the EvaluationController
        # to nicely get the data, so we provide it here.
        # Not really nice, but importing it in BrillouinExport
        # leads to a circular dependency.
        BrillouinExport(EvaluationController()).export(configuration)
