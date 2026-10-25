"""
Frozen copy of the serial evaluation as it was before the point-fit
function was extracted (bmlab 0.12.3, commit ced2f11).

This module is a test reference only. Do not "fix" or refactor it:
its purpose is to keep the pre-change behaviour available, so that the
parity tests can compare the current implementation against it.
"""
import warnings

import numpy as np

from bmlab.controllers import EvaluationController, \
    calculate_derived_values
from bmlab.fits import fit_lorentz_region


class LegacyEvaluationController(EvaluationController):

    def evaluate(self, abort=None, count=None, max_count=None):
        em = self.session.extraction_model()
        if not em:
            if max_count is not None:
                max_count.value = -1
            return

        cm = self.session.calibration_model()
        if not cm:
            if max_count is not None:
                max_count.value = -1
            return

        pm = self.session.peak_selection_model()
        if not pm:
            if max_count is not None:
                max_count.value = -1
            return

        evm = self.session.evaluation_model()
        if not evm:
            if max_count is not None:
                max_count.value = -1
            return

        image_keys = self.session.get_image_keys(True)

        if max_count is not None:
            max_count.value = len(image_keys)

        brillouin_regions = pm.get_brillouin_regions()
        rayleigh_regions = pm.get_rayleigh_regions()

        resolution = self.session.get_payload_resolution()

        # Get first spectrum to find number of images
        spectra, _, _ = self.extract_spectra('0')

        if not spectra:
            if max_count is not None:
                max_count.value = -1
            return

        # We create a variable for this value here,
        # so changing nr_brillouin_peaks during evaluation
        # does not create issues
        nr_brillouin_peaks = evm.nr_brillouin_peaks
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

        # Initialize the Rayleigh shift
        # used for compensating drifts
        rayleigh_peak_initial =\
            np.nan * np.ones((len(spectra), len(rayleigh_regions), 1))
        # Loop over all measurement positions
        for idx, image_key in enumerate(image_keys):

            if count is not None:
                count.value += 1

            # Calculate the indices for the given key
            (ind_x, ind_y, ind_z) =\
                self.get_indices_from_key(resolution, image_key)

            if (abort is not None) and abort.value:
                calculate_derived_values()
                if max_count is not None:
                    max_count.value = -1
                return
            spectra, times, intensities =\
                self.extract_spectra(image_key)
            if spectra is None:
                continue
            evm.results['time'][ind_x, ind_y, ind_z, :, 0, 0] =\
                times
            evm.results['intensity'][ind_x, ind_y, ind_z, :, 0, 0] =\
                intensities

            frequencies = cm.get_frequencies_by_time(times)
            # If we don't have frequency axis, we cannot evaluate on it
            if frequencies is None:
                continue
            frequencies = list(frequencies)
            evm.set_frequencies(image_key, frequencies)

            for region_key, region in enumerate(brillouin_regions):
                results = self.fit_spectra(spectra, frequencies, region)
                for frame_num, _ in enumerate(spectra):
                    ind = (ind_x, ind_y, ind_z,
                           frame_num, region_key, 0)
                    evm.results['brillouin_peak_position_f'][ind] =\
                        results[frame_num][0]
                    evm.results['brillouin_peak_fwhm_f'][ind] =\
                        results[frame_num][1]
                    evm.results['brillouin_peak_intensity'][ind] =\
                        results[frame_num][2]
                    evm.results['brillouin_peak_offset'][ind] =\
                        results[frame_num][3]

            for region_key, region in enumerate(rayleigh_regions,):
                results = self.fit_spectra(spectra, frequencies, region)
                for frame_num, _ in enumerate(spectra):
                    ind = (ind_x, ind_y, ind_z, frame_num,
                           region_key - len(brillouin_regions))
                    evm.results['rayleigh_peak_position_f'][ind] =\
                        results[frame_num][0]
                    evm.results['rayleigh_peak_fwhm_f'][ind] =\
                        results[frame_num][1]
                    evm.results['rayleigh_peak_intensity'][ind] =\
                        results[frame_num][2]
                    evm.results['rayleigh_peak_offset'][ind] =\
                        results[frame_num][3]

            # We can only do a multi-peak fit after the single-peak
            # Rayleigh fit is done, because we have to know the
            # Rayleigh peak positions in GHz in order to convert
            # the multi-peak fit bounds given in GHz into the position
            # in pixels.
            if nr_brillouin_peaks > 1:
                ind =\
                    (ind_x, ind_y, ind_z, slice(None), slice(None), 0)
                rayleigh_peaks = np.transpose(
                    evm.results['rayleigh_peak_position_f'][ind]
                )
                bounds_w0 = self.create_bounds(
                    brillouin_regions,
                    rayleigh_peaks
                )
                bounds_fwhm = self.create_bounds_fwhm(
                    brillouin_regions,
                    rayleigh_peaks
                )

                for region_key, region in enumerate(
                        brillouin_regions):
                    results_multi_peak\
                        = self.fit_spectra(spectra,
                                           frequencies,
                                           region,
                                           nr_brillouin_peaks,
                                           bounds_w0[region_key],
                                           bounds_fwhm[region_key])
                    for frame_num, spectrum in enumerate(spectra):
                        ind = (ind_x, ind_y, ind_z,
                               frame_num, region_key,
                               slice(1, nr_brillouin_peaks+1))
                        evm.results[
                            'brillouin_peak_position_f'][ind] = \
                            results_multi_peak[frame_num][0]
                        evm.results[
                            'brillouin_peak_fwhm_f'][ind] = \
                            results_multi_peak[frame_num][1]
                        evm.results[
                            'brillouin_peak_intensity'][ind] = \
                            results_multi_peak[frame_num][2]
                        evm.results[
                            'brillouin_peak_offset'][ind] = \
                            results_multi_peak[frame_num][3]

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

            # Calculate the derived values every ten steps
            if not (idx % 10):
                calculate_derived_values()

        calculate_derived_values()

        return

    @staticmethod
    def fit_spectra(spectra, frequencies, region, nr_peaks=1,
                    bounds_w0=None, bounds_fwhm=None):
        fits = []
        for frame_num, spectrum in enumerate(spectra):
            if bounds_w0 is None and bounds_fwhm is None:
                fit = fit_lorentz_region(
                    region,
                    frequencies[frame_num],
                    spectrum,
                    nr_peaks
                )
            elif bounds_fwhm is None:
                fit = fit_lorentz_region(
                    region,
                    frequencies[frame_num],
                    spectrum,
                    nr_peaks,
                    bounds_w0[frame_num]
                )
            else:
                fit = fit_lorentz_region(
                    region,
                    frequencies[frame_num],
                    spectrum,
                    nr_peaks,
                    bounds_w0[frame_num],
                    bounds_fwhm[frame_num]
                )
            fits.append(fit)
        return fits

    def create_bounds(self, brillouin_regions, rayleigh_peaks):
        """
        This function converts the bounds settings into
        a bounds object for the fitting function
        Allowed parameters for the bounds settings are
        - 'min'/'max'   -> Will be converted to the respective
            lower or upper limit of the given region
        - '-Inf', 'Inf' -> Will be converted to -np.inf or np.inf
        - number [GHz]  -> Will be converted into the pixel
            position of the given frequency
        Parameters
        ----------
        brillouin_regions
        rayleigh_peaks

        Returns
        -------

        """
        evm = self.session.evaluation_model()
        bounds = evm.bounds_w0
        if bounds is None:
            return None

        # We need two Rayleigh peak positions to determine the FSR
        # and decide whether a peak is Stokes or Anti-Stokes
        if rayleigh_peaks.shape[0] != 2:
            return None

        w0_bounds = []
        # We have to create a separate bound for every region
        for region_idx, region in enumerate(brillouin_regions):
            local_time = []
            for rayleigh_idx in range(rayleigh_peaks.shape[1]):
                anti_stokes_limit = np.nanmean(rayleigh_peaks[:, rayleigh_idx])
                # We need to figure out, whether a peak is a
                # Stokes or Anti-Stokes peak in order to decide
                # to which Rayleigh peak we need to relate to.
                tmp = region >= anti_stokes_limit
                is_pure_region = (tmp == tmp[0]).all()

                is_anti_stokes_region = np.mean(region) >\
                    anti_stokes_limit

                local_bound = []
                for bound in bounds:
                    parsed_bound = []
                    for limit in bound:
                        try:
                            parsed_bound.append(float(limit))
                        except ValueError:
                            parsed_bound.append(np.nan)
                    # We don't treat Inf as a value
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            action='ignore',
                            message='Mean of empty slice'
                        )
                        is_anti_stokes_peak =\
                            np.nanmean(
                                np.array(parsed_bound)[
                                    np.isfinite(parsed_bound)]
                            ) < 0

                    is_anti_stokes = is_anti_stokes_region if\
                        is_pure_region else is_anti_stokes_peak

                    local_limit = []
                    for limit in bound:
                        if limit.lower() == 'min':
                            val = region[int(is_anti_stokes)]
                            # print(val)
                        elif limit.lower() == 'max':
                            val = region[int(not is_anti_stokes)]
                        elif limit.lower() == '-inf':
                            val = -((-1) ** is_anti_stokes)\
                                          * np.inf
                        elif limit.lower() == 'inf':
                            val = ((-1) ** is_anti_stokes)\
                                          * np.inf
                        else:
                            # Try to convert the value in GHz into
                            # a value in pixel depending on the time
                            try:
                                val = ((-1) ** is_anti_stokes)\
                                          * 1e9 * abs(float(limit))\
                                          + rayleigh_peaks[
                                          int(is_anti_stokes), rayleigh_idx]
                            except BaseException:
                                val = np.inf
                        local_limit.append(val)
                    # Check that the bounds are sorted ascendingly
                    # (for anti-stokes, they might not).
                    local_limit.sort()
                    local_bound.append(local_limit)
                local_time.append(local_bound)
            w0_bounds.append(local_time)

        return w0_bounds

    def create_bounds_fwhm(self, brillouin_regions, rayleigh_peaks):
        """
        This function converts the fwhm bounds settings into
        a fwhm bounds object for the fitting function
        Allowed parameters for the bounds settings are
        - 'min'/'max'   -> Will be converted to 0/np.inf
        - '-Inf', 'Inf' -> Will be converted to 0 or np.inf
        - number [GHz]  -> Will be converted into the value in Hz
        Parameters
        ----------
        brillouin_regions
        rayleigh_peaks

        Returns
        -------

        """
        evm = self.session.evaluation_model()
        bounds_fwhm = evm.bounds_fwhm
        if bounds_fwhm is None:
            return None

        fwhm_bounds = []
        # We have to create a separate bound for every region
        for region_idx, _ in enumerate(brillouin_regions):
            region_bound = []
            for rayleigh_idx in range(rayleigh_peaks.shape[1]):
                peak_bound = []
                for bound in bounds_fwhm:
                    local_bound = []
                    for limit in bound:
                        if limit.lower() == 'min':
                            val = 0
                        elif limit.lower() == '-inf':
                            val = 0
                        elif limit.lower() == 'max':
                            val = np.inf
                        elif limit.lower() == 'inf':
                            val = np.inf
                        else:
                            try:
                                val = 1e9*abs(float(limit))
                            except ValueError:
                                val = np.inf
                        local_bound.append(val)
                    peak_bound.append(local_bound)
                region_bound.append(peak_bound)
            fwhm_bounds.append(region_bound)
        return fwhm_bounds
