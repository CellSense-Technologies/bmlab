"""
Process-safe fitting of all spectra of one measurement point.

The functions in this module only work on plain numerical arrays and a
picklable settings snapshot. They never touch the Session, Qt objects or
open HDF5 handles. Hence they can be imported and called in the
evaluation process as well as in worker processes.
"""
import copy
import warnings
from dataclasses import dataclass

import numpy as np

from bmlab.fits import fit_lorentz_region


@dataclass(frozen=True)
class PointFitSettings:
    """
    Snapshot of all settings that influence the fits of one point.

    Create it with `PointFitSettings.snapshot`, which copies the values,
    so that later changes of the models do not affect a running evaluation.

    Attributes
    ----------
    nr_brillouin_peaks: int
        Number of peaks of the multi-peak fit (1: only single-peak fit)
    brillouin_regions: list
        Brillouin regions in Hz, one (start, end) per region
    rayleigh_regions: list
        Rayleigh regions in Hz, one (start, end) per region
    bounds_w0: list or None
        Peak position bounds of the multi-peak fit as given by the user,
        e.g. [['min', '5'], ['5.5', 'Inf']]
    bounds_fwhm: list or None
        Peak width bounds of the multi-peak fit as given by the user
    """
    nr_brillouin_peaks: int
    brillouin_regions: list
    rayleigh_regions: list
    bounds_w0: list = None
    bounds_fwhm: list = None

    @classmethod
    def snapshot(cls, nr_brillouin_peaks, brillouin_regions,
                 rayleigh_regions, bounds_w0=None, bounds_fwhm=None):
        return cls(
            nr_brillouin_peaks=nr_brillouin_peaks,
            brillouin_regions=copy.deepcopy(brillouin_regions),
            rayleigh_regions=copy.deepcopy(rayleigh_regions),
            bounds_w0=copy.deepcopy(bounds_w0),
            bounds_fwhm=copy.deepcopy(bounds_fwhm),
        )


def fit_point(spectra, frequencies, settings):
    """
    Fit all spectra (frames) of one measurement point.

    All fits of a point are kept together: first the single-peak fits
    of all Brillouin and Rayleigh regions, then the bounds of the
    multi-peak fit, which depend on the fitted Rayleigh peak positions
    of this point, and finally the multi-peak fits (if requested).

    Parameters
    ----------
    spectra: sequence of np.ndarray
        The spectrum of every frame
    frequencies: sequence of np.ndarray
        The frequency axis in Hz of every frame
    settings: PointFitSettings
        The fit settings

    Returns
    -------
    results: dict
        The fit results of this point with the keys of the results arrays
        of the evaluation model. The values are float arrays holding one
        entry per frame in the first dimension. They have the layout of
        the results arrays of the evaluation model for a single point
        (with the dimensions x, y, z removed):
        - brillouin_peak_position_f, brillouin_peak_fwhm_f,
          brillouin_peak_intensity, brillouin_peak_offset with the shape
          (frames, Brillouin regions, peaks). Peak 0 is the single-peak
          fit, peaks 1 to nr_brillouin_peaks the multi-peak fit.
        - rayleigh_peak_position_f, rayleigh_peak_fwhm_f,
          rayleigh_peak_intensity, rayleigh_peak_offset with the shape
          (frames, Rayleigh regions, 1)
        Entries without a valid fit are NaN.
    """
    nr_brillouin_peaks = settings.nr_brillouin_peaks
    brillouin_regions = settings.brillouin_regions
    rayleigh_regions = settings.rayleigh_regions
    nr_frames = len(spectra)

    # Same number of peaks as EvaluationModel.initialize_results_arrays
    # stores: the single-peak fit, plus the multi-peak fit if requested.
    nr_peaks_to_store = nr_brillouin_peaks
    if nr_brillouin_peaks > 1:
        nr_peaks_to_store = nr_peaks_to_store + 1

    shape_brillouin = (nr_frames, len(brillouin_regions), nr_peaks_to_store)
    shape_rayleigh = (nr_frames, len(rayleigh_regions), 1)

    brillouin_keys = ['brillouin_peak_position_f', 'brillouin_peak_fwhm_f',
                      'brillouin_peak_intensity', 'brillouin_peak_offset']
    rayleigh_keys = ['rayleigh_peak_position_f', 'rayleigh_peak_fwhm_f',
                     'rayleigh_peak_intensity', 'rayleigh_peak_offset']
    results = {}
    for key in brillouin_keys:
        results[key] = np.full(shape_brillouin, np.nan)
    for key in rayleigh_keys:
        results[key] = np.full(shape_rayleigh, np.nan)

    for region_key, region in enumerate(brillouin_regions):
        fits = fit_spectra(spectra, frequencies, region)
        for frame_num, _ in enumerate(spectra):
            ind = (frame_num, region_key, 0)
            for fit_idx, key in enumerate(brillouin_keys):
                results[key][ind] = fits[frame_num][fit_idx]

    for region_key, region in enumerate(rayleigh_regions):
        fits = fit_spectra(spectra, frequencies, region)
        for frame_num, _ in enumerate(spectra):
            # The negative index is inherited from the original
            # implementation, it is not a typo.
            ind = (frame_num, region_key - len(brillouin_regions))
            for fit_idx, key in enumerate(rayleigh_keys):
                results[key][ind] = fits[frame_num][fit_idx]

    # We can only do a multi-peak fit after the single-peak
    # Rayleigh fit is done, because we have to know the
    # Rayleigh peak positions in GHz in order to convert
    # the multi-peak fit bounds given in GHz into the position
    # in pixels.
    if nr_brillouin_peaks > 1:
        rayleigh_peaks = np.transpose(
            results['rayleigh_peak_position_f'][:, :, 0]
        )
        bounds_w0 = create_bounds(
            settings.bounds_w0,
            brillouin_regions,
            rayleigh_peaks
        )
        bounds_fwhm = create_bounds_fwhm(
            settings.bounds_fwhm,
            brillouin_regions,
            rayleigh_peaks
        )

        for region_key, region in enumerate(brillouin_regions):
            results_multi_peak = fit_spectra(
                spectra,
                frequencies,
                region,
                nr_brillouin_peaks,
                bounds_w0[region_key],
                bounds_fwhm[region_key]
            )
            for frame_num, _ in enumerate(spectra):
                ind = (frame_num, region_key,
                       slice(1, nr_brillouin_peaks + 1))
                for fit_idx, key in enumerate(brillouin_keys):
                    results[key][ind] = \
                        results_multi_peak[frame_num][fit_idx]

    return results


def fit_spectra(spectra, frequencies, region, nr_peaks=1,
                bounds_w0=None, bounds_fwhm=None):
    """
    Fit the given region of every spectrum with a Lorentzian model.

    Returns a list with one fit result per spectrum, see
    bmlab.fits.fit_lorentz_region.
    """
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


def create_bounds(bounds, brillouin_regions, rayleigh_peaks):
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
    bounds
    brillouin_regions
    rayleigh_peaks

    Returns
    -------

    """
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


def create_bounds_fwhm(bounds_fwhm, brillouin_regions,
                       rayleigh_peaks):
    """
    This function converts the fwhm bounds settings into
    a fwhm bounds object for the fitting function
    Allowed parameters for the bounds settings are
    - 'min'/'max'   -> Will be converted to 0/np.inf
    - '-Inf', 'Inf' -> Will be converted to 0 or np.inf
    - number [GHz]  -> Will be converted into the value in Hz
    Parameters
    ----------
    bounds_fwhm
    brillouin_regions
    rayleigh_peaks

    Returns
    -------

    """
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
