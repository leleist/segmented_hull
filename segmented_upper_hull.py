"""
Segmented Upper Hull (SUH) continuum removal following Clark et al. (1987).

Author: Leander Leist (2025)
Based on: Clark, R.N., King, T.V.V., and Gorelick, N.S. (1987)
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def segmented_upper_hull(reflectance, wavelengths = "auto", return_type = "band_depth"):
    """
    Segmented Upper Hull (SUH) continuum removal following Clark et al. (1987).

    Identifies local maxima and constructs a piecewise linear continuum without
    enforcing convexity, allowing extraction of more detailed absorption features
    compared to standard convex hull methods.

    References
    ----------
    Clark, R.N., King, T.V.V., and Gorelick, N.S. (1987). Automatic continuum
    analysis of reflectance spectra. Proceedings of the Third Airborne Imaging
    Spectrometer Data Analysis Workshop, 138-142.
    https://ntrs.nasa.gov/citations/19880004388
    https://ntrs.nasa.gov/api/citations/19880004388/downloads/19880004388.pdf

    SUH was also implemented in hsdar package in R.
    https://rdrr.io/cran/hsdar/f/inst/doc/Hsdar-intro.pdf

    Author
    ------
    Implementation by Leander Leist (2025)
    Based on Clark et al. (1987)

    Parameters
    ----------
    reflectance : array-like
        Reflectance values (typically 0-1 range). Can be pandas Series,
        numpy array, or list.
    wavelengths : array-like or "auto", default "auto"
        Wavelength values in nanometers. If "auto", extracts from
        reflectance.index (for Series) or reflectance.columns (for DataFrame).
    return_type : str, default "band_depth"
        Type of output to return:
        - "band_depth": Absorption depth (1 - continuum_removed)
        - "continuum_removed": Normalized reflectance (reflectance / continuum)
        - "continuum_line": Interpolated segmented hull values
        - "tiepoints": List of (wavelength, reflectance) hull points
        - "all": Dictionary containing all above outputs

    Returns
    -------
    array-like or dict
    """
    from scipy.signal import find_peaks

    if wavelengths == "auto":
        if hasattr(reflectance, 'index'):
            wavelengths = reflectance.index.values  # For Series
        elif hasattr(reflectance, 'columns'):
            wavelengths = reflectance.columns.values  # For DataFrame
        else:
            raise ValueError("Cannot auto-detect wavelengths")

    if len(reflectance) != len(wavelengths):
        raise ValueError("Reflectance and wavelengths must have same length")

    # Step 1: Find global maximum
    global_max_idx = np.argmax(reflectance)
    global_max_wl = wavelengths[global_max_idx]
    global_max_refl = reflectance[global_max_idx]

    # Step 2: Split spectrum at global maximum
    left_wl = wavelengths[:global_max_idx + 1]  # From start to global max (inclusive)
    left_refl = reflectance[:global_max_idx + 1]

    right_wl = wavelengths[global_max_idx:]  # From global max to end (inclusive)
    right_refl = reflectance[global_max_idx:]

    # step 3: Find local maxima each side iteratively
    def _find_next_local_max(refl, start_idx=0):
        """Find next actual local maximum beyond start_idx."""

        # Find all peaks in the remaining segment
        remaining_refl = refl[start_idx + 1:]
        if len(remaining_refl) < 3:  # Need at least 3 points for a peak
           return None

        peaks, _ = find_peaks(remaining_refl)  # No prominence filter

        if len(peaks) == 0:
           return None

        # Return first peak found, adjusted for full array indexing
        return peaks[0] + start_idx + 1

    def find_tiepoints(wl_segment, refl_segment, direction="right"):
        """Find continuum tie points using Clark's iterative algorithm."""

        if direction == "left":
            wl = wl_segment[::-1]
            refl = refl_segment[::-1]
        else:
            wl = wl_segment
            refl = refl_segment

        tiepoints = [(wl[0], refl[0])]
        current_idx = 0

        while current_idx < len(wl) - 1:
            # Find the next local maximum in the remaining segment
            next_max_idx = _find_next_local_max(refl, start_idx=current_idx)

            if next_max_idx is None:
                tiepoints.append((wl[-1], refl[-1]))
                break

            # Extract current and next continuum tie points
            current_wl, current_refl = wl[current_idx], refl[current_idx]
            next_wl, next_refl = wl[next_max_idx], refl[next_max_idx]

            #subset
            segment_wl = wl[current_idx:next_max_idx + 1]
            segment_refl = refl[current_idx:next_max_idx + 1]

            #draw the line and calculate the continuum removed values
            continuum_line = current_refl + (next_refl - current_refl) * (segment_wl - current_wl) / (
                        next_wl - current_wl)
            continuum_removed = segment_refl / continuum_line

            # if there are any relative peaks (>1.0), restart from the first one i.e., set new tie point
            relative_peaks = continuum_removed > 1.0
            peak_indices = np.where(relative_peaks)[0]

            if len(peak_indices) > 0:
                # Restart from first relative peak
                first_peak_idx = peak_indices[0]
                if first_peak_idx > 0 and first_peak_idx < len(segment_wl) - 1:
                    global_peak_idx = current_idx + first_peak_idx
                    tiepoints.append((wl[global_peak_idx], refl[global_peak_idx]))
                    current_idx = global_peak_idx
                else:
                    current_idx = next_max_idx
            else:
                # No relative peaks, move to next max
                current_idx = next_max_idx

        if direction == "left":
            tiepoints = tiepoints[::-1]

        return tiepoints

    right_tiepoints = find_tiepoints(right_wl, right_refl, direction="right")
    left_tiepoints = find_tiepoints(left_wl, left_refl, direction="left")

    # Combine tiepoints (remove duplicate global max)
    all_tiepoints = left_tiepoints + right_tiepoints[1:]

    # sort by wavelength
    all_tiepoints.sort(key=lambda x: x[0])

    # Extract wavelengths and reflectances from tiepoints
    hull_wl = np.array([pt[0] for pt in all_tiepoints])
    hull_refl = np.array([pt[1] for pt in all_tiepoints])

    # Interpolate to create continuum line
    from scipy.interpolate import interp1d
    continuum_interp = interp1d(hull_wl, hull_refl, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
    continuum_line = continuum_interp(wavelengths)

    # Calculate continuum removed and band depth
    continuum_removed = reflectance / continuum_line
    band_depth = 1 - continuum_removed

    # Return results
    if return_type == "band_depth":
        return band_depth
    if return_type == "continuum_removed":
        return continuum_removed
    if return_type == "continuum_line":
        return continuum_line
    if return_type == "tiepoints":
        return all_tiepoints
    if return_type == "all":
        return {
            'segmented_hull': continuum_line,
            'continuum_removed': continuum_removed,
            'band_depth': band_depth,
            'tiepoints': all_tiepoints,
        }
