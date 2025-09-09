"""
Segmented Upper Hull (SUH) continuum removal following Clark et al. (1987).

Author: Leander Leist (2025)
Based on: Clark, R.N., King, T.V.V., and Gorelick, N.S. (1987)
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def segmented_upper_hull(reflectance, wavelengths=None, return_type="band_depth"):
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
        - "upper_hull": List of (wavelength, reflectance) hull points
        - "all": Dictionary containing all above outputs

    Returns
    -------
    array-like or dict
    """

    if wavelengths is None:
        wavelengths = np.arange(len(reflectance))
    elif isinstance(wavelengths, (np.ndarray, list)):
        pass  # array or list
    elif hasattr(reflectance, 'index'):
        wavelengths = reflectance.index.values  # For Series
    elif hasattr(reflectance, 'columns'):
        wavelengths = reflectance.columns.values  # For DataFrame
    else:
        raise ValueError("Cannot auto-detect wavelengths")

    if len(reflectance) != len(wavelengths):
        raise ValueError("Reflectance and wavelengths must have same length")

    # Convert to numpy arrays for consistent handling
    reflectance = np.asarray(reflectance)
    wavelengths = np.asarray(wavelengths)

    # Step 1: Find global maximum
    global_max_idx = np.argmax(reflectance)
    #global_max_wl = wavelengths[global_max_idx]
    #global_max_refl = reflectance[global_max_idx]

    # Step 2: Split spectrum at global maximum
    left_wl = wavelengths[:global_max_idx + 1]  # From start to global max (inclusive)
    left_refl = reflectance[:global_max_idx + 1]

    right_wl = wavelengths[global_max_idx:]  # From global max to end (inclusive)
    right_refl = reflectance[global_max_idx:]

    # Step 3: Find local maxima for each side iteratively
    def _find_next_local_max(refl, start_idx=0):
        """Find next actual local maximum beyond start_idx."""

        # Find peaks in current segment
        remaining_refl = refl[start_idx + 1:]
        if len(remaining_refl) < 3:  # Need at least 3 points for a peak
            return None

        # Add prominence filter to avoid detecting noise as peak
        min_prominence = np.ptp(refl) * 0.01  # 1% of the range
        peaks, properties = find_peaks(remaining_refl, prominence=min_prominence)

        if len(peaks) == 0:
            return None

        # Return first peak found, adjusted for full array indexing
        return peaks[0] + start_idx + 1

    def process_segment(wl, refl, current_idx, next_idx, threshold=1.001):
        """
        Check if a segment from current_idx to next_idx has points above the continuum line.

        Returns
        -------
        has_peaks : bool
            Whether there are points above the continuum line
        first_peak_idx : int or None
            Index of the first point above the line, or None if no peaks
        """
        current_wl, current_refl = wl[current_idx], refl[current_idx]
        next_wl, next_refl = wl[next_idx], refl[next_idx]

        # Extract segment
        segment_wl = wl[current_idx:next_idx + 1]
        segment_refl = refl[current_idx:next_idx + 1]

        # Calculate continuum line for this segment
        continuum_line = current_refl + (next_refl - current_refl) * (segment_wl - current_wl) / (next_wl - current_wl)

        # Check for points above the line
        tolerance = 1e-10  # Avoid numerical issues
        continuum_removed = segment_refl / (continuum_line + tolerance)

        # check for points above threshold
        relative_peaks = continuum_removed > threshold
        peak_indices = np.where(relative_peaks)[0]

        if len(peak_indices) > 0:
            first_peak_idx = peak_indices[0]
            # Make sure it's not the current point or the next point
            if first_peak_idx > 0 and first_peak_idx < len(segment_wl) - 1:
                return True, current_idx + first_peak_idx

        return False, None

    def find_tiepoints(wl_segment, refl_segment, direction="right"):
        """Find continuum tie points using Clark's iterative algorithm."""

        # left -right side handling
        if direction == "left":
            wl = wl_segment[::-1]
            refl = refl_segment[::-1]
        else:
            wl = wl_segment
            refl = refl_segment

        tiepoints = [(wl[0], refl[0])]
        current_idx = 0

        while current_idx < len(wl) - 1:
            # Find the next local maximum in the current segment
            next_max_idx = _find_next_local_max(refl, start_idx=current_idx)

            if next_max_idx is None:
                # No more peaks found - use endpoint as next target
                next_max_idx = len(wl) - 1

            # Process the segment (same logic for peaks and endpoints)
            has_peaks, first_peak_idx = process_segment(wl, refl, current_idx, next_max_idx)

            if has_peaks:
                # Found a point above the line -> add it and continue from there
                tiepoints.append((wl[first_peak_idx], refl[first_peak_idx]))
                current_idx = first_peak_idx
            else:
                # No points above the line -> accept this segment
                tiepoints.append((wl[next_max_idx], refl[next_max_idx]))
                current_idx = next_max_idx

            # Break at  endpoint
            if current_idx >= len(wl) - 1:
                break

        if direction == "left":
            tiepoints = tiepoints[::-1]

        return tiepoints

    right_tiepoints = find_tiepoints(right_wl, right_refl, direction="right")
    left_tiepoints = find_tiepoints(left_wl, left_refl, direction="left")

    # Combine tiepoints (remove duplicate global max)
    all_tiepoints = left_tiepoints + right_tiepoints[1:]

    # Sort by wavelength
    all_tiepoints.sort(key=lambda x: x[0])

    # construct tiepoint xy-pais
    hull_wl = np.array([pt[0] for pt in all_tiepoints])
    hull_refl = np.array([pt[1] for pt in all_tiepoints])
    hull = np.column_stack([hull_wl, hull_refl])

    # Interpolate continuum line
    continuum_interp = interp1d(hull_wl, hull_refl, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
    continuum_line = continuum_interp(wavelengths)

    # Calculate continuum removed and band depth
    continuum_removed = reflectance / continuum_line
    band_depth = 1 - continuum_removed

    if return_type == "band_depth":
        return band_depth
    if return_type == "continuum_removed":
        return continuum_removed
    if return_type == "continuum_line":
        return continuum_line
    if return_type == "hull":
        return hull
    if return_type == "all":
        return {
            'continuum_line': continuum_line,
            'continuum_removed': continuum_removed,
            'band_depth': band_depth,
            'hull': hull,
        }