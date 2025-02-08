import numpy as np
from scipy.signal import find_peaks

def calculate_fwhm(x, y):
    """Calculate Full Width at Half Maximum (FWHM) for peaks in y(x)."""
    y = np.array(y)
    x = np.array(x)
    y -= np.min(y)
    y_max = np.max(y)

    if y_max == 0:
        return []

    peaks, _ = find_peaks(y, height=0.5 * y_max)
    fwhm_results = []

    for peak_idx in peaks:
        peak_x = x[peak_idx]
        peak_y = y[peak_idx]
        half_max = peak_y / 2

        left_indices = np.where(y[:peak_idx] <= half_max)[0]
        left_idx = left_indices[-1] if len(left_indices) > 0 else 0
        left_x = np.interp(half_max, [y[left_idx], y[left_idx + 1]], [x[left_idx], x[left_idx + 1]])

        right_indices = np.where(y[peak_idx:] <= half_max)[0]
        right_idx = right_indices[0] + peak_idx if len(right_indices) > 0 else len(y) - 1
        right_x = np.interp(half_max, [y[right_idx - 1], y[right_idx]], [x[right_idx - 1], x[right_idx]])

        fwhm = abs(right_x - left_x)
        fwhm_results.append((peak_x, fwhm))

    return fwhm_results
