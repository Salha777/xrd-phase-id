import numpy as np
import pytest

from xrd_phase_id.fit import area_normalise, fit_sample
from xrd_phase_id.sanity import check_sample, find_peaks

ANGLES = np.arange(5.0, 90.0, 0.02)


def gaussian(angles, peaks):
    y = np.zeros_like(angles)
    for center, height, fwhm in peaks:
        sigma = fwhm / 2.3548
        y += height * np.exp(-0.5 * ((angles - center) / sigma) ** 2)
    return y + 2.0


PHASE_A = [(20.0, 100, 0.1), (30.0, 40, 0.1)]
PHASE_B = [(25.0, 100, 0.1), (45.0, 60, 0.1)]


def test_area_normalise_sums_to_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    normed = area_normalise(y)
    assert np.isclose(np.sum(np.abs(normed)), 1.0)


def test_area_normalise_handles_all_zero():
    y = np.zeros(5)
    assert np.array_equal(area_normalise(y), y)


def test_fit_recovers_known_mixture():
    sample = 0.6 * gaussian(ANGLES, PHASE_A) + 0.4 * gaussian(ANGLES, PHASE_B)
    references = {
        "A": (ANGLES, gaussian(ANGLES, PHASE_A)),
        "B": (ANGLES, gaussian(ANGLES, PHASE_B)),
    }
    result = fit_sample(ANGLES, sample, references)
    assert result.weights["A"] == pytest.approx(0.6, abs=0.05)
    assert result.weights["B"] == pytest.approx(0.4, abs=0.05)
    assert result.residual_rms < 1e-3


def test_fit_rejects_pattern_shorter_than_sg_window():
    short_angles = ANGLES[:10]
    short_sample = gaussian(short_angles, PHASE_A)
    references = {"A": (ANGLES, gaussian(ANGLES, PHASE_A))}
    with pytest.raises(ValueError):
        fit_sample(short_angles, short_sample, references, sg_window=31)


def test_find_peaks_detects_isolated_peak():
    y = gaussian(ANGLES, [(40.0, 50, 0.15)])
    peaks, ratio = find_peaks(ANGLES, y)
    assert len(peaks) == 1
    assert peaks[0][0] == pytest.approx(40.0, abs=0.1)
    assert ratio > 1


def test_check_sample_flags_missing_phase():
    sample = gaussian(ANGLES, PHASE_A + [(60.0, 80, 0.1)])  # extra peak, not in references
    references = {"A": (ANGLES, gaussian(ANGLES, PHASE_A))}
    result = check_sample(ANGLES, sample, references)
    assert result.verdict == "missing_phase"


def test_check_sample_ok_when_all_peaks_matched():
    sample = gaussian(ANGLES, PHASE_A)
    references = {"A": (ANGLES, gaussian(ANGLES, PHASE_A))}
    result = check_sample(ANGLES, sample, references)
    assert result.verdict == "ok"
