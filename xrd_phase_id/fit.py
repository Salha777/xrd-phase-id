"""The NNLS phase fit itself.

Method (unchanged from the lab tool this was extracted from): area-normalise
the sample and every reference pattern, take a Savitzky-Golay first derivative
of each (robust to a sloping background and to arbitrary intensity scaling),
then fit the sample derivative as a non-negative least-squares mixture of the
reference derivatives. The fit coefficients are normalised to sum to 1 for
readability, but they are NNLS weights in derivative space, not quantitative
phase fractions -- see the "sanity check" module and the README for why that
distinction matters.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls
from scipy.signal import savgol_filter

DEFAULT_SG_WINDOW = 31
DEFAULT_SG_ORDER = 4


def area_normalise(y):
    area = np.sum(np.abs(y))
    return y / area if area > 0 else y


def sg_derivative(y, dx, window=DEFAULT_SG_WINDOW, order=DEFAULT_SG_ORDER):
    """First derivative via Savitzky-Golay, with window/order clamped to fit y."""
    window = min(window, len(y) - (1 - len(y) % 2))
    if window % 2 == 0:
        window -= 1
    window = max(window, order + 2 if (order + 2) % 2 else order + 3)
    order = min(order, window - 2)
    return savgol_filter(y, window_length=window, polyorder=order, deriv=1, delta=dx)


@dataclass
class FitResult:
    angles: np.ndarray
    sample_norm: np.ndarray
    reference_interp: dict          # name -> area-normalised, angle-matched intensities
    weights: dict                   # name -> fractional NNLS weight (sums to 1)
    reconstruction: np.ndarray
    residual: np.ndarray
    residual_rms: float
    sg_window: int
    sg_order: int
    warnings: list = field(default_factory=list)


def fit_sample(angles, intensities, references, sg_window=DEFAULT_SG_WINDOW,
               sg_order=DEFAULT_SG_ORDER):
    """Fit one sample pattern as a non-negative mixture of `references`.

    references: dict of name -> (ref_angles, ref_intensities). Each is
    interpolated onto the sample's angle grid before fitting, so references on
    a different step size or range are fine as long as they overlap it.
    """
    if len(angles) < sg_window + 2:
        raise ValueError(
            f"pattern has only {len(angles)} points, need at least "
            f"{sg_window + 2} for a Savitzky-Golay window of {sg_window} "
            "(lower the window in Advanced settings for short patterns)"
        )
    warnings = []
    dx = float(np.median(np.diff(angles)))
    norm_sample = area_normalise(intensities)
    deriv_sample = sg_derivative(norm_sample, dx, sg_window, sg_order)

    names = list(references.keys())
    ref_interp, cols = {}, []
    for name in names:
        ra, ri = references[name]
        lo, hi = max(angles[0], ra[0]), min(angles[-1], ra[-1])
        if lo >= hi:
            warnings.append(f"'{name}' does not overlap the sample's 2theta range -- skipped")
            continue
        interp = np.interp(angles, ra, ri, left=0.0, right=0.0)
        norm_ref = area_normalise(interp)
        ref_interp[name] = norm_ref
        cols.append(sg_derivative(norm_ref, dx, sg_window, sg_order))

    if not cols:
        raise ValueError("no reference pattern overlaps this sample's 2theta range")

    A = np.column_stack(cols)
    b = deriv_sample
    weights_raw, _ = nnls(A, b)

    total = weights_raw.sum()
    weights = weights_raw / total if total > 0 else weights_raw
    if total <= 0:
        warnings.append("all fit weights are zero -- the sample matches none of the references")

    reconstruction = sum(w * ref_interp[n] for w, n in zip(weights, ref_interp))
    if not isinstance(reconstruction, np.ndarray):
        reconstruction = np.zeros_like(norm_sample)
    residual = norm_sample - reconstruction
    residual_rms = float(np.sqrt(np.mean(residual ** 2)))

    return FitResult(
        angles=angles,
        sample_norm=norm_sample,
        reference_interp=ref_interp,
        weights=dict(zip(ref_interp, weights)),
        reconstruction=reconstruction,
        residual=residual,
        residual_rms=residual_rms,
        sg_window=sg_window,
        sg_order=sg_order,
        warnings=warnings,
    )
