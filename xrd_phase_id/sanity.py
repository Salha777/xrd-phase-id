"""Sanity-checking a fit independently of the fit itself.

NNLS always returns an answer -- it cannot say "none of these references
match". If the real phase is missing from the reference set, the weight goes
to whichever reference correlates best, and you get a confident-looking number
for a phase that isn't in the sample. These checks never look at the fit
weights; they only look at the raw pattern and the reference peak positions.

Peak finding matters here: low-angle air scatter is a smooth ramp several
times the height of a genuine minor reflection, so a naive "top N points"
search returns a ladder of background as if it were a phase. Instead this
estimates a rolling low-percentile background, subtracts it, and keeps local
maxima that clear the point-to-point noise by a set number of standard
deviations.
"""
from dataclasses import dataclass, field

import numpy as np

PEAK_TOL = 0.35     # degrees 2theta: how close a sample peak must be to a reference line
PEAK_FLOOR = 0.15   # ignore sample peaks below this fraction of the pattern's strongest peak
PEAK_SNR = 6.0      # net peak height must exceed this many sigma of point-to-point scatter
WEAK_RATIO = 6.0    # peak/background below this: pattern is near-amorphous, don't trust a fit


def _rolling_background(y, window):
    half = window // 2
    pad = np.pad(y, half, mode="edge")
    return np.array([np.percentile(pad[i:i + window], 10) for i in range(len(y))])


def find_peaks(angles, intensities):
    """Return (list of (2theta, net_height) sorted by angle, peak/background ratio)."""
    a = np.asarray(angles, float)
    y = np.asarray(intensities, float)
    step = float(np.median(np.diff(a)))
    window = max(11, int(round(3.0 / step)) | 1)
    bg = _rolling_background(y, window)
    net = y - bg
    noise = max(np.median(np.abs(np.diff(y))) / 0.954, 1e-9)
    peak_ratio = float(net.max()) / max(float(np.median(bg)), 1e-9)

    hi = float(net.max())
    candidates = [
        (a[i], net[i]) for i in range(1, len(net) - 1)
        if net[i] >= net[i - 1] and net[i] > net[i + 1]
        and net[i] > PEAK_SNR * noise and net[i] > 0.03 * hi
    ]
    candidates.sort(key=lambda t: -t[1])
    kept = []
    for q, v in candidates:  # merge lines within 0.4 deg of one already kept
        if all(abs(q - r) > 0.4 for r, _ in kept):
            kept.append((float(q), float(v)))
    return sorted(kept), peak_ratio


@dataclass
class SanityResult:
    peak_ratio: float
    peaks: list
    unmatched: list             # [(2theta, relative_height)] not explained by any reference line
    verdict: str                # "weak" | "missing_phase" | "minor_unmatched" | "ok" | "no_peaks"
    message: str


def check_sample(angles, intensities, references):
    """Compare a sample's peaks against every reference line, independent of the fit.

    references: dict of name -> (angles, intensities), same as fit.fit_sample.
    """
    ref_lines = []
    for ra, ri in references.values():
        pk, _ = find_peaks(ra, ri)
        ref_lines += [q for q, _ in pk]
    ref_lines = np.array(sorted(ref_lines)) if ref_lines else np.array([])

    peaks, ratio = find_peaks(angles, intensities)
    if not peaks:
        return SanityResult(ratio, [], [], "no_peaks",
                             "no peaks found at all -- amorphous or empty pattern")

    mx = max(v for _, v in peaks)
    unmatched = [
        (q, v / mx) for q, v in peaks
        if v / mx >= PEAK_FLOOR
        and (ref_lines.size == 0 or np.min(np.abs(ref_lines - q)) > PEAK_TOL)
    ]

    if ratio < WEAK_RATIO:
        verdict = "weak"
        message = (f"peak/background = {ratio:.1f} (below {WEAK_RATIO:.0f}): pattern is "
                   "near-amorphous. Weights are not meaningful -- report as too weak to identify.")
    elif not unmatched:
        verdict = "ok"
        message = f"peak/background = {ratio:.1f}. Every strong peak is matched by a reference line -- weights are usable."
    else:
        worst = max(r for _, r in unmatched)
        lines = " ".join(f"{q:.2f} deg ({r:.2f})" for q, r in unmatched[:6])
        if worst >= 0.5:
            verdict = "missing_phase"
            message = (f"peak/background = {ratio:.1f}. An unmatched peak at >=0.50 relative "
                       f"height means the dominant phase is probably not in the reference set: "
                       f"{lines}. The fit will name the best correlate, not the real phase.")
        else:
            verdict = "minor_unmatched"
            message = (f"peak/background = {ratio:.1f}. Minor peaks unmatched (majority call is "
                       f"usually still fine): {lines}")

    return SanityResult(ratio, peaks, unmatched, verdict, message)


def suggest_phases(unmatched, sample_peaks, candidates, top=3):
    """Score candidate references (not in the fit) against the unmatched peaks.

    candidates: dict of name -> (angles, intensities), disjoint from the fit set.
    Two conditions must both hold, or a dense line list would "explain" anything:
      explains -- fraction of the unmatched peak intensity within PEAK_TOL of a
                  candidate's own line
      present  -- fraction of the candidate's 5 strongest lines that are actually
                  present somewhere in the sample's peak list
    Returns up to `top` (score, name, explains, present) tuples, best first.
    """
    if not unmatched or not candidates:
        return []
    obs = np.array([q for q, _ in sample_peaks])
    wsum = sum(r for _, r in unmatched)
    hits = []
    for name, (ra, ri) in candidates.items():
        pk, _ = find_peaks(ra, ri)
        if not pk:
            continue
        lines = np.array([q for q, _ in pk])
        explains = sum(
            r for q, r in unmatched if np.min(np.abs(lines - q)) <= PEAK_TOL
        ) / max(wsum, 1e-9)
        strongest = [q for q, _ in sorted(pk, key=lambda t: -t[1])[:5]]
        present = sum(1 for q in strongest if np.min(np.abs(obs - q)) <= PEAK_TOL) / len(strongest)
        if explains >= 0.5 and present >= 0.6:
            hits.append((explains * present, name, explains, present))
    hits.sort(reverse=True)
    return hits[:top]
