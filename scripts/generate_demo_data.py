"""Generate the synthetic demo dataset shipped under demo_data/.

These are NOT measured or RRUFF-derived patterns. They are Gaussian peaks
placed at commonly published Cu-Kalpha reflection positions for a few textbook
minerals (calcite, quartz, halite), purely so the app has something to fit on
first launch. Do not use them for real phase identification -- bring your own
reference patterns for that (see README).

Run once from the repo root: `python scripts/generate_demo_data.py`
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "demo_data"))
REFS = os.path.join(OUT, "references")

ANGLES = np.arange(5.0, 90.0 + 1e-9, 0.02)

# name -> [(2theta, relative_intensity, fwhm_deg), ...] -- approximate, illustrative only
PHASES = {
    "calcite_demo": [
        (23.0, 15, 0.12), (29.4, 100, 0.10), (35.9, 15, 0.12), (39.4, 20, 0.12),
        (43.1, 15, 0.12), (47.1, 20, 0.13), (48.5, 25, 0.13),
    ],
    "quartz_demo": [
        (20.9, 25, 0.10), (26.6, 100, 0.09), (36.5, 12, 0.12), (39.5, 8, 0.12),
        (40.3, 7, 0.12), (50.1, 12, 0.13),
    ],
    "halite_demo": [
        (31.7, 100, 0.11), (45.4, 55, 0.13), (56.5, 15, 0.14), (66.2, 5, 0.15),
        (75.3, 6, 0.16),
    ],
}


def gaussian_pattern(peaks, angles=ANGLES, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros_like(angles)
    for center, height, fwhm in peaks:
        sigma = fwhm / 2.3548
        y += height * np.exp(-0.5 * ((angles - center) / sigma) ** 2)
    y += 2.0 + 0.02 * (angles - angles[0])          # gentle sloping background
    y += rng.normal(0, 0.3, size=angles.shape)       # detector noise
    return np.clip(y, 0, None)


def main():
    os.makedirs(REFS, exist_ok=True)
    patterns = {}
    for i, (name, peaks) in enumerate(PHASES.items()):
        y = gaussian_pattern(peaks, seed=i)
        patterns[name] = y
        path = os.path.join(REFS, f"{name}.csv")
        with open(path, "w") as f:
            for a, v in zip(ANGLES, y):
                f.write(f"{a:.3f},{v:.4f}\n")
        print(f"wrote {path}")

    # demo sample: ~70% calcite + 30% quartz, independent noise realisation
    rng = np.random.default_rng(99)
    mix = 0.7 * gaussian_pattern(PHASES["calcite_demo"], seed=100) + \
          0.3 * gaussian_pattern(PHASES["quartz_demo"], seed=101)
    mix += rng.normal(0, 0.2, size=ANGLES.shape)
    mix = np.clip(mix, 0, None)
    sample_path = os.path.join(OUT, "sample_pattern_demo.xy")
    with open(sample_path, "w") as f:
        for a, v in zip(ANGLES, mix):
            f.write(f"{a:.3f} {v:.4f}\n")
    print(f"wrote {sample_path}  (synthetic ~70% calcite_demo / 30% quartz_demo mixture)")


if __name__ == "__main__":
    main()
