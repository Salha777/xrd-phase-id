"""Reading two-column XRD data from files or uploaded byte streams.

Two shapes are accepted throughout this package:
  - "sample" files: whitespace-separated ``2theta  intensity`` (.xy convention),
    one point per line. Non-numeric lines are skipped rather than raising, since
    instrument-exported .xy files sometimes carry a stray header line.
  - "reference" files: comma-separated ``2theta,intensity``.

Both readers accept a path, an open binary/text file, or anything with a
``.read()`` method (Streamlit's UploadedFile satisfies this), so the same code
serves the CLI and the GUI.
"""
import io

import numpy as np


def _decode_lines(source):
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        return data.splitlines()
    with open(source, "r", encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


def load_xy(source):
    """Parse whitespace-separated ``2theta intensity`` pairs.

    Returns (angles, intensities) as float arrays, sorted by angle.
    """
    angles, intensities = [], []
    for line in _decode_lines(source):
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            angles.append(float(parts[0]))
            intensities.append(float(parts[1]))
        except ValueError:
            continue
    if not angles:
        raise ValueError("no numeric (2theta, intensity) pairs found")
    order = np.argsort(angles)
    return np.asarray(angles, float)[order], np.asarray(intensities, float)[order]


def load_reference_csv(source):
    """Parse a comma-separated ``2theta,intensity`` reference pattern."""
    lines = [ln for ln in _decode_lines(source) if ln.strip()]
    rows = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue  # tolerate a header row
    if not rows:
        raise ValueError("no numeric (2theta,intensity) rows found")
    rows.sort(key=lambda t: t[0])
    arr = np.asarray(rows, float)
    return arr[:, 0], arr[:, 1]


def name_from_upload(uploaded_file, strip_suffixes=(".csv", ".xy", ".txt", ".dat")):
    name = getattr(uploaded_file, "name", str(uploaded_file))
    for suf in strip_suffixes:
        if name.lower().endswith(suf):
            return name[: -len(suf)]
    return name
