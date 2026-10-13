"""Reading two-column XRD data from files or uploaded byte streams.

Two shapes are accepted throughout this package:
  - "sample" and "reference" files: two columns, ``2theta intensity``, separated
    by whitespace, tabs, commas or semicolons (so .xy / .txt / .dat / .csv all
    work). Non-numeric lines are skipped rather than raising, since exports
    often carry a header line.
  - ``.xrdml`` files (PANalytical XML export): the 2theta scan, intensities and
    sample name are read directly out of the XML, so no separate .xy
    conversion step is needed.

All readers accept a path, an open binary/text file, or anything with a
``.read()`` method (Streamlit's UploadedFile satisfies this), so the same code
serves the CLI and the GUI.
"""
import re
import xml.etree.ElementTree as ET

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
    """Parse a two-column ``2theta intensity`` pattern in any common text layout.

    Columns may be separated by spaces, tabs, commas or semicolons (so .xy,
    .txt, .dat and .csv all work). Header, comment and blank lines are skipped,
    and any columns beyond the first two (e.g. an error column) are ignored.
    Returns (angles, intensities) as float arrays, sorted by angle.
    """
    angles, intensities = [], []
    for line in _decode_lines(source):
        parts = re.sub(r"[,;	]", " ", line.replace("﻿", "")).split()
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


load_reference_csv = load_xy


def load_xrdml(source):
    """Parse a PANalytical .xrdml file: returns (angles, intensities, sample_name).

    Reads the 2theta positions and counts straight out of the XML -- no
    separate .xy conversion needed. `sample_name` is the sample <id> (falling
    back to <name>, then to the first <id> anywhere in the file) and is None
    if the file has neither, in which case callers should fall back to the
    filename.

    Only the first continuous 2theta scan in the file is read -- multi-range
    or multi-axis measurements aren't supported.
    """
    tree = ET.parse(source)
    root = tree.getroot()
    ns_uri = root.tag.split("}")[0].strip("{") if root.tag.startswith("{") else ""

    def tag(name):
        return "{%s}%s" % (ns_uri, name) if ns_uri else name

    sample_name = None
    sample_el = root.find(".//" + tag("sample"))
    if sample_el is not None:
        for child_tag in ("id", "name"):
            child = sample_el.find(tag(child_tag))
            if child is not None and child.text and child.text.strip():
                sample_name = child.text.strip()
                break
    if sample_name is None:
        id_el = root.find(".//" + tag("id"))
        if id_el is not None and id_el.text and id_el.text.strip():
            sample_name = id_el.text.strip()

    positions_el, intensities_el = None, None
    for dp in root.findall(".//" + tag("dataPoints")):
        pos_all = dp.findall(tag("positions"))
        chosen = next((p for p in pos_all if "2th" in p.get("axis", "").lower()), None)
        if chosen is None and pos_all:
            chosen = pos_all[0]
        counts = dp.find(tag("intensities"))
        if counts is None:
            counts = dp.find(tag("counts"))
        if chosen is not None and counts is not None and counts.text:
            positions_el, intensities_el = chosen, counts
            break

    if positions_el is None or intensities_el is None:
        raise ValueError(
            "couldn't find a 2theta scan with intensities in this .xrdml -- "
            "is it a continuous single-range measurement?"
        )

    start = float(positions_el.find(tag("startPosition")).text)
    end = float(positions_el.find(tag("endPosition")).text)
    intensities = np.array([float(v) for v in intensities_el.text.split()])
    if len(intensities) < 2:
        raise ValueError("fewer than 2 intensity points found in this .xrdml")
    angles = np.linspace(start, end, len(intensities))
    return angles, intensities, sample_name


def name_from_upload(uploaded_file, strip_suffixes=(".csv", ".xy", ".txt", ".dat", ".xrdml")):
    name = getattr(uploaded_file, "name", str(uploaded_file))
    for suf in strip_suffixes:
        if name.lower().endswith(suf):
            return name[: -len(suf)]
    return name
