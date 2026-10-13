import io

import numpy as np
import pytest

from xrd_phase_id.io_utils import load_reference_csv, load_xrdml, load_xy

MINIMAL_XRDML = """<?xml version="1.0" encoding="UTF-8"?>
<xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/1.5">
  <sample type="To be analyzed">
    <id>demo sample A</id>
  </sample>
  <xrdMeasurement measurementType="Scan" status="Completed">
    <scan appendNumber="0" mode="Continuous" scanAxis="Gonio" status="Completed">
      <dataPoints>
        <positions axis="2Theta" unit="deg">
          <startPosition>10.0</startPosition>
          <endPosition>10.4</endPosition>
        </positions>
        <intensities unit="counts">100 110 120 115 105</intensities>
      </dataPoints>
    </scan>
  </xrdMeasurement>
</xrdMeasurements>
"""


def test_load_xy_skips_non_numeric_lines():
    text = "# header\n10.0 5.0\n10.1 6.0\nnot a number\n10.2 7.0\n"
    angles, intensities = load_xy(io.StringIO(text))
    assert list(angles) == [10.0, 10.1, 10.2]
    assert list(intensities) == [5.0, 6.0, 7.0]


def test_load_reference_csv_tolerates_header_row():
    text = "2theta,intensity\n10.0,1.0\n10.1,2.0\n"
    angles, intensities = load_reference_csv(io.StringIO(text))
    assert list(angles) == [10.0, 10.1]
    assert list(intensities) == [1.0, 2.0]


def test_load_xrdml_extracts_scan_and_sample_name():
    angles, intensities, name = load_xrdml(io.StringIO(MINIMAL_XRDML))
    assert name == "demo sample A"
    assert len(angles) == 5
    assert angles[0] == pytest.approx(10.0)
    assert angles[-1] == pytest.approx(10.4)
    assert list(intensities) == [100, 110, 120, 115, 105]


def test_load_xrdml_without_scan_raises():
    bad = "<xrdMeasurements><sample><id>x</id></sample></xrdMeasurements>"
    with pytest.raises(ValueError):
        load_xrdml(io.StringIO(bad))


@pytest.mark.parametrize("text", [
    "10.0;5\n10.1;6\n",
    "2theta\tcounts\n10.0\t5\n10.1\t6\n",
    "﻿angles,intensities\n10.0,5\n10.1,6\n",
    "x,y,err\n10.0,5,0.1\n10.1,6,0.1\n",
])
def test_pattern_reader_accepts_common_delimiters(text):
    for loader in (load_xy, load_reference_csv):
        angles, intensities = loader(io.StringIO(text))
        assert list(angles) == [10.0, 10.1]
        assert list(intensities) == [5.0, 6.0]
