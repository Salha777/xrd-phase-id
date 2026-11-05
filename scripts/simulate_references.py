"""Simulate Cu-Kalpha powder patterns for the shipped reference library.

Each pattern is calculated from a crystal structure with pymatgen's XRDCalculator and broadened
with a Gaussian (FWHM 0.10 deg 2theta), written as `2theta,intensity` on a 5-90 deg grid at
0.01 deg steps, strongest peak = 100.

Two sources of structures:

1. STRUCTURES below: textbook structures (space group, lattice constants, asymmetric unit)
   for common minerals. Crystal structures are facts, and the patterns are generated here, so
   the output files are released under CC0 1.0 along with the rest of reference_library/minerals.
   Values are standard literature parameters rounded to the precision shown; peak positions
   depend mainly on the lattice constants, intensities on the atomic positions.

2. --cif FILE ...: any CIF you have the right to use, e.g. from the Crystallography Open
   Database (crystallography.net, whose entries are in the public domain). Use this for phases
   with large or hydrated cells (nesquehonite, hydromagnesite, dypingite, ...).

    pip install pymatgen
    python scripts/simulate_references.py                       # rebuild the shipped library
    python scripts/simulate_references.py --cif my_phase.cif --out reference_library/cod
"""
import argparse
from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "reference_library" / "minerals"
GRID = np.round(np.arange(5.0, 90.0 + 1e-9, 0.01), 2)
FWHM = 0.10

# name: (space group, lattice kwargs, [species], [fractional coords])
STRUCTURES = {
    "calcite_CaCO3": ("R-3c", dict(a=4.990, c=17.061), ["Ca", "C", "O"],
                      [[0, 0, 0], [0, 0, 0.25], [0.2578, 0, 0.25]]),
    "magnesite_MgCO3": ("R-3c", dict(a=4.633, c=15.016), ["Mg", "C", "O"],
                        [[0, 0, 0], [0, 0, 0.25], [0.2774, 0, 0.25]]),
    "dolomite_CaMg(CO3)2": ("R-3", dict(a=4.807, c=16.010), ["Ca", "Mg", "C", "O"],
                            [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.2428], [0.2477, -0.0347, 0.2441]]),
    "siderite_FeCO3": ("R-3c", dict(a=4.692, c=15.380), ["Fe", "C", "O"],
                       [[0, 0, 0], [0, 0, 0.25], [0.2731, 0, 0.25]]),
    "quartz_SiO2": ("P3_221", dict(a=4.913, c=5.405), ["Si", "O"],
                    [[0.4697, 0, 0], [0.4135, 0.2669, 0.2141]]),
    "halite_NaCl": ("Fm-3m", dict(a=5.640), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    "sylvite_KCl": ("Fm-3m", dict(a=6.293), ["K", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    "periclase_MgO": ("Fm-3m", dict(a=4.212), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    "lime_CaO": ("Fm-3m", dict(a=4.811), ["Ca", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    "brucite_Mg(OH)2": ("P-3m1", dict(a=3.142, c=4.766), ["Mg", "O", "H"],
                        [[0, 0, 0], [1 / 3, 2 / 3, 0.2216], [1 / 3, 2 / 3, 0.4303]]),
    "portlandite_Ca(OH)2": ("P-3m1", dict(a=3.589, c=4.911), ["Ca", "O", "H"],
                            [[0, 0, 0], [1 / 3, 2 / 3, 0.2341], [1 / 3, 2 / 3, 0.4248]]),
}


def lattice(sg, kw):
    if "c" not in kw:
        return Lattice.cubic(kw["a"])
    return Lattice.hexagonal(kw["a"], kw["c"])


def pattern(structure):
    calc = XRDCalculator(wavelength="CuKa")
    p = calc.get_pattern(structure, two_theta_range=(GRID[0], GRID[-1]))
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    y = np.zeros_like(GRID)
    for t, i in zip(p.x, p.y):
        y += i * np.exp(-0.5 * ((GRID - t) / sigma) ** 2)
    return 100 * y / y.max(), p


def write(name, structure, out):
    y, p = pattern(structure)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    with open(path, "w") as fh:
        fh.write("2theta,intensity\n")
        for t, v in zip(GRID, y):
            fh.write(f"{t:.2f},{v:.4f}\n")
    top = sorted(zip(p.y, p.x), reverse=True)[:3]
    print(f"{path.relative_to(HERE.parent) if path.is_relative_to(HERE.parent) else path}: "
          f"strongest at " + ", ".join(f"{x:.2f}°" for _, x in top))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cif", nargs="*", help="simulate these CIFs instead of the built-in structures")
    ap.add_argument("--out", type=Path, default=None, help="output folder")
    a = ap.parse_args()
    if a.cif:
        out = a.out or HERE.parent / "reference_library" / "cif"
        for f in a.cif:
            write(Path(f).stem, Structure.from_file(f), out)
        return
    out = a.out or DEFAULT_OUT
    for name, (sg, kw, species, coords) in STRUCTURES.items():
        write(name, Structure.from_spacegroup(sg, lattice(sg, kw), species, coords), out)


if __name__ == "__main__":
    main()
