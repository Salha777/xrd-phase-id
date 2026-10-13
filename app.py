"""XRD Phase-ID -- Streamlit GUI for NNLS phase fitting of powder XRD patterns.

Run with:  streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import streamlit as st

from xrd_phase_id.fit import DEFAULT_SG_ORDER, DEFAULT_SG_WINDOW, fit_sample
from xrd_phase_id.io_utils import load_reference_csv, load_xrdml, load_xy, name_from_upload
from xrd_phase_id.plotting import fit_figure
from xrd_phase_id.sanity import check_sample, suggest_phases

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(HERE, "demo_data")

st.set_page_config(page_title="XRD Phase-ID", layout="wide")

VERDICT_STYLE = {
    "ok": "success",
    "minor_unmatched": "warning",
    "missing_phase": "error",
    "weak": "error",
    "no_peaks": "error",
}


def _load_many(uploaded_files, loader):
    out = {}
    for f in uploaded_files or []:
        try:
            out[name_from_upload(f)] = loader(f)
        except Exception as e:
            st.warning(f"couldn't read {getattr(f, 'name', f)}: {e}")
    return out


def _load_samples(uploaded_files):
    """Like _load_many, but dispatches .xrdml through load_xrdml and uses the
    sample name it carries (falling back to the filename) instead of the
    filename alone."""
    out = {}
    for f in uploaded_files or []:
        fname = getattr(f, "name", str(f))
        try:
            if fname.lower().endswith(".xrdml"):
                angles, intensities, xrdml_name = load_xrdml(f)
                label = xrdml_name or name_from_upload(f)
            else:
                angles, intensities = load_xy(f)
                label = name_from_upload(f)
        except Exception as e:
            st.warning(f"couldn't read {fname}: {e}")
            continue
        if label in out:  # two files resolved to the same sample name
            label = f"{label} ({fname})"
        out[label] = (angles, intensities)
    return out


def _demo_references():
    refs = {}
    ref_dir = os.path.join(DEMO_DIR, "references")
    for fname in sorted(os.listdir(ref_dir)):
        if fname.endswith(".csv"):
            refs[fname[:-4]] = load_reference_csv(os.path.join(ref_dir, fname))
    return refs


def _demo_sample():
    path = os.path.join(DEMO_DIR, "sample_pattern_demo.xy")
    return {"sample_pattern_demo": load_xy(path)}


LIB_DIR = os.path.join(HERE, "reference_library")
PATTERN_EXTS = (".csv", ".xy", ".txt", ".dat")


@st.cache_data(show_spinner=False)
def _library_index():
    """{group label: {pattern name: path}} for every pattern under reference_library/
    plus the synthetic demo references."""
    groups = {}
    for root, _, files in os.walk(LIB_DIR):
        found = {os.path.splitext(f)[0]: os.path.join(root, f)
                 for f in sorted(files) if f.lower().endswith(PATTERN_EXTS)}
        if found:
            rel = os.path.relpath(root, LIB_DIR)
            groups[rel.replace("local" + os.sep, "").replace(os.sep, " / ")] = found
    demo = os.path.join(DEMO_DIR, "references")
    groups["demo (synthetic)"] = {f[:-4]: os.path.join(demo, f)
                                   for f in sorted(os.listdir(demo)) if f.endswith(".csv")}
    return groups


@st.cache_data(show_spinner=False)
def _load_library_pattern(path):
    return load_reference_csv(path)


def _set_group(group, value):
    for name in _library_index()[group]:
        st.session_state[f"lib::{group}::{name}"] = value


st.title("XRD Phase-ID")
st.caption(
    "Fits a powder XRD pattern as a non-negative least-squares mixture of reference "
    "patterns, in Savitzky-Golay derivative space (robust to background and intensity "
    "scaling). This is a screening tool, not quantitative phase analysis -- read the "
    "caveats at the bottom before quoting a number."
)

with st.sidebar:
    st.header("1. Data")
    use_demo = st.checkbox("Use bundled synthetic demo data", value=False,
                            help="A ~70/30 calcite/quartz synthetic mixture, for trying "
                                 "the tool out. Not real measured data.")

    st.subheader("Reference patterns (used in the fit)")
    library = _library_index()
    with st.expander("Reference library -- tick to include", expanded=True):
        for group, patterns in library.items():
            st.markdown(f"**{group}**")
            c1, c2 = st.columns(2)
            c1.button("all", key=f"all::{group}", on_click=_set_group, args=(group, True))
            c2.button("none", key=f"none::{group}", on_click=_set_group, args=(group, False))
            for name in patterns:
                st.checkbox(name, key=f"lib::{group}::{name}")
    ref_files = st.file_uploader(
        "...or upload your own (2theta,intensity; .csv / .xy / .txt / .dat)",
        type=["csv", "xy", "txt", "dat"], accept_multiple_files=True, key="refs",
    )
    suggest_from_library = st.checkbox(
        "Search unticked library patterns for suggestions", value=True,
        help="If a sample has strong peaks none of the ticked references explain, the "
             "unticked library patterns are searched to suggest what might be missing. "
             "They are never added to the fit.")

    with st.expander("Candidate patterns (optional -- suggestions only, not fit)"):
        st.caption(
            "Patterns here are never added to the fit. If a sample has peaks none of "
            "your fit references explain, these are searched to suggest what might be "
            "missing."
        )
        candidate_files = st.file_uploader(
            "2theta,intensity files", type=["csv", "xy", "txt", "dat"],
            accept_multiple_files=True, key="candidates",
        )

    st.subheader("Sample pattern(s)")
    sample_files = st.file_uploader(
        "raw .xrdml, or two-column 2theta/intensity (.csv / .xy / .txt / .dat)",
        type=["xrdml", "xy", "txt", "dat", "csv"], accept_multiple_files=True, key="samples",
        help="A .xrdml is read directly -- its sample name is used as the label "
             "automatically, no conversion to .xy needed.",
    )

    st.header("2. Advanced settings")
    sg_window = st.number_input("Savitzky-Golay window", min_value=5, value=DEFAULT_SG_WINDOW,
                                step=2, help="Must be odd; auto-shrunk for short patterns.")
    sg_order = st.number_input("Savitzky-Golay polynomial order", min_value=1,
                               value=DEFAULT_SG_ORDER, step=1)
    xlim_max = st.number_input("Plot upper 2theta limit", min_value=10, value=60)

    run = st.button("Run fit", type="primary")

ticked = {name: _load_library_pattern(path)
          for group, patterns in library.items() for name, path in patterns.items()
          if st.session_state.get(f"lib::{group}::{name}")}
ticked.update(_load_many(ref_files, load_reference_csv))
references = ticked
if use_demo and not references:
    references = _demo_references()
    st.sidebar.info("No references ticked -- using the demo references.")
candidates = _load_many(candidate_files, load_reference_csv)
if suggest_from_library:
    for group, patterns in library.items():
        for name, path in patterns.items():
            if name not in references and name not in candidates:
                candidates[name] = _load_library_pattern(path)
samples = _demo_sample() if use_demo else _load_samples(sample_files)

if not run:
    st.info("Load references and at least one sample in the sidebar, then click **Run fit**.")
elif not references:
    st.error("No reference patterns selected -- tick some in the library or upload your own.")
elif not samples:
    st.error("No sample patterns loaded.")
else:
    st.write(f"**{len(references)} reference(s):** {', '.join(references)}")
    if candidates:
        st.caption(f"{len(candidates)} unticked/candidate pattern(s) searched for suggestions.")

    summary_rows = []
    for sample_name, (angles, intensities) in samples.items():
        st.divider()
        st.subheader(sample_name)

        try:
            result = fit_sample(angles, intensities, references, sg_window, sg_order)
        except Exception as e:
            st.error(f"fit failed: {e}")
            continue
        for w in result.warnings:
            st.warning(w)

        sanity = check_sample(angles, intensities, references)
        style = VERDICT_STYLE.get(sanity.verdict, "info")
        getattr(st, style)(sanity.message)

        if sanity.unmatched and candidates:
            hints = suggest_phases(sanity.unmatched, sanity.peaks, candidates)
            for score, name, explains, present in hints:
                st.info(f"-> try **{name}**: explains {explains:.0%} of the unmatched "
                       f"intensity, {present:.0%} of its own strongest lines are present "
                       f"in this sample")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = fit_figure(sample_name, result, xlim=(angles[0], min(angles[-1], xlim_max)))
            st.pyplot(fig, clear_figure=True)
        with col2:
            df = pd.DataFrame({
                "reference": list(result.weights.keys()),
                "weight": [round(w, 4) for w in result.weights.values()],
            }).sort_values("weight", ascending=False)
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.metric("Residual RMS", f"{result.residual_rms:.4g}")
            st.caption(
                "Weights are NNLS fit coefficients in derivative space, normalised to "
                "sum to 1. They are **not** quantitative phase fractions -- treat them "
                "as \"mostly consistent with\", never \"X% phase Y\". Residual RMS is on "
                "area-normalised intensities, which are small to start with, so a small "
                "residual is not on its own reassurance -- trust the sanity check above it."
            )

        row = {"sample": sample_name, **{k: round(v, 4) for k, v in result.weights.items()},
               "residual_rms": result.residual_rms, "sanity_verdict": sanity.verdict}
        summary_rows.append(row)

    if summary_rows:
        st.divider()
        st.subheader("Summary")
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
        st.download_button("Download summary CSV", summary_df.to_csv(index=False),
                          file_name="xrd_phase_id_summary.csv", mime="text/csv")

st.divider()
with st.expander("Method and caveats -- read before quoting a number"):
    st.markdown(
        """
- **What the fit does.** Each pattern is area-normalised, then transformed with a
  Savitzky-Golay first derivative. Fitting in derivative space is what makes it robust to
  background slope and to arbitrary intensity scaling; the sample's derivative is then
  fit as a non-negative least-squares (NNLS) mixture of the reference derivatives.
- **NNLS always returns an answer.** It cannot say "none of these references match". If the
  real phase is missing from your reference set, the weight goes to whichever reference
  correlates best, and you get a confident-looking number for a phase that isn't in the
  sample. **Always read the sanity check before trusting the weights** -- it is computed
  independently of the fit, from raw peak positions only.
- **Weights are not phase fractions.** They are fit coefficients, normalised to sum to 1.
  Say "mostly consistent with", not "X% phase Y". Quantitative phase analysis needs
  Rietveld refinement with a preferred-orientation correction -- platy/layered phases in
  particular redistribute intensity in ways a fixed-intensity template match will get wrong.
- **Residual RMS is not a trust signal.** Area-normalised intensities are small to begin
  with, so small residuals are not reassuring on their own.
"""
    )
