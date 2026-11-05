"""Matplotlib figure for one fit result -- stacked patterns, overlay, residual, weights."""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from .fit import area_normalise, sg_derivative
from .sanity import _rolling_background

OFFSET_STEP = 0.025


def display_curves(result):
    """Background-subtracted sample and matching reconstruction, for plotting only.

    The fit works in derivative space, where a smooth background drops out, but the sample is
    area-normalised *with* its background. The references have none, so on real data the raw
    sample_norm looks far too small next to the reconstruction. For display, remove a rolling
    10th-percentile background (the same estimate the sanity check uses) and put both curves
    on the same area scale. The fit, the weights and residual_rms are unchanged.
    """
    a = result.angles
    step = float(np.median(np.diff(a)))
    window = max(11, int(round(3.0 / step)) | 1)
    net = np.clip(result.sample_norm - _rolling_background(result.sample_norm, window), 0, None)
    sample = area_normalise(net)
    recon = area_normalise(result.reconstruction) if np.any(result.reconstruction) else result.reconstruction
    return sample, recon, sample - recon


def fit_figure(sample_name, result, xlim=None):
    angles = result.angles
    sample_d, recon_d, resid_d = display_curves(result)
    xlim = xlim or (angles[0], min(angles[-1], 60))

    fig = plt.figure(figsize=(11, 12))
    fig.suptitle(f"Phase fit -- {sample_name}", fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.4, top=0.95, bottom=0.06,
                           left=0.11, right=0.97)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3])

    names = list(result.reference_interp.keys())
    offset = 0.0
    for name in names:
        w = result.weights.get(name, 0.0)
        label = f"{name} (w={w:.3f})" if w > 1e-6 else f"{name} (not used)"
        ax1.plot(angles, result.reference_interp[name] + offset, linestyle="--",
                 linewidth=1, label=label)
        offset += OFFSET_STEP
    ax1.plot(angles, sample_d + offset, linewidth=1.6, color="black",
             label=sample_name)
    offset += OFFSET_STEP
    ax1.plot(angles, recon_d + offset, linewidth=1.6, color="tab:red",
             linestyle="-.", label="reconstruction")
    ax1.set_ylabel("area-norm. intensity + offset")
    ax1.set_title("References + sample + reconstruction (stacked)")
    ax1.legend(loc="upper right", fontsize="small")
    ax1.set_xlim(*xlim)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.plot(angles, sample_d, linewidth=1.6, color="black", label=f"{sample_name} (background removed)")
    ax2.plot(angles, recon_d, linewidth=1.4, color="tab:red", linestyle="-.",
             label="reconstruction")
    ax2.set_ylabel("area-norm. intensity")
    ax2.set_title("Sample vs. reconstruction (overlaid)")
    ax2.legend(loc="upper right", fontsize="small")
    ax2.set_xlim(*xlim)
    plt.setp(ax2.get_xticklabels(), visible=False)

    dx = float(np.median(np.diff(angles)))
    deriv_residual = sg_derivative(resid_d, dx, result.sg_window, result.sg_order)
    ax3.plot(angles, resid_d, linewidth=1.2, color="steelblue",
             label="residual (sample - reconstruction)")
    ax3r = ax3.twinx()
    ax3r.plot(angles, deriv_residual, linewidth=1.0, color="darkorange", alpha=0.7,
              label="d/d(2theta) residual")
    ax3r.set_ylabel("d(residual)/d(2theta)", color="darkorange")
    ax3r.tick_params(axis="y", labelcolor="darkorange")
    ax3.axhline(0, color="grey", lw=0.7, ls=":")
    ax3.set_ylabel("delta intensity")
    ax3.set_xlabel("2theta (degrees)")
    ax3.set_title(f"Residual (S-G window={result.sg_window}, order={result.sg_order})")
    l1, lb1 = ax3.get_legend_handles_labels()
    l2, lb2 = ax3r.get_legend_handles_labels()
    ax3.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize="small")
    ax3.set_xlim(*xlim)

    weights = np.array([result.weights[n] for n in names])
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(names), 1)))
    bars = ax4.bar(names, weights, color=colors, edgecolor="black", linewidth=0.6)
    ax4.set_ylabel("fractional NNLS weight")
    ax4.set_title("Phase contributions")
    ax4.set_ylim(0, max(weights.max() * 1.15 if len(weights) else 0.05, 0.05))
    for bar, w in zip(bars, weights):
        if w > 1e-4:
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{w:.3f}", ha="center", va="bottom", fontsize=9)
    ax4.tick_params(axis="x", rotation=30)

    return fig
