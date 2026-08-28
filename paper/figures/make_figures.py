"""Paper figures. Print-first: method identity is carried by axis position, so
marks wear a text token; the two validated categorical hues are reserved for
the annotation layers (classical baseline, reference uncertainty)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "figure.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})

M = ["none", "Fourier", "penalty", "orthog.", "spectrum"]
DATA = {  # mean, sd, classical, (reference-uncertainty band) or None
    "Circular, wet bed": ([1602,1681,1662,1725,2347], [12,110,120,130,110], 391.4, (311.1,739.5)),
    "Circular, dry bed": ([1681,1838,1602,1507,2314], [76,120,54,140,190], 379.8, None),
    "Step release":      ([1028,1124,1106,1054,1126], [110,68,47,71,15], 247.6, (2169,2208)),
    "Gaussian mound":    ([383.4,403.2,468.7,969.3,915.9], [22,48,55,180,34], 6.357, (376.3,394.9)),
    "Three humps":       ([887.4,796.2,748.6,918.5,608.4], [68,150,110,230,110], 79.65, None),
}
fig, axes = plt.subplots(1, 5, figsize=(9.4, 2.9), layout="constrained")
x = np.arange(len(M))
for ax, (name, (mu, sd, cl, band)) in zip(axes, DATA.items()):
    top = max(max(np.array(mu)+np.array(sd)), band[1] if band else 0, cl) * 1.15
    if band:
        ax.axhspan(band[0], band[1], color=ORANGE, alpha=0.28, lw=0, zorder=0)
    ax.axhline(cl, color=BLUE, lw=1.5, zorder=1)
    ax.errorbar(x, mu, yerr=sd, fmt="o", ms=4.2, color=INK, ecolor=INK,
                elinewidth=1.1, capsize=2.4, mfc="white", mew=1.1, zorder=3)
    ax.set_title(name, pad=4)
    ax.set_xticks(x); ax.set_xticklabels(M, rotation=38, ha="right")
    ax.set_xlim(-0.65, len(M)-0.35); ax.set_ylim(0, top)
    ax.grid(axis="y", color=MUTED, alpha=0.16, lw=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel(r"$L^1$ depth error")
fig.legend(handles=[Line2D([], [], color=BLUE, lw=1.5, label="classical scheme at $128^2$"),
                    Patch(facecolor=ORANGE, alpha=0.28, label="reference uncertainty (Sect. 6.5)")],
           loc="outside lower center", ncol=2, fontsize=7.2)
fig.savefig("paper/figures/fig_matrix.pdf")
print("wrote fig_matrix.pdf")

# ---- advection: per-seed distributions over the six regimes ---------------
SEEDS = {
 "penalty\n$w_U{=}1/35000$": [7.9e-3,7.0e-3,1.6e-2,8.4e-3,8.2e-3,1.0e-2,8.1e-3,5.6e-3,1.5e-2,1.3e-2,5.4e-3,1.3e-2,8.4e-3,5.3e-3,6.9e-2,9.8e-3,3.8e-2,3.4e-2,7.1e-3,4.0e-3],
 "exactly\northogonal":      [3.6e-2,1.6e-2,6.7e-2,3.0e-2,5.3e-3,1.7e-2,1.7e-2,9.4e-3,1.7e-2,1.7e-2,3.1e-3,1.7e-2,1.2e-2,9.5e-3,1.5e-1,1.4e-1,2.7e-2,4.0e-2,1.7e-2,2.3e-2],
 "unconstrained":            [4.6e-1,2.5e-1,2.3e-1,1.8e-2,7.7e-2,3.7e-1,3.9e-2,1.0e-1,3.4e-2,8.8e-3,1.3e-2,5.1e-2,9.0e-2,9.1e-3,4.7e-1,9.4e-2,4.7e-2,4.2e-2,2.0e-1,1.4e-1],
 "spectrum\nonly":           [4.6e-1,2.0e-2,2.4e0,1.2e-1,2.4e-2,6.7e-1,1.6e-2,1.5e-2,3.8e-2,1.1e-1,9.6e-2,1.8e-1,2.5e-1,1.6e-2,2.1e-1,1.4e-1,2.5e-2,8.6e-2,5.0e-1,1.2e0],
 "same param.\n$w_U{=}0$":   [3.0e-1,5.3e-2,1.1e0,1.1e-1,7.0e-3,2.4e-1,1.2e-2,3.0e-2,4.1e-1,1.8e-1,1.8e-2,6.7e-1,3.9e-2,1.0e-2,7.4e-1,7.6e-2,6.5e-1,3.6e-1,6.3e-1,1.1e0],
 "bounded\n$\\sigma_{\\max}(U)$": [3.8e-1,2.7e-1,6.6e-1,2.7e-2,2.2e0,6.5e-1,2.5e-1,1.1e0,2.8e-2,4.2e-1,3.8e-2,3.4e-2,3.2e-2,2.5e-2,4.4e-2,6.9e-1,6.3e-1,3.6e-1,7.2e-1,7.7e-1],
}
fig2, ax = plt.subplots(figsize=(5.4, 2.9), layout="constrained")
rng = np.random.default_rng(0)
for i, (name, v) in enumerate(SEEDS.items()):
    v = np.array(v)
    ax.scatter(i + rng.uniform(-0.17, 0.17, v.size), v, s=11, facecolors="none",
               edgecolors=INK, linewidths=0.8, zorder=3)
    med = np.median(v)
    ax.plot([i-0.32, i+0.32], [med, med], color=ORANGE, lw=2.0, zorder=4,
            solid_capstyle="butt")
ax.set_yscale("log")
ax.set_xticks(range(len(SEEDS))); ax.set_xticklabels(SEEDS.keys(), fontsize=6.6)
ax.set_ylabel(r"relative $L^2$ error (20 seeds)")
ax.grid(axis="y", color=MUTED, alpha=0.16, lw=0.5); ax.set_axisbelow(True)
ax.legend(handles=[Line2D([], [], color=ORANGE, lw=2.0, label="median")],
          loc="lower right", fontsize=7)
fig2.savefig("paper/figures/fig_advection.pdf")
print("wrote fig_advection.pdf")
