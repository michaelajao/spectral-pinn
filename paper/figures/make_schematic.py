"""Schematic of the factorised hidden layer and the five parameterisations."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

ROOT = Path(__file__).resolve().parents[2]
INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
TRAIN, FROZEN = "#cfe2f7", "#e8e8e4"      # trained / frozen fills
plt.rcParams.update({"font.family": "serif", "font.size": 8, "text.color": INK,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.9),
                         gridspec_kw={"width_ratios": [1.05, 1.55]})

# ---- (A) the factorisation -------------------------------------------------
ax = axes[0]; ax.set_xlim(-0.2, 8.6); ax.set_ylim(-0.9, 4.8); ax.axis("off")
ax.set_title("(A) factorisation at initialisation", loc="left", fontsize=8.5)


def box(ax, x, y, w, h, fc, label, sub=None):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=INK, lw=0.9))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    if sub:
        ax.text(x + w / 2, y - 0.30, sub, ha="center", va="top", fontsize=6.6,
                color=MUTED)


box(ax, 0.1, 1.6, 1.5, 1.9, FROZEN, "$W_0$")
ax.text(2.05, 2.55, "$=$", ha="center", va="center", fontsize=11)
box(ax, 2.5, 1.6, 1.5, 1.9, TRAIN, "$U$", "trained")
box(ax, 4.4, 1.6, 1.5, 1.9, TRAIN, "$S$", "trained")
box(ax, 6.3, 1.6, 1.5, 1.9, FROZEN, "$V^{\\!\\top}$", "frozen")
ax.text(0.85, 4.35, "SVD once, at initialisation only", ha="center", fontsize=7,
        color=MUTED)
ax.annotate("", xy=(0.85, 3.65), xytext=(0.85, 4.10),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
ax.text(3.95, -0.55,
        "$W = U\\,\\mathrm{diag}(s)\\,V^{\\!\\top}$ rebuilt on every forward pass",
        ha="center", va="center", fontsize=7, color=MUTED)

# ---- (B) the spectrum of U under each parameterisation ---------------------
ax = axes[1]
ax.set_title("(B) admissible spectrum of $U$", loc="left", fontsize=8.5)
rows = [
    ("Unconstrained",            None,          "no factorisation"),
    ("Penalised spectral",       "soft",        "$w_U\\,\\|U^{\\!\\top}U-I\\|_2^2$ in the loss"),
    ("Exactly orthogonal",       (1.0, 1.0),    "$\\sigma_i(U)=1$"),
    ("Spectrum only",            (1.0, 1.0),    "$U$ frozen orthogonal"),
    ("Bounded $\\sigma_{\\max}$", (0.0, 1.0),   "$\\sigma_{\\max}(U)=1$"),
]
ax.set_xlim(-0.15, 2.25); ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_yticks(range(len(rows))[::-1])
ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.set_xlabel("singular values of $U$", fontsize=7.5)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0]); ax.tick_params(labelsize=7)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.axvline(1.0, color=MUTED, lw=0.7, ls=(0, (3, 3)), zorder=0)

for i, (_, spec, note) in enumerate(rows):
    y = len(rows) - 1 - i
    if spec == "soft":
        ax.add_patch(Rectangle((0.0, y - 0.17), 2.15, 0.34, facecolor=ORANGE,
                               alpha=0.22, edgecolor="none"))
        ax.plot([1.0], [y], marker="o", ms=5, color=ORANGE, zorder=3)
    elif spec is None:
        ax.add_patch(Rectangle((0.0, y - 0.17), 2.15, 0.34, facecolor=MUTED,
                               alpha=0.16, edgecolor="none"))
    elif spec[0] == spec[1]:
        ax.plot([1.0], [y], marker="|", ms=15, mew=2.4, color=BLUE, zorder=3)
    else:
        ax.add_patch(Rectangle((spec[0], y - 0.17), spec[1] - spec[0], 0.34,
                               facecolor=BLUE, alpha=0.35, edgecolor=INK, lw=0.7))
    ax.text(2.24, y, note, fontsize=6.6, va="center", ha="right", color=MUTED)

fig.savefig(ROOT / "paper" / "figures" / "fig_schematic.pdf")
print("wrote fig_schematic.pdf")
