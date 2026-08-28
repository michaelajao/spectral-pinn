"""Circular dam break: depth fields and centreline profiles at t = 2 s.

The Gaussian-mound figure shows a smooth target. This is the shocked case, and
the centreline profile is where shock capture is legible: an outward-travelling
bore and the rarefaction behind it.
"""
import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models import PINN, PINNConfig               # noqa: E402
from src.run import build_reference, classical_baseline  # noqa: E402
from src.solver import Config, run as fv_run          # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BENCH, SEED, EVAL_N, REF_N = "ca_circular_wet", 0, 128, 512
ENTRIES = [("pinn", "Unconstrained"), ("cnpinn_soft", "Penalised spectral"),
           ("ortho_hard", "Exactly orthogonal")]

INK, MUTED, BLUE, ORANGE, GREEN = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 8,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def load(entry):
    d = ROOT / "runs" / "ml" / entry / BENCH / f"seed{SEED}"
    meta = json.loads((d / "meta.json").read_text())
    m = PINN(PINNConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                           for k, v in meta["model_cfg"].items()}))
    m.load_state_dict(torch.load(d / "best.pt", map_location="cpu"))
    return m.eval()


eval_bi, ref_fields = build_reference(BENCH, REF_N, EVAL_N, "cpu")
grid = eval_bi.grid
X, Y = grid.centers()
h_ref = ref_fields[-1][0].numpy()

# classical scheme at the evaluation resolution, for the profile panel
cfg = Config(grid=grid, bc=eval_bi.bc, scheme="hllc", order=2, limiter="van_leer",
             g=eval_bi.g, manning_n=eval_bi.manning_n)
out = fv_run(cfg, eval_bi.U0, eval_bi.z, eval_bi.t_end, eval_bi.output_times)
h_cls = out["U"][-1][0].numpy()

preds = []
for entry, label in ENTRIES:
    m = load(entry)
    with torch.no_grad():
        xyt = torch.stack([X.reshape(-1), Y.reshape(-1),
                           torch.full((grid.ny * grid.nx,), float(eval_bi.t_end))], 1)
        h, _, _ = m.state(xyt)
    preds.append((label, h.reshape(grid.ny, grid.nx).double().numpy()))

extent = [0, 100, 0, 100]
vmin, vmax = float(h_ref.min()), float(h_ref.max())
jmid = grid.ny // 2
xc = grid.xc.numpy()

fig = plt.figure(figsize=(9.6, 5.6), layout="constrained")
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85])

ims = []
for j, (title, field) in enumerate([("(A) HLLC reference, $N=512$", h_ref)]
                                   + [(f"({chr(66+i)}) {lab}", f_)
                                      for i, (lab, f_) in enumerate(preds)]):
    ax = fig.add_subplot(gs[0, j])
    im = ax.imshow(field, origin="lower", extent=extent, vmin=vmin, vmax=vmax,
                   cmap="viridis")
    ax.set_title(title); ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
    ax.axhline(50, color="w", lw=0.7, ls=(0, (4, 3)))
    if j == 0: ax.set_ylabel("$y$ (m)")
    ax.set_xlabel("$x$ (m)")
    ims.append(im)
fig.colorbar(ims[0], ax=fig.axes[:4], location="right", shrink=0.9,
             label="water depth $h$ (m)")

axp = fig.add_subplot(gs[1, :2])
axp.plot(xc, h_ref[jmid], color=INK, lw=1.8, label="HLLC reference, $N=512$")
axp.plot(xc, h_cls[jmid], color=BLUE, lw=1.2, ls="--", label="classical, $N=128$")
for (lab, f_), c in zip(preds, (ORANGE, GREEN, "#4a3aa7")):
    axp.plot(xc, f_[jmid], color=c, lw=1.2, label=lab)
axp.set_xlabel("$x$ (m)"); axp.set_ylabel("$h$ (m)")
axp.set_title("(E) centreline profile, $y = 50$ m")
axp.legend(fontsize=6.6, loc="upper right", ncol=1)
axp.grid(color=MUTED, alpha=0.16, lw=0.5); axp.set_axisbelow(True)

axz = fig.add_subplot(gs[1, 2:])
for (lab, f_), c in zip(preds, (ORANGE, GREEN, "#4a3aa7")):
    axz.plot(xc, f_[jmid] - h_ref[jmid], color=c, lw=1.2, label=lab)
axz.plot(xc, h_cls[jmid] - h_ref[jmid], color=BLUE, lw=1.2, ls="--",
         label="classical, $N=128$")
axz.axhline(0, color=INK, lw=0.8)
axz.set_xlabel("$x$ (m)"); axz.set_ylabel("$h - h_{\\mathrm{ref}}$ (m)")
axz.set_title("(F) centreline error")
axz.grid(color=MUTED, alpha=0.16, lw=0.5); axz.set_axisbelow(True)

fig.savefig(ROOT / "paper" / "figures" / "fig_shock.pdf")
print("wrote fig_shock.pdf")
print("centreline max |err|:",
      {lab: round(float(np.abs(f_[jmid] - h_ref[jmid]).max()), 3) for lab, f_ in preds},
      "classical:", round(float(np.abs(h_cls[jmid] - h_ref[jmid]).max()), 3))
