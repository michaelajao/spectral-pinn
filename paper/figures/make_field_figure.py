"""Depth fields at the final output time for the Gaussian-mound benchmark.

This is the benchmark on which the constrained parameterisations separate from
the unconstrained network, so it is where a field plot carries information that
the error table does not: it shows *where* the extra error sits.

Reconstructs each trained network from its run directory (meta.json gives the
resolved architecture, best.pt the weights), evaluates it on the shared
evaluation grid, and compares against the in-process HLLC reference.
"""
import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.benchmarks import build                      # noqa: E402
from src.models import PINN, PINNConfig               # noqa: E402
from src.run import build_reference                   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BENCH, SEED, EVAL_N, REF_N = "ca_gaussian", 0, 128, 512
ENTRIES = [("pinn", "Unconstrained"), ("cnpinn_soft", "Penalised spectral"),
           ("ortho_hard", "Exactly orthogonal")]

INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 8,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def load(entry):
    d = ROOT / "runs" / "ml" / entry / BENCH / f"seed{SEED}"
    meta = json.loads((d / "meta.json").read_text())
    model = PINN(PINNConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                               for k, v in meta["model_cfg"].items()}))
    model.load_state_dict(torch.load(d / "best.pt", map_location="cpu"))
    return model.eval()


eval_bi, ref_fields = build_reference(BENCH, REF_N, EVAL_N, "cpu")
grid = eval_bi.grid
X, Y = grid.centers()
t_end = eval_bi.t_end
h_ref = ref_fields[-1][0].numpy()

preds = []
for entry, label in ENTRIES:
    m = load(entry)
    with torch.no_grad():
        xyt = torch.stack([X.reshape(-1), Y.reshape(-1),
                           torch.full((grid.ny * grid.nx,), float(t_end))], 1)
        h, _, _ = m.state(xyt)
    preds.append((label, h.reshape(grid.ny, grid.nx).double().numpy()))

extent = [0, 100, 0, 100]
vmin, vmax = float(h_ref.min()), float(h_ref.max())
dmax = max(np.abs(p - h_ref).max() for _, p in preds)

fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.0), layout="constrained")
im = axes[0, 0].imshow(h_ref, origin="lower", extent=extent, vmin=vmin, vmax=vmax,
                       cmap="viridis")
axes[0, 0].set_title("(A) HLLC reference, $N=512$")
axes[0, 0].set_ylabel("$y$ (m)")
for j, (label, hp) in enumerate(preds, start=1):
    axes[0, j].imshow(hp, origin="lower", extent=extent, vmin=vmin, vmax=vmax,
                      cmap="viridis")
    axes[0, j].set_title(f"({chr(65+j)}) {label}")
fig.colorbar(im, ax=axes[0, :], location="right", shrink=0.85,
             label="water depth $h$ (m)")

axes[1, 0].axis("off")
for j, (label, hp) in enumerate(preds, start=1):
    imd = axes[1, j].imshow(hp - h_ref, origin="lower", extent=extent,
                            vmin=-dmax, vmax=dmax, cmap="RdBu_r")
    l1 = np.abs(hp - h_ref).sum() * grid.dx * grid.dy
    axes[1, j].set_title(f"({chr(68+j)}) error, $L^1={l1:.3g}$")
    axes[1, j].set_xlabel("$x$ (m)")
axes[1, 1].set_ylabel("$y$ (m)")
fig.colorbar(imd, ax=axes[1, :], location="right", shrink=0.85,
             label="$h_\\theta - h_{\\mathrm{ref}}$ (m)")
for ax in axes.ravel():
    if ax.has_data():
        ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
fig.savefig(ROOT / "paper" / "figures" / "fig_fields.pdf")
print("wrote fig_fields.pdf   L1:",
      {lab: round(float(np.abs(p - h_ref).sum() * grid.dx * grid.dy), 1)
       for lab, p in preds})
