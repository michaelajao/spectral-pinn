"""Reproduction study: Wang et al.'s 1D advection case, PyTorch reimplementation.

Their setup (Sect. 2.2 and 4.1 of Appl. Soft Comput. 202:115794):
u_t + u_x = 0 on [-1,1] x [-0.5,0.5] with exact solution
u(x,t) = 0.8 sin(4 pi (x - t) + 0.25 pi); a 3x25 tanh MLP with Xavier
initialization; N_u = 100 points sampled from the initial/boundary data,
N_f = 500 Latin-hypercube collocation points; loss weights w_r = w_b = w_f = 1
and, for cnPINN, w_U = 1/35000; L-BFGS optimization; L2 relative error
evaluated on a 501 x 301 grid.

This script runs the same problem for each spectral weight mode (dense =
vanilla PINN, svd_soft = cnPINN, svd_hard, svd_sigma) over several seeds and
writes a summary table. The published magnitudes to match (their Figs. 5-7):
cnPINN ~1e-3 L2 relative error, vanilla PINN ~1e-2..1e-1.

Their reference implementation (third_party/cnPINNs) is TensorFlow 1.x with
tf.contrib's ScipyOptimizerInterface, covers only the Navier-Stokes case, and
implements a variant of the paper's Eq. 15 (U frozen, V trained, unsquared
matrix 1-norm penalty), so the comparison target is the paper's reported
numbers, not a rerun of their code. We follow Eq. 15: w_U * ||U^T U - I||_2^2.

Usage: python -m src.advection [--iters 3000] [--seeds 5] [--out reports/advection.md]
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import torch
import torch.nn as nn

from .models import SVDLinear, WEIGHT_PARAMS, _make_linear

ROOT = Path(__file__).resolve().parents[1]

X_RANGE = (-1.0, 1.0)
T_RANGE = (-0.5, 0.5)


def exact(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """u(x,t) = 0.8 sin(4 pi (x - t) + 0.25 pi)."""
    return 0.8 * torch.sin(4.0 * math.pi * (x - t) + 0.25 * math.pi)


class AdvectionMLP(nn.Module):
    """(x, t) -> u; 3 hidden layers x 25 neurons, tanh, Xavier init, with the
    two square hidden weights optionally SVD-reparameterized (the input and
    output layers stay dense, as in Wang et al.'s Algorithm 1)."""

    def __init__(self, weight_param: str = "dense", hidden: int = 25,
                 n_hidden: int = 3):
        super().__init__()
        if weight_param not in WEIGHT_PARAMS:
            raise ValueError(f"unknown weight_param '{weight_param}'")
        act = nn.Tanh
        layers: list[nn.Module] = [
            _make_linear(2, hidden, weight_param, "xavier", reparam=False), act()]
        for _ in range(n_hidden - 1):
            layers += [_make_linear(hidden, hidden, weight_param, "xavier",
                                    reparam=True), act()]
        layers += [_make_linear(hidden, 1, weight_param, "xavier", reparam=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.net(xt).squeeze(-1)

    def orthogonality_penalty(self) -> torch.Tensor:
        """sum_j D_{U_j}^2 with D_U = ||U^T U - I||_2 (paper Eq. 15)."""
        svd = [m for m in self.net if isinstance(m, SVDLinear)]
        if not svd:
            return torch.zeros(())
        return torch.stack([m.orthogonality_defect() ** 2 for m in svd]).sum()


def latin_hypercube(n: int, dims: int, gen: torch.Generator) -> torch.Tensor:
    """Simple Latin hypercube sample in [0,1]^dims: one point per stratum,
    stratum order permuted independently per dimension."""
    cols = []
    for _ in range(dims):
        perm = torch.randperm(n, generator=gen).double()
        cols.append((perm + torch.rand(n, generator=gen).double()) / n)
    return torch.stack(cols, dim=1)


def make_training_data(seed: int) -> dict[str, torch.Tensor]:
    """N_u = 100 initial/boundary points, N_f = 500 LHS collocation points."""
    gen = torch.Generator().manual_seed(seed)
    # candidate initial/boundary set: exact data on t = t0 and x = +-1
    n_side = 200
    x_ic = torch.linspace(*X_RANGE, n_side)
    t_bc = torch.linspace(*T_RANGE, n_side)
    cand_x = torch.cat([x_ic, torch.full((n_side,), X_RANGE[0]),
                        torch.full((n_side,), X_RANGE[1])])
    cand_t = torch.cat([torch.full((n_side,), T_RANGE[0]), t_bc, t_bc])
    idx = torch.randperm(cand_x.numel(), generator=gen)[:100]
    xu, tu = cand_x[idx], cand_t[idx]

    lhs = latin_hypercube(500, 2, gen).float()
    xf = X_RANGE[0] + (X_RANGE[1] - X_RANGE[0]) * lhs[:, 0]
    tf_ = T_RANGE[0] + (T_RANGE[1] - T_RANGE[0]) * lhs[:, 1]
    return {"xu": xu, "tu": tu, "uu": exact(xu, tu), "xf": xf, "tf": tf_}


def train_one(weight_param: str, seed: int, iters: int, w_U: float) -> float:
    """Train one model with L-BFGS; return the L2 relative error on the
    501 x 301 evaluation grid."""
    torch.manual_seed(seed)
    model = AdvectionMLP(weight_param)
    data = make_training_data(seed)

    opt = torch.optim.LBFGS(model.parameters(), max_iter=iters,
                            history_size=50, line_search_fn="strong_wolfe")

    xu_tu = torch.stack([data["xu"], data["tu"]], dim=1)

    def closure():
        opt.zero_grad(set_to_none=True)
        # data misfit on initial/boundary points
        L_ub = ((model(xu_tu) - data["uu"]) ** 2).mean()
        # PDE residual at collocation points
        xf = data["xf"].clone().unsqueeze(1).requires_grad_(True)
        tf_ = data["tf"].clone().unsqueeze(1).requires_grad_(True)
        u = model(torch.cat([xf, tf_], dim=1))
        ones = torch.ones_like(u)
        u_x = torch.autograd.grad(u, xf, ones, create_graph=True)[0]
        u_t = torch.autograd.grad(u, tf_, ones, create_graph=True)[0]
        L_f = ((u_t + u_x) ** 2).mean()
        loss = L_ub + L_f
        if w_U > 0.0 and weight_param == "svd_soft":
            loss = loss + w_U * model.orthogonality_penalty()
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        x = torch.linspace(*X_RANGE, 501)
        t = torch.linspace(*T_RANGE, 301)
        T_, X_ = torch.meshgrid(t, x, indexing="ij")
        xt = torch.stack([X_.reshape(-1), T_.reshape(-1)], dim=1)
        u_pred = model(xt)
        u_true = exact(X_.reshape(-1), T_.reshape(-1))
        return float(torch.linalg.norm(u_pred - u_true)
                     / torch.linalg.norm(u_true))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--iters", type=int, default=3000, help="L-BFGS max_iter")
    p.add_argument("--seeds", type=int, default=5, help="independent runs")
    p.add_argument("--w-U", type=float, default=1.0 / 35000.0,
                   help="orthogonality weight for svd_soft (paper: 1/35000)")
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "advection_reproduction.md")
    p.add_argument("--modes", default=",".join(WEIGHT_PARAMS),
                   help="comma-separated subset of " + ",".join(WEIGHT_PARAMS))
    args = p.parse_args()

    labels = {"dense": "pinn (dense)", "svd_soft": "cnpinn (svd_soft)",
              "svd_hard": "ortho (svd_hard)", "svd_sigma": "sigma (svd_sigma)"}
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in WEIGHT_PARAMS]
    if unknown:
        p.error(f"unknown modes {unknown}; choose from {WEIGHT_PARAMS}")
    entries = [(labels[m], m) for m in modes]
    rows = []
    for label, wp in entries:
        errs = [train_one(wp, s, args.iters, args.w_U) for s in range(args.seeds)]
        m, sd = statistics.mean(errs), (statistics.pstdev(errs) if len(errs) > 1 else 0.0)
        rows.append((label, m, sd, errs))
        print(f"{label:22s} L2 rel err = {m:.3e} ± {sd:.1e}   {['%.1e' % e for e in errs]}",
              flush=True)

    lines = ["# Advection reproduction (Wang et al. 2026, Sect. 4.1 setup)\n",
             f"3x25 tanh MLP, Xavier init, N_u=100, N_f=500 (LHS), L-BFGS "
             f"max_iter={args.iters}, w_U={args.w_U:.3e} for svd_soft, "
             f"{args.seeds} seeds. Published magnitudes: cnPINN ~1e-3, "
             f"vanilla PINN ~1e-2..1e-1 (their Figs. 5-7).\n",
             "| entry | L2 rel. error (mean ± std) | per-seed |",
             "|---|---|---|"]
    for label, m, sd, errs in rows:
        lines.append(f"| {label} | {m:.3e} ± {sd:.1e} | "
                     f"{', '.join('%.1e' % e for e in errs)} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
