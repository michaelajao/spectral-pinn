"""Neural models and losses: the strong-form PINN with optional spectral
weight reparameterizations, the FVM-informed PINN, and both loss stacks.

Ported from ``swe-dambreak`` (src/ml/models.py + src/ml/losses.py) and
extended with the spectral variants that are the subject of this project.

Strong-form PINN: an MLP maps (x, y, t) -> either primitive (h, u, v) or
conservative (h, hu, hv) outputs (config switch), so baseline entries differ
only in the residual formulation. Optional Fourier-feature embedding and
softplus positivity on h are available but off by default.

Spectral weight reparameterizations (``PINNConfig.weight_param``): each square
hidden weight matrix is factored once at initialization as W = U diag(s) V^T
and the factors are trained in W's place — see ``SVDLinear``. Following
Wang et al. (2026, Applied Soft Computing 202:115794), whose cnPINN this
reimplements and extends, the first (input) and last (output) layers keep
dense weights.

  dense       plain nn.Linear everywhere (the baseline).
  svd_soft    cnPINN as published: train U and s, freeze V^T, and add
              w_U * D_U^2 to the loss with D_U = ||U^T U - I||_2 (their
              Eq. 15 and Theorem 3.3; the matrix 2-norm, i.e. the largest
              |eigenvalue| of the symmetric defect). ``defect_norm`` can
              switch to the Frobenius norm, which upper-bounds the 2-norm
              by up to a factor sqrt(n) — kept as an ablation, not as the
              published baseline. Note the authors' released code differs
              from their Eq. 15 (it freezes U, trains V, and penalizes an
              unsquared matrix 1-norm); we follow the paper.
  svd_hard    this project's method: train U and s with U kept orthogonal
              to machine precision by torch's orthogonal parametrization —
              no penalty term, no w_U, and the singular values of W equal
              |s| to round-off.
  svd_sigma   train only s with U and V^T frozen at their orthogonal initial
              values: n trainable parameters per layer, exact spectral
              control, the cheapest member of the family.
  svd_bounded the constraint the advection diagnostic points to: train U and s
              with U rescaled every forward pass so its largest singular value
              is 1, which bounds U's scale without forcing its other singular
              values to 1. Strictly weaker than svd_hard's orthogonality and,
              like it, carries no penalty and no w_U. The measured behaviour
              of svd_soft at the published weight is a bounded but
              rank-deficient U, which orthogonality forbids and this allows.

FVM-informed PINN: the network predicts cell-averaged states on the solver
grid at collocation times; the loss is the residual of the discrete SSP-RK2
update built from the differentiable HLLC flux + Audusse hydrostatic
reconstruction (solver.step), so discrete conservation and well-balancing are
inherited from the classical solver rather than learned. It keeps dense
weights for now; composing it with the spectral variants is a later phase.

ML models run in float32 by default; the differentiable flux loss casts to
float64 before touching any solver function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.nn.utils import parametrizations, parametrize

from .solver import Config, pad_scalar, step, velocity

_ACT = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}

#: Weight-reparameterization modes accepted by PINNConfig.weight_param.
WEIGHT_PARAMS = ("dense", "svd_soft", "svd_hard", "svd_sigma", "svd_bounded")


class FourierFeatures(nn.Module):
    """Random Fourier features: z -> [sin(2π B z), cos(2π B z)] with fixed B."""

    def __init__(self, in_dim: int, n_freq: int, scale: float, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(in_dim, n_freq, generator=g) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ==========================================================================
# Spectral weight reparameterization
# ==========================================================================

def _init_weight(w: torch.Tensor, init: str) -> None:
    """Initialize a weight matrix in place: torch's Linear default or Xavier.

    ``xavier`` (Glorot normal) matches the setup of Wang et al. (2026) and is
    used when reproducing their reported numbers; ``default`` keeps torch's
    kaiming-uniform so spectral variants stay init-identical to the dense
    baseline PINN of the ported comparison.
    """
    if init == "default":
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    elif init == "xavier":
        nn.init.xavier_normal_(w)
    else:
        raise ValueError(f"unknown init '{init}'")


class _SpectralCap(nn.Module):
    """Rescale a matrix so its largest singular value is exactly 1.

    Weaker than orthogonality: it pins sigma_max(U) and leaves the remaining
    singular values free in [0, 1], so a rank-deficient U is reachable. That
    is what distinguishes it from ``parametrizations.orthogonal``, which
    forces every singular value to 1. The scale removed here is not lost to
    the layer -- W = U diag(s) V^T, so ``s`` absorbs it.
    """

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        return U / torch.linalg.matrix_norm(U, ord=2).clamp_min(1e-12)

    def right_inverse(self, U: torch.Tensor) -> torch.Tensor:
        return U


class SVDLinear(nn.Module):
    """Linear layer trained through the SVD factors of its initial weight.

    At construction the (freshly initialized) weight W0 is factored once as
    W0 = U diag(s) V^T; thereafter the layer's effective weight is rebuilt
    from the stored factors every forward pass, and which factors train is
    set by ``mode`` (see module docstring): ``svd_soft`` trains U and s with
    V^T frozen, ``svd_hard`` additionally constrains U to the orthogonal
    manifold via ``torch.nn.utils.parametrizations.orthogonal`` (so no
    penalty is needed), ``svd_bounded`` instead pins only U's largest
    singular value, and ``svd_sigma`` trains s alone. SVD is performed
    only at initialization, never during training (Wang et al., Algorithm 1).
    """

    def __init__(self, in_features: int, out_features: int, mode: str,
                 init: str = "default", bias: bool = True,
                 defect_norm: str = "spectral"):
        super().__init__()
        if mode not in ("svd_soft", "svd_hard", "svd_sigma", "svd_bounded"):
            raise ValueError(f"unknown SVDLinear mode '{mode}'")
        if defect_norm not in ("spectral", "frobenius"):
            raise ValueError(f"unknown defect_norm '{defect_norm}'")
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.defect_norm = defect_norm

        w0 = torch.empty(out_features, in_features)
        _init_weight(w0, init)
        U, s, Vh = torch.linalg.svd(w0, full_matrices=False)
        # LAPACK returns strided views (Vh is a transposed V); parameters must
        # be contiguous or their grads inherit the strides and L-BFGS's
        # flat-gradient gather (p.grad.view(-1)) fails.
        U, s, Vh = U.contiguous(), s.contiguous(), Vh.contiguous()

        self.s = nn.Parameter(s)
        self.register_buffer("Vh", Vh)
        if mode == "svd_sigma":
            self.register_buffer("U", U)
        else:
            self.U = nn.Parameter(U)
            if mode == "svd_hard":
                # Replaces U by an exactly-orthogonal parametrization seeded
                # at its current (orthogonal) value; gradients flow through
                # the map, so every optimizer step stays on the manifold.
                parametrizations.orthogonal(self, "U")
            elif mode == "svd_bounded":
                parametrize.register_parametrization(self, "U", _SpectralCap())

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            if init == "xavier":
                nn.init.zeros_(self.bias)        # as the dense layers under xavier
            else:
                bound = 1.0 / math.sqrt(in_features)
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def weight(self) -> torch.Tensor:
        """Effective weight W = U diag(s) V^T."""
        return (self.U * self.s) @ self.Vh

    def orthogonality_defect(self) -> torch.Tensor:
        """D_U = ||U^T U - I|| in ``defect_norm`` (2-norm by default, per
        Wang et al. Theorem 3.3); ~0 to round-off for svd_hard/svd_sigma."""
        UtU = self.U.T @ self.U
        eye = torch.eye(UtU.shape[0], dtype=UtU.dtype, device=UtU.device)
        if self.defect_norm == "spectral":
            return torch.linalg.matrix_norm(UtU - eye, ord=2)
        return torch.linalg.matrix_norm(UtU - eye, ord="fro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight(), self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, mode={self.mode}, "
                f"defect_norm={self.defect_norm}")


def _make_linear(in_f: int, out_f: int, weight_param: str, init: str,
                 *, reparam: bool, defect_norm: str = "spectral") -> nn.Module:
    """A hidden-stack layer: SVD-reparameterized iff requested AND eligible.

    Only square hidden->hidden layers are reparameterized (``reparam=True``);
    the input and output layers always stay dense, following Wang et al.
    """
    if reparam and weight_param != "dense":
        return SVDLinear(in_f, out_f, mode=weight_param, init=init,
                         defect_norm=defect_norm)
    lin = nn.Linear(in_f, out_f)
    _init_weight(lin.weight, init)
    if init == "xavier":
        nn.init.zeros_(lin.bias)
    return lin


# ==========================================================================
# Strong-form PINN
# ==========================================================================

@dataclass
class PINNConfig:
    variables: str = "primitive"       # "primitive" (h,u,v) | "conservative" (h,hu,hv)
    hidden: int = 128
    layers: int = 6
    activation: str = "tanh"
    fourier_features: int = 0          # 0 disables; else number of frequencies
    fourier_scale: float = 1.0
    fourier_seed: int = 0              # wire to the run seed for seed-varied B
    softplus_h: bool = False           # hard-enforce h>=0 (off for honest baseline)
    weight_param: str = "dense"        # see WEIGHT_PARAMS
    defect_norm: str = "spectral"      # "spectral" (paper Eq. 15) | "frobenius"
    init: str = "default"              # "default" | "xavier"
    # input normalization bounds (x, y, t); set from the case domain
    x_range: tuple[float, float] = (0.0, 1.0)
    y_range: tuple[float, float] = (0.0, 1.0)
    t_range: tuple[float, float] = (0.0, 1.0)


class PINN(nn.Module):
    """(x, y, t) -> 3 state outputs, with input normalization to [-1, 1]."""

    def __init__(self, cfg: PINNConfig):
        super().__init__()
        if cfg.weight_param not in WEIGHT_PARAMS:
            raise ValueError(f"unknown weight_param '{cfg.weight_param}'; "
                             f"have {WEIGHT_PARAMS}")
        self.cfg = cfg
        lo = torch.tensor([cfg.x_range[0], cfg.y_range[0], cfg.t_range[0]])
        hi = torch.tensor([cfg.x_range[1], cfg.y_range[1], cfg.t_range[1]])
        self.register_buffer("in_lo", lo)
        self.register_buffer("in_hi", hi)

        if cfg.fourier_features > 0:
            self.embed = FourierFeatures(3, cfg.fourier_features,
                                         cfg.fourier_scale, cfg.fourier_seed)
            in_dim = 2 * cfg.fourier_features
        else:
            self.embed = None
            in_dim = 3

        act = _ACT[cfg.activation]
        layers: list[nn.Module] = [
            _make_linear(in_dim, cfg.hidden, cfg.weight_param, cfg.init,
                         reparam=False),
            act(),
        ]
        for _ in range(cfg.layers - 1):
            layers += [
                _make_linear(cfg.hidden, cfg.hidden, cfg.weight_param,
                             cfg.init, reparam=True,
                             defect_norm=cfg.defect_norm),
                act(),
            ]
        layers += [_make_linear(cfg.hidden, 3, cfg.weight_param, cfg.init,
                                reparam=False)]
        self.net = nn.Sequential(*layers)

    @property
    def svd_layers(self) -> list[SVDLinear]:
        return [m for m in self.net if isinstance(m, SVDLinear)]

    def orthogonality_penalty(self) -> torch.Tensor:
        """sum_j D_{U_j}^2 over reparameterized layers (Wang et al. Eq. 15,
        read as a per-layer sum of squared defects); 0-dim, zero when there
        are none. Multiply by w_U in the loss. Only meaningful for
        ``svd_soft`` — the hard variant is orthogonal by construction."""
        layers = self.svd_layers
        if not layers:
            return self.in_lo.new_zeros(())
        return torch.stack([m.orthogonality_defect() ** 2 for m in layers]).sum()

    def normalize(self, xyt: torch.Tensor) -> torch.Tensor:
        # cast to the network's dtype/device (grid coords arrive as float64)
        xyt = xyt.to(self.in_lo)
        return 2 * (xyt - self.in_lo) / (self.in_hi - self.in_lo) - 1

    def raw(self, xyt: torch.Tensor) -> torch.Tensor:
        """Raw 3-vector network output (pre positivity handling)."""
        z = self.normalize(xyt)
        if self.embed is not None:
            z = self.embed(z)
        return self.net(z)

    def forward(self, xyt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the three physical fields. For primitive vars these are
        (h, u, v); for conservative (h, hu, hv). h is optionally softplus-ed."""
        out = self.raw(xyt)
        h = out[..., 0]
        if self.cfg.softplus_h:
            h = torch.nn.functional.softplus(h)
        return h, out[..., 1], out[..., 2]

    def state(self, xyt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Physical (h, u, v) regardless of the variable formulation, with the
        dry-safe divide for conservative outputs."""
        a, b, c = self.forward(xyt)
        if self.cfg.variables == "primitive":
            return a, b, c
        # conservative: a=h, b=hu, c=hv
        return a, velocity(a, b), velocity(a, c)


def layer_diagnostics(model: nn.Module) -> list[dict[str, float]]:
    """Per SVD layer of a trained model: the orthogonality defect
    D_U = ||U^T U - I||_2, the largest relative gap between the trained |s|
    and the effective weight's singular values, and that weight's extreme
    singular values.

    These are the quantities that decide whether the cnPINN mechanism holds
    after training: D_U near 0 with a small mismatch means diag(s) really is
    the weight's spectrum; anything else means the reparameterization is
    doing something other than spectral control.
    """
    rows = []
    with torch.no_grad():
        for lay in (l for l in model.net if isinstance(l, SVDLinear)):
            UtU = lay.U.T @ lay.U
            eye = torch.eye(UtU.shape[0], dtype=UtU.dtype, device=UtU.device)
            sv = torch.linalg.svdvals(lay.weight())
            s_abs = torch.sort(lay.s.abs(), descending=True).values
            rows.append({
                "D_U": float(torch.linalg.matrix_norm(UtU - eye, ord=2)),
                "sv_mismatch": float(((sv - s_abs).abs()
                                      / sv.abs().clamp(min=1e-12)).max()),
                "s_max": float(sv.max()),
                "s_min": float(sv.min()),
            })
    return rows


# ==========================================================================
# FVM-informed PINN
# ==========================================================================

@dataclass
class FVMPINNConfig:
    hidden: int = 128
    layers: int = 5
    activation: str = "tanh"
    fourier_features: int = 32
    fourier_scale: float = 1.0
    fourier_seed: int = 0
    x_range: tuple[float, float] = (0.0, 1.0)
    y_range: tuple[float, float] = (0.0, 1.0)
    t_range: tuple[float, float] = (0.0, 1.0)
    vel_scale: float = 0.0   # if >0, the net predicts velocity = vel_scale*tanh(raw)
                             # and momentum = h*velocity, so momentum vanishes with
                             # depth (bounded velocity -> the FV step cannot blow up
                             # on near-dry cells, the dry-bed NaN failure mode)


class FVMPINN(nn.Module):
    """(x, y, t) -> (xi, hu, hv); depth recovered as softplus(xi + h_s)."""

    def __init__(self, cfg: FVMPINNConfig, h_s: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("h_s", h_s)                     # (ny, nx) still depth
        lo = torch.tensor([cfg.x_range[0], cfg.y_range[0], cfg.t_range[0]])
        hi = torch.tensor([cfg.x_range[1], cfg.y_range[1], cfg.t_range[1]])
        self.register_buffer("in_lo", lo)
        self.register_buffer("in_hi", hi)

        if cfg.fourier_features > 0:
            self.embed = FourierFeatures(3, cfg.fourier_features,
                                         cfg.fourier_scale, cfg.fourier_seed)
            in_dim = 2 * cfg.fourier_features
        else:
            self.embed = None
            in_dim = 3
        act = _ACT[cfg.activation]
        layers: list[nn.Module] = [nn.Linear(in_dim, cfg.hidden), act()]
        for _ in range(cfg.layers - 1):
            layers += [nn.Linear(cfg.hidden, cfg.hidden), act()]
        layers += [nn.Linear(cfg.hidden, 3)]
        self.net = nn.Sequential(*layers)

    def _forward_points(self, xyt: torch.Tensor) -> torch.Tensor:
        z = 2 * (xyt - self.in_lo) / (self.in_hi - self.in_lo) - 1
        if self.embed is not None:
            z = self.embed(z)
        return self.net(z)

    def predict_grid(self, t: float, grid) -> torch.Tensor:
        """Predicted conserved state U=(h,hu,hv) on all cell centers at time t.

        Returns float64 (3, ny, nx) — ready for the differentiable flux step.
        """
        X, Y = grid.centers()
        xyt = torch.stack([
            X.reshape(-1).to(self.in_lo),
            Y.reshape(-1).to(self.in_lo),
            torch.full((grid.ny * grid.nx,), float(t), dtype=self.in_lo.dtype,
                       device=self.in_lo.device),
        ], dim=1)
        out = self._forward_points(xyt).reshape(grid.ny, grid.nx, 3)
        xi = out[..., 0]
        h = torch.nn.functional.softplus(xi + self.h_s)
        if self.cfg.vel_scale > 0:
            # predict bounded velocity; momentum = h*u vanishes in dry cells
            u = self.cfg.vel_scale * torch.tanh(out[..., 1])
            v = self.cfg.vel_scale * torch.tanh(out[..., 2])
            hu, hv = h * u, h * v
        else:
            hu, hv = out[..., 1], out[..., 2]
        U = torch.stack([h, hu, hv], dim=0)
        return U.to(torch.float64)


@dataclass
class FVMResidualSpec:
    cfg: Config                       # solver config (scheme=hllc, order, g, n)
    times: list[float]                # collocation times (increasing)
    n_sub: int = 1                    # CFL-safe substeps per collocation interval
    z: torch.Tensor | None = None     # bed (ny, nx); None -> flat
    stochastic: bool = False          # sample one interval per call (SGD-style)
    step_dtype: torch.dtype = torch.float64   # precision of the FV step in the loss


def _interval_residual(model, spec, z_pad, k):
    grid = spec.cfg.grid
    t0, t1 = spec.times[k], spec.times[k + 1]
    dt = (t1 - t0) / spec.n_sub
    U = model.predict_grid(t0, grid).to(spec.step_dtype)
    for _ in range(spec.n_sub):
        U = step(U, z_pad, dt, spec.cfg)
    U_next = model.predict_grid(t1, grid).to(spec.step_dtype)
    diff = U_next - U
    return (diff**2).mean(), (diff**2).mean(dim=(-2, -1)).detach()


def fvm_residual_loss(
    model: FVMPINN, spec: FVMResidualSpec
) -> tuple[torch.Tensor, dict]:
    """Mean-squared residual of the discrete FV update between consecutive
    collocation times: || U_pred(t_{k+1}) - FV_step^{n_sub}(U_pred(t_k)) ||^2.

    Both states come from the network, so the gradient trains the network to be
    consistent with the classical discrete operator. With ``stochastic=True`` a
    single random interval is used per call (mini-batch SGD over intervals) —
    this is the practical training mode, since backprop through the sequential
    float64 solver steps is the dominant cost.
    """
    grid = spec.cfg.grid
    z = spec.z if spec.z is not None else torch.zeros(
        grid.ny, grid.nx, dtype=spec.step_dtype, device=grid.device)
    z_pad = pad_scalar(z.to(spec.step_dtype), spec.cfg.bc, grid.ng)
    n = len(spec.times) - 1

    if spec.stochastic:
        k = int(torch.randint(0, n, (1,)))
        loss, pc = _interval_residual(model, spec, z_pad, k)
        return loss.to(model.h_s.dtype), {"per_channel_mse": pc.tolist(), "interval": k}

    total = model.h_s.new_zeros(())
    per_channel = torch.zeros(3, dtype=torch.float64, device=grid.device)
    for k in range(n):
        loss, pc = _interval_residual(model, spec, z_pad, k)
        per_channel += pc.double()
        total = total + loss.to(total.dtype)
    total = total / max(n, 1)
    return total, {"per_channel_mse": (per_channel / max(n, 1)).tolist()}


def ic_anchor_loss(model: FVMPINN, grid, U0: torch.Tensor) -> torch.Tensor:
    """MSE of the predicted grid state at t=0 against the initial condition."""
    U_pred = model.predict_grid(0.0, grid)
    return ((U_pred - U0.to(torch.float64)) ** 2).mean().to(model.h_s.dtype)


def data_anchor_loss(
    model: FVMPINN,
    xyt: torch.Tensor,
    h_s_pts: torch.Tensor,
    targets_U: torch.Tensor,
) -> torch.Tensor:
    """MSE at sparse gauge points/times against reference (h, hu, hv).

    xyt: (M, 3) gauge coordinates; h_s_pts: (M,) still-water depth at those
    points (so the softplus depth can be recovered off-grid); targets_U: (M, 3)
    conserved (h, hu, hv) from the reference run.
    """
    out = model._forward_points(xyt.to(model.in_lo))
    h = torch.nn.functional.softplus(out[..., 0] + h_s_pts.to(out))
    # recover momentum with the SAME reparametrization as predict_grid, so the
    # gauge misfit is measured on the network's actual (h, hu, hv), not the raw
    # pre-activation outputs
    if model.cfg.vel_scale > 0:
        hu = h * model.cfg.vel_scale * torch.tanh(out[..., 1])
        hv = h * model.cfg.vel_scale * torch.tanh(out[..., 2])
    else:
        hu, hv = out[..., 1], out[..., 2]
    pred = torch.stack([h, hu, hv], dim=-1)
    return ((pred - targets_U.to(out)) ** 2).mean()


# ==========================================================================
# Strong-form losses (autograd residuals of the 2D SWEs)
# ==========================================================================

@dataclass
class Physics:
    g: float = 9.81
    manning_n: float = 0.0
    # bed gradient at collocation points; returns (z_x, z_y). None -> flat bed.
    bed_grad: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None
    h_floor: float = 1.0e-3     # depth floor for friction/velocity guards


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy/dx with the graph retained (for higher-order / multi-term use)."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]


def _friction_terms(
    h: torch.Tensor, u: torch.Tensor, v: torch.Tensor, phys: Physics
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manning friction accelerations (per unit mass) fr_x, fr_y; 0 if n=0."""
    if phys.manning_n == 0.0:
        z = torch.zeros_like(h)
        return z, z
    hs = torch.clamp(h, min=phys.h_floor)
    speed = torch.sqrt(u * u + v * v)
    coef = phys.g * phys.manning_n**2 * speed / hs ** (4.0 / 3.0)
    return coef * u, coef * v


def pde_residual(model: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
                 phys: Physics) -> torch.Tensor:
    """(N, 3) SWE residual at collocation points, in the model's formulation.

    x, y, t are (N, 1) leaf tensors with requires_grad=True.
    """
    xyt = torch.cat([x, y, t], dim=1)
    zx, zy = (phys.bed_grad(x, y) if phys.bed_grad is not None
              else (torch.zeros_like(x), torch.zeros_like(y)))

    if model.cfg.variables == "primitive":
        h, u, v = (o.unsqueeze(-1) for o in model.forward(xyt))
        fr_x, fr_y = _friction_terms(h, u, v, phys)
        cont = _grad(h, t) + _grad(h * u, x) + _grad(h * v, y)
        mom_x = (_grad(u, t) + u * _grad(u, x) + v * _grad(u, y)
                 + phys.g * (_grad(h, x) + zx) + fr_x)
        mom_y = (_grad(v, t) + u * _grad(v, x) + v * _grad(v, y)
                 + phys.g * (_grad(h, y) + zy) + fr_y)
        return torch.cat([cont, mom_x, mom_y], dim=1)

    # conservative: network outputs (h, hu, hv)
    h, hu, hv = (o.unsqueeze(-1) for o in model.forward(xyt))
    hs = torch.clamp(h, min=phys.h_floor)
    u, v = hu / hs, hv / hs
    fr_x, fr_y = _friction_terms(h, u, v, phys)
    F0, G0 = hu, hv
    F1 = hu * u + 0.5 * phys.g * h * h
    F2 = hu * v
    G1 = hv * u
    G2 = hv * v + 0.5 * phys.g * h * h
    cont = _grad(h, t) + _grad(F0, x) + _grad(G0, y)
    mom_x = _grad(hu, t) + _grad(F1, x) + _grad(F2, y) + phys.g * h * zx + h * fr_x
    mom_y = _grad(hv, t) + _grad(G1, x) + _grad(G2, y) + phys.g * h * zy + h * fr_y
    return torch.cat([cont, mom_x, mom_y], dim=1)


def mse(a: torch.Tensor, b: torch.Tensor | float = 0.0) -> torch.Tensor:
    if isinstance(b, float):
        return (a * a).mean()
    return ((a - b) ** 2).mean()


def ic_loss(model: PINN, xyt0: torch.Tensor,
            h0: torch.Tensor, u0: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """MSE of the network state at t=0 against the initial condition.

    Targets are given in the model's own variables: (h, u, v) for primitive,
    (h, hu, hv) for conservative — the caller supplies matching targets.
    """
    a, b, c = model.forward(xyt0)
    return mse(a, h0) + mse(b, u0) + mse(c, v0)


def data_loss(model: PINN, xyt: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE against sparse gauge data (targets in the model's variables, (M,3))."""
    a, b, c = model.forward(xyt)
    pred = torch.stack([a, b, c], dim=-1)
    return mse(pred, targets)
