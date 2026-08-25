"""Differentiable, well-balanced finite-volume solver for the 2D shallow-water
equations (single-module port of the ``swe`` package from ``swe-dambreak``).

Sections, in dependency order:

  1. Constants and conserved state U = (h, hu, hv), with dry-safe primitive
     conversion (NaN-safe in the backward pass as well: ``torch.where(wet,
     hu/h, 0)`` back-propagates NaN through the untaken branch when h == 0,
     so denominators are clamped before the select).
  2. Structured grid, ghost cells, and boundary conditions. States are stored
     interior-only; ghosts are attached out-of-place by ``apply_bc`` before
     each flux evaluation, all pure tensor ops (autograd-safe).
  3. Interface reconstruction: first-order and MUSCL with slope limiters on
     (h, u, v, eta); limiting eta and h with the same limiter keeps the
     Audusse et al. (2004) hydrostatic reconstruction well-balanced at
     second order.
  4. Riemann flux kernels (Rusanov, HLL, HLLC), written in interface-local
     coordinates and keyed by name in ``FLUXES``. Pure, vectorized,
     autograd-safe — the HLLC kernel doubles as the FVM-informed loss core.
  5. Pointwise physics: positivity clamp, semi-implicit Manning friction,
     Audusse hydrostatic well-balancing terms, CFL time-step control.
  6. The solver proper: BC ghost fill -> (MUSCL) reconstruction ->
     hydrostatic reconstruction -> Riemann flux -> SSP-RK2 -> positivity ->
     friction split step. ``step``/``rhs`` are pure and differentiable;
     ``run`` adds the adaptive-dt Python loop.
  7. Exact 1D solutions for verification: lake at rest over a bump, Ritter
     (dry-bed) and Stoker (wet-bed) dam breaks. Cf. Delestre et al. (2013).

Everything computes in float64 (``DTYPE``); ML models may run float32 and
cast at the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Literal

import torch


# ==========================================================================
# Constants and conserved state
# ==========================================================================

# --------------------------------------------------------------------------
# Physical and numerical constants
# --------------------------------------------------------------------------

#: Gravitational acceleration [m s^-2].
G: float = 9.81

#: Wet/dry depth threshold [m]. Cells with h <= H_EPS are treated as dry:
#: velocities are zeroed and no momentum flux is admitted.
H_EPS: float = 1.0e-6

#: Default CFL number for adaptive time stepping (SSP-RK2 with first- or
#: second-order reconstruction; cf. Kurganov & Petrova 2007 guidance).
CFL_DEFAULT: float = 0.45

#: Thin-layer threshold factor: cells with h <= THIN_FACTOR * H_EPS are
#: treated as under-resolved films at wet/dry fronts. There, MUSCL drops to
#: first order and HLLC falls back to HLL (Toro's S* contact estimate
#: degrades when the two-rarefaction depth estimate far exceeds both side
#: depths, flipping the sign of the star momentum flux). Standard wet/dry
#: front practice; the resulting 1e-3 m threshold matches GeoClaw's default
#: dry tolerance. See tests/test_solver.py::test_dry_dam_break_positivity.
THIN_FACTOR: float = 1000.0

#: Solver arithmetic precision. ML models may use float32, but every function
#: in the ``swe`` package computes in float64.
DTYPE: torch.dtype = torch.float64

_SQRT2 = math.sqrt(2.0)


# --------------------------------------------------------------------------
# Conserved variables and dry-safe primitive conversion
# --------------------------------------------------------------------------

def conserved(h: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Stack primitives into U = (h, hu, hv) on channel axis -3."""
    return torch.stack([h, h * u, h * v], dim=-3)


def split(U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unbind the channel axis: returns (h, hu, hv)."""
    return U[..., 0, :, :], U[..., 1, :, :], U[..., 2, :, :]


def velocity(h: torch.Tensor, hq: torch.Tensor, h_eps: float = H_EPS) -> torch.Tensor:
    """q = hq / h where h > h_eps, else 0 — safe forward and backward."""
    h_safe = torch.clamp(h, min=h_eps)
    wet = h > h_eps
    return torch.where(wet, hq / h_safe, torch.zeros_like(hq))


def velocity_desingularized(
    h: torch.Tensor, hq: torch.Tensor, eps: float = H_EPS
) -> torch.Tensor:
    """Kurganov–Petrova (2007) smoothed velocity for near-dry cells.

    q = sqrt(2) * h * hq / sqrt(h^4 + max(h^4, eps^4)); smooth in h, so it
    gives usable gradients at wet/dry fronts (used by the ML models).
    """
    h4 = h**4
    denom = torch.sqrt(h4 + torch.clamp(h4, min=eps**4))
    return _SQRT2 * h * hq / denom


def primitives(
    U: torch.Tensor, h_eps: float = H_EPS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(h, u, v) from conserved U with the dry-cell guard."""
    h, hu, hv = split(U)
    return h, velocity(h, hu, h_eps), velocity(h, hv, h_eps)

# ==========================================================================
# Grid, ghost cells, and boundary conditions
# ==========================================================================

BCType = Literal["transmissive", "reflective", "periodic"]

#: Ghost-cell width. Fixed at 2 (MUSCL stencil) regardless of scheme so that
#: shapes are static for ``torch.compile``.
NG: int = 2

# Sign applied to (h, hu, hv) when mirroring across a wall normal to x / y.
_REFLECT_SIGN_X = (1.0, -1.0, 1.0)
_REFLECT_SIGN_Y = (1.0, 1.0, -1.0)


@dataclass(frozen=True)
class Grid:
    """Uniform structured grid over ``[x0, x0+nx*dx] x [y0, y0+ny*dy]``.

    Pure geometry: topography, ICs and friction live in case configs, not here.
    1D problems use ``ny=1`` and run through the identical 2D code path.
    """

    nx: int
    ny: int
    dx: float
    dy: float
    x0: float = 0.0
    y0: float = 0.0
    ng: int = NG
    dtype: torch.dtype = DTYPE
    device: str = "cpu"

    @staticmethod
    def from_extent(
        nx: int,
        ny: int,
        extent: tuple[float, float, float, float],
        *,
        device: str = "cpu",
        dtype: torch.dtype = DTYPE,
    ) -> "Grid":
        """Build a grid from ``(x_min, x_max, y_min, y_max)`` and cell counts."""
        x0, x1, y0, y1 = extent
        return Grid(
            nx=nx, ny=ny,
            dx=(x1 - x0) / nx, dy=(y1 - y0) / ny,
            x0=x0, y0=y0, device=device, dtype=dtype,
        )

    # -- coordinates -------------------------------------------------------

    @cached_property
    def xc(self) -> torch.Tensor:
        """Interior cell-center x coordinates, shape (nx,)."""
        i = torch.arange(self.nx, dtype=self.dtype, device=self.device)
        return self.x0 + (i + 0.5) * self.dx

    @cached_property
    def yc(self) -> torch.Tensor:
        """Interior cell-center y coordinates, shape (ny,)."""
        j = torch.arange(self.ny, dtype=self.dtype, device=self.device)
        return self.y0 + (j + 0.5) * self.dy

    @cached_property
    def xc_g(self) -> torch.Tensor:
        """Cell-center x coordinates including ghost cells, shape (nx + 2*ng,)."""
        i = torch.arange(-self.ng, self.nx + self.ng, dtype=self.dtype, device=self.device)
        return self.x0 + (i + 0.5) * self.dx

    @cached_property
    def yc_g(self) -> torch.Tensor:
        """Cell-center y coordinates including ghost cells, shape (ny + 2*ng,)."""
        j = torch.arange(-self.ng, self.ny + self.ng, dtype=self.dtype, device=self.device)
        return self.y0 + (j + 0.5) * self.dy

    def centers(self, *, ghost: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Meshgrid of cell centers ``(X, Y)``, each ``(ny, nx)`` (row = y)."""
        xs = self.xc_g if ghost else self.xc
        ys = self.yc_g if ghost else self.yc
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        return X, Y

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    def interior(self, t: torch.Tensor) -> torch.Tensor:
        """Strip ghost cells from the last two axes of a padded tensor."""
        ng = self.ng
        return t[..., ng:-ng, ng:-ng]


@dataclass(frozen=True)
class BoundaryConditions:
    """Per-side boundary types. Periodic sides must come in matching pairs."""

    left: BCType = "transmissive"
    right: BCType = "transmissive"
    bottom: BCType = "transmissive"
    top: BCType = "transmissive"

    def __post_init__(self) -> None:
        if (self.left == "periodic") != (self.right == "periodic"):
            raise ValueError("periodic BC requires both left and right periodic")
        if (self.bottom == "periodic") != (self.top == "periodic"):
            raise ValueError("periodic BC requires both bottom and top periodic")


TRANSMISSIVE = BoundaryConditions()
REFLECTIVE = BoundaryConditions("reflective", "reflective", "reflective", "reflective")


def _ghost_1d(
    u: torch.Tensor,
    ng: int,
    side: Literal["low", "high"],
    kind: BCType,
    sign: tuple[float, float, float] | None,
) -> torch.Tensor:
    """Ghost block of width ``ng`` along axis -1 from interior tensor ``u``.

    ``sign`` (per-channel multipliers on axis -3) implements wall reflection of
    the normal momentum; ``None`` means the tensor has no channel axis (bed
    elevation), in which case reflection is a plain mirror.
    """
    w = u.shape[-1]
    if kind == "transmissive":
        edge = u[..., :1] if side == "low" else u[..., -1:]
        return edge.expand(*edge.shape[:-1], ng)
    if kind == "periodic":
        if w < ng:
            raise ValueError(f"periodic BC needs at least {ng} interior cells, got {w}")
        return u[..., -ng:] if side == "low" else u[..., :ng]
    if kind == "reflective":
        # ghost layer k (adjacent to the wall is innermost) mirrors interior
        # cell k; the mirror index is clamped so narrow interiors (w < ng,
        # e.g. 1D runs with ny=1) still get full-width ghost blocks
        m = torch.arange(ng, device=u.device)
        if side == "low":
            sel = torch.clamp(ng - 1 - m, max=w - 1)   # outermost..innermost
        else:
            sel = torch.clamp(w - 1 - m, min=0)        # innermost..outermost
        strip = u.index_select(-1, sel)
        if sign is not None:
            s = strip.new_tensor(sign).view(3, 1, 1)
            strip = strip * s
        return strip
    raise ValueError(f"unknown BC type: {kind}")


def _pad_axis(
    t: torch.Tensor,
    ng: int,
    low: BCType,
    high: BCType,
    sign: tuple[float, float, float] | None,
) -> torch.Tensor:
    lo = _ghost_1d(t, ng, "low", low, sign)
    hi = _ghost_1d(t, ng, "high", high, sign)
    return torch.cat([lo, t, hi], dim=-1)


def apply_bc(U: torch.Tensor, bc: BoundaryConditions, ng: int = NG) -> torch.Tensor:
    """Attach ghost cells to an interior state ``U`` (..., 3, ny, nx).

    Returns a new tensor (..., 3, ny+2ng, nx+2ng); never writes in place.
    x sides are filled first, then y sides read the already-padded columns,
    which gives corner ghosts consistent values.
    """
    U = _pad_axis(U, ng, bc.left, bc.right, _REFLECT_SIGN_X)
    U = U.transpose(-1, -2)
    U = _pad_axis(U, ng, bc.bottom, bc.top, _REFLECT_SIGN_Y)
    return U.transpose(-1, -2)


def pad_scalar(z: torch.Tensor, bc: BoundaryConditions, ng: int = NG) -> torch.Tensor:
    """Attach ghost cells to a scalar field ``z`` (..., ny, nx), e.g. bed elevation.

    Uses the same geometric rule as ``apply_bc`` per side (mirror for
    reflective, edge copy for transmissive, wrap for periodic) with no sign
    flip, so eta = h + z stays consistent across walls.
    """
    z = _pad_axis(z, ng, bc.left, bc.right, None)
    z = z.transpose(-1, -2)
    z = _pad_axis(z, ng, bc.bottom, bc.top, None)
    return z.transpose(-1, -2)

# ==========================================================================
# Interface reconstruction (first-order and MUSCL)
# ==========================================================================

Limiter = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """minmod(a, b): smallest-magnitude slope when signs agree, else 0."""
    return 0.5 * (torch.sign(a) + torch.sign(b)) * torch.minimum(a.abs(), b.abs())


def van_leer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """van Leer harmonic-mean limiter: 2ab/(a+b) when ab > 0, else 0."""
    ab = a * b
    pos = ab > 0
    denom = torch.where(pos, a + b, torch.ones_like(a))  # safe for autograd
    return torch.where(pos, 2.0 * ab / denom, torch.zeros_like(a))


def superbee(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """superbee: maxmod(minmod(2a, b), minmod(a, 2b))."""
    s1 = minmod(2.0 * a, b)
    s2 = minmod(a, 2.0 * b)
    return torch.where(s1.abs() >= s2.abs(), s1, s2)


LIMITERS: dict[str, Limiter] = {
    "minmod": minmod,
    "van_leer": van_leer,
    "superbee": superbee,
}


def traces_first_order(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Piecewise-constant traces along axis -1 of a padded field."""
    return W[..., 1:-2], W[..., 2:-1]


def traces_muscl(W: torch.Tensor, limiter: Limiter) -> tuple[torch.Tensor, torch.Tensor]:
    """Piecewise-linear limited traces along axis -1 of a padded field.

    For cell c with limited slope sigma_c, the right-face value is
    W_c + sigma_c/2 and the left-face value W_c - sigma_c/2; interface k gets
    (WL, WR) from the cells on its two sides.
    """
    dm = W[..., 1:-1] - W[..., :-2]
    dp = W[..., 2:] - W[..., 1:-1]
    sigma = limiter(dm, dp)
    Wc = W[..., 1:-1]
    WL = (Wc + 0.5 * sigma)[..., :-1]
    WR = (Wc - 0.5 * sigma)[..., 1:]
    return WL, WR


def reconstruct_line(
    h: torch.Tensor,
    un: torch.Tensor,
    ut: torch.Tensor,
    z: torch.Tensor,
    *,
    order: int = 2,
    limiter: Limiter = van_leer,
    h_thin: float | None = None,
) -> tuple[torch.Tensor, ...]:
    """Interface traces of (h, un, ut, z) along axis -1.

    Inputs are cell-center fields padded with ghosts along axis -1 (other
    axes arbitrary). ``un``/``ut`` are the velocity components normal and
    tangential to the interfaces. Reconstructs (h, un, ut, eta) and returns

        (hL, unL, utL, zL, hR, unR, utR, zR)

    at the interfaces, with h traces clamped to >= 0. For ``order=1`` the
    traces are the adjacent cell values (z included), so the scheme degrades
    exactly to first order.

    ``h_thin``: if given, slopes are zeroed (local first order) wherever the
    three-cell stencil touches a cell with h <= h_thin — the standard wet/dry
    front treatment; slope overshoots in under-resolved films otherwise feed
    a velocity blow-up.
    """
    eta = h + z
    if order == 1:
        tr = traces_first_order
        (hL, hR), (unL, unR), (utL, utR), (eL, eR) = (
            tr(h), tr(un), tr(ut), tr(eta)
        )
    elif order == 2:
        W = torch.stack([h, un, ut, eta], dim=0)
        WL, WR = traces_muscl(W, limiter)
        if h_thin is not None:
            wet3 = (
                (h[..., 1:-1] > h_thin)
                & (h[..., :-2] > h_thin)
                & (h[..., 2:] > h_thin)
            )
            keepL = wet3[..., :-1]
            keepR = wet3[..., 1:]
            W1L, W1R = traces_first_order(W)
            WL = torch.where(keepL, WL, W1L)
            WR = torch.where(keepR, WR, W1R)
        hL, unL, utL, eL = WL[0], WL[1], WL[2], WL[3]
        hR, unR, utR, eR = WR[0], WR[1], WR[2], WR[3]
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")

    hL = torch.clamp(hL, min=0.0)
    hR = torch.clamp(hR, min=0.0)
    return hL, unL, utL, eL - hL, hR, unR, utR, eR - hR

# ==========================================================================
# Riemann flux kernels
# ==========================================================================

_TINY = 1.0e-14


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------

def physical_flux(
    h: torch.Tensor, un: torch.Tensor, ut: torch.Tensor, g: float = G
) -> torch.Tensor:
    """Exact SWE flux normal to the interface: (h un, h un^2 + g h^2/2, h un ut)."""
    hun = h * un
    return torch.stack([hun, hun * un + 0.5 * g * h * h, hun * ut], dim=-3)


def celerity(h: torch.Tensor, g: float = G) -> torch.Tensor:
    """sqrt(g h) with a derivative-safe floor.

    d(sqrt)/dh is infinite at h = 0, which turns into NaN gradients at dry
    interfaces via 0 * inf products in the backward pass even when the wet/dry
    ``torch.where`` masks zero the forward value. Clamping h at a subnormal
    floor makes the gradient exactly 0 at h = 0 and leaves every physical
    depth untouched.
    """
    return torch.sqrt(g * torch.clamp(h, min=1e-300))


def safe_div(num: torch.Tensor, den: torch.Tensor, tiny: float = _TINY) -> torch.Tensor:
    """num / den with |den| floored at ``tiny`` (sign preserved; sign(0) -> +)."""
    sgn = torch.where(den >= 0, torch.ones_like(den), -torch.ones_like(den))
    den_safe = sgn * torch.clamp(den.abs(), min=tiny)
    return num / den_safe


def wave_speeds(
    hL: torch.Tensor,
    unL: torch.Tensor,
    hR: torch.Tensor,
    unR: torch.Tensor,
    g: float = G,
    h_eps: float = H_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left/right wave-speed estimates SL, SR with dry-bed handling.

    Davies-type direct bounds
        SL = min(uL - aL, uR - aR),  SR = max(uL + aL, uR + aR),
    plus the exact dry-front speeds when one side is dry (h <= h_eps):
        right state dry: SL = uL - aL,  SR = uL + 2 aL
        left  state dry: SL = uR - 2 aR, SR = uR + aR.

    Toro's two-rarefaction estimate with the q_K shock correction was tried
    first and abandoned: in thin films at wet/dry fronts q_K amplifies the
    interface speeds far beyond every *cell* speed (|S| ~ 100x |u|+a), so the
    CFL time step chosen from cell speeds is locally violated and the front
    blows up. The Davies bounds are bounded by the cell speeds by
    construction, hence consistent with the CFL control — and match the
    estimate used by the reference paper's HLL, which helps reconciliation.
    """
    aL = celerity(hL, g)
    aR = celerity(hR, g)

    SL = torch.minimum(unL - aL, unR - aR)
    SR = torch.maximum(unL + aL, unR + aR)

    dryL = hL <= h_eps
    dryR = hR <= h_eps
    SL = torch.where(dryR, unL - aL, SL)
    SR = torch.where(dryR, unL + 2.0 * aL, SR)
    SL = torch.where(dryL, unR - 2.0 * aR, SL)
    SR = torch.where(dryL, unR + aR, SR)
    return SL, SR


def zero_dry_dry(
    F: torch.Tensor, hL: torch.Tensor, hR: torch.Tensor, h_eps: float = H_EPS
) -> torch.Tensor:
    """Force zero flux across interfaces where both sides are dry."""
    wet_any = (hL > h_eps) | (hR > h_eps)
    return F * wet_any.unsqueeze(-3)


# --------------------------------------------------------------------------
# Rusanov (local Lax-Friedrichs)
# --------------------------------------------------------------------------

def rusanov_flux(
    hL: torch.Tensor,
    unL: torch.Tensor,
    utL: torch.Tensor,
    hR: torch.Tensor,
    unR: torch.Tensor,
    utR: torch.Tensor,
    g: float = G,
    h_eps: float = H_EPS,
) -> torch.Tensor:
    """Rusanov flux: F = (F_L + F_R)/2 - a (U_R - U_L)/2 with
    a = max(|un_L| + a_L, |un_R| + a_R). The most dissipative of the three
    kernels; serves as the robustness baseline."""
    FL = physical_flux(hL, unL, utL, g)
    FR = physical_flux(hR, unR, utR, g)
    a = torch.maximum(
        unL.abs() + celerity(hL, g), unR.abs() + celerity(hR, g)
    ).unsqueeze(-3)
    UL = torch.stack([hL, hL * unL, hL * utL], dim=-3)
    UR = torch.stack([hR, hR * unR, hR * utR], dim=-3)
    F = 0.5 * (FL + FR) - 0.5 * a * (UR - UL)
    return zero_dry_dry(F, hL, hR, h_eps)


# --------------------------------------------------------------------------
# HLL (Harten-Lax-van Leer)
# --------------------------------------------------------------------------

def hll_flux(
    hL: torch.Tensor,
    unL: torch.Tensor,
    utL: torch.Tensor,
    hR: torch.Tensor,
    unR: torch.Tensor,
    utR: torch.Tensor,
    g: float = G,
    h_eps: float = H_EPS,
) -> torch.Tensor:
    """HLL flux with Toro's wave-speed estimates (Toro 2001, §10.3):
    F_hll = (S_R F_L - S_L F_R + S_L S_R (U_R - U_L)) / (S_R - S_L)
    selected against F_L (S_L >= 0) and F_R (S_R <= 0)."""
    FL = physical_flux(hL, unL, utL, g)
    FR = physical_flux(hR, unR, utR, g)
    UL = torch.stack([hL, hL * unL, hL * utL], dim=-3)
    UR = torch.stack([hR, hR * unR, hR * utR], dim=-3)

    SL, SR = wave_speeds(hL, unL, hR, unR, g, h_eps)
    SLc = SL.unsqueeze(-3)
    SRc = SR.unsqueeze(-3)

    F_mid = safe_div(SRc * FL - SLc * FR + SLc * SRc * (UR - UL), SRc - SLc)
    F = torch.where(SLc >= 0, FL, torch.where(SRc <= 0, FR, F_mid))
    return zero_dry_dry(F, hL, hR, h_eps)


# --------------------------------------------------------------------------
# HLLC (Toro 2001, §10.4-10.5)
# --------------------------------------------------------------------------

def hllc_flux(
    hL: torch.Tensor,
    unL: torch.Tensor,
    utL: torch.Tensor,
    hR: torch.Tensor,
    unR: torch.Tensor,
    utR: torch.Tensor,
    g: float = G,
    h_eps: float = H_EPS,
) -> torch.Tensor:
    """HLLC flux: restores the contact/shear wave dropped by HLL.

    The tangential momentum is upwinded across the middle wave S*, which is
    what keeps 2D dam-break shear fronts sharp. Fully vectorized over all
    interfaces; pure and differentiable (no in-place ops, no .item(), guarded
    denominators) — this kernel doubles as the FVM-informed PINN loss core.

    Middle-wave speed (Toro 2001, eq. 10.58):
        S* = (S_L h_R (u_R - S_R) - S_R h_L (u_L - S_L))
             / (h_R (u_R - S_R) - h_L (u_L - S_L))
    Star states U*_K = h_K (S_K - u_K)/(S_K - S*) [1, S*, v_K]^T and
    F*_K = F_K + S_K (U*_K - U_K); the flux is picked by the signs of
    S_L, S*, S_R.

    Thin-layer/dry interfaces (either side <= THIN_FACTOR * h_eps) fall back
    to the HLL flux: in under-resolved films the two-rarefaction depth
    estimate far exceeds both side depths, S* lands far from the physical
    contact, and the star momentum flux flips sign — which pumps momentum
    into near-empty cells and blows up the front. The contact restoration
    HLLC exists for only matters in resolved wet regions.
    """
    FL = physical_flux(hL, unL, utL, g)
    FR = physical_flux(hR, unR, utR, g)
    UL = torch.stack([hL, hL * unL, hL * utL], dim=-3)
    UR = torch.stack([hR, hR * unR, hR * utR], dim=-3)

    SL, SR = wave_speeds(hL, unL, hR, unR, g, h_eps)

    num = SL * hR * (unR - SR) - SR * hL * (unL - SL)
    den = hR * (unR - SR) - hL * (unL - SL)
    S_star = safe_div(num, den)
    # degenerate case (both sides nearly still/dry): fall back to mean velocity
    S_star = torch.where(den.abs() < 1e-12, 0.5 * (unL + unR), S_star)

    def star_flux(
        F: torch.Tensor, U: torch.Tensor,
        h: torch.Tensor, un: torch.Tensor, ut: torch.Tensor, S: torch.Tensor,
    ) -> torch.Tensor:
        coef = h * safe_div(S - un, S - S_star)
        U_star = torch.stack([coef, coef * S_star, coef * ut], dim=-3)
        return F + S.unsqueeze(-3) * (U_star - U)

    FsL = star_flux(FL, UL, hL, unL, utL, SL)
    FsR = star_flux(FR, UR, hR, unR, utR, SR)

    SLc = SL.unsqueeze(-3)
    SRc = SR.unsqueeze(-3)
    Ssc = S_star.unsqueeze(-3)
    F = torch.where(
        SLc >= 0,
        FL,
        torch.where(Ssc >= 0, FsL, torch.where(SRc > 0, FsR, FR)),
    )

    # HLL fallback wherever either side is a thin layer or dry (see docstring)
    F_hll = safe_div(SRc * FL - SLc * FR + SLc * SRc * (UR - UL), SRc - SLc)
    F_hll = torch.where(SLc >= 0, FL, torch.where(SRc <= 0, FR, F_hll))
    h_thin = THIN_FACTOR * h_eps
    thin = ((hL <= h_thin) | (hR <= h_thin)).unsqueeze(-3)
    F = torch.where(thin, F_hll, F)

    return zero_dry_dry(F, hL, hR, h_eps)


#: Flux kernels keyed by name for config-driven selection.
FLUXES = {
    "rusanov": rusanov_flux,
    "hll": hll_flux,
    "hllc": hllc_flux,
}

# ==========================================================================
# Pointwise physics: wet/dry, friction, well-balancing, CFL
# ==========================================================================

# --------------------------------------------------------------------------
# Wet/dry front treatment
# --------------------------------------------------------------------------

def enforce_positivity(U: torch.Tensor, h_eps: float = H_EPS) -> torch.Tensor:
    """Clamp h to >= 0 and zero momenta in dry cells (h <= h_eps), so dry
    cells can never advect momentum."""
    h, hu, hv = split(U)
    h = torch.clamp(h, min=0.0)
    wet = h > h_eps
    zero = torch.zeros_like(hu)
    return torch.stack(
        [h, torch.where(wet, hu, zero), torch.where(wet, hv, zero)], dim=-3
    )


# --------------------------------------------------------------------------
# Manning-Strickler bed friction
# --------------------------------------------------------------------------

def apply_friction(
    U: torch.Tensor,
    dt: float | torch.Tensor,
    n_manning: float,
    g: float = G,
    h_eps: float = H_EPS,
) -> torch.Tensor:
    """Semi-implicit Manning friction over one split step of size dt.

    The friction source in the momentum equations is
        d(hu)/dt = -g n^2 u sqrt(u^2 + v^2) / h^(1/3)
                 = -(g n^2 |u_vec| / h^(4/3)) * hu.
    Treating the coefficient explicitly and the momentum implicitly over the
    split step gives the pointwise unconditionally stable update
        hu^{n+1} = hu* / (1 + dt g n^2 |u_vec*| / h^(4/3)),
    which relaxes momentum toward zero and cannot overshoot — the standard
    point-implicit treatment that avoids the stiffness blowup of explicit
    friction near dry fronts. Depth is unchanged.
    """
    if n_manning == 0.0:
        return U
    h, hu, hv = split(U)
    u = velocity(h, hu, h_eps)
    v = velocity(h, hv, h_eps)
    speed = torch.sqrt(u * u + v * v)
    h_safe = torch.clamp(h, min=h_eps)
    denom = 1.0 + dt * g * n_manning**2 * speed / h_safe ** (4.0 / 3.0)
    wet = h > h_eps
    zero = torch.zeros_like(hu)
    return torch.stack(
        [h, torch.where(wet, hu / denom, zero), torch.where(wet, hv / denom, zero)],
        dim=-3,
    )


# --------------------------------------------------------------------------
# Hydrostatic reconstruction (Audusse et al. 2004, SIAM J. Sci. Comput. 25(6))
# --------------------------------------------------------------------------

def hydrostatic_depths(
    hL: torch.Tensor, zL: torch.Tensor, hR: torch.Tensor, zR: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstructed non-negative interface depths (h*_L, h*_R).

    The interface bed is z* = max(z_L, z_R); depths are shifted to
    h*_K = max(0, h_K + z_K - z*) before the Riemann solve.
    """
    z_star = torch.maximum(zL, zR)
    hLs = torch.clamp(hL + zL - z_star, min=0.0)
    hRs = torch.clamp(hR + zR - z_star, min=0.0)
    return hLs, hRs


def pressure_corrections(
    hL: torch.Tensor,
    hLs: torch.Tensor,
    hR: torch.Tensor,
    hRs: torch.Tensor,
    g: float = G,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normal-momentum flux corrections (S_L, S_R), Audusse et al. eq. (4.5).

    S_L is added to the interface flux as seen by the LEFT cell
    (F_{i+1/2}^-), S_R to the flux seen by the RIGHT cell (F_{i-1/2}^+).
    """
    SL = 0.5 * g * (hL * hL - hLs * hLs)
    SR = 0.5 * g * (hR * hR - hRs * hRs)
    return SL, SR


def centered_source(
    h_left_face: torch.Tensor,
    h_right_face: torch.Tensor,
    z_left_face: torch.Tensor,
    z_right_face: torch.Tensor,
    dx: float,
    g: float = G,
) -> torch.Tensor:
    """Second-order in-cell bed source (Audusse et al. eq. (4.7)).

    Arguments are the reconstructed traces *belonging to each cell* at its own
    left/right faces: h_{i-1/2,+}, h_{i+1/2,-}, z_{i-1/2,+}, z_{i+1/2,-}.
    Returns the normal-momentum source density
        -g (h_{i-1/2,+} + h_{i+1/2,-})/2 * (z_{i+1/2,-} - z_{i-1/2,+}) / dx,
    which vanishes identically at first order (flat in-cell traces).
    """
    h_hat = 0.5 * (h_left_face + h_right_face)
    return -g * h_hat * (z_right_face - z_left_face) / dx


# --------------------------------------------------------------------------
# CFL time-step control
# --------------------------------------------------------------------------

def max_wave_speeds(
    U: torch.Tensor, g: float = G, h_eps: float = H_EPS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Max over wet cells of |u| + sqrt(gh) and |v| + sqrt(gh) (0-dim tensors).

    The maxima are over wet cells only (dry cells carry no waves and, before
    the guard, garbage velocities). The masking uses where-then-amax so shapes
    stay static (torch.compile-safe).
    """
    h, u, v = primitives(U, h_eps)
    c = celerity(h, g)
    wet = h > h_eps
    zero = torch.zeros_like(c)
    sx = torch.where(wet, u.abs() + c, zero).amax()
    sy = torch.where(wet, v.abs() + c, zero).amax()
    return sx, sy


def compute_dt(
    U: torch.Tensor,
    grid: Grid,
    g: float = G,
    cfl: float = CFL_DEFAULT,
    h_eps: float = H_EPS,
) -> torch.Tensor:
    """Adaptive dt from the CFL condition (0-dim tensor; caller floats it):
    dt = CFL * min(dx / max_wet(|u| + sqrt(g h)), dy / max_wet(|v| + sqrt(g h))).
    Cf. Kurganov & Petrova (2007) for the CFL guidance."""
    sx, sy = max_wave_speeds(U, g, h_eps)
    tiny = torch.finfo(U.dtype).tiny
    dt_x = grid.dx / torch.clamp(sx, min=tiny)
    dt_y = grid.dy / torch.clamp(sy, min=tiny)
    return cfl * torch.minimum(dt_x, dt_y)

# ==========================================================================
# Solver: SSP-RK2 time stepping over the composed pipeline
# ==========================================================================

@dataclass(frozen=True)
class Config:
    """Everything ``step``/``run`` need besides the state itself."""

    grid: Grid
    bc: BoundaryConditions
    scheme: str = "hllc"
    order: int = 2                      # 1 = first-order, 2 = MUSCL
    limiter: str = "van_leer"
    g: float = G
    h_eps: float = H_EPS
    cfl: float = CFL_DEFAULT
    manning_n: float = 0.0

    def flux_fn(self) -> Callable:
        return FLUXES[self.scheme]

    def limiter_fn(self) -> Callable:
        return LIMITERS[self.limiter]


def _directional_rhs(
    h: torch.Tensor,
    un: torch.Tensor,
    ut: torch.Tensor,
    z: torch.Tensor,
    dx: float,
    cfg: Config,
    wall_iface: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flux divergence + bed source along axis -1 (padded inputs, rows already
    restricted to the interior in axis -2). Returns (..., 3, ny, nx) with
    channels (mass, normal momentum, tangential momentum).

    ``wall_iface`` (bool, broadcastable to the interface axis) marks solid
    internal walls: those interfaces carry zero mass/tangential flux and the
    hydrostatic pressure of each side's own face trace (a free-slip
    impermeable dam), used for partial-breach benchmarks.
    """
    hL, unL, utL, zL, hR, unR, utR, zR = reconstruct_line(
        h, un, ut, z,
        order=cfg.order,
        limiter=cfg.limiter_fn(),
        h_thin=THIN_FACTOR * cfg.h_eps,
    )
    hLs, hRs = hydrostatic_depths(hL, zL, hR, zR)
    F = cfg.flux_fn()(hLs, unL, utL, hRs, unR, utR, cfg.g, cfg.h_eps)

    SL, SR = pressure_corrections(hL, hLs, hR, hRs, cfg.g)
    zero = torch.zeros_like(SL)
    corrL = torch.stack([zero, SL, zero], dim=-3)  # seen by the left cell
    corrR = torch.stack([zero, SR, zero], dim=-3)  # seen by the right cell

    F_minus = F + corrL   # F_{i+1/2} in the update of cell i
    F_plus = F + corrR    # F_{i-1/2} in the update of cell i

    if wall_iface is not None:
        m = wall_iface.unsqueeze(-3)
        wallL = torch.stack([zero, 0.5 * cfg.g * hL * hL, zero], dim=-3)
        wallR = torch.stack([zero, 0.5 * cfg.g * hR * hR, zero], dim=-3)
        F_minus = torch.where(m, wallL, F_minus)
        F_plus = torch.where(m, wallR, F_plus)

    dU = -(F_minus[..., 1:] - F_plus[..., :-1]) / dx

    # second-order in-cell source: cell i's own face traces are
    # (h_{i-1/2,+}, h_{i+1/2,-}) = (hR at left iface, hL at right iface)
    src = centered_source(hR[..., :-1], hL[..., 1:], zR[..., :-1], zL[..., 1:], dx, cfg.g)
    zero_c = torch.zeros_like(src)
    return dU + torch.stack([zero_c, src, zero_c], dim=-3)


def rhs(
    U: torch.Tensor,
    z_pad: torch.Tensor,
    cfg: Config,
    wall: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """dU/dt for the interior state U (..., 3, ny, nx); z_pad is the
    ghost-padded bed (ny+2ng, nx+2ng).

    ``wall`` = (mx, my): bool masks of solid internal-wall interfaces, mx of
    shape (ny, nx+1) over x-interfaces, my of shape (ny+1, nx) over
    y-interfaces. None means no internal walls.
    """
    ng = NG
    Up = apply_bc(U, cfg.bc, ng)
    h, u, v = primitives(Up, cfg.h_eps)
    mx = my = None
    if wall is not None:
        mx, my = wall

    # x-direction: interior rows, reconstruct along x
    rows = slice(ng, -ng)
    dU_x = _directional_rhs(
        h[..., rows, :], u[..., rows, :], v[..., rows, :], z_pad[rows, :],
        cfg.grid.dx, cfg, wall_iface=mx,
    )

    # y-direction: transpose so y becomes axis -1; normal velocity is v
    ht = h.transpose(-1, -2)[..., rows, :]
    ut_ = u.transpose(-1, -2)[..., rows, :]
    vt = v.transpose(-1, -2)[..., rows, :]
    zt = z_pad.transpose(-1, -2)[rows, :]
    my_t = my.transpose(-1, -2) if my is not None else None
    dU_y_t = _directional_rhs(ht, vt, ut_, zt, cfg.grid.dy, cfg, wall_iface=my_t)
    # back to (..., 3, ny, nx); channels arrive as (mass, y-mom, x-mom)
    dU_y = dU_y_t.transpose(-1, -2)[..., (0, 2, 1), :, :]

    return dU_x + dU_y


def step(
    U: torch.Tensor,
    z_pad: torch.Tensor,
    dt: float | torch.Tensor,
    cfg: Config,
    wall: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """One SSP-RK2 (Heun) step + positivity + friction split. Pure and
    differentiable; dt is a fixed input (the ML loss path never touches the
    adaptive CFL reduction)."""
    U1 = enforce_positivity(U + dt * rhs(U, z_pad, cfg, wall), cfg.h_eps)
    U2 = 0.5 * (U + U1 + dt * rhs(U1, z_pad, cfg, wall))
    U2 = enforce_positivity(U2, cfg.h_eps)
    U2 = apply_friction(U2, dt, cfg.manning_n, cfg.g, cfg.h_eps)
    return U2


def run(
    cfg: Config,
    U0: torch.Tensor,
    z: torch.Tensor | None,
    t_end: float,
    output_times: list[float] | None = None,
    *,
    wall_fn=None,
    max_steps: int = 10_000_000,
) -> dict:
    """Advance U0 to t_end with adaptive dt; snapshot at ``output_times``.

    z is the cell-centered bed (ny, nx) or None for a flat bed. ``wall_fn``,
    if given, maps time t -> (mx, my) internal-wall masks (or None) evaluated
    fresh each step, for static or progressive breaches. Returns
    {"t": [...], "U": [tensors], "mass": [...], "n_steps": int}.
    """
    grid = cfg.grid
    if z is None:
        z = torch.zeros(grid.ny, grid.nx, dtype=U0.dtype, device=U0.device)
    z_pad = pad_scalar(z, cfg.bc, grid.ng)

    outputs = sorted(set(output_times or []) | {float(t_end)})
    snaps: dict = {"t": [], "U": [], "mass": [], "n_steps": 0}

    def take_snapshot(t: float, U: torch.Tensor) -> None:
        snaps["t"].append(t)
        snaps["U"].append(U.detach().clone())
        snaps["mass"].append(float(U[..., 0, :, :].sum()) * grid.cell_area)

    U = U0
    t = 0.0
    if outputs and outputs[0] == 0.0:
        outputs.pop(0)
        take_snapshot(0.0, U)

    for _ in range(max_steps):
        if t >= t_end - 1e-12:
            break
        dt = float(compute_dt(U, grid, cfg.g, cfg.cfl, cfg.h_eps))
        # never step past the next output time
        dt = min(dt, outputs[0] - t)
        wall = wall_fn(t) if wall_fn is not None else None
        U = step(U, z_pad, dt, cfg, wall)
        t += dt
        snaps["n_steps"] += 1
        if t >= outputs[0] - 1e-12:
            take_snapshot(t, U)
            outputs.pop(0)
            if not outputs:
                break
    else:
        raise RuntimeError(f"max_steps={max_steps} exceeded at t={t:.6g}")
    return snaps

# ==========================================================================
# Exact 1D solutions for verification
# ==========================================================================

# --------------------------------------------------------------------------
# Lake at rest over a smooth bump
#
# Bed:    z(x) = max(0, 0.2 - 0.05 (x - 10)^2)  on  x in [0, 25]
# State:  eta = h + z = LAKE_ETA0 (default 0.5), u = v = 0.
# The exact solution is the initial state for all time; any velocity a scheme
# produces is spurious. Cf. Audusse et al. (2004).
# --------------------------------------------------------------------------

LAKE_X_MIN: float = 0.0
LAKE_X_MAX: float = 25.0
LAKE_ETA0: float = 0.5


def lake_bed(x: torch.Tensor) -> torch.Tensor:
    """Smooth parabolic bump centered at x = 10."""
    return torch.clamp(0.2 - 0.05 * (x - 10.0) ** 2, min=0.0)


def lake_initial_depth(x: torch.Tensor, eta0: float = LAKE_ETA0) -> torch.Tensor:
    """h(x, 0) = eta0 - z(x) (positive everywhere for eta0 > 0.2)."""
    return eta0 - lake_bed(x)


# --------------------------------------------------------------------------
# Ritter (1892): 1D dam break on a dry, frictionless bed
#
# Initial state: h = h0 for x <= x0, dry for x > x0, u = 0. For t > 0, with
# c0 = sqrt(g h0) and xi = (x - x0)/t:
#     x <= x0 - c0 t      : h = h0,                u = 0
#     inside the fan      : h = (2 c0 - xi)^2/(9g), u = 2 (xi + c0)/3
#     x >= x0 + 2 c0 t    : h = 0,                 u = 0
# --------------------------------------------------------------------------

def ritter_solution(
    x: torch.Tensor, t: float, h0: float, x0: float = 0.0, g: float = G
) -> tuple[torch.Tensor, torch.Tensor]:
    """(h, u) at positions x and time t > 0."""
    c0 = (g * h0) ** 0.5
    xi = (x - x0) / t
    h_fan = (2.0 * c0 - xi) ** 2 / (9.0 * g)
    u_fan = 2.0 * (xi + c0) / 3.0

    h = torch.where(
        xi <= -c0,
        torch.full_like(x, h0),
        torch.where(xi >= 2.0 * c0, torch.zeros_like(x), h_fan),
    )
    u = torch.where(
        (xi > -c0) & (xi < 2.0 * c0), u_fan, torch.zeros_like(x)
    )
    return h, u


# --------------------------------------------------------------------------
# Stoker (1957): 1D dam break on a wet, frictionless bed
#
# Initial state: h = h_l for x <= x0, h = h_r (0 < h_r < h_l) for x > x0,
# u = 0. The solution is a left rarefaction, a constant middle state
# (h_m, u_m), and a right-moving shock at speed s. The middle depth solves
#     2 (sqrt(g h_l) - sqrt(g h_m))
#         = (h_m - h_r) sqrt( g (h_m + h_r) / (2 h_m h_r) ),
# (rarefaction invariant = shock jump relation), solved here by bisection in
# float64. Cf. Toro (2001) §5.
# --------------------------------------------------------------------------

def stoker_middle_state(h_l: float, h_r: float, g: float = G) -> tuple[float, float, float]:
    """(h_m, u_m, s): middle depth/velocity and shock speed."""
    if not 0.0 < h_r < h_l:
        raise ValueError("Stoker requires 0 < h_r < h_l")

    def f(hm: float) -> float:
        return (
            2.0 * (math.sqrt(g * h_l) - math.sqrt(g * hm))
            - (hm - h_r) * math.sqrt(0.5 * g * (hm + h_r) / (hm * h_r))
        )

    lo, hi = h_r * (1.0 + 1e-14), h_l
    for _ in range(200):  # bisection to ~1e-16 relative
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    h_m = 0.5 * (lo + hi)
    u_m = 2.0 * (math.sqrt(g * h_l) - math.sqrt(g * h_m))
    s = h_m * u_m / (h_m - h_r)
    return h_m, u_m, s


def stoker_solution(
    x: torch.Tensor, t: float, h_l: float, h_r: float, x0: float = 0.0, g: float = G
) -> tuple[torch.Tensor, torch.Tensor]:
    """(h, u) at positions x and time t > 0."""
    h_m, u_m, s = stoker_middle_state(h_l, h_r, g)
    c_l = math.sqrt(g * h_l)
    c_m = math.sqrt(g * h_m)
    xi = (x - x0) / t

    h_fan = (2.0 * c_l - xi) ** 2 / (9.0 * g)
    u_fan = 2.0 * (xi + c_l) / 3.0

    h = torch.where(
        xi <= -c_l,
        torch.full_like(x, h_l),
        torch.where(
            xi < u_m - c_m,
            h_fan,
            torch.where(xi < s, torch.full_like(x, h_m), torch.full_like(x, h_r)),
        ),
    )
    u = torch.where(
        xi <= -c_l,
        torch.zeros_like(x),
        torch.where(
            xi < u_m - c_m,
            u_fan,
            torch.where(xi < s, torch.full_like(x, u_m), torch.zeros_like(x)),
        ),
    )
    return h, u
