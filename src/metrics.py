"""Error and diagnostic metrics for solver outputs.

Ported unchanged from ``swe-dambreak`` (experiments/metrics.py).

All metrics operate on cell-centered fields. Fields at different resolutions
are compared by bilinearly sampling the finer field onto the coarser grid's
centers (``resample_to``), so a fine-grid reference and a coarse run can be
differenced directly. Everything is float64.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .solver import Grid, primitives


def resample_to(
    field: torch.Tensor,
    src: Grid,
    dst: Grid,
) -> torch.Tensor:
    """Bilinearly sample a cell-centered ``field`` on grid ``src`` onto the
    cell centers of ``dst``. ``field`` is (..., src.ny, src.nx)."""
    # normalized coords of dst centers within src's cell-center span
    x0 = src.x0 + 0.5 * src.dx
    x1 = src.x0 + (src.nx - 0.5) * src.dx
    y0 = src.y0 + 0.5 * src.dy
    y1 = src.y0 + (src.ny - 0.5) * src.dy
    gx = (dst.xc - x0) / (x1 - x0) * 2 - 1
    gy = (dst.yc - y0) / (y1 - y0) * 2 - 1
    GY, GX = torch.meshgrid(gy.to(field), gx.to(field), indexing="ij")
    grid = torch.stack([GX, GY], dim=-1).unsqueeze(0)
    lead = field.shape[:-2]
    inp = field.reshape(-1, 1, src.ny, src.nx)
    out = torch.nn.functional.grid_sample(
        inp, grid.expand(inp.shape[0], -1, -1, -1),
        mode="bilinear", align_corners=True,
    )
    return out.reshape(*lead, dst.ny, dst.nx)


def lp_error(
    a: torch.Tensor, b: torch.Tensor, dx: float, dy: float, p: int = 1
) -> float:
    """Discrete L^p error over cell area (same-shape fields)."""
    diff = (a - b).abs()
    if p == 1:
        return float(diff.sum() * dx * dy)
    if p == 2:
        return float(torch.sqrt((diff**2).sum() * dx * dy))
    raise ValueError("p must be 1 or 2")


def field_errors(
    U: torch.Tensor,
    U_ref: torch.Tensor,
    grid: Grid,
    h_eps: float,
) -> dict[str, float]:
    """L1/L2 errors of h and velocity magnitude between two same-grid states."""
    h, u, v = primitives(U, h_eps)
    hr, ur, vr = primitives(U_ref, h_eps)
    speed = torch.sqrt(u * u + v * v)
    speed_r = torch.sqrt(ur * ur + vr * vr)
    return {
        "L1_h": lp_error(h, hr, grid.dx, grid.dy, 1),
        "L2_h": lp_error(h, hr, grid.dx, grid.dy, 2),
        "L1_speed": lp_error(speed, speed_r, grid.dx, grid.dy, 1),
        "L2_speed": lp_error(speed, speed_r, grid.dx, grid.dy, 2),
    }


def total_volume(U: torch.Tensor, grid: Grid) -> float:
    return float(U[..., 0, :, :].sum()) * grid.cell_area


def relative_mass_drift(masses: list[float]) -> float:
    """max |V_t / V_0 - 1| over a snapshot series."""
    if not masses:
        return 0.0
    m0 = masses[0] if masses[0] != 0 else 1e-300
    return max(abs(m / m0 - 1.0) for m in masses)


def radial_front_position(
    h: torch.Tensor, grid: Grid, threshold: float, center: tuple[float, float]
) -> float:
    """Mean radius of the outermost cells whose depth crosses ``threshold``,
    for radially spreading fronts. Uses cells where h dips below threshold
    with a wet neighbor inside — a robust proxy for the shock ring radius."""
    X, Y = grid.centers()
    r = torch.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2).to(h)
    wet = h > threshold
    if not wet.any():
        return 0.0
    return float(r[wet].max())


def front_position_1d(
    h: torch.Tensor, x: torch.Tensor, threshold: float
) -> float:
    """Rightmost x where depth crosses ``threshold`` downward (1D shock/front)."""
    above = (h > threshold).nonzero()
    if above.numel() == 0:
        return float(x[0])
    i = int(above[-1])
    if i + 1 < len(x):
        f = (h[i] - threshold) / (h[i] - h[i + 1] + 1e-300)
        return float(x[i] + f * (x[i + 1] - x[i]))
    return float(x[i])


@dataclass
class Timer:
    """Context-manager wall-clock timer (CUDA-synchronized if applicable)."""

    device: str = "cpu"
    elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        if self.device == "cuda":
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._t0
