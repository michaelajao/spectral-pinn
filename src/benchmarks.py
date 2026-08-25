"""2D dam-break benchmark definitions (B1-B4 and the paper IC variants).

Ported from ``swe-dambreak`` (benchmarks/cases.py), with the analytic
initial-condition definitions folded in from its src/data/reference.py
(the external reference-CSV machinery was not carried over: every
experiment here builds its reference in-process with the HLLC solver).

Each builder returns a ``BenchmarkInstance`` at a requested resolution, so a
sweep can run any scheme/limiter on it and a fine self-convergence reference
uses the identical builder at large N. Physics constants are documented per
case; all fields are float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .solver import (
    REFLECTIVE,
    TRANSMISSIVE,
    BoundaryConditions,
    Grid,
    conserved,
)

Device = str


@dataclass
class BenchmarkInstance:
    name: str
    grid: Grid
    bc: BoundaryConditions
    U0: torch.Tensor
    z: torch.Tensor | None
    g: float
    manning_n: float
    t_end: float
    output_times: list[float]
    reference: str = "self_convergence"    # or "analytic:<name>"
    center: tuple[float, float] | None = None
    wall_fn: Callable[[float], tuple[torch.Tensor, torch.Tensor] | None] | None = None
    metadata: dict = field(default_factory=dict)


def _still(h: torch.Tensor) -> torch.Tensor:
    return conserved(h, torch.zeros_like(h), torch.zeros_like(h))


def vertical_wall_masks(
    grid: Grid, x_wall: float, open_lo: float, open_hi: float, device: Device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masks for a thin vertical wall at x=x_wall, solid except for the open
    y-interval [open_lo, open_hi].

    mx: (ny, nx+1) bool over x-interfaces (interface i sits at x0 + i*dx);
    my: (ny+1, nx) bool over y-interfaces (all False — a vertical wall blocks
    only x-interfaces).
    """
    i_wall = int(round((x_wall - grid.x0) / grid.dx))
    mx = torch.zeros(grid.ny, grid.nx + 1, dtype=torch.bool, device=device)
    closed_rows = (grid.yc < open_lo) | (grid.yc > open_hi)
    mx[:, i_wall] = closed_rows.to(device)
    my = torch.zeros(grid.ny + 1, grid.nx, dtype=torch.bool, device=device)
    return mx, my


# --------------------------------------------------------------------------
# B1 — circular (radial) dam break, flat frictionless bed
# --------------------------------------------------------------------------

def build_b1(n: int, device: Device = "cpu", *, wet: bool) -> BenchmarkInstance:
    """Cylindrical column collapse. Domain [0,50]^2, column radius 11 m at the
    centre, h_in = 10 m, downstream h_out = 1 m (wet) or dry. g = 9.81, flat
    bed, transmissive far field. Reference: fine-grid HLLC self-convergence."""
    grid = Grid.from_extent(n, n, (0.0, 50.0, 0.0, 50.0), device=device)
    X, Y = grid.centers()
    r2 = (X - 25.0) ** 2 + (Y - 25.0) ** 2
    h_out = 1.0 if wet else 0.0
    h0 = torch.where(r2 <= 11.0**2, torch.full_like(X, 10.0), torch.full_like(X, h_out))
    return BenchmarkInstance(
        name=f"b1_circular_{'wet' if wet else 'dry'}",
        grid=grid, bc=TRANSMISSIVE, U0=_still(h0), z=None,
        g=9.81, manning_n=0.0, t_end=1.2, output_times=[0.4, 0.8, 1.2],
        center=(25.0, 25.0),
        metadata={"h_in": 10.0, "h_out": h_out, "radius": 11.0},
    )


# --------------------------------------------------------------------------
# B2 — partial-breach rectangular dam break (Fennema & Chaudhry)
# --------------------------------------------------------------------------

def build_b2(n: int, device: Device = "cpu") -> BenchmarkInstance:
    """200x200 m domain, thin wall at x=100 with a 75 m breach (y in
    [95,170]); upstream h=10 m, downstream h=5 m (wet). g=9.81, reflective
    outer walls, flat bed. Instantaneous full-height breach (static wall)."""
    grid = Grid.from_extent(n, n, (0.0, 200.0, 0.0, 200.0), device=device)
    X, _ = grid.centers()
    h0 = torch.where(X < 100.0, torch.full_like(X, 10.0), torch.full_like(X, 5.0))
    mx, my = vertical_wall_masks(grid, 100.0, 95.0, 170.0, device)
    return BenchmarkInstance(
        name="b2_partial_breach",
        grid=grid, bc=REFLECTIVE, U0=_still(h0), z=None,
        g=9.81, manning_n=0.0, t_end=7.2, output_times=[2.4, 4.8, 7.2],
        wall_fn=lambda t: (mx, my),
        metadata={"h_up": 10.0, "h_down": 5.0, "breach": [95.0, 170.0], "x_wall": 100.0},
    )


# --------------------------------------------------------------------------
# B3 — dam break over three humps, Manning friction, dry downstream
# --------------------------------------------------------------------------

def three_hump_bed(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Bed of Brufau, Garcia-Navarro & Vazquez-Cendon (2002): two small humps
    and one tall hump on a flat floor."""
    z1 = 1.0 - 0.125 * torch.sqrt((X - 30.0) ** 2 + (Y - 6.0) ** 2)
    z2 = 1.0 - 0.125 * torch.sqrt((X - 30.0) ** 2 + (Y - 24.0) ** 2)
    z3 = 3.0 - 0.300 * torch.sqrt((X - 47.5) ** 2 + (Y - 15.0) ** 2)
    z = torch.maximum(torch.maximum(z1, z2), z3)
    return torch.clamp(z, min=0.0)


def build_b3(n: int, device: Device = "cpu") -> BenchmarkInstance:
    """75x30 m domain, dam at x=16, upstream free surface 1.875 m, downstream
    dry; three parabolic humps; Manning n=0.018; g=9.81; reflective walls.
    Tests well-balancing + wet/dry + friction together. ny scaled to keep
    cells ~square (domain is 75x30)."""
    ny = max(2, round(n * 30 / 75))
    grid = Grid.from_extent(n, ny, (0.0, 75.0, 0.0, 30.0), device=device)
    X, Y = grid.centers()
    z = three_hump_bed(X, Y)
    eta_up = 1.875
    h0 = torch.where(X <= 16.0, torch.clamp(eta_up - z, min=0.0), torch.zeros_like(X))
    return BenchmarkInstance(
        name="b3_three_humps",
        grid=grid, bc=REFLECTIVE, U0=_still(h0), z=z,
        g=9.81, manning_n=0.018, t_end=20.0, output_times=[2.0, 6.0, 12.0, 20.0],
        metadata={"eta_up": eta_up, "dam_x": 16.0, "manning_n": 0.018},
    )


# --------------------------------------------------------------------------
# B4 — initial-condition matrix (profile x downstream x breach type)
# --------------------------------------------------------------------------

def build_b4(
    n: int,
    device: Device = "cpu",
    *,
    profile: str = "step",         # "step" | "cone"
    downstream: str = "dry",       # "dry" | "wet"
    breach: str = "instant",       # "instant" | "progressive"
    T_b: float = 2.0,
) -> BenchmarkInstance:
    """100x100 m domain, dam at x=50. Upstream free-surface profile is a step
    (uniform h0=5) or a cone (apex 5 m at (25,50), base radius 20). Downstream
    is dry or wet (0.1*h0). The breach is instantaneous (wall gone for t>0) or
    progressive: a centrally growing opening whose half-width increases
    linearly to full over T_b. g=9.81, reflective outer walls, flat bed."""
    h0, r_cone = 5.0, 20.0
    grid = Grid.from_extent(n, n, (0.0, 100.0, 0.0, 100.0), device=device)
    X, Y = grid.centers()
    h_out = 0.1 * h0 if downstream == "wet" else 0.0

    if profile == "step":
        up = torch.full_like(X, h0)
    elif profile == "cone":
        r = torch.sqrt((X - 25.0) ** 2 + (Y - 50.0) ** 2)
        up = torch.clamp(h0 * (1.0 - r / r_cone), min=h_out)
    else:
        raise ValueError(f"unknown profile {profile}")
    h_init = torch.where(X < 50.0, up, torch.full_like(X, h_out))

    wall_fn = None
    if breach == "progressive":
        half_max = 50.0  # opens to the full domain height at t = T_b

        def wall_fn(t: float):  # noqa: E306
            w = half_max * min(t / T_b, 1.0)
            return vertical_wall_masks(grid, 50.0, 50.0 - w, 50.0 + w, device)
    elif breach != "instant":
        raise ValueError(f"unknown breach {breach}")

    return BenchmarkInstance(
        name=f"b4_{profile}_{downstream}_{breach}",
        grid=grid, bc=REFLECTIVE, U0=_still(h_init), z=None,
        g=9.81, manning_n=0.0, t_end=10.0, output_times=[2.5, 5.0, 10.0],
        wall_fn=wall_fn,
        metadata={"profile": profile, "downstream": downstream, "breach": breach,
                  "T_b": T_b, "h0": h0, "h_out": h_out},
    )


# --------------------------------------------------------------------------
# Coauthor-convention cases (CA*): reproduce the paper's own dam-break variants
# under their exact setup so the PINN section overlaps the classical section.
# Domain [0,100]^2, g=2, Gaussian-hump bed, reflective walls, t_end=2 s,
# snapshots every 0.5 s (see reports/data_audit.md / paper Sect. 2.3).
# --------------------------------------------------------------------------

G_CA = 2.0  # coauthor gravity (nonstandard but matches the reference runs)


def ca_bed(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Gaussian hump Z = 2 exp(-((x-50)^2 + (y-50)^2)/200)."""
    return 2.0 * torch.exp(-((X - 50.0) ** 2 + (Y - 50.0) ** 2) / 200.0)



def initial_depth(variant: int, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Initial water depth h(x, y, 0) for paper IC variants 1..6."""
    one = torch.ones_like(X)
    if variant == 1:  # step: 10 m for x <= 50, else 1 m
        return torch.where(X <= 50.0, 10.0 * one, one)
    if variant == 2:  # rectangular column 5 m in [30,70]^2 over 0.2 m
        inside = (X >= 30.0) & (X <= 70.0) & (Y >= 30.0) & (Y <= 70.0)
        return torch.where(inside, 5.0 * one, 0.2 * one)
    if variant == 3:  # circular column 10 m, R = 20, over 1 m
        inside = (X - 50.0) ** 2 + (Y - 50.0) ** 2 <= 20.0**2
        return torch.where(inside, 10.0 * one, one)
    if variant == 4:  # smooth Gaussian mound, peak 10 m, sigma = 10
        return 10.0 * torch.exp(-((X - 50.0) ** 2 + (Y - 50.0) ** 2) / 200.0)
    if variant == 5:  # parabolic band, 10 m inside, 1 m outside
        y_top = 50.0 + 15.0 - 0.03 * (X - 50.0) ** 2
        inside = (Y <= y_top) & (Y >= y_top - 8.0)
        return torch.where(inside, 10.0 * one, one)
    if variant == 6:  # equilateral triangle (side 40), 10 m inside, 1 m outside
        h_tri = (3.0**0.5) / 2.0 * 40.0
        s3 = 3.0**0.5
        inside = (
            (Y >= 50.0 - 2.0 * h_tri / 3.0)
            & (Y <= 50.0 - s3 * (X - 50.0) + h_tri / 3.0)
            & (Y <= 50.0 + s3 * (X - 50.0) + h_tri / 3.0)
        )
        return torch.where(inside, 10.0 * one, one)
    raise ValueError(f"unknown variant {variant}")


def build_ca_circular(n: int, device: Device = "cpu", *, wet: bool = True) -> BenchmarkInstance:
    """Paper Variant 3 (circular) at the paper's conventions. h_in=10 m in
    R=20 m; downstream h_out=1 m (wet, matches the classical section) or 0
    (dry, added to exercise the FVM-PINN low-momentum collapse story)."""
    grid = Grid.from_extent(n, n, (0.0, 100.0, 0.0, 100.0), device=device)
    X, Y = grid.centers()
    z = ca_bed(X, Y)
    inside = (X - 50.0) ** 2 + (Y - 50.0) ** 2 <= 20.0**2
    h_out = 1.0 if wet else 0.0
    h0 = torch.where(inside, torch.full_like(X, 10.0), torch.full_like(X, h_out))
    return BenchmarkInstance(
        name=f"ca_circular_{'wet' if wet else 'dry'}",
        grid=grid, bc=REFLECTIVE, U0=_still(h0), z=z,
        g=G_CA, manning_n=0.0, t_end=2.0, output_times=[0.5, 1.0, 1.5, 2.0],
        center=(50.0, 50.0),
        metadata={"paper_variant": 3, "wet": wet, "h_in": 10.0, "h_out": h_out},
    )


_CA_VARIANT_NAMES = {1: "step", 2: "rectangular", 3: "circular", 4: "gaussian",
                     5: "parabolic", 6: "triangular"}
# radially symmetric variants get a front-position metric about the centre
_CA_RADIAL = {3, 4}


def build_ca_variant(variant: int, n: int, device: Device = "cpu") -> BenchmarkInstance:
    """Any of the paper's six IC variants at the paper's conventions, with the
    initial depth taken from the same definition used to audit the reference
    data of the submitted dam-break paper (initial_depth below)."""
    grid = Grid.from_extent(n, n, (0.0, 100.0, 0.0, 100.0), device=device)
    X, Y = grid.centers()
    z = ca_bed(X, Y)
    h0 = initial_depth(variant, X, Y)
    return BenchmarkInstance(
        name=f"ca_{_CA_VARIANT_NAMES[variant]}",
        grid=grid, bc=REFLECTIVE, U0=_still(h0), z=z,
        g=G_CA, manning_n=0.0, t_end=2.0, output_times=[0.5, 1.0, 1.5, 2.0],
        center=(50.0, 50.0) if variant in _CA_RADIAL else None,
        metadata={"paper_variant": variant,
                  "h_out": float(h0.min())},
    )


def build_ca_step(n: int, device: Device = "cpu") -> BenchmarkInstance:
    """Paper Variant 1 (step): kept as a named builder for existing configs."""
    return build_ca_variant(1, n, device)


# --------------------------------------------------------------------------
# registry: benchmark id -> builder(n, device)
# --------------------------------------------------------------------------

BENCHMARKS: dict[str, Callable[..., BenchmarkInstance]] = {
    "b1_circular_wet": lambda n, device="cpu": build_b1(n, device, wet=True),
    "b1_circular_dry": lambda n, device="cpu": build_b1(n, device, wet=False),
    "b2_partial_breach": build_b2,
    "b3_three_humps": build_b3,
    "ca_circular_wet": lambda n, device="cpu": build_ca_circular(n, device, wet=True),
    "ca_circular_dry": lambda n, device="cpu": build_ca_circular(n, device, wet=False),
    "ca_step": build_ca_step,
}
# all six paper variants under their own ids (ca_step/ca_circular kept above
# for configs that already reference them; ca_circular == ca_circular_wet)
for _v, _nm in _CA_VARIANT_NAMES.items():
    BENCHMARKS.setdefault(
        f"ca_{_nm}", lambda n, device="cpu", v=_v: build_ca_variant(v, n, device)
    )
for _prof in ("step", "cone"):
    for _down in ("dry", "wet"):
        for _br in ("instant", "progressive"):
            _id = f"b4_{_prof}_{_down}_{_br}"
            BENCHMARKS[_id] = (
                lambda n, device="cpu", p=_prof, d=_down, b=_br: build_b4(
                    n, device, profile=p, downstream=d, breach=b
                )
            )


def build(benchmark_id: str, n: int, device: Device = "cpu") -> BenchmarkInstance:
    if benchmark_id not in BENCHMARKS:
        raise KeyError(f"unknown benchmark '{benchmark_id}'; "
                       f"have {sorted(BENCHMARKS)}")
    return BENCHMARKS[benchmark_id](n, device)
