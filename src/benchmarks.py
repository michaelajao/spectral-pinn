"""2D dam-break benchmark definitions (B1-B4 and the paper IC variants).

Ported from ``swe-dambreak`` (benchmarks/cases.py), with the analytic
initial-condition definitions folded in from its src/data/reference.py.

Two independent sources of truth live here. Every experiment builds its own
reference in-process with the HLLC solver, which is what the error tables in
reports/ are measured against. Separately, the bottom of this module reads
the co-authors' vendored solver output in data/ (depth only, 501x501 nodes)
so our solver can be cross-checked against theirs on the three IC variants
both cover -- see data/README.md.

Each builder returns a ``BenchmarkInstance`` at a requested resolution, so a
sweep can run any scheme/limiter on it and a fine self-convergence reference
uses the identical builder at large N. Physics constants are documented per
case; all fields are float64.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
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


# --------------------------------------------------------------------------
# vendored reference data: the co-authors' solver output for the same ICs
#
# Depth only, on a 501x501 *node* grid over [0,100]^2 (dx = dy = 0.2 m), five
# snapshots to t = 2 s. Axis 0 is y, axis 1 is x -- verified against the
# analytic IC by tests/test_benchmarks.py, not assumed. Momentum is absent
# from the drop, so anything built on this compares h and nothing else.
# --------------------------------------------------------------------------

#: Location of the vendored data. Resolved from this file rather than the
#: working directory, so `python -m src.run` and pytest agree no matter where
#: they are invoked from; SPECTRAL_PINN_DATA points at an out-of-tree copy.
DATA_ROOT = Path(
    os.environ.get("SPECTRAL_PINN_DATA")
    or Path(__file__).resolve().parents[1] / "data"
)

#: benchmark id -> the co-authors' variant directory for the same IC.
#: Absent by design: ca_circular_dry sets h_out = 0 where their Variant 3 is
#: wet (h_out = 1), and the B-series is synthetic. See data/README.md.
REFERENCE_VARIANTS: dict[str, str] = {
    "ca_step": "Variant 1 Step Dam-Break",
    "ca_circular_wet": "Variant 3 Circular Dam-Break",
    "ca_gaussian": "Variant 4 Gaussian Dam-Break",
}

REFERENCE_SCHEMES = ("HLL", "LW", "MUSCLRS")
REFERENCE_TIMES = (0.0, 0.5, 1.0, 1.5, 2.0)
REFERENCE_EXTENT = (0.0, 100.0, 0.0, 100.0)
REFERENCE_NODES = 501


def has_reference(benchmark_id: str) -> bool:
    """Whether vendored co-author output exists for this benchmark's IC."""
    return benchmark_id in REFERENCE_VARIANTS


def reference_dir(benchmark_id: str, scheme: str) -> Path:
    """Directory holding one scheme's snapshots for a benchmark's IC.

    The drop's directory names are not formed by a single rule -- Variant 4's
    HLL run is spelled ``gaussians`` where its LW and MUSCL-RS runs are
    ``gaussian`` -- so match on the scheme suffix instead of building a name.
    """
    if not has_reference(benchmark_id):
        raise KeyError(
            f"no vendored reference for '{benchmark_id}'; "
            f"have {sorted(REFERENCE_VARIANTS)} (see data/README.md)"
        )
    if scheme not in REFERENCE_SCHEMES:
        raise ValueError(f"unknown scheme '{scheme}'; have {REFERENCE_SCHEMES}")
    variant = DATA_ROOT / REFERENCE_VARIANTS[benchmark_id]
    hits = sorted(variant.glob(f"solution_outputs_*_numerical_{scheme}"))
    if not hits:
        raise FileNotFoundError(
            f"no '{scheme}' run under {variant}; is data/ present?"
        )
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous '{scheme}' runs under {variant}: {hits}")
    return hits[0]


def available_reference_schemes(benchmark_id: str) -> tuple[str, ...]:
    """Schemes actually present on disk for this benchmark (possibly empty)."""
    if not has_reference(benchmark_id):
        return ()
    out = []
    for scheme in REFERENCE_SCHEMES:
        try:
            reference_dir(benchmark_id, scheme)
        except (FileNotFoundError, RuntimeError):
            continue
        out.append(scheme)
    return tuple(out)


@functools.lru_cache(maxsize=None)
def _load_csv(path: str) -> np.ndarray:
    """Parse one snapshot. Cached: these are ~6 MB of ASCII and ~2 s each."""
    a = np.loadtxt(path, delimiter=",")
    if a.shape != (REFERENCE_NODES, REFERENCE_NODES):
        raise ValueError(f"{path}: expected {REFERENCE_NODES}^2 nodes, got {a.shape}")
    return a


def load_reference_depth(
    benchmark_id: str,
    scheme: str = "MUSCLRS",
    t: float = 2.0,
    *,
    device: Device = "cpu",
) -> torch.Tensor:
    """Co-author depth field at time ``t`` as a (501, 501) float64 tensor,
    indexed [y, x] on the node grid spanning ``REFERENCE_EXTENT``."""
    match = [s for s in REFERENCE_TIMES if abs(s - t) < 1e-9]
    if not match:
        raise ValueError(f"no snapshot at t={t}; have {REFERENCE_TIMES}")
    path = reference_dir(benchmark_id, scheme) / f"h_t{match[0]:.1f}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.from_numpy(_load_csv(str(path)).copy()).to(device)


def reference_depth_on(
    benchmark_id: str,
    grid: Grid,
    t: float = 2.0,
    scheme: str = "MUSCLRS",
) -> torch.Tensor:
    """Co-author depth bilinearly sampled onto ``grid``'s cell centers,
    shape (grid.ny, grid.nx) -- directly differenceable against our fields.

    Their field is node-centered and ours is cell-centered, so this is a
    node-to-center interpolation, not the center-to-center ``resample_to``
    used elsewhere.
    """
    field = load_reference_depth(benchmark_id, scheme, t, device=str(grid.device))
    x0, x1, y0, y1 = REFERENCE_EXTENT
    gx = (grid.xc.to(field) - x0) / (x1 - x0) * 2 - 1
    gy = (grid.yc.to(field) - y0) / (y1 - y0) * 2 - 1
    GY, GX = torch.meshgrid(gy, gx, indexing="ij")
    samp = torch.stack([GX, GY], dim=-1).unsqueeze(0)
    out = torch.nn.functional.grid_sample(
        field.reshape(1, 1, *field.shape), samp,
        mode="bilinear", align_corners=True,
    )
    return out.reshape(grid.ny, grid.nx)
