"""Smoke + invariant tests for the 2D benchmarks and the internal wall."""

import pytest
import torch

from src.benchmarks import BENCHMARKS, build, vertical_wall_masks
from src.solver import REFLECTIVE, Config, Grid, conserved, run


def cfg_for(bi, scheme="hllc", order=2):
    return Config(grid=bi.grid, bc=bi.bc, scheme=scheme, order=order,
                  g=bi.g, manning_n=bi.manning_n)


@pytest.mark.parametrize("bid", list(BENCHMARKS))
def test_benchmark_builds_and_runs(bid):
    """Every benchmark runs a short interval with finite, non-negative depth."""
    bi = build(bid, n=32)
    cfg = cfg_for(bi)
    out = run(cfg, bi.U0, bi.z, t_end=min(bi.t_end, 0.5),
              output_times=[min(bi.t_end, 0.5)], wall_fn=bi.wall_fn)
    Uf = out["U"][-1]
    assert torch.isfinite(Uf).all()
    assert Uf[0].min().item() >= 0.0


def test_vertical_wall_blocks_mass_flux():
    """A fully-closed vertical wall keeps the two sides isolated: with a big
    head difference and no opening, the downstream total stays put briefly."""
    grid = Grid.from_extent(40, 40, (0, 100, 0, 100))
    cfg = Config(grid=grid, bc=REFLECTIVE, scheme="hllc", order=2)
    X, _ = grid.centers()
    h0 = torch.where(X < 50, torch.full_like(X, 10.0), torch.full_like(X, 1.0))
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    # fully closed wall at x=50 (open interval empty)
    mx, my = vertical_wall_masks(grid, 50.0, 1.0, 0.0, "cpu")
    down0 = float(h0[:, grid.xc > 50].sum())
    out = run(cfg, U, None, t_end=1.0, wall_fn=lambda t: (mx, my))
    down1 = float(out["U"][-1][0][:, grid.xc > 50].sum())
    assert abs(down1 - down0) / down0 < 1e-10  # no mass crossed the wall


def test_open_breach_lets_mass_through():
    """With a central opening, mass flows from the high side to the low side."""
    grid = Grid.from_extent(40, 40, (0, 100, 0, 100))
    cfg = Config(grid=grid, bc=REFLECTIVE, scheme="hllc", order=2)
    X, _ = grid.centers()
    h0 = torch.where(X < 50, torch.full_like(X, 10.0), torch.full_like(X, 1.0))
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    mx, my = vertical_wall_masks(grid, 50.0, 30.0, 70.0, "cpu")  # 40 m breach
    down0 = float(h0[:, grid.xc > 50].sum())
    out = run(cfg, U, None, t_end=2.0, wall_fn=lambda t: (mx, my))
    down1 = float(out["U"][-1][0][:, grid.xc > 50].sum())
    assert down1 > down0 * 1.01  # downstream volume rose through the breach


def test_wall_run_conserves_total_mass_closed_domain():
    """Static wall in a reflective box conserves total volume to round-off."""
    grid = Grid.from_extent(48, 48, (0, 100, 0, 100))
    cfg = Config(grid=grid, bc=REFLECTIVE, scheme="hll", order=2)
    X, _ = grid.centers()
    h0 = torch.where(X < 50, torch.full_like(X, 8.0), torch.full_like(X, 2.0))
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    mx, my = vertical_wall_masks(grid, 50.0, 40.0, 60.0, "cpu")
    out = run(cfg, U, None, t_end=3.0, output_times=[1.0, 2.0, 3.0],
              wall_fn=lambda t: (mx, my))
    m0 = out["mass"][0]
    for m in out["mass"][1:]:
        assert abs(m / m0 - 1.0) < 1e-12


def test_progressive_breach_opens_over_time():
    """B4 progressive wall has more open interfaces at t=T_b than at t=0."""
    bi = build("b4_step_dry_progressive", n=40)
    mx0, _ = bi.wall_fn(0.0)
    mxT, _ = bi.wall_fn(10.0)  # >> T_b
    assert int(mx0.sum()) > int(mxT.sum())  # starts closed, ends open
    assert int(mxT.sum()) == 0              # fully open after T_b


def test_b3_has_bed_and_is_well_balanced_at_dry_rest():
    """B3's still upstream pool over the humps should not spuriously flow in a
    short run (well-balanced hydrostatic reconstruction, before the dam front
    reaches the humps)."""
    bi = build("b3_three_humps", n=60)
    assert bi.z is not None and bi.z.max() > 2.5  # tall hump present
    # a lake-at-rest patch away from the dam stays still: build a still pool
    grid = bi.grid
    X, Y = grid.centers()
    from src.benchmarks import three_hump_bed
    z = three_hump_bed(X, Y)
    h0 = torch.clamp(2.0 - z, min=0.0)  # eta = 2 everywhere, wet where bed<2
    cfg = cfg_for(bi)
    out = run(cfg, conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0)),
              z, t_end=2.0)
    Uf = out["U"][-1]
    assert Uf[1].abs().max().item() < 1e-11
    assert Uf[2].abs().max().item() < 1e-11


# --- vendored co-author reference data -------------------------------------

from src.benchmarks import (  # noqa: E402
    DATA_ROOT, REFERENCE_EXTENT, REFERENCE_NODES, available_reference_schemes,
    has_reference, initial_depth, load_reference_depth, reference_depth_on,
)

REF_BIDS = ["ca_step", "ca_circular_wet", "ca_gaussian"]
#: benchmark id -> the co-authors' IC variant number (data/README.md)
REF_VARIANT = {"ca_step": 1, "ca_circular_wet": 3, "ca_gaussian": 4}

needs_data = pytest.mark.skipif(
    not DATA_ROOT.is_dir(), reason=f"vendored reference data absent ({DATA_ROOT})"
)


def _node_coords():
    """The reference node grid: 501 nodes spanning [0,100] inclusive."""
    x0, x1, _, _ = REFERENCE_EXTENT
    c = torch.linspace(x0, x1, REFERENCE_NODES, dtype=torch.float64)
    Y, X = torch.meshgrid(c, c, indexing="ij")   # axis 0 = y, axis 1 = x
    return X, Y


@needs_data
@pytest.mark.parametrize("bid", REF_BIDS)
def test_reference_t0_matches_analytic_ic(bid):
    """t=0 of every scheme is the analytic IC, which pins the [y, x] axis
    order and the domain mapping. Nodes sitting exactly on an IC discontinuity
    disagree by O(1) through the reference solver's own inside/outside
    rounding, so allow a small fraction of them (the source audit counts 2 of
    251001 for circular, 1628 for gaussian)."""
    X, Y = _node_coords()
    expect = initial_depth(REF_VARIANT[bid], X, Y)
    for scheme in available_reference_schemes(bid):
        got = load_reference_depth(bid, scheme, 0.0)
        diff = (got - expect).abs()
        bad = diff > 1e-8
        assert bad.float().mean() < 0.01, f"{bid}/{scheme}: {int(bad.sum())} nodes differ"
        assert diff[~bad].max() <= 1e-8


@needs_data
def test_reference_axis_order_is_y_then_x():
    """The step IC depends on x alone, so rows must be identical and columns
    must carry the jump. Transposed data would fail this and nothing else."""
    h = load_reference_depth("ca_step", "HLL", 0.0)
    assert torch.allclose(h[0], h[-1])                 # no variation along y
    assert h[:, 0].std() == 0 and h[0, 0] > h[0, -1]   # the jump is along x
    assert int((h[0] > 5).sum()) == 251                # h = 10 for x <= 50


@needs_data
@pytest.mark.parametrize("bid", REF_BIDS)
def test_reference_depth_on_grid(bid):
    """Sampling onto our cell centers keeps shape, dtype and range."""
    grid = Grid.from_extent(128, 128, REFERENCE_EXTENT)
    src = load_reference_depth(bid, "MUSCLRS", 2.0)
    got = reference_depth_on(bid, grid, 2.0, "MUSCLRS")
    assert got.shape == (128, 128)
    assert got.dtype == torch.float64
    assert torch.isfinite(got).all()
    # bilinear interpolation cannot leave the source's range
    assert got.min() >= src.min() - 1e-12
    assert got.max() <= src.max() + 1e-12


@needs_data
def test_reference_absent_for_dry_and_synthetic_cases():
    """ca_circular_dry is ours (h_out = 0 vs their wet Variant 3) and the
    B-series is synthetic; asking for either must fail loudly, not silently
    hand back the wrong IC."""
    assert not has_reference("ca_circular_dry")
    assert not has_reference("b3_three_humps")
    with pytest.raises(KeyError):
        load_reference_depth("ca_circular_dry", "HLL", 0.0)


@needs_data
def test_reference_rejects_unknown_time():
    with pytest.raises(ValueError):
        load_reference_depth("ca_step", "HLL", 0.7)
