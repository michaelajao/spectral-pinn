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
