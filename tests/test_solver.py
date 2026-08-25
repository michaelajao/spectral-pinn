"""Tests for src/solver.py — the merged differentiable SWE solver.

Concatenation of swe-dambreak's test_{state,grid,reconstruction,fluxes,
friction_timestep,analytic,solver}.py with imports rewritten for the
single-module layout; test logic is unchanged.
"""

import math

import pytest
import torch

from src import solver as analytic
from src.solver import (
    FLUXES,
    G,
    LIMITERS,
    NG,
    REFLECTIVE,
    TRANSMISSIVE,
    BoundaryConditions,
    Config,
    Grid,
    apply_bc,
    apply_friction,
    compute_dt,
    conserved,
    max_wave_speeds,
    minmod,
    pad_scalar,
    physical_flux,
    primitives,
    reconstruct_line,
    rhs,
    run,
    split,
    step,
    superbee,
    traces_first_order,
    traces_muscl,
    van_leer,
    velocity,
    velocity_desingularized,
)


# ==========================================================================
# from tests/test_state.py
# ==========================================================================

"""Tests for swe.state: dry guard values AND gradients."""




def test_roundtrip_wet():
    torch.manual_seed(0)
    h = torch.rand(4, 5, dtype=torch.float64) + 0.5
    u = torch.randn(4, 5, dtype=torch.float64)
    v = torch.randn(4, 5, dtype=torch.float64)
    U = conserved(h, u, v)
    h2, u2, v2 = primitives(U)
    assert torch.allclose(h2, h) and torch.allclose(u2, u) and torch.allclose(v2, v)


def test_dry_cells_get_zero_velocity():
    h = torch.tensor([[0.0, 1e-12, 1e-6, 1.0]], dtype=torch.float64)
    hu = torch.tensor([[1.0, 1.0, 1.0, 2.0]], dtype=torch.float64)
    u = velocity(h, hu, h_eps=1e-6)
    assert u[0, 0] == 0 and u[0, 1] == 0 and u[0, 2] == 0  # h <= h_eps -> dry
    assert u[0, 3] == 2.0


def test_velocity_gradients_finite_at_dry_cells():
    h = torch.tensor([[0.0, 1e-12, 0.5]], dtype=torch.float64, requires_grad=True)
    hu = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64, requires_grad=True)
    u = velocity(h, hu)
    u.sum().backward()
    assert torch.isfinite(h.grad).all()
    assert torch.isfinite(hu.grad).all()


def test_desingularized_matches_plain_when_wet_and_finite_grad_when_dry():
    h = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    hu = torch.tensor([[0.5, -1.0]], dtype=torch.float64)
    u = velocity_desingularized(h, hu, eps=1e-6)
    assert torch.allclose(u, hu / h, atol=1e-10)

    h0 = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    hu0 = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
    u0 = velocity_desingularized(h0, hu0, eps=1e-6)
    assert u0.abs().max() == 0
    u0.sum().backward()
    assert torch.isfinite(h0.grad).all() and torch.isfinite(hu0.grad).all()


def test_split_stack_batched():
    U = torch.randn(7, 3, 4, 5, dtype=torch.float64)
    h, hu, hv = split(U)
    assert h.shape == (7, 4, 5)
    U2 = torch.stack([h, hu, hv], dim=-3)
    assert torch.equal(U, U2)

# ==========================================================================
# from tests/test_grid.py
# ==========================================================================

"""Tests for swe.grid: coordinates, ghost-cell fills, autograd safety."""




def make_state(ny=4, nx=5, batch=()):
    torch.manual_seed(0)
    return torch.randn(*batch, 3, ny, nx, dtype=torch.float64)


def test_grid_coordinates():
    g = Grid.from_extent(nx=10, ny=4, extent=(0.0, 10.0, -2.0, 2.0))
    assert g.dx == pytest.approx(1.0) and g.dy == pytest.approx(1.0)
    assert g.xc[0].item() == pytest.approx(0.5)
    assert g.xc[-1].item() == pytest.approx(9.5)
    assert g.yc[0].item() == pytest.approx(-1.5)
    assert g.xc_g.shape == (10 + 2 * NG,)
    assert g.xc_g[0].item() == pytest.approx(0.5 - NG * 1.0)
    X, Y = g.centers()
    assert X.shape == (4, 10)
    # x varies along columns (axis -1), y along rows (axis -2)
    assert torch.all(X[0] == X[-1]) and torch.all(Y[:, 0] == Y[:, -1])


def test_transmissive_copies_edge():
    U = make_state()
    Up = apply_bc(U, TRANSMISSIVE)
    assert Up.shape == (3, 4 + 2 * NG, 5 + 2 * NG)
    # left ghosts equal first interior column
    for k in range(NG):
        assert torch.equal(Up[:, NG:-NG, k], U[:, :, 0])
        assert torch.equal(Up[:, NG:-NG, -1 - k], U[:, :, -1])
        assert torch.equal(Up[:, k, NG:-NG], U[:, 0, :])
        assert torch.equal(Up[:, -1 - k, NG:-NG], U[:, -1, :])


def test_reflective_mirrors_and_negates_normal_momentum():
    U = make_state()
    Up = apply_bc(U, REFLECTIVE)
    # ghost column ng-1 mirrors interior column 0; ghost ng-2 mirrors column 1
    for k in range(NG):
        ghost = Up[:, NG:-NG, NG - 1 - k]
        inner = U[:, :, k]
        assert torch.equal(ghost[0], inner[0])          # h mirrored
        assert torch.equal(ghost[1], -inner[1])         # hu negated (normal to x-wall)
        assert torch.equal(ghost[2], inner[2])          # hv tangential
    for k in range(NG):
        ghost = Up[:, NG - 1 - k, NG:-NG]
        inner = U[:, k, :]
        assert torch.equal(ghost[0], inner[0])
        assert torch.equal(ghost[1], inner[1])          # hu tangential to y-wall
        assert torch.equal(ghost[2], -inner[2])         # hv negated


def test_periodic_wraps():
    U = make_state()
    bc = BoundaryConditions("periodic", "periodic", "periodic", "periodic")
    Up = apply_bc(U, bc)
    assert torch.equal(Up[:, NG:-NG, :NG], U[:, :, -NG:])
    assert torch.equal(Up[:, NG:-NG, -NG:], U[:, :, :NG])
    assert torch.equal(Up[:, :NG, NG:-NG], U[:, -NG:, :])


def test_periodic_must_pair():
    with pytest.raises(ValueError):
        BoundaryConditions(left="periodic", right="transmissive")


def test_batch_dim_broadcasts():
    U = make_state(batch=(7,))
    Up = apply_bc(U, REFLECTIVE)
    assert Up.shape == (7, 3, 4 + 2 * NG, 5 + 2 * NG)
    # each batch element matches the unbatched fill
    one = apply_bc(U[3], REFLECTIVE)
    assert torch.equal(Up[3], one)


def test_apply_bc_is_differentiable_and_out_of_place():
    U = make_state().requires_grad_(True)
    Up = apply_bc(U, REFLECTIVE)
    Up.sum().backward()
    assert U.grad is not None
    assert torch.isfinite(U.grad).all()


def test_pad_scalar_mirrors_without_sign_flip():
    z = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    zp = pad_scalar(z, REFLECTIVE)
    assert zp.shape == (4 + 2 * NG, 5 + 2 * NG)
    assert torch.equal(zp[NG:-NG, NG - 1], z[:, 0])
    zp2 = pad_scalar(z, TRANSMISSIVE)
    assert torch.equal(zp2[NG:-NG, 0], z[:, 0])


def test_reflective_narrow_interior_full_ghost_width():
    """ny=1 (1D as degenerate 2D) with reflective walls must still produce
    full ng-wide ghost blocks (regression: slicing under-filled them)."""
    U = make_state(ny=1, nx=6)
    Up = apply_bc(U, REFLECTIVE)
    assert Up.shape == (3, 1 + 2 * NG, 6 + 2 * NG)
    # every ghost row mirrors the single interior row (hv negated)
    for k in range(NG):
        assert torch.equal(Up[0, k, NG:-NG], U[0, 0, :])
        assert torch.equal(Up[2, k, NG:-NG], -U[2, 0, :])
        assert torch.equal(Up[0, -1 - k, NG:-NG], U[0, 0, :])


def test_periodic_narrow_interior_raises():
    U = make_state(ny=1, nx=6)
    bc = BoundaryConditions("transmissive", "transmissive", "periodic", "periodic")
    with pytest.raises(ValueError, match="periodic"):
        apply_bc(U, bc)


def test_interior_roundtrip():
    g = Grid.from_extent(nx=5, ny=4, extent=(0, 5, 0, 4))
    U = make_state()
    assert torch.equal(g.interior(apply_bc(U, TRANSMISSIVE)), U)

# ==========================================================================
# from tests/test_reconstruction.py
# ==========================================================================

"""Tests for swe.reconstruction: limiter properties and trace correctness."""




def test_limiter_pointwise_values():
    a = torch.tensor([1.0, 1.0, -1.0, 1.0, 0.0], dtype=torch.float64)
    b = torch.tensor([2.0, 1.0, -2.0, -1.0, 3.0], dtype=torch.float64)
    assert torch.allclose(minmod(a, b), torch.tensor([1.0, 1.0, -1.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.allclose(van_leer(a, b), torch.tensor([4 / 3, 1.0, -4 / 3, 0.0, 0.0], dtype=torch.float64))
    assert torch.allclose(superbee(a, b), torch.tensor([2.0, 1.0, -2.0, 0.0, 0.0], dtype=torch.float64))


def test_limiters_are_symmetric_and_zero_at_extrema():
    torch.manual_seed(1)
    a = torch.randn(100, dtype=torch.float64)
    b = torch.randn(100, dtype=torch.float64)
    for lim in LIMITERS.values():
        assert torch.allclose(lim(a, b), lim(b, a), atol=1e-14)
        opp = a * b <= 0
        assert torch.all(lim(a, b)[opp] == 0)


def test_first_order_traces_are_adjacent_cells():
    W = torch.arange(8, dtype=torch.float64) ** 2  # padded axis of length 8
    WL, WR = traces_first_order(W)
    assert WL.shape == (5,) and WR.shape == (5,)
    assert torch.equal(WL, W[1:-2]) and torch.equal(WR, W[2:-1])


def test_muscl_exact_on_linear_data():
    # a linear field has equal one-sided slopes -> limiter returns the exact
    # slope and traces from both sides agree with the true interface value
    x = torch.arange(10, dtype=torch.float64)
    W = 3.0 * x + 1.0
    for lim in LIMITERS.values():
        WL, WR = traces_muscl(W, lim)
        exact = 3.0 * (torch.arange(7, dtype=torch.float64) + 1.5) + 1.0
        assert torch.allclose(WL, exact, atol=1e-13)
        assert torch.allclose(WR, exact, atol=1e-13)


def test_muscl_traces_stay_in_neighbor_hull():
    torch.manual_seed(2)
    W = torch.rand(50, dtype=torch.float64)
    for lim in LIMITERS.values():
        WL, WR = traces_muscl(W, lim)
        lo = torch.minimum(W[1:-2], torch.minimum(W[:-3], W[2:-1]))
        hi = torch.maximum(W[1:-2], torch.maximum(W[:-3], W[2:-1]))
        assert torch.all(WL >= lo - 1e-13) and torch.all(WL <= hi + 1e-13)


def test_reconstruct_line_well_balanced_traces():
    # lake at rest: eta = h + z constant; the reconstructed traces must
    # satisfy hK + zK == eta exactly on both sides of every interface
    torch.manual_seed(3)
    z = torch.rand(1, 12, dtype=torch.float64)
    eta = torch.full_like(z, 2.0)
    h = eta - z
    u = torch.zeros_like(z)
    for order in (1, 2):
        hL, unL, utL, zL, hR, unR, utR, zR = reconstruct_line(
            h, u, u, z, order=order, limiter=van_leer
        )
        assert torch.allclose(hL + zL, torch.full_like(hL, 2.0), atol=1e-14)
        assert torch.allclose(hR + zR, torch.full_like(hR, 2.0), atol=1e-14)


def test_reconstruct_line_clamps_negative_depth_traces():
    h = torch.tensor([[0.0, 0.0, 1e-9, 2.0, 4.0, 4.0, 4.0]], dtype=torch.float64)
    z = torch.zeros_like(h)
    u = torch.zeros_like(h)
    hL, *_ , hR, _, _, _ = reconstruct_line(h, u, u, z, order=2, limiter=superbee)
    assert torch.all(hL >= 0) and torch.all(hR >= 0)


def test_reconstruction_differentiable():
    h = (torch.rand(1, 10, dtype=torch.float64) + 0.5).requires_grad_(True)
    z = torch.rand(1, 10, dtype=torch.float64)
    u = torch.randn(1, 10, dtype=torch.float64).requires_grad_(True)
    out = reconstruct_line(h, u, u, z, order=2, limiter=van_leer)
    sum(o.sum() for o in out).backward()
    assert torch.isfinite(h.grad).all() and torch.isfinite(u.grad).all()

# ==========================================================================
# from tests/test_fluxes.py
# ==========================================================================

"""Tests for the Riemann flux kernels: consistency, symmetry, dry states,
differentiability."""



ALL = list(FLUXES.items())


def rand_states(n=64, dry_frac=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    def r(lo, hi):
        return lo + (hi - lo) * torch.rand(1, n, dtype=torch.float64, generator=g)
    hL, hR = r(0.1, 3.0), r(0.1, 3.0)
    if dry_frac:
        mask = torch.rand(1, n, generator=g) < dry_frac
        hR = torch.where(mask, torch.zeros_like(hR), hR)
    return hL, r(-2, 2), r(-2, 2), hR, r(-2, 2), r(-2, 2)


@pytest.mark.parametrize("name,flux", ALL)
def test_consistency_with_physical_flux(name, flux):
    """F(U, U) must equal the exact flux."""
    h, un, ut, *_ = rand_states()
    F = flux(h, un, ut, h, un, ut)
    assert torch.allclose(F, physical_flux(h, un, ut), atol=1e-12), name


@pytest.mark.parametrize("name,flux", ALL)
def test_still_water_gives_pure_pressure(name, flux):
    h = torch.full((1, 5), 2.0, dtype=torch.float64)
    zero = torch.zeros_like(h)
    F = flux(h, zero, zero, h, zero, zero)
    assert torch.allclose(F[..., 0, :, :], zero, atol=1e-14)
    assert torch.allclose(F[..., 1, :, :], 0.5 * G * h * h, atol=1e-12)
    assert torch.allclose(F[..., 2, :, :], zero, atol=1e-14)


@pytest.mark.parametrize("name,flux", ALL)
def test_mirror_symmetry(name, flux):
    """Reflecting the Riemann problem (x -> -x) must flip the sign of the
    mass and tangential-momentum fluxes and preserve the normal-momentum
    flux: F(mirrored) = diag(-1, 1, -1) F(original)."""
    hL, unL, utL, hR, unR, utR = rand_states(seed=1)
    F = flux(hL, unL, utL, hR, unR, utR)
    Fm = flux(hR, -unR, utR, hL, -unL, utL)
    sign = torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float64).view(3, 1, 1)
    assert torch.allclose(Fm, sign * F, atol=1e-11), name


@pytest.mark.parametrize("name,flux", ALL)
def test_dry_dry_interface_zero_flux(name, flux):
    z = torch.zeros(1, 4, dtype=torch.float64)
    F = flux(z, z, z, z, z, z)
    assert torch.all(F == 0), name


@pytest.mark.parametrize("name,flux", ALL)
def test_dry_bed_front_no_nan_and_finite_grads(name, flux):
    """Wet-dry interface (Ritter front): finite flux, finite gradients."""
    hL = torch.tensor([[1.0]], dtype=torch.float64, requires_grad=True)
    hR = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True)
    unL = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    zero = torch.zeros(1, 1, dtype=torch.float64)
    F = flux(hL, unL, zero, hR, zero, zero)
    assert torch.isfinite(F).all(), name
    F.sum().backward()
    assert torch.isfinite(hL.grad).all() and torch.isfinite(hR.grad).all()
    assert torch.isfinite(unL.grad).all()


def test_hllc_upwinds_tangential_momentum():
    """For a right-moving contact, HLLC must carry the LEFT tangential
    velocity exactly (the star states share h and un, so the flux reduces to
    the physical flux with the upwind ut); HLL smears it."""
    h = torch.full((1, 1), 1.0, dtype=torch.float64)
    un = torch.full((1, 1), 0.3, dtype=torch.float64)
    utL = torch.full((1, 1), 1.0, dtype=torch.float64)
    utR = torch.full((1, 1), -1.0, dtype=torch.float64)
    F = FLUXES["hllc"](h, un, utL, h, un, utR)  # (3, 1, 1)
    # mass flux = h*un = 0.3; tangential flux must be mass_flux * utL = 0.3
    assert abs(F[0, 0, 0].item() - 0.3) < 1e-12
    assert abs(F[2, 0, 0].item() - 0.3 * utL.item()) < 1e-12
    # HLL, by contrast, averages the tangential states
    Fh = FLUXES["hll"](h, un, utL, h, un, utR)
    assert abs(Fh[2, 0, 0].item() - 0.3) > 1e-3


@pytest.mark.parametrize("name,flux", ALL)
def test_gradcheck_wet_states(name, flux):
    """Rigorous autograd check on smooth wet states (float64)."""
    torch.manual_seed(4)
    args = tuple(
        (0.5 + torch.rand(1, 3, dtype=torch.float64)).requires_grad_(True)
        if i in (0, 3)
        else torch.randn(1, 3, dtype=torch.float64).mul(0.3).requires_grad_(True)
        for i in range(6)
    )
    assert torch.autograd.gradcheck(
        lambda *a: flux(*a), args, eps=1e-7, atol=1e-6, nondet_tol=0.0
    ), name


@pytest.mark.parametrize("name,flux", ALL)
def test_batched_matches_unbatched(name, flux):
    hL, unL, utL, hR, unR, utR = rand_states(seed=5)
    Fb = flux(
        hL.expand(4, -1, -1), unL.expand(4, -1, -1), utL.expand(4, -1, -1),
        hR.expand(4, -1, -1), unR.expand(4, -1, -1), utR.expand(4, -1, -1),
    )
    F = flux(hL, unL, utL, hR, unR, utR)
    assert Fb.shape == (4, 3, 1, hL.shape[-1])
    for b in range(4):
        assert torch.allclose(Fb[b], F, atol=1e-14)

# ==========================================================================
# from tests/test_friction_timestep.py
# ==========================================================================

"""Tests for the Manning friction split step and CFL dt control."""





def test_friction_matches_pointwise_implicit_relation():
    h = torch.full((1, 4), 0.8, dtype=torch.float64)
    u = torch.full((1, 4), 1.5, dtype=torch.float64)
    v = torch.full((1, 4), -0.5, dtype=torch.float64)
    U = conserved(h, u, v)
    n, dt = 0.03, 0.2
    Uf = apply_friction(U, dt, n)
    speed = math.sqrt(1.5**2 + 0.5**2)
    denom = 1.0 + dt * G * n**2 * speed / 0.8 ** (4.0 / 3.0)
    assert torch.allclose(Uf[1], h * u / denom, atol=1e-14)
    assert torch.allclose(Uf[2], h * v / denom, atol=1e-14)
    assert torch.equal(Uf[0], h)


def test_friction_noop_for_zero_n_and_stable_near_dry():
    h = torch.tensor([[1e-12, 0.5]], dtype=torch.float64)
    hu = torch.tensor([[1e-9, 0.3]], dtype=torch.float64)
    U = torch.stack([h, hu, torch.zeros_like(h)], dim=-3)
    assert apply_friction(U, 0.1, 0.0) is U
    Uf = apply_friction(U, 1e3, 0.05)  # huge dt: must not blow up or flip sign
    assert torch.isfinite(Uf).all()
    assert Uf[1, 0, 0] == 0.0                    # dry cell momentum zeroed
    assert 0.0 <= Uf[1, 0, 1] <= 0.3             # relaxation toward zero only


def test_friction_never_reverses_momentum():
    torch.manual_seed(0)
    h = torch.rand(1, 50, dtype=torch.float64) * 2 + 1e-4
    u = torch.randn(1, 50, dtype=torch.float64) * 3
    v = torch.randn(1, 50, dtype=torch.float64) * 3
    U = conserved(h, u, v)
    Uf = apply_friction(U, 10.0, 0.1)
    assert torch.all(Uf[1] * U[1] >= 0) and torch.all(Uf[2] * U[2] >= 0)
    assert torch.all(Uf[1].abs() <= U[1].abs() + 1e-15)


def test_dt_matches_hand_computation():
    grid = Grid.from_extent(nx=10, ny=5, extent=(0, 10, 0, 10))  # dx=1, dy=2
    h = torch.full((5, 10), 1.0, dtype=torch.float64)
    u = torch.full((5, 10), 2.0, dtype=torch.float64)
    v = torch.zeros_like(h)
    U = conserved(h, u, v)
    c = math.sqrt(G)
    sx, sy = max_wave_speeds(U)
    assert abs(sx.item() - (2.0 + c)) < 1e-14
    assert abs(sy.item() - c) < 1e-14
    dt = compute_dt(U, grid, cfl=0.45)
    assert abs(dt.item() - 0.45 * min(1.0 / (2 + c), 2.0 / c)) < 1e-14


def test_dt_ignores_dry_cells():
    grid = Grid.from_extent(nx=4, ny=1, extent=(0, 4, 0, 1))
    h = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]], dtype=torch.float64)[0]
    hu = torch.tensor([[[9.9, 0.0, 0.0, 0.0]]], dtype=torch.float64)[0]  # garbage in dry cell
    U = torch.stack([h, hu, torch.zeros_like(h)], dim=-3)
    sx, _ = max_wave_speeds(U)
    assert abs(sx.item() - math.sqrt(G)) < 1e-12  # dry cell's momentum ignored

# ==========================================================================
# from tests/test_analytic.py
# ==========================================================================

"""Tests for the exact solutions (Ritter, Stoker, lake at rest)."""





def test_ritter_regions_and_continuity():
    h0, x0, t = 4.0, 0.0, 2.0
    c0 = math.sqrt(G * h0)
    x = torch.linspace(-100, 100, 4001, dtype=torch.float64)
    h, u = analytic.ritter_solution(x, t, h0, x0)
    assert torch.all(h[x < -c0 * t] == h0)
    assert torch.all(u[x <= -c0 * t] == 0)
    assert torch.all(h[x > 2 * c0 * t] == 0)
    # continuity at the fan edges and positivity inside
    assert torch.all(h >= 0)
    assert (h.diff().abs().max()).item() < h0 * 0.01  # no jumps on this grid
    # depth at the dam equals 4/9 h0 (classic Ritter value)
    i = (x - x0).abs().argmin()
    assert abs(h[i].item() - 4.0 * h0 / 9.0) < 1e-3


def test_stoker_middle_state_satisfies_relations():
    h_l, h_r = 10.0, 1.0
    h_m, u_m, s = analytic.stoker_middle_state(h_l, h_r)
    assert h_r < h_m < h_l
    # rarefaction invariant
    assert abs(u_m - 2 * (math.sqrt(G * h_l) - math.sqrt(G * h_m))) < 1e-10
    # shock Rankine-Hugoniot (mass): s (h_m - h_r) = h_m u_m
    assert abs(s * (h_m - h_r) - h_m * u_m) < 1e-10
    # momentum RH: s(h_m u_m) = h_m u_m^2 + g/2 (h_m^2 - h_r^2)
    lhs = s * h_m * u_m
    rhs = h_m * u_m**2 + 0.5 * G * (h_m**2 - h_r**2)
    assert abs(lhs - rhs) < 1e-8


def test_stoker_profile_regions():
    h_l, h_r, t = 10.0, 1.0, 3.0
    h_m, u_m, s = analytic.stoker_middle_state(h_l, h_r)
    x = torch.linspace(-200, 200, 8001, dtype=torch.float64)
    h, u = analytic.stoker_solution(x, t, h_l, h_r)
    assert torch.all(h[x / t <= -math.sqrt(G * h_l)] == h_l)
    assert torch.all(h[x / t >= s + 1e-9] == h_r)
    mid = (x / t > u_m - math.sqrt(G * h_m) + 1e-9) & (x / t < s - 1e-9)
    assert torch.allclose(h[mid], torch.full_like(h[mid], h_m), atol=1e-12)
    assert torch.allclose(u[mid], torch.full_like(u[mid], u_m), atol=1e-12)
    assert torch.all(h >= h_r - 1e-14)


def test_lake_at_rest_geometry():
    x = torch.linspace(analytic.LAKE_X_MIN, analytic.LAKE_X_MAX, 1001, dtype=torch.float64)
    z = analytic.lake_bed(x)
    h = analytic.lake_initial_depth(x)
    assert z.max().item() == 0.2  # bump peak at x=10
    assert torch.all(h > 0)
    assert torch.allclose(h + z, torch.full_like(x, analytic.LAKE_ETA0))

# ==========================================================================
# from tests/test_solver.py
# ==========================================================================

"""Integration tests for swe.solver: well-balance, conservation, positivity,
symmetry, batching, differentiability."""




def make_1d_case(nx=100, extent=(0.0, 25.0), **kw):
    grid = Grid.from_extent(nx=nx, ny=1, extent=(*extent, 0.0, 1.0))
    return Config(grid=grid, bc=TRANSMISSIVE, **kw)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("scheme", ["rusanov", "hll", "hllc"])
def test_lake_at_rest_is_machine_still(scheme, order):
    """Well-balanced gate (short version; the 100 s check runs in the
    validation script): velocities stay at machine precision over the bump."""
    cfg = make_1d_case(nx=200, scheme=scheme, order=order)
    x = cfg.grid.xc.unsqueeze(0)
    z = analytic.lake_bed(x)
    h0 = analytic.lake_initial_depth(x)
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    out = run(cfg, U, z, t_end=2.0)
    Uf = out["U"][-1]
    assert Uf[1].abs().max().item() < 1e-13
    assert Uf[2].abs().max().item() < 1e-13
    assert (Uf[0] - h0).abs().max().item() < 1e-13


def test_lake_at_rest_2d_with_hump():
    """2D still lake over the reference-style Gaussian hump, reflective box."""
    grid = Grid.from_extent(nx=48, ny=48, extent=(0, 100, 0, 100))
    cfg = Config(grid=grid, bc=REFLECTIVE, scheme="hllc", order=2)
    X, Y = grid.centers()
    z = 2.0 * torch.exp(-((X - 50) ** 2 + (Y - 50) ** 2) / 200.0)
    h0 = 5.0 - z
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    out = run(cfg, U, z, t_end=1.0)
    Uf = out["U"][-1]
    assert Uf[1].abs().max().item() < 1e-13
    assert Uf[2].abs().max().item() < 1e-13


def test_mass_conservation_closed_box():
    """Reflective box, random smooth blob: relative volume drift < 1e-12."""
    torch.manual_seed(0)
    grid = Grid.from_extent(nx=40, ny=32, extent=(0, 10, 0, 8))
    cfg = Config(grid=grid, bc=REFLECTIVE, scheme="hllc", order=2)
    X, Y = grid.centers()
    h0 = 1.0 + torch.exp(-((X - 4) ** 2 + (Y - 3) ** 2))
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    out = run(cfg, U, None, t_end=2.0, output_times=[0.5, 1.0, 2.0])
    m0 = out["mass"][0]
    for m in out["mass"][1:]:
        assert abs(m / m0 - 1.0) < 1e-12


def test_mass_conservation_with_bed_and_walls():
    """Hydrostatic reconstruction must not break conservation."""
    grid = Grid.from_extent(nx=32, ny=32, extent=(0, 100, 0, 100))
    cfg = Config(grid=grid, bc=REFLECTIVE, scheme="hll", order=2)
    X, Y = grid.centers()
    z = 2.0 * torch.exp(-((X - 50) ** 2 + (Y - 50) ** 2) / 200.0)
    h0 = torch.where(X < 50, 10.0 - z, 1.0 - torch.clamp(z, max=0.9))
    h0 = torch.clamp(h0, min=0.1)
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    out = run(cfg, U, z, t_end=2.0)
    assert abs(out["mass"][-1] / out["mass"][0] - 1.0) < 1e-12


@pytest.mark.parametrize("scheme", ["rusanov", "hll", "hllc"])
def test_dry_dam_break_positivity(scheme):
    """Ritter setup: h must never go negative, and stays 0 ahead of the front."""
    cfg = make_1d_case(nx=200, extent=(0.0, 100.0), scheme=scheme, order=2)
    x = cfg.grid.xc.unsqueeze(0)
    h0 = torch.where(x <= 50.0, torch.full_like(x, 1.0), torch.zeros_like(x))
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    out = run(cfg, U, None, t_end=5.0, output_times=[1.0, 2.5, 5.0])
    for Uf in out["U"]:
        assert Uf[0].min().item() >= 0.0


def test_circular_dam_break_preserves_symmetry():
    """Radial IC on a square grid: x-flip, y-flip and 90-degree rotation
    symmetry must hold to near machine precision (unlike the reference
    upwind runs, which drift)."""
    grid = Grid.from_extent(nx=50, ny=50, extent=(0, 100, 0, 100))
    cfg = Config(grid=grid, bc=REFLECTIVE, scheme="hllc", order=2)
    X, Y = grid.centers()
    inside = (X - 50) ** 2 + (Y - 50) ** 2 <= 400.0
    h0 = torch.where(inside, 10.0, 1.0).to(torch.float64)
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0))
    out = run(cfg, U, None, t_end=1.0)
    hf = out["U"][-1][0]
    assert (hf - hf.flip(-1)).abs().max().item() < 1e-12
    assert (hf - hf.flip(-2)).abs().max().item() < 1e-12
    assert (hf - torch.rot90(hf, 1, (-2, -1))).abs().max().item() < 1e-12


def test_step_batched_matches_unbatched():
    torch.manual_seed(1)
    grid = Grid.from_extent(nx=16, ny=12, extent=(0, 4, 0, 3))
    cfg = Config(grid=grid, bc=TRANSMISSIVE, scheme="hllc", order=2)
    X, Y = grid.centers()
    z = 0.1 * torch.exp(-((X - 2) ** 2 + (Y - 1.5) ** 2))
    z_pad = pad_scalar(z, cfg.bc)
    Us = []
    for b in range(3):
        h = 1.0 + torch.rand(12, 16, dtype=torch.float64) * 0.1
        u = torch.randn(12, 16, dtype=torch.float64) * 0.05
        v = torch.randn(12, 16, dtype=torch.float64) * 0.05
        Us.append(conserved(h, u, v))
    UB = torch.stack(Us)
    out_b = step(UB, z_pad, 0.005, cfg)
    for b in range(3):
        out_1 = step(Us[b], z_pad, 0.005, cfg)
        assert torch.allclose(out_b[b], out_1, atol=1e-14)


def test_step_is_differentiable_end_to_end():
    """Gradient of a scalar loss of step() w.r.t. the input state is finite —
    the property the FVM-informed PINN loss relies on. Includes dry cells."""
    grid = Grid.from_extent(nx=20, ny=1, extent=(0, 20, 0, 1))
    cfg = Config(grid=grid, bc=TRANSMISSIVE, scheme="hllc", order=2)
    x = grid.xc.unsqueeze(0)
    h0 = torch.where(x <= 10.0, torch.full_like(x, 1.0), torch.zeros_like(x))
    U = conserved(h0, torch.zeros_like(h0), torch.zeros_like(h0)).requires_grad_(True)
    z_pad = pad_scalar(torch.zeros(1, 20, dtype=torch.float64), cfg.bc)
    out = step(U, z_pad, 0.01, cfg)
    loss = (out**2).sum()
    loss.backward()
    assert torch.isfinite(U.grad).all()
    assert U.grad.abs().sum() > 0


def test_rhs_zero_for_uniform_state():
    grid = Grid.from_extent(nx=10, ny=10, extent=(0, 1, 0, 1))
    cfg = Config(grid=grid, bc=TRANSMISSIVE, scheme="hllc", order=2)
    h = torch.full((10, 10), 2.0, dtype=torch.float64)
    U = conserved(h, torch.zeros_like(h), torch.zeros_like(h))
    z_pad = pad_scalar(torch.zeros(10, 10, dtype=torch.float64), cfg.bc)
    dU = rhs(U, z_pad, cfg)
    assert dU.abs().max().item() < 1e-14
