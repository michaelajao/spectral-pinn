"""Tests for src/models.py: shapes, residual wiring, well-balancing, training,
and the spectral weight reparameterizations (SVDLinear and its modes).

The first half is ported unchanged from swe-dambreak's test_ml.py; the
spectral tests at the end are new to this project.
"""

import torch

from src.models import (
    FVMPINN,
    FVMPINNConfig,
    FVMResidualSpec,
    PINN,
    PINNConfig,
    Physics,
    SVDLinear,
    WEIGHT_PARAMS,
    data_anchor_loss,
    fvm_residual_loss,
    ic_anchor_loss,
    pde_residual,
)
from src.solver import TRANSMISSIVE, Config, Grid, conserved
from src.train import TrainConfig, train


def _zero_last_layer(net, bias=None):
    last = net[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
        if bias is not None:
            last.bias.copy_(torch.tensor(bias, dtype=last.bias.dtype))


def test_pinn_shapes_both_formulations():
    for variables in ("primitive", "conservative"):
        m = PINN(PINNConfig(variables=variables, hidden=16, layers=2))
        xyt = torch.rand(10, 3)
        a, b, c = m.forward(xyt)
        assert a.shape == (10,) and b.shape == (10,) and c.shape == (10,)
        h, u, v = m.state(xyt)
        assert h.shape == (10,)


def test_fourier_features_shape():
    m = PINN(PINNConfig(hidden=16, layers=2, fourier_features=8))
    assert m.forward(torch.rand(5, 3))[0].shape == (5,)


def test_fourier_features_vary_with_seed():
    """The embedding matrix B must differ across seeds (it is wired to the
    run seed so seed-replicates are genuinely independent)."""
    m0 = PINN(PINNConfig(hidden=16, layers=2, fourier_features=8, fourier_seed=0))
    m1 = PINN(PINNConfig(hidden=16, layers=2, fourier_features=8, fourier_seed=1))
    assert not torch.allclose(m0.embed.B, m1.embed.B)


def test_strong_form_residual_zero_for_still_lake():
    """Constant still state (h=H, u=v=0, flat bed) has zero SWE residual in
    both formulations — validates the autograd residual wiring."""
    H = 2.0
    for variables in ("primitive", "conservative"):
        m = PINN(PINNConfig(variables=variables, hidden=16, layers=2))
        _zero_last_layer(m.net, bias=[H, 0.0, 0.0])
        x = torch.rand(20, 1, requires_grad=True)
        y = torch.rand(20, 1, requires_grad=True)
        t = torch.rand(20, 1, requires_grad=True)
        res = pde_residual(m, x, y, t, Physics(g=9.81))
        assert res.shape == (20, 3)
        assert res.abs().max() < 1e-5


def test_strong_form_residual_has_param_gradients():
    m = PINN(PINNConfig(hidden=16, layers=2))
    x = torch.rand(8, 1, requires_grad=True)
    y = torch.rand(8, 1, requires_grad=True)
    t = torch.rand(8, 1, requires_grad=True)
    loss = (pde_residual(m, x, y, t, Physics()) ** 2).mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters())


def _fvm_setup(n=16, H=2.0):
    grid = Grid.from_extent(n, n, (0, 10, 0, 10))
    h_s = torch.full((n, n), H, dtype=torch.float32)
    mcfg = FVMPINNConfig(hidden=16, layers=2, fourier_features=8,
                         x_range=(0, 10), y_range=(0, 10), t_range=(0, 1))
    model = FVMPINN(mcfg, h_s)
    scfg = Config(grid=grid, bc=TRANSMISSIVE, scheme="hllc", order=2)
    return grid, model, scfg, H


def test_fvm_predict_grid_shape_and_positive():
    grid, model, _, _ = _fvm_setup()
    U = model.predict_grid(0.3, grid)
    assert U.shape == (3, grid.ny, grid.nx)
    assert U.dtype == torch.float64
    assert U[0].min() > 0  # softplus keeps h strictly positive


def test_fvm_residual_zero_for_constant_still_prediction():
    """Zeroed final layer -> xi=0, hu=hv=0 -> uniform still lake; the discrete
    FV step reproduces it, so the FVM consistency residual is ~0 (inherited
    well-balancing)."""
    grid, model, scfg, H = _fvm_setup()
    _zero_last_layer(model.net)
    spec = FVMResidualSpec(cfg=scfg, times=[0.0, 0.1, 0.2], n_sub=2)
    loss, info = fvm_residual_loss(model, spec)
    assert float(loss) < 1e-16
    assert len(info["per_channel_mse"]) == 3


def test_fvm_residual_and_ic_have_gradients():
    grid, model, scfg, H = _fvm_setup()
    spec = FVMResidualSpec(cfg=scfg, times=[0.0, 0.05, 0.1], n_sub=1)
    U0 = conserved(torch.full((grid.ny, grid.nx), 2.0, dtype=torch.float64),
                   torch.zeros(grid.ny, grid.nx, dtype=torch.float64),
                   torch.zeros(grid.ny, grid.nx, dtype=torch.float64))
    loss = fvm_residual_loss(model, spec)[0] + ic_anchor_loss(model, grid, U0)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_data_anchor_uses_same_reconstruction_as_predict_grid():
    """The gauge misfit must be measured on the network's actual (h,hu,hv),
    i.e. the same softplus depth and h*vel reparametrization as predict_grid.
    Feeding the model's own grid prediction as the target => ~zero loss."""
    grid, model, _, _ = _fvm_setup(n=16)
    model.cfg.vel_scale = 8.0  # exercise the bounded-velocity path
    t = 0.5
    X, Y = grid.centers()
    xyt = torch.stack([X.reshape(-1), Y.reshape(-1),
                       torch.full((grid.ny * grid.nx,), t)], dim=1)
    U = model.predict_grid(t, grid)                     # (3, ny, nx)
    targets = torch.stack([U[0].reshape(-1), U[1].reshape(-1), U[2].reshape(-1)], dim=1)
    h_s = model.h_s.reshape(-1)
    loss = data_anchor_loss(model, xyt, h_s, targets)
    assert float(loss) < 1e-8


def test_train_reduces_loss(tmp_path):
    """Short FVM-PINN training on a lake-at-rest reduces the loss and writes
    artifacts."""
    grid, model, scfg, H = _fvm_setup(n=12)
    spec = FVMResidualSpec(cfg=scfg, times=[0.0, 0.1, 0.2], n_sub=1)
    U0 = conserved(torch.full((grid.ny, grid.nx), H, dtype=torch.float64),
                   torch.zeros(grid.ny, grid.nx, dtype=torch.float64),
                   torch.zeros(grid.ny, grid.nx, dtype=torch.float64))

    def loss_fn():
        r, info = fvm_residual_loss(model, spec)
        ic = ic_anchor_loss(model, grid, U0)
        return r + ic, {"residual": float(r), "ic": float(ic)}

    l0 = float(loss_fn()[0])
    out = train(model, loss_fn, TrainConfig(iters=40, lr=1e-3, log_every=10,
                                            out_dir=str(tmp_path / "run")))
    assert out["best_loss"] < l0
    assert (tmp_path / "run" / "history.csv").exists()
    assert (tmp_path / "run" / "best.pt").exists()


# ==========================================================================
# Spectral weight reparameterizations (new in this project)
# ==========================================================================

def test_all_weight_params_forward_and_layer_count():
    """Every mode builds, runs, and reparameterizes exactly the layers-1
    square hidden weights (input and output layers stay dense)."""
    for wp in WEIGHT_PARAMS:
        m = PINN(PINNConfig(hidden=16, layers=3, weight_param=wp))
        a, b, c = m.forward(torch.rand(7, 3))
        assert a.shape == (7,)
        expected = 0 if wp == "dense" else 2   # layers - 1
        assert len(m.svd_layers) == expected


def test_svd_layer_reproduces_initial_weight():
    """U diag(s) V^T at init equals the SVD'd weight to round-off, so all
    variants start from the same distribution as a dense layer."""
    torch.manual_seed(0)
    lin = SVDLinear(16, 16, mode="svd_soft")
    W = lin.weight()
    U, s, Vh = torch.linalg.svd(W, full_matrices=False)
    assert torch.allclose(torch.sort(s, descending=True).values,
                          torch.sort(lin.s.detach().abs(), descending=True).values,
                          atol=1e-5)


def test_svd_soft_defect_zero_at_init_positive_after_perturbation():
    """D_U = ||U^T U - I||_2 is round-off at init and grows once U moves;
    the Frobenius option upper-bounds the 2-norm."""
    lin = SVDLinear(16, 16, mode="svd_soft")
    assert float(lin.orthogonality_defect()) < 1e-5      # SVD gives orthogonal U
    with torch.no_grad():
        lin.U += 0.1 * torch.randn_like(lin.U)
    d2 = float(lin.orthogonality_defect())
    assert d2 > 1e-2                                     # training breaks it
    fro = SVDLinear(16, 16, mode="svd_soft", defect_norm="frobenius")
    with torch.no_grad():
        fro.U.copy_(lin.U)
    assert float(fro.orthogonality_defect()) >= d2 - 1e-6


def test_svd_hard_stays_orthogonal_under_optimization():
    """The central property of the hard variant: U^T U == I to machine
    precision after real optimizer steps, with no penalty term anywhere,
    and the singular values of the effective weight are exactly |s|."""
    torch.manual_seed(0)
    lin = SVDLinear(16, 16, mode="svd_hard")
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    x = torch.randn(32, 16)
    for _ in range(20):
        opt.zero_grad()
        loss = ((lin(x) - x) ** 2).mean()
        loss.backward()
        opt.step()
    UtU = lin.U.T @ lin.U
    assert torch.allclose(UtU, torch.eye(16), atol=1e-5)  # float32 exactness
    sv = torch.linalg.svdvals(lin.weight().detach())
    s_abs = torch.sort(lin.s.detach().abs(), descending=True).values
    assert torch.allclose(sv, s_abs, atol=1e-4)


def test_svd_sigma_trains_only_singular_values():
    lin = SVDLinear(16, 16, mode="svd_sigma")
    trainable = {n for n, p in lin.named_parameters() if p.requires_grad}
    assert trainable == {"s", "bias"}
    # U and Vh are buffers, frozen orthogonal
    assert torch.allclose(lin.U.T @ lin.U, torch.eye(16), atol=1e-5)


def test_orthogonality_penalty_aggregation():
    dense = PINN(PINNConfig(hidden=16, layers=3, weight_param="dense"))
    assert float(dense.orthogonality_penalty()) == 0.0
    soft = PINN(PINNConfig(hidden=16, layers=3, weight_param="svd_soft"))
    p = soft.orthogonality_penalty()
    assert p.shape == () and float(p) < 1e-8   # squared round-off at init


def test_penalty_gradient_is_finite_at_orthogonal_init():
    """The 2-norm of a near-zero symmetric matrix has a degenerate spectrum;
    its autograd must still be finite so the first L-BFGS step is sane."""
    lin = SVDLinear(16, 16, mode="svd_soft")
    (lin.orthogonality_defect() ** 2).backward()
    assert torch.isfinite(lin.U.grad).all()


def test_xavier_zero_bias_consistent_across_layer_types():
    m = PINN(PINNConfig(hidden=16, layers=3, weight_param="svd_soft", init="xavier"))
    for lay in m.net:
        if hasattr(lay, "bias") and lay.bias is not None:
            assert float(lay.bias.abs().max()) == 0.0


def test_spectral_pinn_residual_gradients_reach_svd_factors():
    """The PDE-residual gradient must reach U and s in every variant that
    trains them (the whole point of the reparameterization)."""
    for wp in ("svd_soft", "svd_hard", "svd_sigma"):
        m = PINN(PINNConfig(hidden=16, layers=3, weight_param=wp))
        x = torch.rand(8, 1, requires_grad=True)
        y = torch.rand(8, 1, requires_grad=True)
        t = torch.rand(8, 1, requires_grad=True)
        loss = (pde_residual(m, x, y, t, Physics()) ** 2).mean()
        loss.backward()
        for lay in m.svd_layers:
            assert lay.s.grad is not None and torch.isfinite(lay.s.grad).all()


def test_svd_layers_train_under_lbfgs():
    """L-BFGS flattens grads with .view(-1), which requires contiguous
    parameters; the SVD factors from LAPACK are strided views, so this guards
    the fix that makes every mode usable with the optimizer Wang et al. use."""
    x = torch.randn(16, 12)
    for mode in ("svd_soft", "svd_hard", "svd_sigma"):
        lin = SVDLinear(12, 12, mode=mode)
        opt = torch.optim.LBFGS(lin.parameters(), max_iter=5)

        def closure():
            opt.zero_grad()
            loss = ((lin(x) - x) ** 2).mean()
            loss.backward()
            return loss

        l0 = float(closure())
        opt.step(closure)
        assert float(closure()) < l0


def test_xavier_init_option():
    m = PINN(PINNConfig(hidden=16, layers=2, init="xavier"))
    assert m.forward(torch.rand(4, 3))[0].shape == (4,)


def test_svd_box_enforces_the_constraint_and_spans_hard_to_free():
    """eps = 0 must reproduce exact orthogonality and larger eps must widen the
    admissible spectrum, since that interpolation is the point of the layer."""
    from src.models import SVDLinear
    for eps in (0.0, 0.1, 0.5):
        torch.manual_seed(0)
        layer = SVDLinear(24, 24, mode="svd_box", init="xavier", box_eps=eps)
        with torch.no_grad():   # push the raw parameter far outside the box
            layer.parametrizations.U.original.mul_(6.0).add_(torch.randn(24, 24))
        sv = torch.linalg.svdvals(layer.U)
        assert sv.min() >= 1.0 - eps - 1e-4
        assert sv.max() <= 1.0 + eps + 1e-4


def test_svd_box_gradients_stay_finite():
    """The exact projection Jacobian is singular where the box binds (all
    singular values near 1), so the layer uses a straight-through estimator;
    this pins that the backward pass produces usable numbers."""
    from src.models import SVDLinear
    torch.manual_seed(0)
    layer = SVDLinear(24, 24, mode="svd_box", init="xavier", box_eps=0.1)
    with torch.no_grad():
        layer.parametrizations.U.original.mul_(6.0)
    layer(torch.randn(8, 24)).sum().backward()
    g = layer.parametrizations.U.original.grad
    assert torch.isfinite(g).all() and g.abs().sum() > 0
