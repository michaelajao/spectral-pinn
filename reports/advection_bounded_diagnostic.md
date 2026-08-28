# Advection reproduction (Wang et al. 2026, Sect. 4.1 setup)

3x25 tanh MLP, Xavier init, N_u=100, N_f=500 (LHS), L-BFGS max_iter=3000, w_U=2.857e-05 for svd_soft, 3 seeds. Published magnitudes: cnPINN ~1e-3, vanilla PINN ~1e-2..1e-1 (their Figs. 5-7).

| entry | L2 rel. error (mean ± std) | per-seed | diagnostics |
|---|---|---|---|
| bounded (svd_bounded) | 4.391e-01 ± 1.6e-01 | 3.8e-01, 2.7e-01, 6.6e-01 | D_U median 1.00e+00 max 1.00e+00; sv-mismatch max 1.29e+03; s_max max 15.66; s_min min 0.000 |
| cnpinn (svd_soft) | 1.019e-02 ± 3.9e-03 | 7.9e-03, 7.0e-03, 1.6e-02 | D_U median 9.80e-01 max 1.08e+00; sv-mismatch max 1.34e+01; s_max max 6.29; s_min min 0.003 |
