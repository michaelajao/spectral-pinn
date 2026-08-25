# Advection reproduction (Wang et al. 2026, Sect. 4.1 setup)

3x25 tanh MLP, Xavier init, N_u=100, N_f=500 (LHS), L-BFGS max_iter=3000, w_U=0.000e+00 for svd_soft, 20 seeds. Published magnitudes: cnPINN ~1e-3, vanilla PINN ~1e-2..1e-1 (their Figs. 5-7).

| entry | L2 rel. error (mean ± std) | per-seed |
|---|---|---|
| cnpinn (svd_soft) | 3.372e-01 ± 3.5e-01 | 3.0e-01, 5.3e-02, 1.1e+00, 1.1e-01, 7.0e-03, 2.4e-01, 1.2e-02, 3.0e-02, 4.1e-01, 1.8e-01, 1.8e-02, 6.7e-01, 3.9e-02, 1.0e-02, 7.4e-01, 7.6e-02, 6.5e-01, 3.6e-01, 6.3e-01, 1.1e+00 |
