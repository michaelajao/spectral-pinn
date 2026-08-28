# Advection reproduction (Wang et al. 2026, Sect. 4.1 setup)

3x25 tanh MLP, Xavier init, N_u=100, N_f=500 (LHS), L-BFGS max_iter=3000, w_U=2.857e-05 for svd_soft, 20 seeds. Published magnitudes: cnPINN ~1e-3, vanilla PINN ~1e-2..1e-1 (their Figs. 5-7).

| entry | L2 rel. error (mean ± std) | per-seed |
|---|---|---|
| bounded (svd_bounded) | 4.683e-01 ± 5.1e-01 | 3.8e-01, 2.7e-01, 6.6e-01, 2.7e-02, 2.2e+00, 6.5e-01, 2.5e-01, 1.1e+00, 2.8e-02, 4.2e-01, 3.8e-02, 3.4e-02, 3.2e-02, 2.5e-02, 4.4e-02, 6.9e-01, 6.3e-01, 3.6e-01, 7.2e-01, 7.7e-01 |
