# Advection reproduction (Wang et al. 2026, Sect. 4.1 setup)

3x25 tanh MLP, Xavier init, N_u=100, N_f=500 (LHS), L-BFGS max_iter=3000, w_U=2.857e-05 for svd_soft, 5 seeds. Published magnitudes: cnPINN ~1e-3, vanilla PINN ~1e-2..1e-1 (their Figs. 5-7).

| entry | L2 rel. error (mean ± std) | per-seed |
|---|---|---|
| pinn (dense) | 2.072e-01 ± 1.5e-01 | 4.6e-01, 2.5e-01, 2.3e-01, 1.8e-02, 7.7e-02 |
| cnpinn (svd_soft) | 9.434e-03 ± 3.1e-03 | 7.9e-03, 7.0e-03, 1.6e-02, 8.4e-03, 8.2e-03 |
| ortho (svd_hard) | 3.097e-02 ± 2.1e-02 | 3.6e-02, 1.6e-02, 6.7e-02, 3.0e-02, 5.3e-03 |
| sigma (svd_sigma) | 6.044e-01 ± 9.1e-01 | 4.6e-01, 2.0e-02, 2.4e+00, 1.2e-01, 2.4e-02 |
