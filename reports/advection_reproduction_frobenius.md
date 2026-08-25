# Advection reproduction (Wang et al. 2026, Sect. 4.1 setup)

3x25 tanh MLP, Xavier init, N_u=100, N_f=500 (LHS), L-BFGS max_iter=3000, w_U=2.857e-05 for svd_soft, 5 seeds. Published magnitudes: cnPINN ~1e-3, vanilla PINN ~1e-2..1e-1 (their Figs. 5-7).

| entry | L2 rel. error (mean ± std) | per-seed |
|---|---|---|
| pinn (dense) | 2.072e-01 ± 1.5e-01 | 4.6e-01, 2.5e-01, 2.3e-01, 1.8e-02, 7.7e-02 |
| cnpinn (svd_soft) | 1.001e-02 ± 5.8e-03 | 2.1e-02, 7.1e-03, 5.9e-03, 7.9e-03, 7.6e-03 |
| ortho (svd_hard) | 1.391e-02 ± 6.7e-03 | 2.3e-02, 1.6e-02, 1.8e-02, 8.3e-03, 4.3e-03 |
| sigma (svd_sigma) | 1.351e-01 ± 1.6e-01 | 1.2e-01, 4.4e-02, 4.4e-01, 3.4e-02, 3.5e-02 |
