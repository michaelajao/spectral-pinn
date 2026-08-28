# Advection reproduction (Wang et al. 2026, Sect. 4.1 setup)

3x25 tanh MLP, Xavier init, N_u=100, N_f=500 (LHS), L-BFGS max_iter=3000, w_U=2.857e-05 for svd_soft, 3 seeds. Published magnitudes: cnPINN ~1e-3, vanilla PINN ~1e-2..1e-1 (their Figs. 5-7).

| entry | L2 rel. error (mean ± std) | per-seed | diagnostics |
|---|---|---|---|
| box eps=0.1 (svd_box) | 1.031e+00 ± 9.9e-03 | 1.0e+00, 1.0e+00, 1.0e+00 | D_U median 2.10e-01 max 2.10e-01; sv-mismatch max 3.93e-02; s_max max 1.98; s_min min 0.004 |
