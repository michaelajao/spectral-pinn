# Roadmap

Working plan; tick items as they land. Dates are rough, assuming part-time
effort alongside other work.

## Phase 0 — foundations and reproduction (now – ~2 weeks)

- [x] New repo scaffolded (flat `src/` layout), solver + harness ported from
      swe-dambreak, tests green
- [x] `SVDLinear` with `svd_soft` (cnPINN), `svd_hard` (ours), `svd_sigma`
      modes + unit tests
- [x] Literature sweep -> `related_work.md` with a novelty verdict (2026-08-25; no collision)
- [x] cnPINNs reference code cloned to `third_party/` (TF1; read-only)
- [x] Reproduction gate passed 2026-08-25 (see Results log; `reports/advection_reproduction.md`)
- [x] svd_hard on the same case matches cnPINN with w_U removed (see Results log)

## Phase 1 — comparison matrix on the dam-break suite (~2–3 weeks)

- [x] Smoke run (`configs/smoke.yaml`) end-to-end on GPU (2026-08-25)
- [ ] Full matrix (`configs/main.yaml`): 6 entries x 5 benchmarks x 5 seeds — running since 2026-08-25 on GPU 1 (logs/main.log)
- [ ] Cost/accuracy frontier table (dense / soft / hard / sigma)
- [ ] Augmented-Lagrangian soft variant as the adaptive-w_U ablation

## Phase 2 — local condition-number diagnostic (~3–4 weeks, parallel)

- [ ] kappa(x, y, t) field: autograd on the network, finite differences on
      the HLLC reference
- [ ] Mismatch-vs-error heatmaps across the dam-break cases
- [ ] Vanishing-viscosity analysis of the Theorem 2.1 constants (NUST)

## Phase 3 — hard cases and Navier–Stokes (~3–4 weeks)

- [ ] Taylor–Green reproduction (targets: their Table 3)
- [ ] Lid-driven cavity, Re 100–1000
- [ ] One laboratory dam-break benchmark
- [ ] Noise-robustness and memory/cost study
- [ ] FVM-informed composition experiment

## Phase 4 — writing (~3–4 weeks)

- [ ] Draft; venue call (ASOC first choice) after Phase 1–2 results

## Results log

**2026-08-25 — Phase 0 reproduction gate (Wang et al. advection case), 5 seeds,
L-BFGS 3000 iterations, L2 relative error mean ± std.** Full table in
`reports/advection_reproduction.md`.

| entry | ours | Wang et al. Table 5 (tanh) |
|---|---|---|
| vanilla PINN (dense) | 2.07e-1 ± 1.5e-1 | (1.13 ± 1.67)e-1 |
| cnPINN (svd_soft, w_U = 1/35000) | 1.00e-2 ± 5.8e-3 | (5.64 ± 3.86)e-3 |
| ours (svd_hard, no penalty, no w_U) | 1.39e-2 ± 6.7e-3 | — |
| svd_sigma (singular values only) | 1.35e-1 ± 1.6e-1 | — |

Reading: both published cells overlap ours within one standard deviation, and
the ~20x cnPINN-over-vanilla gain reproduces; the residual ~2x gap on cnPINN is
consistent with the 3000-iteration budget and our Frobenius-norm penalty (the
paper uses the matrix 2-norm). The hard-orthogonal variant matches cnPINN with
the penalty and its weight removed — contribution C1 holds on the source
paper's own benchmark. Training only the singular values beats dense on 4 of
5 seeds but with high variance: the trained directions carry most of the
gain, which bounds how cheap C2 can go.

**Phase 1 matrix (`configs/main.yaml`)** launched 2026-08-25 on GPU 1;
~14 min per strong-form run, ~8 h per benchmark block, 2–3 days total.
First finished run: pinn/ca_circular_wet/seed0, L1(h) = 1.586e3 vs
classical HLLC 3.914e2 at N = 128.
