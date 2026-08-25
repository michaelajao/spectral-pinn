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
- [x] Reproduction gate (2026-08-25, `reports/advection_reproduction.md`,
      5 seeds, L-BFGS 3000 iters): vanilla 2.07e-1 ± 1.5e-1 vs their Table 5
      (1.13 ± 1.67)e-1; cnPINN 1.00e-2 ± 5.8e-3 vs their (5.64 ± 3.86)e-3 —
      both overlap the published values within one std; the 20x gain
      reproduces. Residual ~2x gap plausibly from the iteration budget and
      the Frobenius (vs 2-norm) penalty; not chased further.
- [x] svd_hard on the same case: 1.39e-2 ± 6.7e-3 — indistinguishable from
      cnPINN (overlapping std) with no w_U and no penalty term. svd_sigma
      (singular values only): 1.35e-1 ± 1.6e-1, better than dense on 4/5
      seeds but high-variance — the trained *directions* carry most of the
      gain, which bounds how cheap C2 can go.

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
