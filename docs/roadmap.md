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
- [x] svd_hard on the same case: bounded but weaker than the soft penalty at
      20 seeds (see Results log) — C1 reframed, not confirmed
- [x] Mechanism diagnostic (`--diagnose`): the trained soft models are not
      orthogonal and their diag(s) is not the weight spectrum (Results log)

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

**2026-08-26 — Phase 0 gate, 20 seeds (supersedes the 5-seed tables).**
Wang et al.'s advection case, L-BFGS 3000 iterations, their Eq. 15 penalty
(2-norm), L2 relative error. Table of record:
`reports/advection_reproduction_20seeds.md`.

| entry | mean ± std | median | Wang et al. Table 5 (tanh) |
|---|---|---|---|
| vanilla PINN (dense) | 1.37e-1 ± 1.4e-1 | 7e-2 | (1.13 ± 1.67)e-1 |
| cnPINN (svd_soft, w_U = 1/35000) | 1.46e-2 ± 1.5e-2 | 8.4e-3 | (5.64 ± 3.86)e-3 |
| svd_hard (no penalty, no w_U) | 3.31e-2 ± 3.9e-2 | 1.7e-2 | — |
| svd_sigma (singular values only) | 3.29e-1 ± 5.5e-1 | 1.2e-1 | — |
| svd_soft parameterization, w_U = 0 | 3.37e-1 ± 3.5e-1 | 2e-1 | — |

The vanilla and cnPINN cells reproduce the published ones, so the setup is
faithful. The other three rows say what the source paper does not test.

Two earlier readings were wrong and are retracted. The 5-seed tables reported
svd_hard as matching cnPINN and svd_sigma as beating dense; at 20 seeds
svd_hard is about twice cnPINN's median error and svd_sigma is worse than
dense with two diverged seeds. The 5-seed tables were also confounded: the
first used a Frobenius penalty (up to 128x the Eq. 15 2-norm at width 128),
and the correction commit simultaneously zeroed SVDLinear's bias under Xavier
init, so the before/after pair were different experiments. Both tables are
kept — `reports/advection_reproduction_frobenius.md` and
`reports/advection_reproduction.md` — as the ablation record.

**2026-08-26 — what the penalty does (`python -m src.advection --diagnose`).**
Per SVD layer of trained models, 3 seeds: D_U = ||U^T U - I||_2, the largest
relative gap between the trained |s| and the effective weight's singular
values, and that weight's extreme singular values.

| setting | error (median) | D_U (median) | |s| vs singular values |
|---|---|---|---|
| w_U = 1/35000 | 7.9e-3 | 0.98 | up to 13x apart |
| w_U = 0 | 3.05e-1 | 63 (max 177) | up to 31x apart |

At the published weight the trained U is rank-deficient — D_U near 1 means an
eigendirection of U^T U has collapsed — and diag(s) is not the weight's
spectrum, so the mechanism the paper describes (singular values steered
through diag(s) with U, V orthogonal) does not survive training. Removing the
penalty lets U grow by two orders of magnitude and costs the entire gain.
Ordering the four regimes by median error: unconstrained 2e-1, dense 7e-2,
exactly orthogonal 1.7e-2, weakly penalized 8.4e-3. The penalty is doing
something necessary, and it is bounding U rather than orthogonalizing it;
exact orthogonality is a stronger constraint than the problem wants.

Caveat: one smooth 1D problem, one architecture, one optimizer. Whether the
ordering survives on shocks is what the Phase 1 matrix tests.

**Phase 1 matrix (`configs/main.yaml`)** launched 2026-08-25 on GPU 1, ~14 min
per strong-form run. First finished run: pinn/ca_circular_wet/seed0,
L1(h) = 1.586e3 against the classical HLLC 3.914e2 at N = 128. Partial results
in `reports/ml_runs/main_table.md`; note that this run predates the Eq. 15
correction, so its cnpinn_soft rows carry the Frobenius penalty (every other
entry is unaffected, since only the soft variant reads that norm).
