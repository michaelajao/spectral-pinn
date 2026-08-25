# Roadmap

Working plan; tick items as they land. Dates are rough, assuming part-time
effort alongside other work.

## Phase 0 — foundations and reproduction (now – ~2 weeks)

- [x] New repo scaffolded (flat `src/` layout), solver + harness ported from
      swe-dambreak, tests green
- [x] `SVDLinear` with `svd_soft` (cnPINN), `svd_hard` (ours), `svd_sigma`
      modes + unit tests
- [ ] Literature sweep -> `related_work.md` with a novelty verdict
- [ ] cnPINNs reference code cloned to `third_party/` (TF1; read-only)
- [ ] Reproduction gate: Wang et al.'s 1D advection case (3x25 tanh,
      N_u = 100, N_f = 500, w_U = 1/35000, L-BFGS) — cnPINN ~1e-3 L2 rel.
      error vs vanilla ~1e-2..1e-1, matching their Figs. 5–7 magnitudes
- [ ] svd_hard on the same case: error <= cnPINN's with w_U removed

## Phase 1 — comparison matrix on the dam-break suite (~2–3 weeks)

- [ ] Smoke run (`configs/smoke.yaml`) end-to-end on GPU
- [ ] Full matrix (`configs/main.yaml`): 6 entries x 5 benchmarks x 5 seeds
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
