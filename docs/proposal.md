# Proposal: penalty-free spectral control of PINNs, validated where the theory breaks

Working title: *Penalty-free spectral control of physics-informed neural
networks, with validation on shock-dominated shallow-water flows.*

Team: M. Ajao-Olarinoye (Coventry), M. A. Farooq (NUST), and collaborators
from the dam-break paper (submitted to Computers & Fluids, Aug 2026).

## Problem statement — taken from the source paper's own conclusion

Wang et al. (2026) close cnPINN (see `paper_breakdown.md`) with three open
problems: (1) the orthogonality-penalty weight w_U must be hand-tuned per
PDE, and the adaptive-weighting schemes they tried (NTK, DB-PINN) diverge on
it because the penalty starts at exactly zero; (2) the penalty adds training
cost (1.28–1.83x vanilla in their runs); (3) validation covers only smooth
synthetic examples. A paper that resolves all three, on problems their
H^1-based theory formally excludes, is the project.

## Contributions

**C1 — remove the penalty, not tune it.** Parameterize U on the orthogonal
manifold (torch's built-in orthogonal parametrization, after Lezcano-Casado &
Martínez-Rubio 2019), so U^T U = I holds by construction, w_U ceases to
exist, and the singular values of each hidden weight are exactly the trained
diag(S). This turns their Theorem 3.3 bound into an identity. Implemented in
`src/models.py` as `weight_param: svd_hard`; an augmented-Lagrangian soft
version serves as the ablation baseline for the "adaptive w_U" reading of
their limitation 1.

**C2 — cost.** The `svd_sigma` variant trains only the n singular values per
layer (U, V frozen orthogonal): exact spectral control at n parameters per
layer, the r -> 0 limit of their own cnPINN-r sweep, which already showed
most rows of U inert. Report a cost/accuracy frontier across dense, soft,
hard, and sigma on identical budgets.

**C3 — validation in the excluded regime.** The 2D dam-break suite ported
from swe-dambreak: six initial-condition variants at the submitted paper's
conventions (g = 2, Gaussian-hump bed, reflective walls), wet and dry
downstream states, the three-hump Manning-friction case, five seeds per
cell, against a fine-grid HLLC reference. Shocks make ||J(u)||_F unbounded,
so every run sits outside Theorem 2.1's hypotheses — the question is whether
spectral control still helps, and where.

**C4 — a local diagnostic.** Replace the domain-integrated scalar
||kappa(u_theta) - kappa(u)||_2 by the pointwise field
kappa(x, y, t) = |grad u|, computed by autograd on the network and by finite
differences on the HLLC reference. Hypothesis: the mismatch concentrates on
the shock set, which no global spectral scale can fix; the FVM-informed loss
(already in the harness) is the composition test for what can.

**C5 — the Navier–Stokes leg.** Reproduce their Taylor–Green case for direct
comparability (their Table 3 gives the targets), then the lid-driven cavity
at Re 100–1000, whose corner singularities are the incompressible analogue
of the unbounded-gradient regime in C3/C4.

Baselines throughout include a Fourier-feature PINN — the standard
high-frequency remedy absent from their comparison — plus vanilla PINN,
cnPINN (soft) as published, and the FVM-informed PINN.

## Why the paper survives a negative result

If svd_hard does not beat Fourier features at a shock front, C4 explains why
(a localized pathology on a measure-zero set) and the FVM-informed
composition shows what does work; C1/C2 stand on their own as the resolution
of limitations 1 and 2. The design does not hinge on the method winning.

## Theory work (NUST side)

How the constants in their Theorem 2.1 degrade under vanishing-viscosity
regularization u_eps as eps -> 0: M(eps) grows like the inverse shock width,
so the bound is vacuous in the limit — a short, self-contained analysis that
formalizes why the smooth-case diagnostic must be localized (C4) to say
anything about hyperbolic problems.

## Risks

- Deep literature sweep (Phase 0a) may surface prior hard-orthogonal PINN
  training; the spot check on 2026-08-25 found SVD-PINNs (arXiv 2211.08760;
  frozen U, V, trained S — but for transfer learning) and AL-PINNs (arXiv
  2205.01059; augmented Lagrangian on IC/BC, not on weight structure), and
  nothing on either hard orthogonality in PINN training or condition-number
  diagnostics at shocks. If a collision appears, C3–C5 carry the paper.
- L-BFGS interacts with parametrized modules through closure re-evaluation;
  fallback is the Adam -> L-BFGS schedule Wang et al. themselves use for
  Navier–Stokes.
- The w_U-free claim must be tested against a *tuned* soft baseline, not a
  strawman: reuse their published per-PDE values.

## Venue

First choice *Applied Soft Computing* — the paper answers, point by point,
the stated future work of an ASOC paper. CMAME or JCP if the C4 diagnostic
plus theory section comes out strong enough to headline.

## Phasing

See `roadmap.md`. Phase 0 (reproduction + sweep) is running now; Phase 1
(comparison matrix) starts as soon as the reproduction gate passes.
