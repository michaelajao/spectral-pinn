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
cost (1.28–1.83x vanilla across their width and collocation sweeps, Figs.
15D and 17D); (3) validation covers only "controlled synthetic examples"
(their words) — all five of which are smooth. A paper that resolves all three, on problems their
H^1-based theory formally excludes, is the project.

## Contributions

**C1 — establish what the orthogonality penalty actually does.** Our
diagnostic on their own advection case (20 seeds; `docs/roadmap.md`, Results
log) shows the mechanism cnPINN is built on does not hold after training: at
the published w_U = 1/35000 the trained U has ||U^T U - I||_2 close to 1 and
is rank-deficient, and diag(s) differs from the effective weight's singular
values by up to a factor of 13. Setting w_U = 0 lets U grow to a defect of
about 60 and returns the error to worse than a plain MLP; constraining U to be
exactly orthogonal (`weight_param: svd_hard`, no penalty and no w_U) is stable
but reaches only 1.7e-2 median error against the weak penalty's 8.4e-3. The
four regimes order as unconstrained (2e-1) < dense (7e-2) < exactly orthogonal
(1.7e-2) < weakly penalized (8.4e-3). We therefore argue that the weak penalty
acts as a bound on U's scale rather than as an orthogonality constraint, and
that exact orthogonality over-constrains the problem. This reframes their
stated limitation 1: the open question is not how to tune w_U adaptively but
what constraint class the term belongs to. It also predicts that a bound
weaker than orthogonality — a spectral-norm cap on U, or an explicit box on
its singular values — should recover the full gain without a tuned penalty,
which is the method we propose to develop and test.

**C2 — cost, and the limit of spectral-only training.** The `svd_sigma`
variant trains only the n singular values per layer with U and V frozen
orthogonal. On advection it reaches 3.29e-1 ± 5.5e-1 over 20 seeds, worse than
the dense baseline with two diverged runs, so spectral-only training is not a
viable cheap variant on this problem; the trained directions, not the trained
spectrum, carry the gain. That is a negative result worth reporting, and it is
consistent with C1: if the gain were spectral control, freezing orthogonal
directions and training s would preserve it. Structurally this layer is the
same as SVD-PINNs (Gao, Cheung & Ng, arXiv 2211.08760), which uses it to
transfer from a trained model; the from-scratch use and the failure mode are
what we add. The cost comparison across dense, soft, hard and sigma on
identical budgets still stands as the answer to their limitation 2.

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
  strawman. Their five published w_U values span 1/600 to 1/200000 and none
  is for shallow-water flow, so the baseline needs a Table-7-style sweep on
  one benchmark before the matrix numbers mean anything; the penalty must
  be Eq. 15's squared 2-norm (our first implementation used a Frobenius
  norm, up to a factor 128 stronger at width 128 — corrected 2026-08-25).

## Venue

First choice *Applied Soft Computing* — the paper answers, point by point,
the stated future work of an ASOC paper. CMAME or JCP if the C4 diagnostic
plus theory section comes out strong enough to headline.

## Phasing

See `roadmap.md`. Phase 0 (reproduction + sweep) is running now; Phase 1
(comparison matrix) starts as soon as the reproduction gate passes.
