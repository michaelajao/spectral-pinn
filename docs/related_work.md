# Related work and novelty sweep

Status: sweep of 2026-08-25 (web search; arXiv-indexed material). cnPINN
appeared online 2026-06-19, so citing work may still be unindexed — re-run
this sweep before submission. Verify every entry against the actual paper
before citing; titles and arXiv ids below were checked only against search
listings.

## Closest prior art, and the difference

| Work | What it does | Difference from our C1/C2 |
|---|---|---|
| cnPINN — Wang et al., Appl. Soft Comput. 202 (2026) 115794 | SVD-reparameterize hidden weights at init; train U + diag(S), freeze V; soft penalty w_U·||U'U−I||² | The paper we extend. Our svd_hard removes the penalty (orthogonal by construction); svd_sigma removes U's training too. |
| SVD-PINNs — [arXiv 2211.08760](https://arxiv.org/abs/2211.08760) | Freeze U *and* V of a trained PINN, retrain singular values on a new PDE (transfer learning) | Same factor structure as our svd_sigma but as a *transfer* method from a trained model; we train from scratch, and cnPINN does not cite it — both must be cited and differentiated. |
| Cheap Orthogonal Constraints in NNs — Lezcano-Casado & Martínez-Rubio 2019, [arXiv 1901.08428](https://arxiv.org/abs/1901.08428) | Exponential/Cayley parametrizations for exactly-orthogonal weights (now torch.nn.utils.parametrizations.orthogonal) | The mechanism we borrow. Applied there to RNNs; no PINN or PDE application found. |
| AL-PINNs — Son et al., Neurocomputing 2023, [arXiv 2205.01059](https://arxiv.org/abs/2205.01059); conditionally adaptive ALM, [arXiv 2508.15695](https://arxiv.org/html/2508.15695v1) | Augmented-Lagrangian treatment of IC/BC constraints in the PINN loss | Constraint is on the *solution's* boundary data, not on weight structure. Relevant as the mechanism for our soft-ablation arm. |
| Spectral normalization (Miyato et al. 2018 line; e.g. PINN-adjacent use in [arXiv 2405.16770](https://arxiv.org/pdf/2405.16770)) | Divide each weight by sigma_max to force a 1-Lipschitz map | Clamps the top singular value to a constant; our variants *train* the full spectrum. Opposite intent: SN caps expressible gradient magnitude, cnPINN's diagnostic says networks fail by having too *little* of it for steep targets. |
| Orthogonality-constrained networks for inverse eigenvalue problems ([arXiv 2406.19981](https://arxiv.org/pdf/2406.19981), [arXiv 2601.17798](https://arxiv.org/pdf/2601.17798)) | Network *outputs* live on a Stiefel manifold | Constraint on the solution, not on trained weights of a PDE solver. Not a collision. |

Searched and not found (2026-08-25): hard/manifold-orthogonal weight training
for PINNs; trainable-spectrum reparameterization for PINNs outside
cnPINN/SVD-PINNs; condition-number or gradient-norm mismatch diagnostics for
PINNs on hyperbolic problems; any indexed citation of cnPINN.

## Shock-regime PINN literature (context for C3/C4 baselines and framing)

- wPINN (Kruzhkov-entropy weak form) and relatives — pointwise residuals are
  ill-defined at shocks; weak forms remain defined.
- CLINN ([arXiv 2509.02091](https://arxiv.org/html/2509.02091)) — jump
  conditions and boundedness built into loss/architecture.
- WHC-PINN ([Sci. Rep. 2025](https://www.nature.com/articles/s41598-025-34263-1)) —
  gradient-weighted loss at shock regions + hard BCs.
- Relaxation neural networks ([arXiv 2404.01163](https://arxiv.org/pdf/2404.01163)),
  non-diffusive NN methods ([arXiv 2405.15559](https://arxiv.org/pdf/2405.15559)),
  estimatable-variation NNs ([arXiv 2409.08909](https://arxiv.org/pdf/2409.08909)).
- Plus the SWE-specific PINN line already surveyed in the swe-dambreak paper
  (Dazzi 2024; Qi et al. 2024/2025; Tian et al. 2025; FVM-informed hybrids).

None of these measure or control a network condition number; they modify the
residual or the architecture. Our C4 diagnostic is complementary to all of
them and is, as far as this sweep shows, unclaimed territory.

## Verdict

C1 (penalty-free hard orthogonality in PINN training) and C4 (local
condition-number diagnostics at shocks): no collision found; components exist
separately (Lezcano-Casado mechanism; cnPINN target). C2's svd_sigma is the
from-scratch twin of SVD-PINNs' transfer setting and must be presented as
such, never as new in structure. C3/C5's value is the excluded-regime
validation, which no cnPINN-line work has attempted.
