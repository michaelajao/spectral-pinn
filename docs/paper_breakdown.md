# Wang et al. (2026): what cnPINN is, what it shows, and where it stops

**Paper.** Z.-P. Wang, J.-Y. Chen, L.-L. Guo, L.-S. Zhang, Z.-Y. Zhang,
"Utilizing condition number to diagnose and mitigate the training failure of
physics-informed neural network," *Applied Soft Computing* 202 (2026) 115794.
Code: https://github.com/zzy-muc/cnPINNs (TensorFlow 1.x with tf.contrib's
ScipyOptimizerInterface, so it will not run on a modern stack; Navier–Stokes
example only, with the Taylor–Green reference computed analytically; relies
on globals set under ``__main__``). Two discrepancies from the paper text
matter for anyone reimplementing it: the code freezes U and trains V (the
paper's Sect. 4.6.2 says the two are equivalent), and it penalizes an
*unsquared* matrix 1-norm of the defect, whereas Eq. 15 squares a 2-norm.
Squared vs unsquared is not cosmetic: D^2 has zero gradient at an orthogonal
initialization, which is what makes their adaptive-weighting attempts blow
up (Sect. 4.6.3). We follow Eq. 15 and footnote the code.

## Framing correction

This is a PINN training-diagnostics and optimization paper, not a fluids
paper. The Navier–Stokes example (their Section 4.5) is the 2D Taylor–Green
vortex on [-2,2]^2 x [0,1] at nu = 1e-3: smooth, analytic, no Reynolds sweep,
no boundary layers. It is a verification case chosen because a closed-form
solution makes the L2 error computable.

## The definition doing the work

For a function f, the paper's "condition number" is the *absolute* condition
number kappa(f) = ||J(f)||_F — the Frobenius norm of the Jacobian with
respect to the network *inputs* (x, y, t), integrated in L2 over the domain
(their Eq. 1). This is a gradient-magnitude measure, an H^1 seminorm, not
the classical sigma_max/sigma_min. Readers from numerical analysis will
misread it unless told; we should not adopt the term without qualification.

## Empirical observation (their Figs. 1–3)

Across supervised fits of x, 10x, cos x, cos 10x and PINN solutions of the
advection equation u_t + u_x = 0, the mismatch ||kappa(u_theta) - kappa(u)||_2
tracks the L2 relative error approximately linearly. The adaptive-activation
method (AAF) helps precisely because its scale factor an > 1 raises
kappa(u_theta) toward the target's value (their Fig. 2B).

## Theory (their Theorem 2.1)

Under (i) kappa(u), kappa(u_theta) <= M and (ii) a directional-consistency
condition <J(u), J(u_theta)>_F >= delta ||J(u)||_F ||J(u_theta)||_F, a
Poincaré inequality plus the polarization identity gives

  ||u - u_theta||_{2,Omega}
    <= C ( ||kappa(u) - kappa(u_theta)||_{2,Omega} + sqrt(1 - delta) )
       + ||u - u_theta||_{2,boundary}.

Assessment: the proof is Poincaré + Cauchy–Schwarz; the sqrt(1-delta) term
does not vanish unless the gradient fields align, and delta is unmeasurable in
practice. Read plainly, the bound says an H^1-close approximation is L2-close
— true, and close to a restatement. The useful content is the diagnostic
correlation, not the bound's sharpness.

## The method (cnPINN), stripped of the framing

A spectral reparameterization of the hidden weights:

1. At initialization only, SVD each square hidden weight: W_j = U_j S_j V_j^T
   (j = 2..L-1; input and output layers stay dense).
2. Freeze V_j; train U_j and diag(S_j) in place of W_j.
3. Add w_U ||U^T U - I||_2^2 to the loss to keep U near-orthogonal.

Cost: +n parameters per layer; 1.28–1.83x vanilla PINN wall-clock across
their Navier–Stokes width and collocation-point sweeps (Figs. 15D, 17D; the
depth sweep's values are not given in the text); GPU memory 1263 MB peak vs
1252 MB for vanilla (their Table 4). Mechanism: with U and V orthogonal, diag(S) *is* the singular-value
spectrum of W, so the optimizer gets a decoupled handle on the network's
Lipschitz scale instead of moving it implicitly through dense updates (their
Prop. 3.2 sandwiches kappa(u_theta) between products of min/max singular
values).

## Headline numbers

- Taylor–Green (5x50 tanh, N_f = 10^4, Adam 20k -> L-BFGS): cnPINN reaches
  6.1e-3 / 6.7e-3 on u / v where PINN, AAF and NTK sit at 2.4–2.9e-2 and
  E-DNN at 6.2e-3 / 6.9e-3 with 2.5–3.8x cost (Table 3 at 0% noise,
  Fig. 15D). Best pressure error 3.86e-4, from the width sweep at 20 neurons
  (Fig. 15C); Table 3's pressure figure at 0% noise is 4.54e-3.
- Noise: at 5% Gaussian IC/BC noise cnPINN holds 8.8e-3 / 7.8e-3 on u / v;
  the others degrade to 1.4–4.4e-2 (Table 3).
- Loss: final total loss 4.46e-7; no compared method breaks 1e-6 (Fig. 19).

## Weaknesses to keep in view

1. w_U is hand-tuned per PDE over two and a half decades — 1/35000
   (advection), 1/20000 (mKdV), 1/1000 (Klein–Gordon), 1/600
   (Lotka–Volterra), 1/200000 (Navier–Stokes) — and their Section 4.6.3
   reports that NTK- and DB-PINN-style adaptive weighting *fails* for this
   term because the orthogonality loss starts at exactly zero, so the
   adapted weight grows without bound.
2. No Fourier-feature, SIREN, modified-MLP, or hard-BC baselines — random
   Fourier features are the standard remedy for exactly the high-frequency
   targets the paper opens with. (A sine activation appears only as an
   ablation of cnPINN itself in Table 5, not as a SIREN baseline.)
3. Three runs per Navier–Stokes cell; some wins sit inside one standard
   deviation of the baseline (pressure at 0% noise: PINN (8.84±9.27)e-3 vs
   cnPINN (4.54±3.75)e-3).
4. The conclusion claims the five PDEs "cannot be trained by the vanilla PINN
   and AAF"; their own results show vanilla PINN reaching 1e-3 to 4e-2
   depending on the case (7.5e-3 on Klein–Gordon, Sect. 4.3; 9.8e-3 / 1.6e-3
   on Lotka–Volterra, Table 2; 1e-3 in 8 of 100 advection runs, Sect. 2.2) —
   worse, not failing.
5. Section 4 states all experiments ran on an i5-12400F CPU; Section 4.5.5
   reports GPU memory.
6. Every assumption of Theorem 2.1 requires u in H^1 with bounded ||J(u)||_F.

Credit where due: Section 4.6 is a careful ablation study — activation
functions (tanh/sin/ReLU/GELU), freezing U vs V (equivalent, their Table 6),
w_U sensitivity over two orders of magnitude (Table 7), and a parameter-
freezing sweep (cnPINN-r) showing many rows of U are inert.

## Why this matters for our dam-break line

A dam-break solution carries a shock: ||J(u)||_F is unbounded and u is not in
H^1, so Theorem 2.1 is formally inapplicable to the entire problem class of
the submitted Computers & Fluids paper. The authors say as much in another
context ("this condition will be violated once the PINN training fully
collapses"). That gap — not a complaint, an opening — is the starting point
of the proposal in `proposal.md`.
