# Reference cross-check: our HLLC against the co-authors' schemes

How far our in-process reference — HLLC MUSCL-van Leer at N = 512, the field
every error in `ml_runs/` is measured against — sits from the co-authors'
solver output for the same initial condition. Depth only: their drop carries
`h` and no momentum. Both fields are taken to the shared N = 128 evaluation
grid at the final output time (t = 2 s), theirs by node-to-centre bilinear
interpolation. Reproduce with `python -m src.run <config>`, which now writes
this table into every run's report.

`classical@128` is our own HLLC at the evaluation resolution against our
N = 512 reference — the discretisation error already reported in the main
table, included here as the scale to read the rest against.

| benchmark | classical@128 | vs their HLL | vs their MUSCL-RS | vs their LW | PINN (5 seeds) |
|---|---|---|---|---|---|
| ca_step | 2.476e+02 | 2.169e+03 | 2.208e+03 | 9.349e+03 | 1.028e+03 |
| ca_circular_wet | 3.914e+02 | 3.111e+02 | 7.395e+02 | 7.551e+03 | 1.602e+03 |
| ca_gaussian | 6.357e+00 | 3.763e+02 | 3.949e+02 | 3.435e+03 | ~3.9e+02 |

All entries are L1(h) over the 100 x 100 m domain, so an L1 of 1e+03 is a mean
depth discrepancy of about 0.1 m. The PINN column is the vanilla `pinn` row
from `ml_runs/main_table.md` (ca_gaussian was still running; its five seeds
span 3.47e+02 to 4.16e+02).

## What it says

**The reference is not a ground truth, and how much that matters depends on
the benchmark.** Three different regimes appear:

- `ca_circular_wet` — the reference is solid. Our N = 512 field agrees with
  their HLL to 3.11e+02, *tighter* than our own N = 128 discretisation error
  of 3.91e+02, and both are well under the 1.60e+03 PINN error. Differences
  between PINN variants here are real.
- `ca_gaussian` — the reference disagreement (3.76e+02) and the PINN error
  (~3.9e+02) are the same number. Meanwhile our own discretisation error is
  6.36e+00, sixty times smaller. Nothing about a PINN's accuracy on this case
  can be read from a comparison against one solver family.
- `ca_step` — the two solver families differ by 2.2e+03, twice the 1.03e+03
  PINN error. The planar shock reflecting off the closed walls is where the
  schemes diverge most, and no ranking of the neural entries on this benchmark
  survives a change of reference.

Lax-Wendroff is the outlier everywhere (3.4e+03 to 9.3e+03), which is expected
of Lax-Wendroff with artificial viscosity on shocks and is a statement about
that scheme rather than about ours.

This does not say our reference is wrong — HLLC MUSCL-van Leer at N = 512 is
the most refined and least diffusive of the four, and on `ca_circular_wet` it
is vindicated. It says the comparison table has a resolution floor that varies
by benchmark, and that the floor must be quoted next to any claimed margin.
The immediate consequence for the paper: margins on `ca_step` and
`ca_gaussian` should be reported as bounded by reference uncertainty, not as
measured differences.
