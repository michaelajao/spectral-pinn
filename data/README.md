# Reference data — the collaborators' dam-break solver output

Copied byte-for-byte from the `swe-dambreak` project (the comparative
dam-break paper submitted to *Computers & Fluids*, Aug 2026), where this drop
was received from the co-authors and audited in that repo's
`reports/data_audit.md`. Three of that paper's six IC variants are kept here —
the ones our benchmarks reproduce. Variants 2 (rectangular), 5 (parabolic) and
6 (triangular) are not used by this project and stay in the source repo.

## Conventions

Verified numerically in the source audit, and matched by `src/benchmarks.py`:

- Domain [0, 100]^2 m, 501x501 **nodes**, dx = dy = 0.2 m
- **g = 2.0 m/s^2** (nonstandard; `G_CA` in `src/benchmarks.py`), Manning n = 0
- Gaussian-hump bed `Z = 2 exp(-((x-50)^2 + (y-50)^2)/200)`
- Reflective boundaries on all four sides — the domain is closed, so any mass
  drift is numerical, not outflow
- t_end = 2 s, snapshots every 0.5 s
- Every file is a headerless CSV of **water depth h** (not the free surface
  eta = h + Z) for one snapshot. Momentum fields are absent.
- Schemes: `HLL`, `LW` (Lax-Wendroff with artificial viscosity), `MUSCLRS`
  (MUSCL minmod on conserved variables + Rusanov + SSP-RK3)

## Mapping to our benchmarks

| directory | paper variant | our benchmark id |
|---|---|---|
| `Variant 1 Step Dam-Break/` | 1, step | `ca_step` |
| `Variant 3 Circular Dam-Break/` | 3, circular | `ca_circular_wet` |
| `Variant 4 Gaussian Dam-Break/` | 4, gaussian | `ca_gaussian` |

Two of our benchmark ids have **no** counterpart here and must not be
validated against this data:

- `ca_circular_dry` sets the downstream depth to 0; the reference Variant 3 is
  the wet case (h_out = 1 m). The dry variant is ours, added to exercise the
  FVM-PINN low-momentum collapse.
- `b3_three_humps` and the rest of the B-series are synthetic cases defined
  only in `src/benchmarks.py`.

## How it is used

`src/benchmarks.py` reads this tree; `DATA_ROOT` resolves from the module's
own location (override with `SPECTRAL_PINN_DATA`), so it is correct whatever
the working directory:

```python
from src.benchmarks import load_reference_depth, reference_depth_on
h = load_reference_depth("ca_step", "MUSCLRS", t=2.0)      # (501, 501), [y, x]
h_on_ours = reference_depth_on("ca_step", grid, 2.0, "HLL")  # (grid.ny, grid.nx)
```

`src/run.py` calls it automatically for every benchmark that has a
counterpart, and writes a **Reference cross-check** section into the run's
table: our in-process HLLC reference against each of their three schemes,
depth only, at the final output time. That is solver-vs-solver disagreement,
and it bounds how finely the neural rows can be read — a difference between
two PINN variants smaller than the spread between reference schemes is not
resolvable.

It remains true that no *error metric for a neural entry* is computed against
this data. Those still use the in-process HLLC reference at N = 512 built by
`src/solver.py`; this tree is the independent check on that reference.

The axis order ([y, x]) and the domain mapping are verified against the
analytic IC in `tests/test_benchmarks.py` rather than assumed — a transposed
or mis-registered read would fail those tests.

45 files, 270 MB, float64 with ~18 significant digits per value.
