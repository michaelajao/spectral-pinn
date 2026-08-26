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

## Status: not read by any code in this repo

Nothing here is loaded at runtime. Every error metric in `reports/` compares
against an in-process HLLC reference built by `src/solver.py` at N = 512 —
see the module docstring of `src/benchmarks.py`. This data is versioned for
provenance and as the reference for a future cross-check against the
collaborators' schemes; wiring it into the evaluation harness is still to do.

45 files, 270 MB, float64 with ~18 significant digits per value.
