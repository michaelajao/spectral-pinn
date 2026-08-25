# spectral-pinn

Spectral control of physics-informed neural networks: replacing cnPINN's soft
orthogonality penalty (Wang et al. 2026, Applied Soft Computing 202:115794)
with weight factors that are **orthogonal by construction**, and validating
where that paper's theory formally breaks — solutions with unbounded gradient
norm: shock-dominated 2D shallow-water dam breaks, then incompressible
Navier–Stokes cases.

The solver, benchmarks, and evaluation harness are ported from the
[`swe-dambreak`](../swe-dambreak) project (the comparative dam-break paper
submitted to *Computers & Fluids*), consolidated into a single flat package:

```
src/solver.py       differentiable, well-balanced 2D SWE finite-volume solver
                    (HLLC/HLL/Rusanov, MUSCL, Audusse well-balancing, SSP-RK2)
src/models.py       strong-form PINN + spectral weight variants (SVDLinear:
                    svd_soft = cnPINN, svd_hard = ours, svd_sigma = cheapest),
                    FVM-informed PINN, and both loss stacks
src/train.py        generic training loop (seeds, checkpoints, CSV logging)
src/benchmarks.py   2D dam-break benchmark definitions (B1–B4 + paper variants)
src/metrics.py      error and diagnostic metrics
src/run.py          config-driven experiment driver
```

## Setup

```bash
pip install -e . --group dev
pytest                                  # verify the port
```

## Running

```bash
python -m src.run configs/smoke.yaml    # fast end-to-end pipeline check
python -m src.run configs/main.yaml     # the comparison matrix
```

Per-run artifacts land in `runs/ml/<entry>/<benchmark>/seed<k>/`; each config
writes its table to `reports/ml_runs/<config>_table.md`.

## Project documents

- `docs/proposal.md` — the research proposal (contributions, experiment plan)
- `docs/paper_breakdown.md` — analysis of the cnPINN paper this builds on
- `docs/related_work.md` — novelty sweep
- `docs/roadmap.md` — phased plan with progress checkboxes

Note: the installed package is named `src` (flat research layout); do not
`pip install` this alongside another project that also installs a top-level
`src` package.
