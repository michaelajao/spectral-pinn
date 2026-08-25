"""spectral-pinn: spectral control of physics-informed neural networks.

Modules:
    solver      differentiable well-balanced 2D shallow-water FV solver
    models      PINN / FVM-informed PINN + spectral weight variants + losses
    train       generic training loop (seeds, checkpoints, CSV logging)
    benchmarks  2D dam-break benchmark definitions
    metrics     error and diagnostic metrics
    run         config-driven experiment driver (python -m src.run <cfg.yaml>)
"""
