"""Training loop for the neural entries: seeds, checkpointing, CSV logging.

Ported unchanged from ``swe-dambreak`` (src/ml/train.py).

Generic over all entries — the caller passes a ``loss_fn`` returning
(total_loss, components_dict); this module owns the optimizer, seed control,
best-checkpoint saving, per-iteration CSV logging, and loss-curve figures.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import torch

matplotlib.use("Agg")  # headless backend: training runs on machines without a display
import matplotlib.pyplot as plt


@dataclass
class TrainConfig:
    iters: int = 20000
    lr: float = 1.0e-3
    optimizer: str = "adam"
    grad_clip: float = 0.0            # 0 disables
    lbfgs_iters: int = 0              # optional L-BFGS polish after Adam
    log_every: int = 200
    seed: int = 0
    device: str = "cpu"
    out_dir: str = "runs/ml"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    model: torch.nn.Module,
    loss_fn: Callable[[], tuple[torch.Tensor, dict]],
    cfg: TrainConfig,
    provenance: dict | None = None,
) -> dict:
    """Optimize ``model`` on ``loss_fn``; returns a history dict and writes
    out_dir/{history.csv, best.pt, loss_curves.png, meta.json}.

    ``provenance`` is merged into meta.json and should carry whatever identifies
    the run beyond the optimizer settings -- the entry name, benchmark, and any
    entry-specific knob such as the gauge count. Without it a run directory is
    only identifiable by its path, so two entries that share a configuration
    (say a named data ablation and its gauge-sweep twin) are indistinguishable
    from their contents alone.
    """
    set_seed(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: list[dict] = []
    best = float("inf")

    def closure_step(i: int) -> dict:
        opt.zero_grad(set_to_none=True)
        total, comps = loss_fn()
        total.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        rec = {"iter": i, "loss": float(total.detach())}
        rec.update({k: (float(v) if not isinstance(v, (list, tuple)) else json.dumps(v))
                    for k, v in comps.items()})
        return rec

    for i in range(cfg.iters):
        rec = closure_step(i)
        if i % cfg.log_every == 0 or i == cfg.iters - 1:
            history.append(rec)
            if rec["loss"] < best:
                best = rec["loss"]
                torch.save(model.state_dict(), out / "best.pt")

    if cfg.lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(), max_iter=cfg.lbfgs_iters,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            total, _ = loss_fn()
            total.backward()
            return total

        lbfgs.step(closure)
        final, comps = loss_fn()
        rec = {"iter": cfg.iters, "loss": float(final.detach())}
        history.append(rec)
        if rec["loss"] < best:
            best = rec["loss"]
            torch.save(model.state_dict(), out / "best.pt")

    _write_history(out, history)
    _plot_curves(out, history)
    meta = {"best_loss": best, "iters": cfg.iters, "lr": cfg.lr, "seed": cfg.seed,
            "grad_clip": cfg.grad_clip, "lbfgs_iters": cfg.lbfgs_iters,
            "device": cfg.device}
    meta.update(provenance or {})
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return {"history": history, "best_loss": best}


def _write_history(out: Path, history: list[dict]) -> None:
    if not history:
        return
    keys = sorted({k for rec in history for k in rec})
    keys = ["iter", "loss"] + [k for k in keys if k not in ("iter", "loss")]
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)


def _plot_curves(out: Path, history: list[dict]) -> None:
    if not history:
        return
    its = [r["iter"] for r in history]
    comp_keys = [k for k in history[0]
                 if k not in ("iter",) and isinstance(history[0][k], float)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in comp_keys:
        ys = [r.get(k, float("nan")) for r in history]
        ax.semilogy(its, ys, label=k, lw=1.1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (log)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=130)
    plt.close(fig)
