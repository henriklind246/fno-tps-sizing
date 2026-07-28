from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fno_tps.config import StudyConfig
from fno_tps.model import FNOConfig, TPSFNO, TRANSFER_SOURCE_REVISION


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RIGNOThreePhaseScheduler:
    """Epoch scheduler transferred in lean form from the IHCP training runtime."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        peak_lr: float,
        warmup_fraction: float = 0.02,
        cosine_fraction: float = 0.88,
        init_ratio: float = 0.05,
        cosine_floor_ratio: float = 0.05,
        final_ratio: float = 0.005,
    ):
        self.optimizer = optimizer
        self.epochs = max(1, int(epochs))
        self.peak_lr = float(peak_lr)
        self.warmup_epochs = max(1, int(round(self.epochs * warmup_fraction)))
        self.cosine_epochs = max(1, int(round(self.epochs * cosine_fraction)))
        if self.warmup_epochs + self.cosine_epochs > self.epochs:
            self.cosine_epochs = max(1, self.epochs - self.warmup_epochs)
        self.init_lr = self.peak_lr * init_ratio
        self.cosine_floor = self.peak_lr * cosine_floor_ratio
        self.final_lr = self.peak_lr * final_ratio
        self.last_epoch = -1
        self._set_lr(self.init_lr)

    def _lr_for_epoch(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            fraction = (epoch + 1) / self.warmup_epochs
            return self.init_lr + fraction * (self.peak_lr - self.init_lr)
        cosine_index = epoch - self.warmup_epochs
        if cosine_index < self.cosine_epochs:
            fraction = cosine_index / max(1, self.cosine_epochs - 1)
            cosine = 0.5 * (1.0 + np.cos(np.pi * fraction))
            return self.cosine_floor + cosine * (self.peak_lr - self.cosine_floor)
        tail_epochs = max(1, self.epochs - self.warmup_epochs - self.cosine_epochs)
        tail_index = min(epoch - self.warmup_epochs - self.cosine_epochs + 1, tail_epochs)
        fraction = tail_index / tail_epochs
        return self.cosine_floor * (self.final_lr / self.cosine_floor) ** fraction

    def _set_lr(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(learning_rate)

    def step(self) -> float:
        self.last_epoch += 1
        learning_rate = self._lr_for_epoch(self.last_epoch)
        self._set_lr(learning_rate)
        return learning_rate

    def state_dict(self) -> dict[str, Any]:
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_epoch = int(state["last_epoch"])
        if self.last_epoch >= 0:
            self._set_lr(self._lr_for_epoch(self.last_epoch))


def capture_git_state(directory: str | Path | None = None) -> dict[str, str]:
    """Return the repository revision and working-tree state when available."""
    cwd = Path.cwd() if directory is None else Path(directory)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        commit, status = "unavailable", "unavailable\n"
    return {
        "git_commit": commit,
        "git_status": status,
    }


def capture_provenance(run_dir: str | Path, study: StudyConfig) -> None:
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        **capture_git_state(),
        "transfer_source_revision": TRANSFER_SOURCE_REVISION,
        "study": study.as_dict(),
        "material_properties": study.property_provenance,
    }
    (destination / "provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_checkpoint(
    path: str | Path,
    model: TPSFNO,
    optimizer: torch.optim.Optimizer,
    scheduler: RIGNOThreePhaseScheduler,
    study: StudyConfig,
    epoch: int,
    best_metric: float,
    normalization: dict[str, float],
    selection_metrics: dict[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "study": study.as_dict(),
            "material_properties": study.property_provenance,
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "normalization": dict(normalization),
            "selection_metrics": (
                dict(selection_metrics)
                if selection_metrics is not None
                else None
            ),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "transfer_source_revision": TRANSFER_SOURCE_REVISION,
        },
        Path(path),
    )


def load_model_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[TPSFNO, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    model = TPSFNO(FNOConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, checkpoint
