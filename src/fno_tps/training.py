from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from fno_tps.config import StudyConfig
from fno_tps.data import create_dataloaders, load_dataset
from fno_tps.evaluation import evaluate_loader
from fno_tps.model import Representation, build_model
from fno_tps.runtime import (
    RIGNOThreePhaseScheduler,
    capture_provenance,
    load_model_checkpoint,
    resolve_device,
    save_checkpoint,
    set_seed,
)


def _training_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    spatial: torch.Tensor,
    config: StudyConfig,
) -> torch.Tensor:
    if (
        (config.training.bond_loss_weight or config.training.interface_loss_weight)
        and not config.structural_interface_regions_are_constant_property()
    ):
        raise NotImplementedError(
            "Bond/interface auxiliary losses require constant-property bond "
            "and backing regions."
        )
    loss = F.mse_loss(prediction, target)
    tps_end = config.mesh.nx_tps
    bond_end = tps_end + config.mesh.nx_bond
    if config.training.bond_loss_weight:
        loss = loss + config.training.bond_loss_weight * F.mse_loss(
            prediction[:, tps_end:bond_end],
            target[:, tps_end:bond_end],
        )
    if config.training.interface_loss_weight:
        loss = loss + config.training.interface_loss_weight * F.mse_loss(
            _structural_interface_trace(prediction, spatial, config),
            _structural_interface_trace(target, spatial, config),
        )
    return loss


def _structural_interface_trace(
    field: torch.Tensor,
    spatial: torch.Tensor,
    config: StudyConfig,
) -> torch.Tensor:
    """Reconstruct the structure-side interface trace in normalized units."""
    if not config.structural_interface_regions_are_constant_property():
        raise NotImplementedError(
            "Structural-interface trace reconstruction currently requires "
            "constant-property bond and backing regions."
        )
    bond_index = config.mesh.nx_tps + config.mesh.nx_bond - 1
    backing_index = bond_index + 1
    left = field[:, bond_index, :, 0]
    right = field[:, backing_index, :, 0]
    # Spatial channels 3 and 4 are dx/d_ref and reference k/k_ref. This is
    # exact here because the bond and backing regions are guarded as constant.
    # Their shared
    # d_ref/k_ref scale and the half-cell factor cancel in the ratio.
    resistance_left = (
        spatial[:, bond_index, :, 3] / spatial[:, bond_index, :, 4]
    )
    resistance_right = (
        spatial[:, backing_index, :, 3] / spatial[:, backing_index, :, 4]
    )
    flux_temperature = (left - right) / (resistance_left + resistance_right)
    return right + flux_temperature * resistance_right


def train_model(
    config: StudyConfig,
    data_dir: str | Path,
    run_dir: str | Path,
    representation: Representation,
    *,
    epochs: int | None = None,
    device: str = "auto",
    batch_size: int | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    capture_provenance(destination, config)
    set_seed(config.training.seed)
    resolved_device = resolve_device(device)
    bundle = load_dataset(data_dir)
    if bundle.manifest["study_config_sha256"] != config.sha256:
        raise ValueError("Dataset study configuration does not match the training configuration.")
    train_loader, val_loader, _ = create_dataloaders(
        config,
        bundle,
        representation,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if len(val_loader.dataset) == 0:
        raise ValueError("Training requires a non-empty validation split.")
    model = build_model(config, representation).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    epoch_count = config.training.epochs if epochs is None else int(epochs)
    scheduler = RIGNOThreePhaseScheduler(
        optimizer,
        epochs=epoch_count,
        peak_lr=config.training.learning_rate,
    )
    best_path = destination / "fno_best.pt"
    latest_path = destination / "fno_latest.pt"
    metrics_path = destination / "train_metrics.csv"
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "val_field_rmse_K",
        "val_bond_peak_mae_K",
        "val_structural_peak_mae_K",
        "val_critical_mean_peak_error_K",
        "val_critical_max_peak_error_K",
        "val_margin_error_mean_K",
        "val_margin_sign_disagreements",
        "epoch_seconds",
        "is_best",
    ]
    best_metric = float("inf")
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(epoch_count):
            epoch_start = time.perf_counter()
            learning_rate = scheduler.step()
            train_loader.dataset.set_epoch(epoch)
            model.train()
            total_loss = 0.0
            sample_count = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                spatial = batch["spatial"].to(resolved_device)
                prediction = model(
                    spatial,
                    batch["cond_static"].to(resolved_device),
                    (
                        batch["forcing_seq"].to(resolved_device)
                        if batch["forcing_seq"].numel()
                        else None
                    ),
                )
                target = batch["target"].to(resolved_device)
                loss = _training_loss(prediction, target, spatial, config)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
                optimizer.step()
                batch_count = target.shape[0]
                total_loss += float(loss.detach()) * batch_count
                sample_count += batch_count
            train_loss = total_loss / max(sample_count, 1)
            validation = evaluate_loader(
                model,
                val_loader,
                config,
                bundle,
                resolved_device,
            )
            selection = float(validation["critical_mean_peak_error_K"])
            is_best = selection < best_metric
            if is_best:
                best_metric = selection
                bad_epochs = 0
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    epoch,
                    best_metric,
                    bundle.normalization,
                )
            else:
                bad_epochs += 1
            save_checkpoint(
                latest_path,
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                best_metric,
                bundle.normalization,
            )
            row = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "val_field_rmse_K": validation["field_rmse_K"],
                "val_bond_peak_mae_K": validation["bond_peak_mae_K"],
                "val_structural_peak_mae_K": validation["structural_peak_mae_K"],
                "val_critical_mean_peak_error_K": validation["critical_mean_peak_error_K"],
                "val_critical_max_peak_error_K": validation["critical_max_peak_error_K"],
                "val_margin_error_mean_K": validation["margin_error_mean_K"],
                "val_margin_sign_disagreements": validation["margin_sign_disagreements"],
                "epoch_seconds": time.perf_counter() - epoch_start,
                "is_best": int(is_best),
            }
            writer.writerow(row)
            handle.flush()
            history.append(row)
            if bad_epochs >= config.training.patience:
                break

    best_model, best_checkpoint = load_model_checkpoint(best_path, resolved_device)
    final_validation = evaluate_loader(
        best_model,
        val_loader,
        config,
        bundle,
        resolved_device,
    )
    report = {
        "representation": representation,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_critical_mean_peak_error_K": best_metric,
        "epochs_completed": len(history),
        "wall_seconds": time.perf_counter() - start,
        "best_checkpoint": str(best_path.resolve()),
        "validation": final_validation,
    }
    (destination / "final_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report
