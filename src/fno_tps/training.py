from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from fno_tps.config import StudyConfig
from fno_tps.data import (
    create_dataloaders,
    dataset_matches_config,
    load_dataset,
)
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
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        (config.training.bond_loss_weight or config.training.interface_loss_weight)
        and not config.structural_interface_regions_are_constant_property()
    ):
        raise NotImplementedError(
            "Bond/interface auxiliary losses require constant-property bond "
            "and backing regions."
        )
    reduction_dims = tuple(range(1, prediction.ndim))
    per_sample_loss = F.mse_loss(
        prediction,
        target,
        reduction="none",
    ).mean(dim=reduction_dims)
    tps_end = config.mesh.nx_tps
    bond_end = tps_end + config.mesh.nx_bond
    if config.training.bond_loss_weight:
        bond_loss = F.mse_loss(
            prediction[:, tps_end:bond_end],
            target[:, tps_end:bond_end],
            reduction="none",
        ).mean(dim=reduction_dims)
        per_sample_loss = (
            per_sample_loss
            + config.training.bond_loss_weight * bond_loss
        )
    if config.training.interface_loss_weight:
        interface_prediction = _structural_interface_trace(
            prediction,
            spatial,
            config,
        )
        interface_target = _structural_interface_trace(
            target,
            spatial,
            config,
        )
        interface_dims = tuple(range(1, interface_prediction.ndim))
        interface_loss = F.mse_loss(
            interface_prediction,
            interface_target,
            reduction="none",
        ).mean(dim=interface_dims)
        per_sample_loss = (
            per_sample_loss
            + config.training.interface_loss_weight * interface_loss
        )
    if sample_weights is None:
        return per_sample_loss.mean()
    weights = sample_weights.to(
        device=per_sample_loss.device,
        dtype=per_sample_loss.dtype,
    ).reshape(-1)
    if weights.shape != per_sample_loss.shape:
        raise ValueError(
            "sample_weights must contain one value per batch item."
        )
    if not torch.isfinite(weights).all() or torch.any(weights <= 0.0):
        raise ValueError("sample_weights must be finite and positive.")
    # The thickness weights are normalized globally to mean one. Do not divide
    # by the current batch's weight sum: that would erase the intended
    # upweighting whenever a batch contains a single thickness stratum.
    return (per_sample_loss * weights).mean()


def _thickness_loss_weights(
    thickness_m: torch.Tensor,
    config: StudyConfig,
) -> torch.Tensor:
    """Return inverse-thickness weights with candidate-set mean equal to one."""
    power = config.training.thickness_loss_weight_power
    if power == 0.0:
        return torch.ones_like(thickness_m)
    candidates = torch.as_tensor(
        config.thickness_candidates,
        device=thickness_m.device,
        dtype=thickness_m.dtype,
    )
    reference = candidates.max()
    normalization = torch.mean((reference / candidates).pow(power))
    return (reference / thickness_m).pow(power) / normalization


def _thickness_loss_weighting_report(
    config: StudyConfig,
) -> dict[str, Any]:
    thicknesses = torch.tensor(
        config.thickness_candidates,
        dtype=torch.float64,
    )
    weights = _thickness_loss_weights(thicknesses, config)
    return {
        "enabled": bool(config.training.thickness_loss_weight_power > 0.0),
        "power": config.training.thickness_loss_weight_power,
        "formula": (
            "w(d) = (d_max / d)^power / "
            "mean_candidates((d_max / d_candidate)^power)"
        ),
        "normalization": (
            "arithmetic mean across configured TPS thickness candidates = 1"
        ),
        "applied_to": (
            "combined per-sample field, bond, and structural-interface "
            "training loss"
        ),
        "validation_and_checkpoint_selection_weighted": False,
        "weights_by_thickness": [
            {
                "d_tps_m": float(thickness),
                "weight": float(weight),
            }
            for thickness, weight in zip(thicknesses, weights)
        ],
        "minimum_weight": float(weights.min()),
        "maximum_weight": float(weights.max()),
        "maximum_to_minimum_ratio": float(weights.max() / weights.min()),
    }


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


def _checkpoint_selection_metrics(
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Rank training candidates on two-sided safety accuracy.

    This remains a continuous training-time signal. Deployment selection is a
    separate post-training step on a dedicated calibration cohort, where zero
    false-feasible events are the first constraint.
    """
    mean_error = float(validation["critical_mean_peak_error_K"])
    tail_error = float(validation["margin_error_p95_K"])
    optimistic_tail_error = float(
        validation["optimistic_margin_error_p95_K"]
    )
    field_error = float(validation["field_rmse_K"])
    false_feasible = int(validation["false_feasible_count"])
    feasibility_disagreements = int(
        validation["feasibility_disagreement_count"]
    )
    score = mean_error + tail_error
    selection_key = (
        score,
        field_error,
        mean_error,
    )
    return {
        "strategy": (
            "continuous(critical_mean_peak_error_K + "
            "margin_error_p95_K, field_rmse_K, "
            "critical_mean_peak_error_K); final deployment selection uses a "
            "dedicated calibration cohort and zero false-feasible events as "
            "the first constraint"
        ),
        "field_rmse_K": field_error,
        "critical_mean_peak_error_K": mean_error,
        "margin_error_p95_K": tail_error,
        "optimistic_margin_error_p95_K": optimistic_tail_error,
        "checkpoint_selection_score_K": score,
        "false_feasible_count": false_feasible,
        "false_infeasible_count": int(validation["false_infeasible_count"]),
        "feasibility_disagreement_count": feasibility_disagreements,
        "selection_key": list(selection_key),
    }


def _meaningfully_improved(current: float, best: float) -> bool:
    """Use a small scale-aware tolerance to avoid resetting on numerical noise."""
    if not np.isfinite(best):
        return True
    tolerance = max(1.0e-6, 1.0e-4 * abs(best))
    return current < best - tolerance


def _pareto_frontier(
    checkpoints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return non-dominated epochs for field, peak, and absolute-margin tail."""
    keys = (
        "field_rmse_K",
        "critical_mean_peak_error_K",
        "margin_error_p95_K",
    )
    frontier: list[dict[str, Any]] = []
    for candidate in checkpoints:
        candidate_values = tuple(float(candidate[key]) for key in keys)
        dominated = False
        for other in checkpoints:
            if other is candidate:
                continue
            other_values = tuple(float(other[key]) for key in keys)
            if (
                all(left <= right for left, right in zip(
                    other_values,
                    candidate_values,
                ))
                and any(left < right for left, right in zip(
                    other_values,
                    candidate_values,
                ))
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: int(item["epoch"]))


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
    if not dataset_matches_config(bundle, config):
        raise ValueError(
            "Dataset physics configuration does not match the training "
            "configuration."
        )
    train_loader, val_loader, _ = create_dataloaders(
        config,
        bundle,
        representation,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if len(val_loader.dataset) == 0:
        raise ValueError("Training requires a non-empty validation split.")
    model = build_model(
        config,
        representation,
        bundle.normalization,
    ).to(resolved_device)
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
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_state_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
    )
    estimated_max_archive_bytes = model_state_bytes * epoch_count
    available_storage_bytes = shutil.disk_usage(destination).free
    if available_storage_bytes < estimated_max_archive_bytes:
        warnings.warn(
            "Available disk space is below the model-only checkpoint "
            f"archive estimate for all {epoch_count} epochs "
            f"({available_storage_bytes / 2**20:.0f} MiB available versus "
            f"{estimated_max_archive_bytes / 2**20:.0f} MiB estimated). "
            "Early stopping may keep the actual archive smaller; otherwise "
            "free space or use a run directory on a larger volume.",
            RuntimeWarning,
            stacklevel=2,
        )
    metrics_path = destination / "train_metrics.csv"
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_thickness_weight_mean",
        "train_thickness_weight_min",
        "train_thickness_weight_max",
        "val_field_rmse_K",
        "val_bond_peak_mae_K",
        "val_structural_peak_mae_K",
        "val_critical_mean_peak_error_K",
        "val_critical_max_peak_error_K",
        "val_margin_error_mean_K",
        "val_margin_error_p95_K",
        "val_margin_error_max_K",
        "val_optimistic_margin_error_mean_K",
        "val_optimistic_margin_error_p95_K",
        "val_optimistic_margin_error_p99_K",
        "val_optimistic_margin_error_max_K",
        "val_margin_sign_disagreements",
        "val_false_feasible_count",
        "val_false_infeasible_count",
        "val_feasibility_disagreement_count",
        "val_checkpoint_selection_score_K",
        "continuous_early_stopping_improvement",
        "continuous_bad_epochs",
        "epoch_seconds",
        "is_best",
    ]
    best_metric = float("inf")
    best_selection_key: tuple[float, ...] | None = None
    best_selection_metrics: dict[str, Any] | None = None
    continuous_bests = {
        "field_rmse_K": float("inf"),
        "critical_mean_peak_error_K": float("inf"),
        "margin_error_p95_K": float("inf"),
    }
    candidate_specs = {
        "lowest_field_rmse": "field_rmse_K",
        "lowest_critical_peak_error": "critical_mean_peak_error_K",
        "lowest_absolute_margin_tail": "margin_error_p95_K",
        "best_continuous_composite": "checkpoint_selection_score_K",
    }
    checkpoint_candidates: dict[str, dict[str, Any]] = {}
    checkpoint_records: list[dict[str, Any]] = []
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
            total_thickness_weight = 0.0
            minimum_thickness_weight = float("inf")
            maximum_thickness_weight = 0.0
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
                thickness_weights = _thickness_loss_weights(
                    batch["tps_thickness_m"].to(resolved_device),
                    config,
                )
                loss = _training_loss(
                    prediction,
                    target,
                    spatial,
                    config,
                    thickness_weights,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
                optimizer.step()
                batch_count = target.shape[0]
                total_loss += float(loss.detach()) * batch_count
                sample_count += batch_count
                total_thickness_weight += float(
                    thickness_weights.detach().sum()
                )
                minimum_thickness_weight = min(
                    minimum_thickness_weight,
                    float(thickness_weights.detach().min()),
                )
                maximum_thickness_weight = max(
                    maximum_thickness_weight,
                    float(thickness_weights.detach().max()),
                )
            train_loss = total_loss / max(sample_count, 1)
            mean_thickness_weight = (
                total_thickness_weight / max(sample_count, 1)
            )
            validation = evaluate_loader(
                model,
                val_loader,
                config,
                bundle,
                resolved_device,
            )
            selection_metrics = _checkpoint_selection_metrics(validation)
            selection_key = tuple(selection_metrics["selection_key"])
            is_best = (
                best_selection_key is None
                or selection_key < best_selection_key
            )
            if is_best:
                best_metric = float(
                    selection_metrics["checkpoint_selection_score_K"]
                )
                best_selection_key = selection_key
                best_selection_metrics = selection_metrics
            improved_continuous_metrics = []
            for key in continuous_bests:
                current = float(validation[key])
                if _meaningfully_improved(current, continuous_bests[key]):
                    continuous_bests[key] = current
                    improved_continuous_metrics.append(key)
            if improved_continuous_metrics:
                bad_epochs = 0
            else:
                bad_epochs += 1
            epoch_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            candidate_values = {
                "field_rmse_K": float(validation["field_rmse_K"]),
                "critical_mean_peak_error_K": float(
                    validation["critical_mean_peak_error_K"]
                ),
                "margin_error_p95_K": float(
                    validation["margin_error_p95_K"]
                ),
                "optimistic_margin_error_p95_K": float(
                    validation["optimistic_margin_error_p95_K"]
                ),
                "checkpoint_selection_score_K": float(
                    selection_metrics["checkpoint_selection_score_K"]
                ),
            }
            improved_candidate_roles: list[str] = []
            for role, metric_name in candidate_specs.items():
                current = candidate_values[metric_name]
                previous = checkpoint_candidates.get(role)
                if previous is None or current < float(previous["value"]):
                    checkpoint_candidates[role] = {
                        "role": role,
                        "metric": metric_name,
                        "value": current,
                        "epoch": epoch,
                        "checkpoint": str(epoch_path.resolve()),
                    }
                    improved_candidate_roles.append(role)
            save_checkpoint(
                epoch_path,
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                best_metric,
                bundle.normalization,
                {
                    **selection_metrics,
                    "is_best": bool(is_best),
                    "best_selection_key": list(best_selection_key),
                    "improved_candidate_roles": improved_candidate_roles,
                    "continuous_early_stopping_improvements": (
                        improved_continuous_metrics
                    ),
                    "continuous_bad_epochs": bad_epochs,
                },
                include_training_state=False,
            )
            latest_selection_metrics = {
                **selection_metrics,
                "is_best": bool(is_best),
                "best_selection_key": list(best_selection_key),
                "improved_candidate_roles": improved_candidate_roles,
                "continuous_early_stopping_improvements": (
                    improved_continuous_metrics
                ),
                "continuous_bad_epochs": bad_epochs,
            }
            save_checkpoint(
                latest_path,
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                best_metric,
                bundle.normalization,
                latest_selection_metrics,
            )
            if is_best:
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    epoch,
                    best_metric,
                    bundle.normalization,
                    latest_selection_metrics,
                )
            checkpoint_record = {
                "epoch": epoch,
                "checkpoint": str(epoch_path.resolve()),
                **candidate_values,
            }
            checkpoint_records.append(checkpoint_record)
            row = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "train_thickness_weight_mean": mean_thickness_weight,
                "train_thickness_weight_min": minimum_thickness_weight,
                "train_thickness_weight_max": maximum_thickness_weight,
                "val_field_rmse_K": validation["field_rmse_K"],
                "val_bond_peak_mae_K": validation["bond_peak_mae_K"],
                "val_structural_peak_mae_K": validation["structural_peak_mae_K"],
                "val_critical_mean_peak_error_K": validation["critical_mean_peak_error_K"],
                "val_critical_max_peak_error_K": validation["critical_max_peak_error_K"],
                "val_margin_error_mean_K": validation["margin_error_mean_K"],
                "val_margin_error_p95_K": validation["margin_error_p95_K"],
                "val_margin_error_max_K": validation["margin_error_max_K"],
                "val_optimistic_margin_error_mean_K": validation[
                    "optimistic_margin_error_mean_K"
                ],
                "val_optimistic_margin_error_p95_K": validation[
                    "optimistic_margin_error_p95_K"
                ],
                "val_optimistic_margin_error_p99_K": validation[
                    "optimistic_margin_error_p99_K"
                ],
                "val_optimistic_margin_error_max_K": validation[
                    "optimistic_margin_error_max_K"
                ],
                "val_margin_sign_disagreements": validation["margin_sign_disagreements"],
                "val_false_feasible_count": validation["false_feasible_count"],
                "val_false_infeasible_count": validation["false_infeasible_count"],
                "val_feasibility_disagreement_count": validation[
                    "feasibility_disagreement_count"
                ],
                "val_checkpoint_selection_score_K": selection_metrics[
                    "checkpoint_selection_score_K"
                ],
                "continuous_early_stopping_improvement": int(
                    bool(improved_continuous_metrics)
                ),
                "continuous_bad_epochs": bad_epochs,
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
        "initial_condition": {
            "hard_enforced": model.config.hard_initial_condition,
            "normalized_zero_delta_temperature": (
                model.config.initial_delta_temperature_norm
            ),
            "gate": "sine_saturating",
            "gate_tau_normalized": model.config.initial_condition_gate_tau,
            "gate_time_seconds": (
                model.config.initial_condition_gate_tau
                * config.time.t_final
            ),
            "t0_in_training_targets": False,
            "t0_retained_in_validation": True,
        },
        "tps_thickness_loss_weighting": (
            _thickness_loss_weighting_report(config)
        ),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_critical_mean_peak_error_K": final_validation[
            "critical_mean_peak_error_K"
        ],
        "best_continuous_composite_K": best_metric,
        "epochs_completed": len(history),
        "wall_seconds": time.perf_counter() - start,
        "best_checkpoint": str(best_path.resolve()),
        "promotion_status": (
            "training candidate only; calibrate a safety buffer and evaluate "
            "buffered false-feasible performance before promotion"
        ),
        "checkpoint_selection": (
            best_checkpoint.get("selection_metrics")
            or best_selection_metrics
            or _checkpoint_selection_metrics(final_validation)
        ),
        "checkpoint_candidates": {
            **checkpoint_candidates,
            "latest": {
                "role": "latest",
                "metric": "epoch",
                "value": int(history[-1]["epoch"]),
                "epoch": int(history[-1]["epoch"]),
                "checkpoint": str(latest_path.resolve()),
            },
        },
        "pareto_frontier": _pareto_frontier(checkpoint_records),
        "checkpoint_archive": {
            "directory": str(checkpoint_dir.resolve()),
            "saved_every_evaluated_epoch": True,
            "checkpoint_count": len(checkpoint_records),
            "model_only": True,
            "full_precision": True,
            "resume_capable": False,
            "model_state_bytes_per_epoch": model_state_bytes,
            "estimated_max_archive_bytes": estimated_max_archive_bytes,
            "available_storage_bytes_at_start": available_storage_bytes,
            "latest_and_best_are_separate_resume_capable_checkpoints": True,
        },
        "early_stopping": {
            "strategy": (
                "stop after configured patience when field RMSE, critical "
                "mean peak error, and absolute-margin P95 all fail to "
                "improve meaningfully"
            ),
            "patience": config.training.patience,
            "meaningful_improvement_relative_tolerance": 1.0e-4,
            "meaningful_improvement_absolute_tolerance_K": 1.0e-6,
            "continuous_metric_bests": continuous_bests,
            "final_bad_epochs": bad_epochs,
            "stopped_early": len(history) < epoch_count,
        },
        "validation": final_validation,
    }
    (destination / "final_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report
