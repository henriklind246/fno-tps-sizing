from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import json

import numpy as np
import torch

from fno_tps.config import StudyConfig, inference_compatibility_sha256
from fno_tps.data import (
    DatasetBundle,
    create_dataloaders,
    dataset_matches_config,
    load_dataset,
)
from fno_tps.physics import TPSFVSolver
from fno_tps.runtime import load_model_checkpoint, resolve_device


def _model_forward(model, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    forcing = batch["forcing_seq"].to(device)
    if forcing.numel() == 0:
        forcing = None
    return model(
        batch["spatial"].to(device),
        batch["cond_static"].to(device),
        forcing,
    )


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    config: StudyConfig,
    bundle: DatasetBundle,
    device: str | torch.device = "cpu",
    *,
    safety_buffer_K: float = 0.0,
    near_boundary_limit_K: float = 10.0,
) -> dict[str, Any]:
    resolved_device = torch.device(device)
    safety_buffer_K = float(safety_buffer_K)
    near_boundary_limit_K = float(near_boundary_limit_K)
    if not np.isfinite(safety_buffer_K) or safety_buffer_K < 0.0:
        raise ValueError("safety_buffer_K must be finite and nonnegative.")
    if not np.isfinite(near_boundary_limit_K) or near_boundary_limit_K < 0.0:
        raise ValueError("near_boundary_limit_K must be finite and nonnegative.")
    model.eval()
    mean = bundle.normalization["delta_t_mean"]
    std = bundle.normalization["delta_t_std"]
    tps_end = config.mesh.nx_tps
    bond_end = tps_end + config.mesh.nx_bond
    back_start = bond_end

    total_squared = 0.0
    total_count = 0
    region_squared = {"tps": 0.0, "bond": 0.0, "backing": 0.0}
    region_count = {"tps": 0, "bond": 0, "backing": 0}
    prediction_by_case: dict[int, np.ndarray] = {}
    target_by_case: dict[int, np.ndarray] = {}
    for batch in loader:
        output = _model_forward(model, batch, resolved_device)
        prediction = output.detach().cpu().numpy()[..., 0] * std + mean
        target = batch["target"].numpy()[..., 0] * std + mean
        error = prediction - target
        total_squared += float(np.sum(error**2))
        total_count += error.size
        for name, region_slice in (
            ("tps", slice(0, tps_end)),
            ("bond", slice(tps_end, bond_end)),
            ("backing", slice(back_start, config.mesh.nx)),
        ):
            values = error[:, region_slice, :]
            region_squared[name] += float(np.sum(values**2))
            region_count[name] += values.size
        for row, case_id_tensor in enumerate(batch["case_index"]):
            case_id = int(case_id_tensor)
            target_index = int(batch["target_index"][row])
            if case_id not in prediction_by_case:
                shape = (len(bundle.times), config.mesh.nx, config.mesh.ny)
                prediction_by_case[case_id] = np.full(shape, np.nan, dtype=np.float64)
                target_by_case[case_id] = np.full(shape, np.nan, dtype=np.float64)
            prediction_by_case[case_id][target_index] = prediction[row]
            target_by_case[case_id][target_index] = target[row]

    records: list[dict[str, Any]] = []
    bond_errors: list[float] = []
    structural_errors: list[float] = []
    critical_peak_errors: list[float] = []
    margin_errors: list[float] = []
    margin_sign_disagreements = 0
    false_feasible_count = 0
    false_infeasible_count = 0
    feasibility_disagreement_count = 0
    true_distances_to_limit: list[float] = []
    predicted_distances_to_limit: list[float] = []
    optimistic_margin_errors: list[float] = []
    near_boundary_optimistic_margin_errors: list[float] = []
    buffered_false_feasible_count = 0
    buffered_false_infeasible_count = 0
    safety_abstention_count = 0
    t0_max_errors: list[float] = []
    for case_id in sorted(prediction_by_case):
        prediction_delta = prediction_by_case[case_id]
        target_delta = target_by_case[case_id]
        if np.isnan(prediction_delta).any():
            raise ValueError(
                "Critical QoI evaluation requires every saved target time for each case."
            )
        case = bundle.cases[case_id]
        solver = TPSFVSolver(config, case)
        prediction_qoi = solver.quantities_of_interest(
            config.initial_temperature + prediction_delta,
            bundle.times,
        )
        target_qoi = solver.quantities_of_interest(
            config.initial_temperature + target_delta,
            bundle.times,
        )
        bond_error = abs(prediction_qoi["bond_max"] - target_qoi["bond_max"])
        structural_error = abs(
            prediction_qoi["structural_interface_max"]
            - target_qoi["structural_interface_max"]
        )
        bond_errors.append(bond_error)
        structural_errors.append(structural_error)
        critical_peak_errors.append(max(bond_error, structural_error))
        t0_error = float(np.max(np.abs(prediction_delta[0])))
        t0_max_errors.append(t0_error)

        bond_pred_margin = (
            config.bond_temperature_limit - prediction_qoi["bond_max"]
        )
        bond_true_margin = config.bond_temperature_limit - target_qoi["bond_max"]
        struct_pred_margin = (
            config.structural_temperature_limit
            - prediction_qoi["structural_interface_max"]
        )
        struct_true_margin = (
            config.structural_temperature_limit
            - target_qoi["structural_interface_max"]
        )
        case_margin_error = max(
            abs(bond_pred_margin - bond_true_margin),
            abs(struct_pred_margin - struct_true_margin),
        )
        margin_errors.append(case_margin_error)
        sign_disagreement = (
            (bond_pred_margin >= 0.0) != (bond_true_margin >= 0.0)
            or (struct_pred_margin >= 0.0) != (struct_true_margin >= 0.0)
        )
        margin_sign_disagreements += int(sign_disagreement)
        predicted_feasible = (
            bond_pred_margin >= 0.0 and struct_pred_margin >= 0.0
        )
        true_feasible = (
            bond_true_margin >= 0.0 and struct_true_margin >= 0.0
        )
        false_feasible = predicted_feasible and not true_feasible
        false_infeasible = not predicted_feasible and true_feasible
        feasibility_disagreement = predicted_feasible != true_feasible
        false_feasible_count += int(false_feasible)
        false_infeasible_count += int(false_infeasible)
        feasibility_disagreement_count += int(feasibility_disagreement)
        true_minimum_margin = min(bond_true_margin, struct_true_margin)
        predicted_minimum_margin = min(bond_pred_margin, struct_pred_margin)
        optimistic_margin_error = max(
            0.0,
            predicted_minimum_margin - true_minimum_margin,
        )
        optimistic_margin_errors.append(optimistic_margin_error)
        if abs(true_minimum_margin) <= near_boundary_limit_K:
            near_boundary_optimistic_margin_errors.append(
                optimistic_margin_error
            )
        if predicted_minimum_margin < 0.0:
            buffered_classification = "surrogate_infeasible"
        elif (
            bond_pred_margin > safety_buffer_K
            and struct_pred_margin > safety_buffer_K
        ):
            buffered_classification = "surrogate_feasible"
        else:
            buffered_classification = "near_boundary_fv_required"
        buffered_false_feasible = (
            buffered_classification == "surrogate_feasible"
            and not true_feasible
        )
        buffered_false_infeasible = (
            buffered_classification == "surrogate_infeasible"
            and true_feasible
        )
        safety_abstention = (
            buffered_classification == "near_boundary_fv_required"
        )
        buffered_false_feasible_count += int(buffered_false_feasible)
        buffered_false_infeasible_count += int(buffered_false_infeasible)
        safety_abstention_count += int(safety_abstention)
        true_distance_to_limit = min(
            abs(bond_true_margin),
            abs(struct_true_margin),
        )
        predicted_distance_to_limit = min(
            abs(bond_pred_margin),
            abs(struct_pred_margin),
        )
        true_distances_to_limit.append(true_distance_to_limit)
        predicted_distances_to_limit.append(predicted_distance_to_limit)
        records.append({
            "case_index": case_id,
            "case_id": case.case_id,
            "d_tps": case.d_tps,
            "event_count": case.event_count,
            "defect_count": case.defect_count,
            "bond_peak_error_K": bond_error,
            "structural_peak_error_K": structural_error,
            "bond_peak_time_error": abs(
                prediction_qoi["bond_peak_time"] - target_qoi["bond_peak_time"]
            ),
            "structural_peak_time_error": abs(
                prediction_qoi["structural_interface_peak_time"]
                - target_qoi["structural_interface_peak_time"]
            ),
            "bond_peak_y_error": abs(
                prediction_qoi["bond_peak_y"] - target_qoi["bond_peak_y"]
            ),
            "structural_peak_y_error": abs(
                prediction_qoi["structural_interface_peak_y"]
                - target_qoi["structural_interface_peak_y"]
            ),
            "true_bond_margin_K": bond_true_margin,
            "predicted_bond_margin_K": bond_pred_margin,
            "true_structural_margin_K": struct_true_margin,
            "predicted_structural_margin_K": struct_pred_margin,
            "true_minimum_margin_K": true_minimum_margin,
            "predicted_minimum_margin_K": predicted_minimum_margin,
            "optimistic_margin_error_K": optimistic_margin_error,
            "true_distance_to_limit_K": true_distance_to_limit,
            "predicted_distance_to_limit_K": predicted_distance_to_limit,
            "true_feasible": bool(true_feasible),
            "predicted_feasible": bool(predicted_feasible),
            "false_feasible": bool(false_feasible),
            "false_infeasible": bool(false_infeasible),
            "feasibility_disagreement": bool(feasibility_disagreement),
            "buffered_classification": buffered_classification,
            "buffered_false_feasible": bool(buffered_false_feasible),
            "buffered_false_infeasible": bool(buffered_false_infeasible),
            "safety_abstention": bool(safety_abstention),
            "margin_error_K": case_margin_error,
            "margin_sign_disagreement": bool(sign_disagreement),
            "t0_max_abs_delta_T_K": t0_error,
        })

    mean_bond = float(np.mean(bond_errors)) if bond_errors else float("nan")
    mean_structural = (
        float(np.mean(structural_errors)) if structural_errors else float("nan")
    )

    def percentile(values: list[float], quantile: float) -> float:
        return (
            float(np.percentile(values, quantile))
            if values
            else float("nan")
        )

    critical_mean = 0.5 * (mean_bond + mean_structural)
    margin_error_p95 = percentile(margin_errors, 95.0)
    optimistic_margin_error_p95 = percentile(
        optimistic_margin_errors,
        95.0,
    )
    return {
        "field_rmse_K": float(np.sqrt(total_squared / max(total_count, 1))),
        "tps_rmse_K": float(np.sqrt(region_squared["tps"] / max(region_count["tps"], 1))),
        "bond_rmse_K": float(np.sqrt(region_squared["bond"] / max(region_count["bond"], 1))),
        "backing_rmse_K": float(
            np.sqrt(region_squared["backing"] / max(region_count["backing"], 1))
        ),
        "bond_peak_mae_K": mean_bond,
        "structural_peak_mae_K": mean_structural,
        "critical_mean_peak_error_K": critical_mean,
        "critical_max_peak_error_K": max(mean_bond, mean_structural),
        "critical_peak_error_p90_K": percentile(critical_peak_errors, 90.0),
        "critical_peak_error_p95_K": percentile(critical_peak_errors, 95.0),
        "critical_peak_error_max_K": (
            max(critical_peak_errors) if critical_peak_errors else float("nan")
        ),
        "margin_error_mean_K": float(np.mean(margin_errors)) if margin_errors else float("nan"),
        "margin_error_p90_K": percentile(margin_errors, 90.0),
        "margin_error_p95_K": margin_error_p95,
        "margin_error_max_K": max(margin_errors) if margin_errors else float("nan"),
        "optimistic_margin_error_mean_K": (
            float(np.mean(optimistic_margin_errors))
            if optimistic_margin_errors
            else float("nan")
        ),
        "optimistic_margin_error_p90_K": percentile(
            optimistic_margin_errors,
            90.0,
        ),
        "optimistic_margin_error_p95_K": optimistic_margin_error_p95,
        "optimistic_margin_error_p99_K": percentile(
            optimistic_margin_errors,
            99.0,
        ),
        "optimistic_margin_error_max_K": (
            max(optimistic_margin_errors)
            if optimistic_margin_errors
            else float("nan")
        ),
        "near_boundary_limit_K": near_boundary_limit_K,
        "near_boundary_case_count": len(
            near_boundary_optimistic_margin_errors
        ),
        "near_boundary_optimistic_margin_error_mean_K": (
            float(np.mean(near_boundary_optimistic_margin_errors))
            if near_boundary_optimistic_margin_errors
            else float("nan")
        ),
        "near_boundary_optimistic_margin_error_p95_K": percentile(
            near_boundary_optimistic_margin_errors,
            95.0,
        ),
        "near_boundary_optimistic_margin_error_p99_K": percentile(
            near_boundary_optimistic_margin_errors,
            99.0,
        ),
        "near_boundary_optimistic_margin_error_max_K": (
            max(near_boundary_optimistic_margin_errors)
            if near_boundary_optimistic_margin_errors
            else float("nan")
        ),
        "margin_sign_disagreements": margin_sign_disagreements,
        "false_feasible_count": false_feasible_count,
        "false_infeasible_count": false_infeasible_count,
        "feasibility_disagreement_count": feasibility_disagreement_count,
        "safety_buffer_K": safety_buffer_K,
        "buffered_false_feasible_count": buffered_false_feasible_count,
        "buffered_false_infeasible_count": buffered_false_infeasible_count,
        "safety_abstention_count": safety_abstention_count,
        "true_distance_to_limit_mean_K": (
            float(np.mean(true_distances_to_limit))
            if true_distances_to_limit
            else float("nan")
        ),
        "true_distance_to_limit_min_K": (
            min(true_distances_to_limit)
            if true_distances_to_limit
            else float("nan")
        ),
        "predicted_distance_to_limit_mean_K": (
            float(np.mean(predicted_distances_to_limit))
            if predicted_distances_to_limit
            else float("nan")
        ),
        "predicted_distance_to_limit_min_K": (
            min(predicted_distances_to_limit)
            if predicted_distances_to_limit
            else float("nan")
        ),
        "checkpoint_selection_score_K": (
            critical_mean + optimistic_margin_error_p95
        ),
        "t0_max_abs_delta_T_K": max(t0_max_errors) if t0_max_errors else float("nan"),
        "case_count": len(records),
        "records": records,
    }


def evaluate_checkpoint(
    config: StudyConfig,
    data_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    *,
    split: str = "test",
    device: str = "auto",
    safety_buffer_K: float | None = None,
    calibration_split: str | None = None,
    calibration_quantile: float = 99.0,
    near_boundary_limit_K: float = 10.0,
) -> dict[str, Any]:
    if safety_buffer_K is not None and calibration_split is not None:
        raise ValueError(
            "Pass either safety_buffer_K or calibration_split, not both."
        )
    if not 0.0 <= calibration_quantile <= 100.0:
        raise ValueError("calibration_quantile must lie in [0, 100].")
    bundle = load_dataset(data_dir)
    model, checkpoint = load_model_checkpoint(checkpoint_path, resolve_device(device))
    if not dataset_matches_config(bundle, config):
        raise ValueError(
            "Dataset physics configuration does not match the evaluation "
            "configuration."
        )
    if (
        inference_compatibility_sha256(checkpoint["study"])
        != config.inference_compatibility_sha256
    ):
        raise ValueError(
            "Checkpoint inference configuration does not match the evaluation "
            "configuration."
        )
    if "normalization" not in checkpoint:
        raise ValueError("Checkpoint does not contain its training normalization.")
    dataset_normalization = dict(bundle.normalization)
    evaluation_normalization = {
        key: float(value)
        for key, value in checkpoint["normalization"].items()
    }
    if split not in ("train", "val", "test", "all"):
        raise ValueError(f"Unknown evaluation split {split!r}.")
    evaluation_bundle = replace(
        bundle,
        normalization=evaluation_normalization,
    )
    representation = model.config.representation
    loaders = create_dataloaders(config, evaluation_bundle, representation)
    loader_by_split = {
        "train": loaders[0],
        "val": loaders[1],
        "test": loaders[2],
    }
    if split in ("train", "all"):
        target_splits = dict(evaluation_bundle.splits)
        target_splits["test"] = (
            np.asarray(evaluation_bundle.splits["train"], dtype=np.int64)
            if split == "train"
            else np.arange(len(bundle.cases), dtype=np.int64)
        )
        target_bundle = replace(
            evaluation_bundle,
            splits=target_splits,
        )
        loader = create_dataloaders(
            config,
            target_bundle,
            representation,
        )[2]
    else:
        loader = loader_by_split[split]
    calibration: dict[str, Any] | None = None
    if calibration_split is not None:
        if calibration_split not in ("val", "test"):
            raise ValueError(
                "calibration_split must be val or test so every saved time "
                "is evaluated."
            )
        calibration_report = evaluate_loader(
            model,
            loader_by_split[calibration_split],
            config,
            evaluation_bundle,
            next(model.parameters()).device,
            near_boundary_limit_K=near_boundary_limit_K,
        )
        calibration_errors = [
            float(record["optimistic_margin_error_K"])
            for record in calibration_report["records"]
        ]
        if not calibration_errors:
            raise ValueError(
                f"Calibration split {calibration_split!r} contains no cases."
            )
        applied_safety_buffer_K = float(
            np.percentile(calibration_errors, calibration_quantile)
        )
        calibration = {
            "split": calibration_split,
            "quantile": float(calibration_quantile),
            "case_count": len(calibration_errors),
            "safety_buffer_K": applied_safety_buffer_K,
            "optimistic_margin_error_mean_K": calibration_report[
                "optimistic_margin_error_mean_K"
            ],
            "optimistic_margin_error_p95_K": calibration_report[
                "optimistic_margin_error_p95_K"
            ],
            "optimistic_margin_error_p99_K": calibration_report[
                "optimistic_margin_error_p99_K"
            ],
            "optimistic_margin_error_max_K": calibration_report[
                "optimistic_margin_error_max_K"
            ],
            "overlaps_evaluation_split": (
                split == calibration_split or split == "all"
            ),
        }
    else:
        applied_safety_buffer_K = (
            0.0 if safety_buffer_K is None else float(safety_buffer_K)
        )
    report = evaluate_loader(
        model,
        loader,
        config,
        evaluation_bundle,
        next(model.parameters()).device,
        safety_buffer_K=applied_safety_buffer_K,
        near_boundary_limit_K=near_boundary_limit_K,
    )
    report.update({
        "split": split,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "representation": representation,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "normalization_source": "checkpoint",
        "checkpoint_normalization": evaluation_normalization,
        "dataset_normalization": dataset_normalization,
        "dataset_validity": bundle.manifest["validity"],
        "safety_buffer_source": (
            "calibration_split"
            if calibration is not None
            else (
                "explicit"
                if safety_buffer_K is not None
                else "unbuffered_zero"
            )
        ),
        "safety_calibration": calibration,
    })
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
