from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import torch

from fno_tps.config import StudyConfig
from fno_tps.data import CausalTargetDataset, DatasetBundle, create_dataloaders, load_dataset
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
) -> dict[str, Any]:
    resolved_device = torch.device(device)
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
    margin_errors: list[float] = []
    margin_sign_disagreements = 0
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
            "margin_error_K": case_margin_error,
            "margin_sign_disagreement": bool(sign_disagreement),
            "t0_max_abs_delta_T_K": t0_error,
        })

    mean_bond = float(np.mean(bond_errors)) if bond_errors else float("nan")
    mean_structural = (
        float(np.mean(structural_errors)) if structural_errors else float("nan")
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
        "critical_mean_peak_error_K": 0.5 * (mean_bond + mean_structural),
        "critical_max_peak_error_K": max(mean_bond, mean_structural),
        "margin_error_mean_K": float(np.mean(margin_errors)) if margin_errors else float("nan"),
        "margin_sign_disagreements": margin_sign_disagreements,
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
) -> dict[str, Any]:
    bundle = load_dataset(data_dir)
    model, checkpoint = load_model_checkpoint(checkpoint_path, resolve_device(device))
    if bundle.manifest["study_config_sha256"] != config.sha256:
        raise ValueError("Dataset study configuration does not match the evaluation configuration.")
    if checkpoint["study"] != config.as_dict():
        raise ValueError("Checkpoint study configuration does not match the evaluation configuration.")
    representation = model.config.representation
    loaders = create_dataloaders(config, bundle, representation)
    loader = {"train": loaders[0], "val": loaders[1], "test": loaders[2]}[split]
    report = evaluate_loader(model, loader, config, bundle, next(model.parameters()).device)
    report.update({
        "split": split,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "representation": representation,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "dataset_validity": bundle.manifest["validity"],
    })
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
