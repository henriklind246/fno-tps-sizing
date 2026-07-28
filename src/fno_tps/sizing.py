from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any
import csv
import json
import time
from itertools import combinations

import numpy as np
import torch
import yaml

from fno_tps.acceptance import assess_trajectory
from fno_tps.config import StudyConfig
from fno_tps.data import TPSInputBuilder
from fno_tps.physics import TPSFVSolver
from fno_tps.problem import BondDefect, HeatingEvent, SimulationCase
from fno_tps.runtime import load_model_checkpoint, resolve_device


def load_scenarios(
    path: str | Path,
) -> tuple[str, bool, list[dict[str, Any]], str]:
    source = Path(path)
    raw = source.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), list):
        raise ValueError("Scenario YAML must contain a scenarios list.")
    scenarios: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in payload["scenarios"]:
        scenario_id = str(value["scenario_id"])
        if scenario_id in seen:
            raise ValueError(f"Duplicate scenario_id {scenario_id!r}.")
        seen.add(scenario_id)
        events = tuple(HeatingEvent.from_dict(item) for item in value["heating_events"])
        defects = tuple(BondDefect.from_dict(item) for item in value.get("bond_defects", []))
        if not 1 <= len(events) <= 3 or len(defects) > 2:
            raise ValueError(f"Scenario {scenario_id!r} violates event/defect counts.")
        scenarios.append({
            "scenario_id": scenario_id,
            "heating_events": events,
            "bond_defects": defects,
        })
    return (
        str(payload["scenario_set_id"]),
        bool(payload.get("authoritative", False)),
        scenarios,
        sha256(raw).hexdigest(),
    )


def predict_case_delta(
    model,
    config: StudyConfig,
    case: SimulationCase,
    solver: TPSFVSolver,
    times: np.ndarray,
    normalization: dict[str, float],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    """Surrogate temperature rise over `times`, shaped (Nt, Nx, Ny), and its wall time."""
    builder = TPSInputBuilder(config, normalization)
    items = [
        builder.build(
            model.config.representation,
            case,
            solver.grid.x_centers,
            solver.grid.dx,
            solver.grid.y_centers,
            solver.bond_conductivity,
            float(target_time),
        )
        for target_time in times
    ]
    predictions = []
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(items), batch_size):
            batch = items[offset:offset + batch_size]
            spatial = torch.from_numpy(np.stack([item["spatial"] for item in batch])).to(device)
            conditioning = torch.from_numpy(
                np.stack([item["cond_static"] for item in batch])
            ).to(device)
            if model.config.representation == "summary":
                forcing = None
            else:
                forcing = torch.from_numpy(
                    np.stack([item["forcing_seq"] for item in batch])
                ).to(device)
            output = model(spatial, conditioning, forcing)
            predictions.append(output.cpu().numpy()[..., 0])
    elapsed = time.perf_counter() - start
    normalized = np.concatenate(predictions, axis=0)
    delta = (
        normalized * normalization["delta_t_std"]
        + normalization["delta_t_mean"]
    )
    return delta, elapsed


def _selected_thickness(
    rows: list[dict[str, Any]],
    candidates: tuple[float, ...],
    key: str,
) -> float | None:
    for thickness in candidates:
        subset = [row for row in rows if row["d_tps"] == thickness]
        if subset and all(bool(row[key]) for row in subset):
            return thickness
    return None


def _field_errors(
    prediction: np.ndarray,
    reference: np.ndarray,
    config: StudyConfig,
) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(reference)
    tps_end = config.mesh.nx_tps
    bond_end = tps_end + config.mesh.nx_bond

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(values**2)))

    reference_norm = float(np.linalg.norm(reference.ravel()))
    return {
        "field_rmse_K": rmse(error),
        "field_relative_l2": float(
            np.linalg.norm(error.ravel()) / max(reference_norm, 1e-12)
        ),
        "tps_rmse_K": rmse(error[:, :tps_end, :]),
        "bond_rmse_K": rmse(error[:, tps_end:bond_end, :]),
        "backing_rmse_K": rmse(error[:, bond_end:, :]),
    }


def _ranking_accuracy(
    rows: list[dict[str, Any]],
    candidates: tuple[float, ...],
) -> tuple[float | None, int]:
    correct = 0
    total = 0
    for thickness in candidates:
        subset = [row for row in rows if row["d_tps"] == thickness]
        for left, right in combinations(subset, 2):
            predicted_difference = (
                left["surrogate_criticality_ratio"]
                - right["surrogate_criticality_ratio"]
            )
            fv_difference = left["fv_criticality_ratio"] - right["fv_criticality_ratio"]
            correct += int(
                int(np.sign(predicted_difference)) == int(np.sign(fv_difference))
            )
            total += 1
    return (correct / total if total else None), total


def _selection_neighborhood(
    selected: float | None,
    candidates: tuple[float, ...],
) -> dict[str, float | None]:
    if selected is None:
        return {"selected": None, "thinner": None, "thicker": None}
    index = candidates.index(selected)
    return {
        "selected": selected,
        "thinner": candidates[index - 1] if index > 0 else None,
        "thicker": candidates[index + 1] if index + 1 < len(candidates) else None,
    }


def _limiting_scenarios(
    rows: list[dict[str, Any]],
    selected: float | None,
    prefix: str,
) -> dict[str, str | None]:
    if selected is None:
        return {"critical": None, "bond": None, "structural": None}
    subset = [row for row in rows if row["d_tps"] == selected]
    return {
        "critical": max(
            subset,
            key=lambda row: row[f"{prefix}_criticality_ratio"],
        )["scenario_id"],
        "bond": max(
            subset,
            key=lambda row: row[f"{prefix}_bond_max"],
        )["scenario_id"],
        "structural": max(
            subset,
            key=lambda row: row[f"{prefix}_structural_max"],
        )["scenario_id"],
    }


def run_full_sizing(
    config: StudyConfig,
    scenario_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    allow_demo: bool = False,
    device: str = "auto",
    batch_size: int = 64,
    safety_buffer_K: float = 0.0,
) -> dict[str, Any]:
    config.require_authoritative(allow_demo=allow_demo)
    safety_buffer_K = float(safety_buffer_K)
    if not np.isfinite(safety_buffer_K) or safety_buffer_K < 0.0:
        raise ValueError("safety_buffer_K must be finite and nonnegative.")
    scenario_set_id, scenario_authoritative, scenarios, scenario_hash = load_scenarios(
        scenario_path
    )
    if not scenario_authoritative and not allow_demo:
        raise ValueError("Scenario set is non-production; pass allow_demo=True explicitly.")
    resolved_device = resolve_device(device)
    model, checkpoint = load_model_checkpoint(checkpoint_path, resolved_device)
    if checkpoint["study"] != config.as_dict():
        raise ValueError("Checkpoint study configuration does not match the sizing configuration.")
    normalization = {
        key: float(value) for key, value in checkpoint["normalization"].items()
    }
    times = np.asarray(config.time.saved_times(), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    total_fv_seconds = 0.0
    total_surrogate_seconds = 0.0

    for thickness in config.thickness_candidates:
        for scenario in scenarios:
            case = SimulationCase(
                case_id=f"{scenario['scenario_id']}@{thickness:g}",
                d_tps=thickness,
                heating_events=scenario["heating_events"],
                bond_defects=scenario["bond_defects"],
            )
            solver_start = time.perf_counter()
            solver = TPSFVSolver(config, case)
            trajectory = solver.solve(save_times=times)
            fv_seconds = time.perf_counter() - solver_start
            fv_hot_face_max = trajectory.maximum_hot_face_temperature
            prediction_delta, surrogate_seconds = predict_case_delta(
                model,
                config,
                case,
                solver,
                times,
                normalization,
                resolved_device,
                batch_size,
            )
            total_fv_seconds += fv_seconds
            total_surrogate_seconds += surrogate_seconds
            fv_qoi = solver.quantities_of_interest(trajectory.temperatures, times)
            fv_acceptance = assess_trajectory(
                config,
                trajectory,
                bond_max=fv_qoi["bond_max"],
                structural_interface_max=fv_qoi["structural_interface_max"],
            )
            predicted_temperature = config.initial_temperature + prediction_delta
            incident_flux = case.heat_flux(
                solver.grid.y_centers,
                times,
            ).T
            predicted_surface = solver.surface_temperature(
                predicted_temperature[:, 0, :],
                incident_flux,
            )
            surrogate_hot_face_max = float(
                np.max(predicted_surface)
            )
            predicted_qoi = solver.quantities_of_interest(predicted_temperature, times)
            field_errors = _field_errors(
                prediction_delta,
                trajectory.temperatures - config.initial_temperature,
                config,
            )

            predicted_bond_margin = (
                config.bond_temperature_limit - predicted_qoi["bond_max"]
            )
            fv_bond_margin = config.bond_temperature_limit - fv_qoi["bond_max"]
            predicted_structural_margin = (
                config.structural_temperature_limit
                - predicted_qoi["structural_interface_max"]
            )
            fv_structural_margin = (
                config.structural_temperature_limit
                - fv_qoi["structural_interface_max"]
            )
            surrogate_raw_feasible = (
                predicted_bond_margin >= 0.0 and predicted_structural_margin >= 0.0
            )
            if min(
                predicted_bond_margin,
                predicted_structural_margin,
            ) < 0.0:
                buffered_design_classification = "infeasible"
            elif (
                predicted_bond_margin > safety_buffer_K
                and predicted_structural_margin > safety_buffer_K
            ):
                buffered_design_classification = "feasible"
            else:
                buffered_design_classification = (
                    "near_boundary_fv_required"
                )
            surrogate_valid = bool(
                surrogate_hot_face_max
                <= config.hot_face_temperature_limit + 1.0e-8
                and float(np.min(predicted_temperature))
                > config.validity.minimum_physical_temperature
            )
            surrogate_feasible = (
                surrogate_valid
                and buffered_design_classification == "feasible"
            )
            fv_feasible = bool(fv_acceptance.feasible)
            bond_peak_error = abs(
                predicted_qoi["bond_max"] - fv_qoi["bond_max"]
            )
            structural_peak_error = abs(
                predicted_qoi["structural_interface_max"]
                - fv_qoi["structural_interface_max"]
            )
            row = {
                "scenario_id": scenario["scenario_id"],
                "d_tps": thickness,
                "fv_hot_face_max": fv_hot_face_max,
                "surrogate_hot_face_max": surrogate_hot_face_max,
                "fv_within_validity": fv_acceptance.valid,
                "fv_classification": fv_acceptance.classification,
                "fv_invalid_reasons": ";".join(
                    fv_acceptance.invalid_reasons
                ),
                "fv_design_violations": ";".join(
                    fv_acceptance.design_violations
                ),
                "surrogate_within_validity": surrogate_valid,
                "surrogate_classification": (
                    "invalid"
                    if not surrogate_valid
                    else buffered_design_classification
                ),
                "surrogate_bond_max": predicted_qoi["bond_max"],
                "fv_bond_max": fv_qoi["bond_max"],
                "bond_peak_error_K": bond_peak_error,
                "surrogate_structural_max": predicted_qoi["structural_interface_max"],
                "fv_structural_max": fv_qoi["structural_interface_max"],
                "structural_peak_error_K": structural_peak_error,
                "critical_mean_peak_error_K": 0.5 * (
                    bond_peak_error + structural_peak_error
                ),
                "critical_max_peak_error_K": max(
                    bond_peak_error,
                    structural_peak_error,
                ),
                "bond_peak_time_error": abs(
                    predicted_qoi["bond_peak_time"] - fv_qoi["bond_peak_time"]
                ),
                "structural_peak_time_error": abs(
                    predicted_qoi["structural_interface_peak_time"]
                    - fv_qoi["structural_interface_peak_time"]
                ),
                "bond_peak_y_error": abs(
                    predicted_qoi["bond_peak_y"] - fv_qoi["bond_peak_y"]
                ),
                "structural_peak_y_error": abs(
                    predicted_qoi["structural_interface_peak_y"]
                    - fv_qoi["structural_interface_peak_y"]
                ),
                "surrogate_bond_margin": predicted_bond_margin,
                "fv_bond_margin": fv_bond_margin,
                "surrogate_structural_margin": predicted_structural_margin,
                "fv_structural_margin": fv_structural_margin,
                "surrogate_minimum_margin": min(
                    predicted_bond_margin,
                    predicted_structural_margin,
                ),
                "fv_minimum_margin": min(
                    fv_bond_margin,
                    fv_structural_margin,
                ),
                "optimistic_margin_error_K": max(
                    0.0,
                    min(
                        predicted_bond_margin,
                        predicted_structural_margin,
                    )
                    - min(fv_bond_margin, fv_structural_margin),
                ),
                "surrogate_criticality_ratio": max(
                    predicted_qoi["bond_max"] / config.bond_temperature_limit,
                    predicted_qoi["structural_interface_max"]
                    / config.structural_temperature_limit,
                ),
                "fv_criticality_ratio": max(
                    fv_qoi["bond_max"] / config.bond_temperature_limit,
                    fv_qoi["structural_interface_max"]
                    / config.structural_temperature_limit,
                ),
                "margin_error_K": max(
                    abs(predicted_bond_margin - fv_bond_margin),
                    abs(predicted_structural_margin - fv_structural_margin),
                ),
                "margin_sign_disagreement": (
                    surrogate_raw_feasible != fv_feasible
                ),
                "surrogate_raw_feasible": surrogate_raw_feasible,
                "surrogate_feasible": surrogate_feasible,
                "safety_abstention": (
                    surrogate_valid
                    and buffered_design_classification
                    == "near_boundary_fv_required"
                ),
                "fv_feasible": fv_feasible,
                "fv_relative_energy_residual": (
                    fv_acceptance.relative_energy_residual
                ),
                "fv_final_nonlinear_residual_norm_K": (
                    trajectory.final_nonlinear_residual_norm
                ),
                "fv_property_query_min_temperature_K": (
                    trajectory.property_query_temperature_range[0]
                ),
                "fv_property_query_max_temperature_K": (
                    trajectory.property_query_temperature_range[1]
                ),
                "false_feasible": (
                    surrogate_valid
                    and buffered_design_classification == "feasible"
                    and not fv_feasible
                ),
                "false_infeasible": (
                    surrogate_valid
                    and buffered_design_classification == "infeasible"
                    and fv_feasible
                ),
                "fv_seconds": fv_seconds,
                "surrogate_seconds": surrogate_seconds,
            }
            row.update(field_errors)
            rows.append(row)

    fno_selected = _selected_thickness(
        rows,
        config.thickness_candidates,
        "surrogate_feasible",
    )
    fv_selected = _selected_thickness(rows, config.thickness_candidates, "fv_feasible")
    if fno_selected is not None and fv_selected is not None:
        absolute_error = abs(fno_selected - fv_selected)
        relative_error = absolute_error / fv_selected
        differences = np.diff(config.thickness_candidates)
        grid_error = (
            absolute_error / float(differences[0])
            if len(differences) and np.allclose(differences, differences[0])
            else None
        )
    else:
        absolute_error = relative_error = grid_error = None

    ranking_accuracy, ranking_pair_count = _ranking_accuracy(
        rows,
        config.thickness_candidates,
    )
    worst_predicted_bond = max(rows, key=lambda row: row["surrogate_bond_max"])
    worst_fv_bond = max(rows, key=lambda row: row["fv_bond_max"])
    worst_predicted_structural = max(
        rows,
        key=lambda row: row["surrogate_structural_max"],
    )
    worst_fv_structural = max(rows, key=lambda row: row["fv_structural_max"])
    result = {
        "study_id": config.study_id,
        "study_config_sha256": config.sha256,
        "study_authoritative": config.authoritative,
        "heating_interpretation": config.heating.interpretation,
        "surface": config.surface_payload,
        "material_properties": config.property_provenance,
        "bond_conductivity_model": (
            "linear_gaussian_degradation_with_positive_floor"
        ),
        "validity": {
            **config.as_dict()["validity"],
            "hot_face_limit_hierarchy": config.hot_face_limit_hierarchy,
            "hot_face_evaluation_grid": "all_internal_solver_steps",
        },
        "invalid_fv_cell_count": sum(
            int(not row["fv_within_validity"]) for row in rows
        ),
        "scenario_set_id": scenario_set_id,
        "scenario_authoritative": scenario_authoritative,
        "scenario_sha256": scenario_hash,
        "representation": model.config.representation,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "transfer_source_revision": checkpoint["transfer_source_revision"],
        "safety_buffer_K": safety_buffer_K,
        "safety_policy": (
            "feasible only when both predicted margins are strictly greater "
            "than safety_buffer_K; nonnegative margins at or below the buffer "
            "require finite-volume verification"
        ),
        "fno_selected_thickness": fno_selected,
        "fv_selected_thickness": fv_selected,
        "absolute_thickness_error": absolute_error,
        "relative_thickness_error": relative_error,
        "grid_increment_error": grid_error,
        "fno_selection_neighborhood": _selection_neighborhood(
            fno_selected,
            config.thickness_candidates,
        ),
        "fv_selection_neighborhood": _selection_neighborhood(
            fv_selected,
            config.thickness_candidates,
        ),
        "fno_limiting_scenarios": _limiting_scenarios(
            rows,
            fno_selected,
            "surrogate",
        ),
        "fv_limiting_scenarios": _limiting_scenarios(
            rows,
            fv_selected,
            "fv",
        ),
        "false_feasible_count": sum(int(row["false_feasible"]) for row in rows),
        "false_infeasible_count": sum(int(row["false_infeasible"]) for row in rows),
        "safety_abstention_count": sum(
            int(row["safety_abstention"]) for row in rows
        ),
        "margin_sign_disagreement_count": sum(
            int(row["margin_sign_disagreement"]) for row in rows
        ),
        "critical_mean_peak_error_K": float(
            np.mean([row["critical_mean_peak_error_K"] for row in rows])
        ),
        "critical_max_peak_error_K": float(
            np.max([row["critical_max_peak_error_K"] for row in rows])
        ),
        "margin_error_mean_K": float(
            np.mean([row["margin_error_K"] for row in rows])
        ),
        "optimistic_margin_error_mean_K": float(
            np.mean([row["optimistic_margin_error_K"] for row in rows])
        ),
        "optimistic_margin_error_p95_K": float(
            np.percentile(
                [row["optimistic_margin_error_K"] for row in rows],
                95.0,
            )
        ),
        "optimistic_margin_error_p99_K": float(
            np.percentile(
                [row["optimistic_margin_error_K"] for row in rows],
                99.0,
            )
        ),
        "optimistic_margin_error_max_K": float(
            np.max([row["optimistic_margin_error_K"] for row in rows])
        ),
        "field_rmse_mean_K": float(
            np.mean([row["field_rmse_K"] for row in rows])
        ),
        "scenario_ranking_accuracy": ranking_accuracy,
        "scenario_ranking_pair_count": ranking_pair_count,
        "worst_bond_scenario_rank_correct": (
            worst_predicted_bond["scenario_id"] == worst_fv_bond["scenario_id"]
            and worst_predicted_bond["d_tps"] == worst_fv_bond["d_tps"]
        ),
        "worst_structural_scenario_rank_correct": (
            worst_predicted_structural["scenario_id"]
            == worst_fv_structural["scenario_id"]
            and worst_predicted_structural["d_tps"] == worst_fv_structural["d_tps"]
        ),
        "total_fv_seconds": total_fv_seconds,
        "total_surrogate_seconds": total_surrogate_seconds,
        "screening_speedup": (
            total_fv_seconds / total_surrogate_seconds
            if total_surrogate_seconds > 0.0
            else None
        ),
        "rows": rows,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "sizing_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "sizing_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result
