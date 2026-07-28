from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import csv
import json
import time

import numpy as np

from fno_tps.acceptance import assess_trajectory
from fno_tps.config import StudyConfig
from fno_tps.physics import TPSFVSolver
from fno_tps.problem import BondDefect, HeatingEvent, SimulationCase


PILOT_SCENARIO_DESCRIPTIONS = (
    "Nominal centered single event with a healthy bond.",
    "Maximum localized single event aligned with one severe narrow defect.",
    "Two maximum events moving laterally across two severe defects.",
    "Three maximum distributed events with a healthy bond.",
    "Three maximum laterally aligned events over two severe broad defects.",
)


def _pilot_scenarios(
    config: StudyConfig,
    scenario_count: int,
) -> list[SimulationCase]:
    if not 1 <= scenario_count <= len(PILOT_SCENARIO_DESCRIPTIONS):
        raise ValueError(
            f"scenario_count must be between 1 and "
            f"{len(PILOT_SCENARIO_DESCRIPTIONS)}."
        )
    heating = config.heating
    bond = config.bond
    amplitude_mid = 0.5 * sum(heating.amplitude)
    y_mid = 0.5 * sum(heating.y_center)
    time_mid = 0.5 * sum(heating.t_center)
    sigma_y_mid = 0.5 * sum(heating.sigma_y)
    sigma_t_mid = 0.5 * sum(heating.sigma_t)
    time_gap = min(
        heating.sigma_t[1],
        0.25 * (heating.t_center[1] - heating.t_center[0]),
    )

    def event(
        *,
        amplitude: float,
        y_center: float,
        t_center: float,
        sigma_y: float,
        sigma_t: float,
    ) -> HeatingEvent:
        return HeatingEvent(amplitude, y_center, t_center, sigma_y, sigma_t)

    def defect(y_center: float, sigma: float) -> BondDefect:
        return BondDefect(
            bond.severity[1],
            float(np.clip(y_center, *bond.y_center)),
            sigma,
        )

    scenarios = [
        SimulationCase(
            case_id="pilot-nominal-single-healthy",
            d_tps=config.thickness_candidates[0],
            heating_events=(
                event(
                    amplitude=amplitude_mid,
                    y_center=y_mid,
                    t_center=time_mid,
                    sigma_y=sigma_y_mid,
                    sigma_t=sigma_t_mid,
                ),
            ),
            bond_defects=(),
        ),
        SimulationCase(
            case_id="pilot-localized-single-defect",
            d_tps=config.thickness_candidates[0],
            heating_events=(
                event(
                    amplitude=heating.amplitude[1],
                    y_center=heating.y_center[0],
                    t_center=time_mid,
                    sigma_y=heating.sigma_y[0],
                    sigma_t=heating.sigma_t[1],
                ),
            ),
            bond_defects=(
                defect(heating.y_center[0], bond.sigma[0]),
            ),
        ),
        SimulationCase(
            case_id="pilot-moving-double-defect",
            d_tps=config.thickness_candidates[0],
            heating_events=(
                event(
                    amplitude=heating.amplitude[1],
                    y_center=heating.y_center[0],
                    t_center=time_mid - time_gap,
                    sigma_y=sigma_y_mid,
                    sigma_t=heating.sigma_t[1],
                ),
                event(
                    amplitude=heating.amplitude[1],
                    y_center=heating.y_center[1],
                    t_center=time_mid + time_gap,
                    sigma_y=sigma_y_mid,
                    sigma_t=heating.sigma_t[1],
                ),
            ),
            bond_defects=(
                defect(heating.y_center[0], bond.sigma[0]),
                defect(heating.y_center[1], bond.sigma[0]),
            ),
        ),
        SimulationCase(
            case_id="pilot-distributed-triple-healthy",
            d_tps=config.thickness_candidates[0],
            heating_events=tuple(
                event(
                    amplitude=heating.amplitude[1],
                    y_center=y_center,
                    t_center=t_center,
                    sigma_y=heating.sigma_y[1],
                    sigma_t=heating.sigma_t[1],
                )
                for y_center, t_center in zip(
                    (
                        heating.y_center[0],
                        y_mid,
                        heating.y_center[1],
                    ),
                    (
                        heating.t_center[0],
                        time_mid,
                        heating.t_center[1],
                    ),
                )
            ),
            bond_defects=(),
        ),
        SimulationCase(
            case_id="pilot-aligned-triple-severe",
            d_tps=config.thickness_candidates[0],
            heating_events=tuple(
                event(
                    amplitude=heating.amplitude[1],
                    y_center=y_mid,
                    t_center=t_center,
                    sigma_y=heating.sigma_y[1],
                    sigma_t=heating.sigma_t[1],
                )
                for t_center in (
                    time_mid - time_gap,
                    time_mid,
                    time_mid + time_gap,
                )
            ),
            bond_defects=(
                defect(y_mid - 0.5 * bond.sigma[1], bond.sigma[1]),
                defect(y_mid + 0.5 * bond.sigma[1], bond.sigma[1]),
            ),
        ),
    ]
    return scenarios[:scenario_count]


def _minimum_feasible(
    rows: list[dict[str, Any]],
    scenario_id: str,
    backing_thickness: float,
    candidates: tuple[float, ...],
) -> float | None:
    by_thickness = {
        row["d_tps"]: bool(row["feasible"])
        for row in rows
        if (
            row["scenario_id"] == scenario_id
            and np.isclose(row["backing_thickness_m"], backing_thickness)
        )
    }
    return next(
        (thickness for thickness in candidates if by_thickness.get(thickness, False)),
        None,
    )


def run_pilot_matrix(
    config: StudyConfig,
    output_dir: str | Path,
    *,
    scenario_count: int = 5,
    backing_thicknesses: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Run the nonlinear FV feasibility matrix with explicit validity gates."""
    scenarios = _pilot_scenarios(config, scenario_count)
    backing_values = (
        (config.backing_thickness,)
        if backing_thicknesses is None
        else tuple(sorted({float(value) for value in backing_thicknesses}))
    )
    if not backing_values or min(backing_values) <= 0.0:
        raise ValueError("backing_thicknesses must contain positive values.")
    simulation_count = (
        scenario_count
        * len(config.thickness_candidates)
        * len(backing_values)
    )
    times = np.asarray(config.time.saved_times(), dtype=np.float64)
    ceiling = config.hot_face_temperature_limit
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()

    for backing_thickness in backing_values:
        run_config = replace(config, backing_thickness=backing_thickness)
        for scenario in scenarios:
            for thickness in config.thickness_candidates:
                case = replace(
                    scenario,
                    case_id=(
                        f"{scenario.case_id}@back={backing_thickness:g}"
                        f"@tps={thickness:g}"
                    ),
                    d_tps=thickness,
                )
                solve_start = time.perf_counter()
                solver = TPSFVSolver(run_config, case)
                trajectory = solver.solve(save_times=times)
                solve_seconds = time.perf_counter() - solve_start
                qoi = solver.quantities_of_interest(
                    trajectory.temperatures,
                    times,
                )
                bond_margin = (
                    config.bond_temperature_limit - qoi["bond_max"]
                )
                structural_margin = (
                    config.structural_temperature_limit
                    - qoi["structural_interface_max"]
                )
                bond_utilization = (
                    (qoi["bond_max"] - config.initial_temperature)
                    / (
                        config.bond_temperature_limit
                        - config.initial_temperature
                    )
                )
                structural_utilization = (
                    (
                        qoi["structural_interface_max"]
                        - config.initial_temperature
                    )
                    / (
                        config.structural_temperature_limit
                        - config.initial_temperature
                    )
                )
                hot_face_max = trajectory.maximum_hot_face_temperature
                acceptance = assess_trajectory(
                    run_config,
                    trajectory,
                    bond_max=qoi["bond_max"],
                    structural_interface_max=(
                        qoi["structural_interface_max"]
                    ),
                )
                design_constraints_satisfied = not (
                    acceptance.design_violations
                )
                rows.append({
                "scenario_id": scenario.case_id,
                "backing_thickness_m": backing_thickness,
                "event_count": scenario.event_count,
                "defect_count": scenario.defect_count,
                "d_tps": thickness,
                "bond_max_K": qoi["bond_max"],
                "structural_interface_max_K": qoi["structural_interface_max"],
                "hot_face_max_K": hot_face_max,
                "hot_cell_max_K": float(
                    np.max(trajectory.temperatures[:, 0, :])
                ),
                "incident_amplitude": float(max(
                    (
                        event.amplitude
                        for event in scenario.heating_events
                    ),
                    default=0.0,
                )),
                "boundary_input_energy_J": float(
                    trajectory.boundary_input_energy[-1]
                ),
                "reradiated_energy_J": float(
                    trajectory.radiated_energy[-1]
                ),
                "net_boundary_energy_J": float(
                    trajectory.net_boundary_energy[-1]
                ),
                "reradiated_energy_fraction": (
                    float(
                        trajectory.radiated_energy[-1]
                        / trajectory.boundary_input_energy[-1]
                    )
                    if trajectory.boundary_input_energy[-1] > 0.0
                    else 0.0
                ),
                "maximum_nonlinear_iterations": int(
                    np.max(trajectory.nonlinear_iteration_counts, initial=0)
                ),
                "nonlinear_iterations_max": (
                    trajectory.max_nonlinear_iterations
                ),
                "nonlinear_iterations_mean": float(
                    np.mean(trajectory.nonlinear_iteration_counts)
                    if trajectory.nonlinear_iteration_counts.size
                    else 0.0
                ),
                "linear_solves": trajectory.linear_solve_count,
                "factorizations": trajectory.factorization_count,
                "accepted_range_excursions": (
                    trajectory.accepted_range_excursions
                ),
                "iteration_range_clamps": (
                    trajectory.iteration_range_clamps
                ),
                "property_clamped": (
                    trajectory.accepted_range_excursions > 0
                ),
                "observed_min_temperature_K": (
                    trajectory.observed_temperature_range[0]
                ),
                "observed_max_temperature_K": (
                    trajectory.observed_temperature_range[1]
                ),
                "valid": acceptance.valid,
                "within_validity": acceptance.valid,
                "classification": acceptance.classification,
                "invalid_reasons": ";".join(acceptance.invalid_reasons),
                "design_violations": ";".join(
                    acceptance.design_violations
                ),
                "eligible_for_authoritative_benchmark": acceptance.valid,
                "bond_margin_K": bond_margin,
                "structural_margin_K": structural_margin,
                "bond_utilization": bond_utilization,
                "structural_utilization": structural_utilization,
                "controlling_constraint": (
                    "bond"
                    if bond_utilization >= structural_utilization
                    else "structural_interface"
                ),
                "design_constraints_satisfied": (
                    design_constraints_satisfied
                ),
                "feasible": bool(acceptance.feasible),
                "solver_converged": trajectory.solver_converged,
                "final_nonlinear_residual_norm_K": (
                    trajectory.final_nonlinear_residual_norm
                ),
                "damped_nonlinear_iteration_count": int(np.sum(
                    trajectory.nonlinear_damped_iteration_counts
                )),
                "nonlinear_backtrack_count": int(np.sum(
                    trajectory.nonlinear_backtrack_counts
                )),
                "failed_nonlinear_iterations": (
                    trajectory.failed_nonlinear_iterations
                ),
                "property_query_min_temperature_K": (
                    trajectory.property_query_temperature_range[0]
                ),
                "property_query_max_temperature_K": (
                    trajectory.property_query_temperature_range[1]
                ),
                "stored_energy_change_J": float(
                    trajectory.internal_energy[-1]
                    - trajectory.internal_energy[0]
                ),
                "energy_balance_residual_J": float(
                    trajectory.energy_residual[-1]
                ),
                "maximum_relative_energy_residual": (
                    acceptance.relative_energy_residual
                ),
                "solve_seconds": solve_seconds,
            })

    scenario_ids = [scenario.case_id for scenario in scenarios]
    configuration_ids = [
        (scenario_id, backing_thickness)
        for backing_thickness in backing_values
        for scenario_id in scenario_ids
    ]
    minimum_feasible = {
        f"{scenario_id}|backing={backing_thickness:g}": _minimum_feasible(
            rows,
            scenario_id,
            backing_thickness,
            config.thickness_candidates,
        )
        for scenario_id, backing_thickness in configuration_ids
    }
    scenario_validity_complete = {
        f"{scenario_id}|backing={backing_thickness:g}": all(
            row["within_validity"]
            for row in rows
            if (
                row["scenario_id"] == scenario_id
                and np.isclose(
                    row["backing_thickness_m"],
                    backing_thickness,
                )
            )
        )
        for scenario_id, backing_thickness in configuration_ids
    }
    validity_eligible_minimum_feasible = {
        configuration_id: (
            minimum_feasible[configuration_id]
            if scenario_validity_complete[configuration_id]
            else None
        )
        for configuration_id in minimum_feasible
    }
    valid_rows = [row for row in rows if row["within_validity"]]
    invalid_rows = [row for row in rows if not row["within_validity"]]
    feasible_fraction = float(np.mean([row["feasible"] for row in rows]))
    validity_filtered_feasible_fraction = (
        float(np.mean([row["feasible"] for row in valid_rows]))
        if valid_rows
        else None
    )
    thinnest_rows = [
        row for row in rows
        if row["d_tps"] == config.thickness_candidates[0]
    ]
    thickest_rows = [
        row for row in rows
        if row["d_tps"] == config.thickness_candidates[-1]
    ]
    thinnest_infeasible = any(
        row["valid"] and not row["design_constraints_satisfied"]
        for row in thinnest_rows
    )
    thickest_feasible_fraction = float(
        np.mean([row["feasible"] for row in thickest_rows])
    )
    interior_minimum_count = sum(
        thickness is not None
        and thickness not in {
            config.thickness_candidates[0],
            config.thickness_candidates[-1],
        }
        for thickness in minimum_feasible.values()
    )
    selected_thicknesses = [
        thickness
        for thickness in minimum_feasible.values()
        if thickness is not None
    ]
    minimum_feasible_thickness_varies = (
        len(set(selected_thicknesses)) > 1
    )
    controlling_counts = {
        constraint: sum(
            row["controlling_constraint"] == constraint for row in rows
        )
        for constraint in ("bond", "structural_interface")
    }
    maximum_combo_ids = {
        scenario.case_id
        for scenario in scenarios
        if scenario.event_count == 3 and scenario.defect_count == 2
    }
    maximum_combo_has_feasible_design = (
        None
        if not maximum_combo_ids
        else any(
            row["scenario_id"] in maximum_combo_ids and row["feasible"]
            for row in rows
        )
    )
    maximum_combo_has_valid_feasible_design = (
        None
        if not maximum_combo_ids
        else any(
            row["scenario_id"] in maximum_combo_ids
            and row["within_validity"]
            and row["feasible"]
            for row in rows
        )
    )
    maximum_hot_face = float(max(row["hot_face_max_K"] for row in rows))
    hot_face_within_ceiling = maximum_hot_face <= ceiling
    all_cells_within_validity = not invalid_rows
    target_feasible_fraction = 0.25 <= feasible_fraction <= 0.75
    matrix_size_in_recommended_range = 30 <= simulation_count <= 50
    both_constraints_control = all(
        count > 0 for count in controlling_counts.values()
    )
    sizing_regime_balanced = bool(
        target_feasible_fraction
        and thinnest_infeasible
        and thickest_feasible_fraction >= 0.80
        and interior_minimum_count > 0
        and maximum_combo_has_valid_feasible_design is not False
        and all_cells_within_validity
    )
    authority_review_ready = bool(
        config.authoritative
        and sizing_regime_balanced
        and both_constraints_control
        and hot_face_within_ceiling is True
    )

    recommendations: list[str] = []
    if not config.authoritative:
        recommendations.append(
            "The numerical regime is non-authoritative; replace or substantiate "
            "all placeholder property and emissivity inputs before production."
        )
    if feasible_fraction < 0.25 or thickest_feasible_fraction < 0.80:
        recommendations.append(
            "Reduce heating.amplitude first; too little of the matrix is feasible."
        )
    elif feasible_fraction > 0.75 or not thinnest_infeasible:
        if ceiling is None:
            recommendations.append(
                "Too much of the matrix is feasible, but do not increase heating "
                "until the hot-face/material temperature ceiling is validated. "
                "Then adjust amplitude only if the hot-face check permits it."
            )
        elif hot_face_within_ceiling:
            recommendations.append(
                "Increase heating.amplitude first; too much of the matrix is feasible "
                "and the current hot face remains within the declared ceiling."
            )
        else:
            recommendations.append(
                "Do not increase heating amplitude: the hot face already exceeds "
                "the declared validity ceiling. Revisit the surface-energy "
                "physics, design window, or study constraints."
            )
    if interior_minimum_count == 0:
        recommendations.append(
            "Recheck the thickness grid after amplitude calibration; no scenario "
            "has an interior minimum feasible thickness."
        )
    if not both_constraints_control:
        recommendations.append(
            "Only one constraint controls in this pilot; retain both metrics and "
            "review the scenario set after amplitude calibration."
        )
    if not hot_face_within_ceiling:
        recommendations.append(
                "The hot face exceeds the declared validity limit; revise the "
                "physics or loading."
        )

    result = {
        "study_id": config.study_id,
        "study_config_sha256": config.sha256,
        "authoritative": config.authoritative,
        "simulation_count": simulation_count,
        "scenario_count": scenario_count,
        "backing_thicknesses_m": list(backing_values),
        "matrix_size_in_recommended_30_to_50_range": matrix_size_in_recommended_range,
        "full_45_case_envelope_per_backing": (
            simulation_count == 45 * len(backing_values)
        ),
        "feasible_fraction": feasible_fraction,
        "validity_filtered_feasible_fraction": (
            validity_filtered_feasible_fraction
        ),
        "valid_cell_count": len(valid_rows),
        "invalid_cell_count": len(invalid_rows),
        "invalid_cell_fraction": len(invalid_rows) / len(rows),
        "all_cells_within_validity": all_cells_within_validity,
        "target_feasible_fraction_25_to_75_percent": target_feasible_fraction,
        "thinnest_design_sometimes_infeasible": thinnest_infeasible,
        "thickest_design_feasible_fraction": thickest_feasible_fraction,
        "interior_minimum_feasible_scenario_count": interior_minimum_count,
        "minimum_feasible_thickness_varies": (
            minimum_feasible_thickness_varies
        ),
        "minimum_feasible_thickness_by_scenario": minimum_feasible,
        "scenario_validity_complete": scenario_validity_complete,
        "validity_eligible_minimum_feasible_thickness_by_scenario": (
            validity_eligible_minimum_feasible
        ),
        "controlling_constraint_counts": controlling_counts,
        "both_constraints_control_some_cases": both_constraints_control,
        "maximum_event_defect_combo_has_feasible_design": (
            maximum_combo_has_feasible_design
        ),
        "maximum_event_defect_combo_has_valid_feasible_design": (
            maximum_combo_has_valid_feasible_design
        ),
        "maximum_hot_face_temperature_K": maximum_hot_face,
        "validity": {
            "tps_property_model": config.validity.tps_property_model,
            "property_table_minimum_temperature_K": (
                config.validity.property_tables.minimum_temperature
            ),
            "hot_face_limit_hierarchy": config.hot_face_limit_hierarchy,
            "resolved_hot_face_temperature_limit_K": ceiling,
            "max_property_relative_deviation": (
                config.validity.max_property_relative_deviation
            ),
            "conductivity_reference_pressure_Pa": (
                config.validity.conductivity_reference_pressure
            ),
            "reject_cases_outside_validity": (
                config.validity.reject_cases_outside_validity
            ),
            "hot_face_evaluation_grid": "all_internal_solver_steps",
            "property_tables": config.property_provenance,
            "accepted_range_excursions": int(sum(
                row["accepted_range_excursions"] for row in rows
            )),
            "iteration_range_clamps_diagnostic_only": int(sum(
                row["iteration_range_clamps"] for row in rows
            )),
            "observed_min_temperature_K": float(min(
                row["observed_min_temperature_K"] for row in rows
            )),
            "observed_max_temperature_K": float(max(
                row["observed_max_temperature_K"] for row in rows
            )),
            "maximum_relative_energy_residual": float(max(
                row["maximum_relative_energy_residual"] for row in rows
            )),
        },
        "surface": config.surface_payload,
        "material_properties": config.property_provenance,
        "hot_face_within_validity": hot_face_within_ceiling,
        "sizing_regime_balanced": sizing_regime_balanced,
        "authority_review_ready": authority_review_ready,
        "recommendations": recommendations,
        "wall_seconds": time.perf_counter() - start,
        "pilot_scenario_design": [
            {
                "scenario_id": scenario.case_id,
                "description": PILOT_SCENARIO_DESCRIPTIONS[index],
                "case": scenario.as_dict(),
            }
            for index, scenario in enumerate(scenarios)
        ],
        "rows": rows,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "pilot_matrix.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "pilot_report.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result
