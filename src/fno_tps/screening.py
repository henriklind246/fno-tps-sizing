from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable
import csv
import json
import time

import numpy as np

from fno_tps.acceptance import assess_trajectory
from fno_tps.config import StudyConfig, TimeConfig
from fno_tps.materials import PropertyRangeError
from fno_tps.physics import TPSFVSolver
from fno_tps.problem import BondDefect, HeatingEvent, SimulationCase


DEFAULT_PULSE_WIDTHS = (30.0, 60.0, 120.0, 180.0)
DEFAULT_HORIZONS = (600.0, 900.0, 1200.0, 1800.0, 2400.0)
DEFAULT_SCREENING_THICKNESSES = (0.003, 0.012, 0.024)
DEFAULT_REFERENCE_AMPLITUDE = 1000.0
DEFAULT_INCIDENT_AMPLITUDES = (1000.0, 2000.0, 3000.0, 4000.0, 5000.0)
DEFAULT_INCIDENT_PULSE_WIDTHS = (120.0, 180.0)
DEFAULT_INCIDENT_HORIZONS = (1200.0, 1800.0, 2400.0)

SCREEN_SCENARIO_DESCRIPTIONS = {
    "screen-centered-single-healthy": (
        "One centered heating event with a healthy bond."
    ),
    "screen-localized-single-defect": (
        "One localized heating event aligned with a severe narrow defect."
    ),
    "screen-moving-triple-two-defects": (
        "Three laterally separated events moving across two severe narrow defects."
    ),
}


def _positive_sorted_unique(
    values: Iterable[float],
    *,
    name: str,
) -> tuple[float, ...]:
    result = tuple(sorted({float(value) for value in values}))
    if not result or result[0] <= 0.0:
        raise ValueError(f"{name} must contain positive values.")
    return result


def _screen_scenarios(
    config: StudyConfig,
    sigma_t: float,
    reference_amplitude: float,
) -> list[SimulationCase]:
    """Create the three fixed spatial/defect scenarios for one pulse width.

    ``reference_amplitude`` is the peak amplitude of each event. The linear
    screen later scales every event in a scenario by the same multiplier.
    Event centers are expressed relative to ``sigma_t`` so the pulse shape is
    not truncated at t=0 when widths change.
    """
    if sigma_t <= 0.0 or reference_amplitude <= 0.0:
        raise ValueError("Pulse width and reference amplitude must be positive.")
    heating = config.heating
    bond = config.bond
    y_mid = 0.5 * sum(heating.y_center)
    sigma_y_mid = 0.5 * sum(heating.sigma_y)
    thin = config.thickness_candidates[0]

    def event(
        y_center: float,
        t_center: float,
        sigma_y: float,
    ) -> HeatingEvent:
        return HeatingEvent(
            amplitude=reference_amplitude,
            y_center=y_center,
            t_center=t_center,
            sigma_y=sigma_y,
            sigma_t=sigma_t,
        )

    def severe_defect(y_center: float) -> BondDefect:
        return BondDefect(
            severity=bond.severity[1],
            y_center=float(np.clip(y_center, *bond.y_center)),
            sigma=bond.sigma[0],
        )

    return [
        SimulationCase(
            case_id="screen-centered-single-healthy",
            d_tps=thin,
            heating_events=(
                event(y_mid, 3.0 * sigma_t, sigma_y_mid),
            ),
            bond_defects=(),
        ),
        SimulationCase(
            case_id="screen-localized-single-defect",
            d_tps=thin,
            heating_events=(
                event(
                    heating.y_center[0],
                    3.0 * sigma_t,
                    heating.sigma_y[0],
                ),
            ),
            bond_defects=(
                severe_defect(heating.y_center[0]),
            ),
        ),
        SimulationCase(
            case_id="screen-moving-triple-two-defects",
            d_tps=thin,
            heating_events=(
                event(
                    heating.y_center[0],
                    3.0 * sigma_t,
                    sigma_y_mid,
                ),
                event(
                    y_mid,
                    3.5 * sigma_t,
                    sigma_y_mid,
                ),
                event(
                    heating.y_center[1],
                    4.0 * sigma_t,
                    sigma_y_mid,
                ),
            ),
            bond_defects=(
                severe_defect(heating.y_center[0]),
                severe_defect(heating.y_center[1]),
            ),
        ),
    ]


def _threshold(limit_rise: float, response_per_amplitude: float) -> float:
    if response_per_amplitude <= 0.0:
        return float("inf")
    return limit_rise / response_per_amplitude


def _amplitude_interval(
    rows: list[dict[str, Any]],
    *,
    initial_temperature: float,
    hot_face_limit: float,
    bond_limit: float,
    structural_limit: float,
    thinnest: float,
    thickest: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Amplitude interval calculation requires response rows.")
    scenario_ids = sorted({str(row["scenario_id"]) for row in rows})
    thin_rows = [
        row for row in rows
        if np.isclose(float(row["d_tps"]), thinnest)
    ]
    thick_rows = [
        row for row in rows
        if np.isclose(float(row["d_tps"]), thickest)
    ]
    if (
        {str(row["scenario_id"]) for row in thin_rows} != set(scenario_ids)
        or {str(row["scenario_id"]) for row in thick_rows} != set(scenario_ids)
    ):
        raise ValueError("Every scenario requires both thin and thick responses.")

    hot_rise_limit = hot_face_limit - initial_temperature
    bond_rise_limit = bond_limit - initial_temperature
    structural_rise_limit = structural_limit - initial_temperature
    if min(hot_rise_limit, bond_rise_limit, structural_rise_limit) <= 0.0:
        raise ValueError("Temperature limits must exceed the initial temperature.")

    hot_bound = min(
        _threshold(hot_rise_limit, float(row["hot_rise_per_amplitude_K_per_W_m2"]))
        for row in rows
    )
    thin_failure_by_scenario = {
        scenario_id: min(
            _threshold(
                bond_rise_limit,
                float(row["bond_rise_per_amplitude_K_per_W_m2"]),
            ),
            _threshold(
                structural_rise_limit,
                float(row["structural_rise_per_amplitude_K_per_W_m2"]),
            ),
        )
        for scenario_id in scenario_ids
        for row in thin_rows
        if row["scenario_id"] == scenario_id
    }
    thick_feasible_by_scenario = {
        scenario_id: min(
            _threshold(
                bond_rise_limit,
                float(row["bond_rise_per_amplitude_K_per_W_m2"]),
            ),
            _threshold(
                structural_rise_limit,
                float(row["structural_rise_per_amplitude_K_per_W_m2"]),
            ),
        )
        for scenario_id in scenario_ids
        for row in thick_rows
        if row["scenario_id"] == scenario_id
    }
    thin_failure = min(thin_failure_by_scenario.values())
    thick_feasible = min(thick_feasible_by_scenario.values())
    upper = min(hot_bound, thick_feasible)
    width = upper - thin_failure
    viable = bool(np.isfinite(width) and width > 0.0)
    relative_width = (
        width / thin_failure
        if viable and thin_failure > 0.0
        else 0.0
    )
    recommended = (
        thin_failure + 0.5 * width
        if viable
        else None
    )
    return {
        "hot_face_amplitude_upper_bound_W_m2": float(hot_bound),
        "thin_failure_amplitude_lower_bound_W_m2": float(thin_failure),
        "thick_feasible_amplitude_upper_bound_W_m2": float(thick_feasible),
        "interval_upper_bound_W_m2": float(upper),
        "amplitude_interval_width_W_m2": float(width),
        "relative_interval_width": float(relative_width),
        "linear_interval_viable": viable,
        "comfortable_interval": bool(viable and relative_width >= 0.10),
        "recommended_amplitude_W_m2": (
            None if recommended is None else float(recommended)
        ),
        "thin_failure_threshold_by_scenario_W_m2": {
            key: float(value)
            for key, value in thin_failure_by_scenario.items()
        },
        "thick_feasible_threshold_by_scenario_W_m2": {
            key: float(value)
            for key, value in thick_feasible_by_scenario.items()
        },
    }


def _matrix_at_amplitude(
    rows: list[dict[str, Any]],
    amplitude: float | None,
    *,
    initial_temperature: float,
    hot_face_limit: float,
    bond_limit: float,
    structural_limit: float,
    thicknesses: tuple[float, ...],
) -> dict[str, Any]:
    if amplitude is None:
        return {
            "evaluated": False,
            "rows": [],
            "all_cells_within_validity": False,
            "feasible_fraction": None,
            "thinnest_design_sometimes_infeasible": False,
            "thickest_design_all_feasible": False,
            "interior_minimum_feasible_scenario_count": 0,
            "minimum_feasible_thickness_by_scenario": {},
            "minimum_feasible_thickness_varies": False,
            "controlling_constraint_counts": {
                "bond": 0,
                "structural_interface": 0,
            },
        }

    matrix_rows: list[dict[str, Any]] = []
    for row in rows:
        hot = initial_temperature + amplitude * float(
            row["hot_rise_per_amplitude_K_per_W_m2"]
        )
        bond = initial_temperature + amplitude * float(
            row["bond_rise_per_amplitude_K_per_W_m2"]
        )
        structural = initial_temperature + amplitude * float(
            row["structural_rise_per_amplitude_K_per_W_m2"]
        )
        bond_utilization = (
            (bond - initial_temperature) / (bond_limit - initial_temperature)
        )
        structural_utilization = (
            (structural - initial_temperature)
            / (structural_limit - initial_temperature)
        )
        matrix_rows.append({
            "scenario_id": row["scenario_id"],
            "d_tps": float(row["d_tps"]),
            "hot_face_max_K": float(hot),
            "bond_max_K": float(bond),
            "structural_interface_max_K": float(structural),
            "within_validity": bool(hot <= hot_face_limit + 1e-8),
            "feasible": bool(
                bond <= bond_limit + 1e-8
                and structural <= structural_limit + 1e-8
            ),
            "controlling_constraint": (
                "bond"
                if bond_utilization >= structural_utilization
                else "structural_interface"
            ),
        })

    scenario_ids = sorted({str(row["scenario_id"]) for row in matrix_rows})
    minimum_feasible = {}
    for scenario_id in scenario_ids:
        by_thickness = {
            float(row["d_tps"]): bool(row["feasible"])
            for row in matrix_rows
            if row["scenario_id"] == scenario_id
        }
        minimum_feasible[scenario_id] = next(
            (
                thickness
                for thickness in thicknesses
                if by_thickness.get(thickness, False)
            ),
            None,
        )
    selected = [
        value for value in minimum_feasible.values()
        if value is not None
    ]
    controls = {
        constraint: sum(
            row["controlling_constraint"] == constraint
            for row in matrix_rows
        )
        for constraint in ("bond", "structural_interface")
    }
    return {
        "evaluated": True,
        "amplitude_W_m2": float(amplitude),
        "rows": matrix_rows,
        "all_cells_within_validity": all(
            row["within_validity"] for row in matrix_rows
        ),
        "feasible_fraction": float(np.mean([
            row["feasible"] for row in matrix_rows
        ])),
        "thinnest_design_sometimes_infeasible": any(
            not row["feasible"]
            for row in matrix_rows
            if np.isclose(row["d_tps"], thicknesses[0])
        ),
        "thickest_design_all_feasible": all(
            row["feasible"]
            for row in matrix_rows
            if np.isclose(row["d_tps"], thicknesses[-1])
        ),
        "interior_minimum_feasible_scenario_count": sum(
            value is not None
            and value not in {thicknesses[0], thicknesses[-1]}
            for value in minimum_feasible.values()
        ),
        "minimum_feasible_thickness_by_scenario": minimum_feasible,
        "minimum_feasible_thickness_varies": len(set(selected)) > 1,
        "controlling_constraint_counts": controls,
    }


def _series_for_trajectory(
    solver: TPSFVSolver,
    temperatures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hot = np.max(temperatures[:, 0, :], axis=1)
    bond = np.max(
        solver.bond_temperature_envelope(temperatures),
        axis=1,
    )
    structural = np.max(
        solver.structural_interface_temperature(temperatures),
        axis=1,
    )
    return hot, bond, structural


def run_pulse_width_screen(
    config: StudyConfig,
    output_dir: str | Path,
    *,
    pulse_widths: Iterable[float] = DEFAULT_PULSE_WIDTHS,
    horizons: Iterable[float] = DEFAULT_HORIZONS,
    thicknesses: Iterable[float] = DEFAULT_SCREENING_THICKNESSES,
    reference_amplitude: float = DEFAULT_REFERENCE_AMPLITUDE,
) -> dict[str, Any]:
    """Screen wider transient pulses using one long solve per physical case."""
    if (
        config.surface.reradiation_enabled
        or config.validity.tps_property_model != "constant_effective"
    ):
        raise ValueError(
            "pulse-screen uses exact linear amplitude scaling and cannot be used "
            "with reradiation or temperature-dependent properties. Use "
            "incident-screen/the nonlinear pilot instead."
        )
    widths = _positive_sorted_unique(pulse_widths, name="pulse_widths")
    candidate_horizons = _positive_sorted_unique(horizons, name="horizons")
    screening_thicknesses = _positive_sorted_unique(
        thicknesses,
        name="thicknesses",
    )
    if len(screening_thicknesses) < 2:
        raise ValueError("Pulse screening requires at least two TPS thicknesses.")
    if reference_amplitude <= 0.0:
        raise ValueError("reference_amplitude must be positive.")
    for thickness in screening_thicknesses:
        if thickness not in config.thickness_candidates:
            raise ValueError(
                f"Screening thickness {thickness:g} is not a configured candidate."
            )

    maximum_horizon = candidate_horizons[-1]
    time_config = TimeConfig(
        dt=config.time.dt,
        t_final=maximum_horizon,
        save_stride=config.time.save_stride,
        horizon_candidates=candidate_horizons,
    )
    save_interval = time_config.dt * time_config.save_stride
    for horizon in candidate_horizons:
        steps = horizon / save_interval
        if abs(steps - round(steps)) > 1e-10:
            raise ValueError(
                f"Horizon {horizon:g} is not aligned with saved interval "
                f"{save_interval:g}."
            )
    screen_config = replace(config, time=time_config)
    times = np.asarray(time_config.saved_times(), dtype=np.float64)
    horizon_indices = {
        horizon: int(np.flatnonzero(np.isclose(times, horizon))[0])
        for horizon in candidate_horizons
    }

    solve_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for sigma_t in widths:
        scenarios = _screen_scenarios(
            screen_config,
            sigma_t,
            reference_amplitude,
        )
        if max(
            event.t_center
            for scenario in scenarios
            for event in scenario.heating_events
        ) > maximum_horizon:
            raise ValueError(
                f"Pulse width {sigma_t:g} places an event center beyond the "
                f"maximum horizon {maximum_horizon:g}."
            )
        for scenario in scenarios:
            for thickness in screening_thicknesses:
                case = replace(
                    scenario,
                    case_id=(
                        f"{scenario.case_id}@sigma={sigma_t:g}@d={thickness:g}"
                    ),
                    d_tps=thickness,
                )
                solve_start = time.perf_counter()
                solver = TPSFVSolver(screen_config, case)
                trajectory = solver.solve(save_times=times)
                solve_seconds = time.perf_counter() - solve_start
                hot_series, bond_series, structural_series = _series_for_trajectory(
                    solver,
                    trajectory.temperatures,
                )
                heat_samples = np.stack([
                    case.heat_flux(solver.grid.y_centers, float(sample_time))
                    for sample_time in times
                ])
                heat_peak = float(np.max(heat_samples))
                exact_hot_rise = (
                    trajectory.maximum_hot_face_temperature
                    - config.initial_temperature
                )
                full_bond_peak_index = int(np.argmax(bond_series))
                full_structural_peak_index = int(np.argmax(structural_series))
                solve_rows.append({
                    "sigma_t_s": sigma_t,
                    "scenario_id": scenario.case_id,
                    "event_count": scenario.event_count,
                    "defect_count": scenario.defect_count,
                    "d_tps": thickness,
                    "reference_amplitude_W_m2": reference_amplitude,
                    "full_horizon_s": maximum_horizon,
                    "exact_hot_face_max_K": (
                        trajectory.maximum_hot_face_temperature
                    ),
                    "full_bond_max_K": float(np.max(bond_series)),
                    "full_structural_interface_max_K": float(
                        np.max(structural_series)
                    ),
                    "full_bond_peak_time_s": float(
                        times[full_bond_peak_index]
                    ),
                    "full_structural_peak_time_s": float(
                        times[full_structural_peak_index]
                    ),
                    "full_bond_peak_at_endpoint": bool(
                        full_bond_peak_index == len(times) - 1
                    ),
                    "full_structural_peak_at_endpoint": bool(
                        full_structural_peak_index == len(times) - 1
                    ),
                    "factorization_count": trajectory.factorization_count,
                    "nonlinear_iterations_max": (
                        trajectory.max_nonlinear_iterations
                    ),
                    "solve_seconds": solve_seconds,
                })

                for horizon in candidate_horizons:
                    end = horizon_indices[horizon]
                    horizon_bond = float(np.max(bond_series[:end + 1]))
                    horizon_structural = float(
                        np.max(structural_series[:end + 1])
                    )
                    q_final = float(np.max(heat_samples[end]))
                    q_final_fraction = (
                        0.0 if heat_peak == 0.0 else q_final / heat_peak
                    )
                    response_rows.append({
                        "sigma_t_s": sigma_t,
                        "horizon_s": horizon,
                        "scenario_id": scenario.case_id,
                        "event_count": scenario.event_count,
                        "defect_count": scenario.defect_count,
                        "d_tps": thickness,
                        "reference_amplitude_W_m2": reference_amplitude,
                        "hot_rise_per_amplitude_K_per_W_m2": (
                            exact_hot_rise / reference_amplitude
                        ),
                        "bond_rise_per_amplitude_K_per_W_m2": (
                            (horizon_bond - config.initial_temperature)
                            / reference_amplitude
                        ),
                        "structural_rise_per_amplitude_K_per_W_m2": (
                            (horizon_structural - config.initial_temperature)
                            / reference_amplitude
                        ),
                        "heating_final_fraction": q_final_fraction,
                        "heating_ended": bool(q_final_fraction <= 1e-3),
                        "bond_full_response_fraction": (
                            (horizon_bond - config.initial_temperature)
                            / max(
                                float(np.max(bond_series))
                                - config.initial_temperature,
                                1e-15,
                            )
                        ),
                        "structural_full_response_fraction": (
                            (horizon_structural - config.initial_temperature)
                            / max(
                                float(np.max(structural_series))
                                - config.initial_temperature,
                                1e-15,
                            )
                        ),
                    })

    candidates: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for sigma_t in widths:
        width_rows = [
            row for row in response_rows
            if np.isclose(row["sigma_t_s"], sigma_t)
        ]
        for horizon in candidate_horizons:
            rows = [
                row for row in width_rows
                if np.isclose(row["horizon_s"], horizon)
            ]
            interval = _amplitude_interval(
                rows,
                initial_temperature=config.initial_temperature,
                hot_face_limit=config.hot_face_temperature_limit,
                bond_limit=config.bond_temperature_limit,
                structural_limit=config.structural_temperature_limit,
                thinnest=screening_thicknesses[0],
                thickest=screening_thicknesses[-1],
            )
            heating_ended = all(row["heating_ended"] for row in rows)
            minimum_response_coverage = min(
                min(
                    float(row["bond_full_response_fraction"]),
                    float(row["structural_full_response_fraction"]),
                )
                for row in rows
            )
            matrix = _matrix_at_amplitude(
                rows,
                interval["recommended_amplitude_W_m2"],
                initial_temperature=config.initial_temperature,
                hot_face_limit=config.hot_face_temperature_limit,
                bond_limit=config.bond_temperature_limit,
                structural_limit=config.structural_temperature_limit,
                thicknesses=screening_thicknesses,
            )
            for matrix_row in matrix["rows"]:
                matrix_rows.append({
                    "sigma_t_s": sigma_t,
                    "horizon_s": horizon,
                    "amplitude_W_m2": (
                        interval["recommended_amplitude_W_m2"]
                    ),
                    **matrix_row,
                })
            transient_duration = 7.0 * sigma_t
            candidate = {
                "sigma_t_s": sigma_t,
                "single_event_fwhm_s": 2.355 * sigma_t,
                "screen_family_duration_s": transient_duration,
                "horizon_s": horizon,
                "heating_ended": heating_ended,
                "minimum_full_response_fraction": minimum_response_coverage,
                "pulse_is_transient_within_horizon": bool(
                    transient_duration <= 0.60 * horizon
                ),
                **interval,
                "recommended_matrix": {
                    key: value
                    for key, value in matrix.items()
                    if key != "rows"
                },
            }
            candidate["screen_gate_passed"] = bool(
                candidate["comfortable_interval"]
                and heating_ended
                and candidate["pulse_is_transient_within_horizon"]
                and matrix["all_cells_within_validity"]
                and matrix["thinnest_design_sometimes_infeasible"]
                and matrix["thickest_design_all_feasible"]
                and matrix["interior_minimum_feasible_scenario_count"] > 0
                and matrix["minimum_feasible_thickness_varies"]
            )
            candidates.append(candidate)

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            bool(candidate["screen_gate_passed"]),
            bool(candidate["comfortable_interval"]),
            bool(candidate["heating_ended"]),
            float(candidate["amplitude_interval_width_W_m2"]),
            float(candidate["minimum_full_response_fraction"]),
            -float(candidate["horizon_s"]),
        ),
        reverse=True,
    )
    passing = [candidate for candidate in ranked if candidate["screen_gate_passed"]]
    linear_candidates = [
        candidate
        for candidate in ranked
        if (
            candidate["comfortable_interval"]
            and candidate["heating_ended"]
            and candidate["pulse_is_transient_within_horizon"]
        )
    ]
    if passing:
        conclusion = (
            "At least one wider-pulse, longer-observation candidate passed the "
            "reduced screen. Run the full 45-cell pilot for the best one or two "
            "pulse widths before accepting this physical configuration."
        )
    elif linear_candidates:
        conclusion = (
            "At least one robust linear amplitude interval exists, but its "
            "three-thickness reduced matrix did not satisfy every decision-"
            "diversity gate. Inspect the best reduced matrix before deciding "
            "whether to run the full 45-cell pilot."
        )
    elif np.isclose(config.backing_thickness, 0.002):
        conclusion = (
            "No candidate passed the reduced screen with the 2 mm backing and "
            "current linear prescribed-net-flux physics. Proceed to the bounded "
            "0.5/1.0/2.0 mm backing-thickness screen."
        )
    else:
        conclusion = (
            "No candidate produced a robust valid amplitude interval for this "
            f"{1000.0 * config.backing_thickness:g} mm backing under the current "
            "linear prescribed-net-flux physics."
        )
    result = {
        "study_id": config.study_id,
        "study_config_sha256": config.sha256,
        "screen_type": "wider_pulse_longer_observation_linear_amplitude_interval",
        "material_properties": config.property_provenance,
        "governing_model_unchanged": {
            "backing_thickness_m": config.backing_thickness,
            "surface_model": config.heating.interpretation,
            "radiation_enabled": False,
            "max_hot_face_temperature_K": (
                config.hot_face_temperature_limit
            ),
        },
        "settings": {
            "pulse_widths_s": list(widths),
            "horizons_s": list(candidate_horizons),
            "screening_thicknesses_m": list(screening_thicknesses),
            "reference_amplitude_per_event_W_m2": reference_amplitude,
            "dt_s": time_config.dt,
            "save_interval_s": save_interval,
            "maximum_horizon_s": maximum_horizon,
            "heating_ended_fraction": 1e-3,
            "comfortable_relative_interval_width": 0.10,
            "amplitude_semantics": (
                "Every event in a scenario has the reported per-event peak "
                "amplitude; all events are scaled together."
            ),
        },
        "scenario_design": [
            {
                "scenario_id": scenario.case_id,
                "description": SCREEN_SCENARIO_DESCRIPTIONS[scenario.case_id],
                "event_count": scenario.event_count,
                "defect_count": scenario.defect_count,
                "lateral_event_centers_m": [
                    event.y_center for event in scenario.heating_events
                ],
                "event_center_multiples_of_sigma_t": [
                    event.t_center / widths[0]
                    for event in scenario.heating_events
                ],
            }
            for scenario in _screen_scenarios(
                screen_config,
                widths[0],
                reference_amplitude,
            )
        ],
        "reference_solve_count": len(solve_rows),
        "expected_reference_solve_count": (
            len(widths) * 3 * len(screening_thicknesses)
        ),
        "candidate_horizon_count": len(candidates),
        "separate_solves_avoided_by_trajectory_reuse": (
            len(solve_rows) * (len(candidate_horizons) - 1)
        ),
        "viable_candidate_count": len(passing),
        "robust_linear_interval_candidate_count": len(linear_candidates),
        "viable_short_transient_found": bool(passing),
        "best_candidate": ranked[0] if ranked else None,
        "passing_candidates": passing,
        "candidates": candidates,
        "reference_solves": solve_rows,
        "wall_seconds": time.perf_counter() - start,
        "conclusion": conclusion,
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "pulse_reference_solves.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(solve_rows[0]))
        writer.writeheader()
        writer.writerows(solve_rows)
    with (destination / "pulse_response_matrix.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(response_rows[0]))
        writer.writeheader()
        writer.writerows(response_rows)
    if matrix_rows:
        with (destination / "pulse_candidate_matrices.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]))
            writer.writeheader()
            writer.writerows(matrix_rows)
    candidate_rows = [
        {
            key: value
            for key, value in candidate.items()
            if key != "recommended_matrix"
            and not isinstance(value, dict)
        }
        for candidate in candidates
    ]
    with (destination / "pulse_amplitude_intervals.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    (destination / "pulse_screen_report.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def _direct_matrix_summary(
    rows: list[dict[str, Any]],
    *,
    thicknesses: tuple[float, ...],
) -> dict[str, Any]:
    scenario_ids = sorted({str(row["scenario_id"]) for row in rows})
    minimum_feasible: dict[str, float | None] = {}
    for scenario_id in scenario_ids:
        by_thickness = {
            float(row["d_tps"]): bool(
                row["within_validity"] and row["feasible"]
            )
            for row in rows
            if row["scenario_id"] == scenario_id
        }
        minimum_feasible[scenario_id] = next(
            (
                thickness
                for thickness in thicknesses
                if by_thickness.get(thickness, False)
            ),
            None,
        )
    selections = [
        value for value in minimum_feasible.values()
        if value is not None
    ]
    controls = {
        constraint: sum(
            row["controlling_constraint"] == constraint
            for row in rows
        )
        for constraint in ("bond", "structural_interface")
    }
    return {
        "cell_count": len(rows),
        "valid_cell_count": sum(row["within_validity"] for row in rows),
        "all_cells_within_validity": all(
            row["within_validity"] for row in rows
        ),
        "feasible_fraction": float(np.mean([
            row["feasible"] for row in rows
        ])),
        "valid_and_feasible_fraction": float(np.mean([
            row["within_validity"] and row["feasible"] for row in rows
        ])),
        "maximum_hot_face_temperature_K": float(max(
            row["hot_face_max_K"] for row in rows
        )),
        "thinnest_design_sometimes_infeasible": any(
            row["within_validity"]
            and not row.get("design_constraints_satisfied", row["feasible"])
            for row in rows
            if np.isclose(row["d_tps"], thicknesses[0])
        ),
        "thickest_design_all_feasible": all(
            row["feasible"]
            for row in rows
            if np.isclose(row["d_tps"], thicknesses[-1])
        ),
        "interior_minimum_feasible_scenario_count": sum(
            value is not None
            and value not in {thicknesses[0], thicknesses[-1]}
            for value in minimum_feasible.values()
        ),
        "minimum_feasible_thickness_by_scenario": minimum_feasible,
        "minimum_feasible_thickness_varies": len(set(selections)) > 1,
        "controlling_constraint_counts": controls,
        "both_constraints_control_some_cells": all(
            value > 0 for value in controls.values()
        ),
        "mean_reradiated_energy_fraction": float(np.mean([
            row["reradiated_energy_fraction"] for row in rows
        ])),
        "maximum_nonlinear_iterations": int(max(
            row["maximum_nonlinear_iterations"] for row in rows
        )),
    }


def evaluate_stage6a_promotion(
    refined_configurations: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the five-condition nonlinear-pilot promotion rule.

    Records are intentionally explicit: amplitudes must come from boundary
    refinement at ``dt`` and ``dt/2`` rather than from the coarse four-point
    screen.
    """
    evaluations: list[dict[str, Any]] = []
    for record in refined_configurations:
        thin = record["thin_at_A_min"]
        thick = record["thick_at_A_min"]
        a_min = float(record["A_min_dt"])
        a_max = float(record["A_max_dt"])
        a_min_half = float(record["A_min_half_dt"])
        a_max_half = float(record["A_max_half_dt"])

        thin_genuine_failure = bool(
            thin.get("within_validity", True)
            and (
                thin["bond_max_K"] > record["bond_limit_K"]
                or thin["structural_interface_max_K"]
                > record["structural_limit_K"]
            )
        )
        thick_feasible = bool(
            thick.get("within_validity", True)
            and thick["bond_max_K"] <= record["bond_limit_K"]
            and thick["structural_interface_max_K"]
            <= record["structural_limit_K"]
        )
        material_domain = bool(
            thin.get(
                "within_validity",
                thin["accepted_range_excursions"] == 0
                and thin["hot_face_max_K"] <= record["hot_face_limit_K"],
            )
            and thick.get(
                "within_validity",
                thick["accepted_range_excursions"] == 0
                and thick["hot_face_max_K"] <= record["hot_face_limit_K"],
            )
        )
        band_relative_width = (a_max - a_min) / a_min
        half_band_relative_width = (
            (a_max_half - a_min_half) / a_min_half
        )
        robust_band = bool(
            band_relative_width >= 0.10
            and half_band_relative_width >= 0.10
        )
        amplitude_convergence = bool(
            abs(a_min_half - a_min) / a_min_half < 0.02
            and abs(a_max_half - a_max) / a_max_half < 0.02
        )
        coarse_interior = record["interior_dt"]
        fine_interior = record["interior_half_dt"]
        classification_unchanged = bool(
            coarse_interior["thin_feasible"]
            == fine_interior["thin_feasible"]
            and coarse_interior["thick_feasible"]
            == fine_interior["thick_feasible"]
        )
        temperature_convergence = True
        for design in ("thin", "thick"):
            for metric in (
                "hot_face_max_K",
                "bond_max_K",
                "structural_interface_max_K",
            ):
                coarse_value = float(coarse_interior[design][metric])
                fine_value = float(fine_interior[design][metric])
                tolerance = max(0.5, 0.0025 * abs(fine_value))
                temperature_convergence &= (
                    abs(coarse_value - fine_value) < tolerance
                )
        timestep_convergence = bool(
            amplitude_convergence
            and classification_unchanged
            and temperature_convergence
            and half_band_relative_width >= 0.10
        )
        conditions = {
            "1_thin_genuine_bond_or_structural_failure": (
                thin_genuine_failure
            ),
            "2_thick_feasible_at_same_amplitude": thick_feasible,
            "3_both_inside_material_domain": material_domain,
            "4_refined_band_relative_width_at_least_10_percent": (
                robust_band
            ),
            "5_boundary_converges_under_dt_half": timestep_convergence,
        }
        evaluations.append({
            "configuration_id": record.get("configuration_id"),
            "A_min_dt": a_min,
            "A_max_dt": a_max,
            "band_relative_width": band_relative_width,
            "A_min_half_dt": a_min_half,
            "A_max_half_dt": a_max_half,
            "half_dt_band_relative_width": half_band_relative_width,
            "conditions": conditions,
            "passed": all(conditions.values()),
            "failed_conditions": [
                name for name, passed in conditions.items() if not passed
            ],
        })
    passing = [row for row in evaluations if row["passed"]]
    return {
        "evaluated_configuration_count": len(evaluations),
        "passing_configuration_count": len(passing),
        "promote_to_stage_6b": bool(passing),
        "evaluations": evaluations,
        "verdict": (
            "promote"
            if passing
            else (
                "negative_result"
                if evaluations
                else "refinement_not_run"
            )
        ),
    }


def _incident_design_result(
    config: StudyConfig,
    *,
    sigma_t: float,
    amplitude: float,
    scenario_id: str,
    thickness: float,
    horizon: float,
    dt: float,
) -> dict[str, Any]:
    save_interval = config.time.dt * config.time.save_stride
    run_config = replace(
        config,
        time=TimeConfig(
            dt=dt,
            t_final=horizon,
            save_stride=max(1, int(round(save_interval / dt))),
            horizon_candidates=(horizon,),
        ),
    )
    scenario = next(
        candidate
        for candidate in _screen_scenarios(run_config, sigma_t, amplitude)
        if candidate.case_id == scenario_id
    )
    case = replace(
        scenario,
        case_id=(
            f"{scenario_id}@refine@A={amplitude:.12g}"
            f"@dt={dt:.12g}@d={thickness:.12g}"
        ),
        d_tps=thickness,
    )
    try:
        solver = TPSFVSolver(run_config, case)
        trajectory = solver.solve()
    except (PropertyRangeError, RuntimeError) as exc:
        return {
            "amplitude_W_m2": amplitude,
            "d_tps": thickness,
            "hot_face_max_K": float("inf"),
            "bond_max_K": float("inf"),
            "structural_interface_max_K": float("inf"),
            "accepted_range_excursions": 1,
            "within_validity": False,
            "feasible": False,
            "solver_error": str(exc),
        }
    qoi = solver.quantities_of_interest(
        trajectory.temperatures,
        trajectory.times,
    )
    acceptance = assess_trajectory(
        run_config,
        trajectory,
        bond_max=qoi["bond_max"],
        structural_interface_max=qoi["structural_interface_max"],
    )
    return {
        "amplitude_W_m2": amplitude,
        "d_tps": thickness,
        "hot_face_max_K": trajectory.maximum_hot_face_temperature,
        "bond_max_K": qoi["bond_max"],
        "structural_interface_max_K": qoi["structural_interface_max"],
        "accepted_range_excursions": trajectory.accepted_range_excursions,
        "iteration_range_clamps": trajectory.iteration_range_clamps,
        "within_validity": acceptance.valid,
        "classification": acceptance.classification,
        "invalid_reasons": list(acceptance.invalid_reasons),
        "design_violations": list(acceptance.design_violations),
        "design_constraints_satisfied": not bool(
            acceptance.design_violations
        ),
        "feasible": bool(acceptance.feasible),
        "nonlinear_iterations_max": trajectory.max_nonlinear_iterations,
        "final_nonlinear_residual_norm_K": (
            trajectory.final_nonlinear_residual_norm
        ),
        "maximum_relative_energy_residual": (
            acceptance.relative_energy_residual
        ),
    }


def _refine_boolean_transition(
    evaluate: Callable[[float], dict[str, Any]],
    lower: float,
    upper: float,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    lower_is_true: bool,
    relative_tolerance: float = 2.0e-3,
    max_iterations: int = 12,
) -> tuple[float, dict[str, Any]]:
    lower_result = evaluate(lower)
    upper_result = evaluate(upper)
    if predicate(lower_result) != lower_is_true:
        raise ValueError("Refinement lower endpoint has the wrong classification.")
    if predicate(upper_result) == lower_is_true:
        raise ValueError("Refinement interval does not bracket a transition.")
    for _ in range(max_iterations):
        if (upper - lower) / max(upper, 1.0e-30) <= relative_tolerance:
            break
        midpoint = 0.5 * (lower + upper)
        midpoint_result = evaluate(midpoint)
        if predicate(midpoint_result) == lower_is_true:
            lower, lower_result = midpoint, midpoint_result
        else:
            upper, upper_result = midpoint, midpoint_result
    return (
        (lower, lower_result)
        if lower_is_true
        else (upper, upper_result)
    )


def _run_stage6a_refinement(
    config: StudyConfig,
    response_rows: list[dict[str, Any]],
    *,
    widths: tuple[float, ...],
    amplitudes: tuple[float, ...],
    horizons: tuple[float, ...],
    thicknesses: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(amplitudes) < 2:
        return [], [{
            "status": "not_run",
            "reason": "At least two coarse amplitudes are required to bracket.",
        }]
    thin_thickness, thick_thickness = thicknesses[0], thicknesses[-1]
    records: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    scenario_ids = sorted({str(row["scenario_id"]) for row in response_rows})

    for sigma_t in widths:
        for horizon in horizons:
            for scenario_id in scenario_ids:
                selected = [
                    row
                    for row in response_rows
                    if (
                        np.isclose(row["sigma_t_s"], sigma_t)
                        and np.isclose(row["horizon_s"], horizon)
                        and row["scenario_id"] == scenario_id
                    )
                ]

                def rows_for(thickness: float) -> list[dict[str, Any]]:
                    return sorted(
                        (
                            row
                            for row in selected
                            if np.isclose(row["d_tps"], thickness)
                        ),
                        key=lambda row: row[
                            "incident_amplitude_per_event_W_m2"
                        ],
                    )

                thin_rows = rows_for(thin_thickness)
                thick_rows = rows_for(thick_thickness)
                thin_bracket = next(
                    (
                        (
                            float(thin_rows[index - 1][
                                "incident_amplitude_per_event_W_m2"
                            ]),
                            float(thin_rows[index][
                                "incident_amplitude_per_event_W_m2"
                            ]),
                        )
                        for index in range(1, len(thin_rows))
                        if (
                            thin_rows[index - 1]["feasible"]
                            and thin_rows[index]["within_validity"]
                            and not thin_rows[index].get(
                                "design_constraints_satisfied",
                                thin_rows[index]["feasible"],
                            )
                        )
                    ),
                    None,
                )
                thick_bracket = next(
                    (
                        (
                            float(thick_rows[index][
                                "incident_amplitude_per_event_W_m2"
                            ]),
                            float(thick_rows[index + 1][
                                "incident_amplitude_per_event_W_m2"
                            ]),
                        )
                        for index in range(len(thick_rows) - 1)
                        if (
                            thick_rows[index]["within_validity"]
                            and thick_rows[index]["feasible"]
                            and not (
                                thick_rows[index + 1]["within_validity"]
                                and thick_rows[index + 1]["feasible"]
                            )
                        )
                    ),
                    None,
                )
                configuration_id = (
                    f"sigma={sigma_t:g}|horizon={horizon:g}|{scenario_id}"
                )
                if thin_bracket is None or thick_bracket is None:
                    attempts.append({
                        "configuration_id": configuration_id,
                        "status": "not_bracketed",
                        "thin_bracket": thin_bracket,
                        "thick_bracket": thick_bracket,
                    })
                    continue

                def refine_at(
                    dt: float,
                ) -> tuple[float, dict[str, Any], float]:
                    cache: dict[tuple[float, float], dict[str, Any]] = {}

                    def design(amplitude: float, thickness: float) -> dict[str, Any]:
                        key = (float(amplitude), float(thickness))
                        if key not in cache:
                            cache[key] = _incident_design_result(
                                config,
                                sigma_t=sigma_t,
                                amplitude=amplitude,
                                scenario_id=scenario_id,
                                thickness=thickness,
                                horizon=horizon,
                                dt=dt,
                            )
                        return cache[key]

                    a_min, thin_at_min = _refine_boolean_transition(
                        lambda value: design(value, thin_thickness),
                        *thin_bracket,
                        lambda result: (
                            result["within_validity"]
                            and not result.get(
                                "design_constraints_satisfied",
                                result["feasible"],
                            )
                        ),
                        lower_is_true=False,
                    )
                    a_max, _ = _refine_boolean_transition(
                        lambda value: design(value, thick_thickness),
                        *thick_bracket,
                        lambda result: (
                            result["within_validity"] and result["feasible"]
                        ),
                        lower_is_true=True,
                    )
                    return a_min, thin_at_min, a_max

                try:
                    a_min, thin_at_min, a_max = refine_at(config.time.dt)
                    a_min_half, _, a_max_half = refine_at(
                        0.5 * config.time.dt
                    )
                except (RuntimeError, ValueError) as exc:
                    attempts.append({
                        "configuration_id": configuration_id,
                        "status": "refinement_failed",
                        "reason": str(exc),
                    })
                    continue

                def result(amplitude: float, thickness: float, dt: float) -> dict[str, Any]:
                    return _incident_design_result(
                        config,
                        sigma_t=sigma_t,
                        amplitude=amplitude,
                        scenario_id=scenario_id,
                        thickness=thickness,
                        horizon=horizon,
                        dt=dt,
                    )

                thick_at_min = result(
                    a_min,
                    thick_thickness,
                    config.time.dt,
                )
                interior_amplitude = 0.5 * (a_min + a_max)
                interior_dt = {
                    "thin": result(
                        interior_amplitude,
                        thin_thickness,
                        config.time.dt,
                    ),
                    "thick": result(
                        interior_amplitude,
                        thick_thickness,
                        config.time.dt,
                    ),
                }
                interior_half = {
                    "thin": result(
                        interior_amplitude,
                        thin_thickness,
                        0.5 * config.time.dt,
                    ),
                    "thick": result(
                        interior_amplitude,
                        thick_thickness,
                        0.5 * config.time.dt,
                    ),
                }
                records.append({
                    "configuration_id": configuration_id,
                    "thin_at_A_min": thin_at_min,
                    "thick_at_A_min": thick_at_min,
                    "bond_limit_K": config.bond_temperature_limit,
                    "structural_limit_K": config.structural_temperature_limit,
                    "hot_face_limit_K": (
                        config.hot_face_temperature_limit
                    ),
                    "A_min_dt": a_min,
                    "A_max_dt": a_max,
                    "A_min_half_dt": a_min_half,
                    "A_max_half_dt": a_max_half,
                    "interior_dt": {
                        **interior_dt,
                        "thin_feasible": interior_dt["thin"]["feasible"],
                        "thick_feasible": interior_dt["thick"]["feasible"],
                    },
                    "interior_half_dt": {
                        **interior_half,
                        "thin_feasible": interior_half["thin"]["feasible"],
                        "thick_feasible": interior_half["thick"]["feasible"],
                    },
                })
                attempts.append({
                    "configuration_id": configuration_id,
                    "status": "refined",
                    "thin_bracket": thin_bracket,
                    "thick_bracket": thick_bracket,
                })
    return records, attempts


def run_incident_radiation_screen(
    config: StudyConfig,
    output_dir: str | Path,
    *,
    pulse_widths: Iterable[float] = DEFAULT_INCIDENT_PULSE_WIDTHS,
    amplitudes: Iterable[float] = DEFAULT_INCIDENT_AMPLITUDES,
    horizons: Iterable[float] = DEFAULT_INCIDENT_HORIZONS,
    thicknesses: Iterable[float] = DEFAULT_SCREENING_THICKNESSES,
) -> dict[str, Any]:
    """Directly screen incident amplitudes for the nonlinear radiation model."""
    if not config.surface.reradiation_enabled:
        raise ValueError(
            "incident-screen requires surface.model="
            "'incident_heat_flux_with_reradiation'."
        )
    widths = _positive_sorted_unique(pulse_widths, name="pulse_widths")
    incident_amplitudes = _positive_sorted_unique(
        amplitudes,
        name="amplitudes",
    )
    candidate_horizons = _positive_sorted_unique(horizons, name="horizons")
    screening_thicknesses = _positive_sorted_unique(
        thicknesses,
        name="thicknesses",
    )
    if len(screening_thicknesses) < 2:
        raise ValueError("Incident screening requires at least two thicknesses.")
    for thickness in screening_thicknesses:
        if thickness not in config.thickness_candidates:
            raise ValueError(
                f"Screening thickness {thickness:g} is not a configured candidate."
            )

    maximum_horizon = candidate_horizons[-1]
    time_config = TimeConfig(
        dt=config.time.dt,
        t_final=maximum_horizon,
        save_stride=config.time.save_stride,
        horizon_candidates=candidate_horizons,
    )
    save_interval = time_config.dt * time_config.save_stride
    for horizon in candidate_horizons:
        if abs(horizon / save_interval - round(horizon / save_interval)) > 1e-10:
            raise ValueError(
                f"Horizon {horizon:g} is not aligned with saved interval "
                f"{save_interval:g}."
            )
    screen_config = replace(config, time=time_config)
    times = np.asarray(time_config.saved_times(), dtype=np.float64)
    horizon_indices = {
        horizon: int(np.flatnonzero(np.isclose(times, horizon))[0])
        for horizon in candidate_horizons
    }

    solve_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for sigma_t in widths:
        for amplitude in incident_amplitudes:
            scenarios = _screen_scenarios(
                screen_config,
                sigma_t,
                amplitude,
            )
            for scenario in scenarios:
                for thickness in screening_thicknesses:
                    case = replace(
                        scenario,
                        case_id=(
                            f"{scenario.case_id}@sigma={sigma_t:g}"
                            f"@A={amplitude:g}@d={thickness:g}"
                        ),
                        d_tps=thickness,
                    )
                    solve_start = time.perf_counter()
                    solver = TPSFVSolver(screen_config, case)
                    trajectory = solver.solve(save_times=times)
                    solve_seconds = time.perf_counter() - solve_start
                    _, bond_series, structural_series = _series_for_trajectory(
                        solver,
                        trajectory.temperatures,
                    )
                    heat_samples = np.stack([
                        case.heat_flux(
                            solver.grid.y_centers,
                            float(sample_time),
                        )
                        for sample_time in times
                    ])
                    heat_peak = float(np.max(heat_samples))
                    trajectory_acceptance = assess_trajectory(
                        screen_config,
                        trajectory,
                    )
                    solve_rows.append({
                        "sigma_t_s": sigma_t,
                        "incident_amplitude_per_event_W_m2": amplitude,
                        "scenario_id": scenario.case_id,
                        "event_count": scenario.event_count,
                        "defect_count": scenario.defect_count,
                        "d_tps": thickness,
                        "full_horizon_s": maximum_horizon,
                        "maximum_hot_face_temperature_K": (
                            trajectory.maximum_hot_face_temperature
                        ),
                        "boundary_input_energy_J": float(
                            trajectory.boundary_input_energy[-1]
                        ),
                        "reradiated_energy_J": float(
                            trajectory.radiated_energy[-1]
                        ),
                        "net_boundary_energy_J": float(
                            trajectory.net_boundary_energy[-1]
                        ),
                        "reradiated_energy_fraction": float(
                            trajectory.radiated_energy[-1]
                            / max(trajectory.boundary_input_energy[-1], 1e-15)
                        ),
                        "maximum_nonlinear_iterations": int(
                            np.max(
                                trajectory.nonlinear_iteration_counts,
                                initial=0,
                            )
                        ),
                        "factorization_count": trajectory.factorization_count,
                        "nonlinear_iterations_max": (
                            trajectory.max_nonlinear_iterations
                        ),
                        "linear_solves": trajectory.linear_solve_count,
                        "accepted_range_excursions": (
                            trajectory.accepted_range_excursions
                        ),
                        "iteration_range_clamps": (
                            trajectory.iteration_range_clamps
                        ),
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
                            trajectory_acceptance.relative_energy_residual
                        ),
                        "solve_seconds": solve_seconds,
                    })
                    for horizon in candidate_horizons:
                        saved_end = horizon_indices[horizon]
                        internal_end = int(round(horizon / time_config.dt))
                        bond_max = float(np.max(bond_series[:saved_end + 1]))
                        structural_max = float(
                            np.max(structural_series[:saved_end + 1])
                        )
                        q_final = float(np.max(heat_samples[saved_end]))
                        q_fraction = (
                            0.0 if heat_peak == 0.0 else q_final / heat_peak
                        )
                        bond_utilization = (
                            (bond_max - config.initial_temperature)
                            / (
                                config.bond_temperature_limit
                                - config.initial_temperature
                            )
                        )
                        structural_utilization = (
                            (structural_max - config.initial_temperature)
                            / (
                                config.structural_temperature_limit
                                - config.initial_temperature
                            )
                        )
                        acceptance = assess_trajectory(
                            screen_config,
                            trajectory,
                            bond_max=bond_max,
                            structural_interface_max=structural_max,
                        )
                        response_rows.append({
                            "sigma_t_s": sigma_t,
                            "incident_amplitude_per_event_W_m2": amplitude,
                            "horizon_s": horizon,
                            "scenario_id": scenario.case_id,
                            "event_count": scenario.event_count,
                            "defect_count": scenario.defect_count,
                            "d_tps": thickness,
                            "hot_face_max_K": (
                                trajectory.maximum_hot_face_temperature
                            ),
                            "bond_max_K": bond_max,
                            "structural_interface_max_K": structural_max,
                            "within_validity": acceptance.valid,
                            "classification": acceptance.classification,
                            "invalid_reasons": ";".join(
                                acceptance.invalid_reasons
                            ),
                            "design_violations": ";".join(
                                acceptance.design_violations
                            ),
                            "accepted_range_excursions": (
                                trajectory.accepted_range_excursions
                            ),
                            "iteration_range_clamps": (
                                trajectory.iteration_range_clamps
                            ),
                            "design_constraints_satisfied": not bool(
                                acceptance.design_violations
                            ),
                            "feasible": bool(acceptance.feasible),
                            "controlling_constraint": (
                                "bond"
                                if bond_utilization >= structural_utilization
                                else "structural_interface"
                            ),
                            "heating_final_fraction": q_fraction,
                            "heating_ended": bool(q_fraction <= 1e-3),
                            "boundary_input_energy_J": float(
                                trajectory.boundary_input_energy[internal_end]
                            ),
                            "reradiated_energy_J": float(
                                trajectory.radiated_energy[internal_end]
                            ),
                            "net_boundary_energy_J": float(
                                trajectory.net_boundary_energy[internal_end]
                            ),
                            "reradiated_energy_fraction": float(
                                trajectory.radiated_energy[internal_end]
                                / max(
                                    trajectory.boundary_input_energy[internal_end],
                                    1e-15,
                                )
                            ),
                            "maximum_nonlinear_iterations": int(
                                np.max(
                                    trajectory.nonlinear_iteration_counts[
                                        :internal_end
                                    ],
                                    initial=0,
                                )
                            ),
                            "final_nonlinear_residual_norm_K": (
                                trajectory.final_nonlinear_residual_norm
                            ),
                            "maximum_relative_energy_residual": (
                                acceptance.relative_energy_residual
                            ),
                        })

    monotonicity_checks: list[dict[str, Any]] = []
    for sigma_t in widths:
        for horizon in candidate_horizons:
            for scenario_id in sorted({
                str(row["scenario_id"]) for row in response_rows
            }):
                for thickness in screening_thicknesses:
                    series = sorted(
                        (
                            row
                            for row in response_rows
                            if (
                                np.isclose(row["sigma_t_s"], sigma_t)
                                and np.isclose(row["horizon_s"], horizon)
                                and row["scenario_id"] == scenario_id
                                and np.isclose(row["d_tps"], thickness)
                            )
                        ),
                        key=lambda row: row[
                            "incident_amplitude_per_event_W_m2"
                        ],
                    )
                    metric_checks = {}
                    for metric in (
                        "hot_face_max_K",
                        "bond_max_K",
                        "structural_interface_max_K",
                    ):
                        values = np.asarray(
                            [float(row[metric]) for row in series],
                            dtype=np.float64,
                        )
                        tolerance = (
                            1.0e-8
                            * max(1.0, float(np.max(np.abs(values))))
                        )
                        metric_checks[metric] = bool(
                            np.all(np.diff(values) >= -tolerance)
                        )
                    monotonicity_checks.append({
                        "sigma_t_s": sigma_t,
                        "horizon_s": horizon,
                        "scenario_id": scenario_id,
                        "d_tps": thickness,
                        "amplitudes_W_m2": [
                            row["incident_amplitude_per_event_W_m2"]
                            for row in series
                        ],
                        "metrics": metric_checks,
                        "passed": all(metric_checks.values()),
                    })

    candidates: list[dict[str, Any]] = []
    for sigma_t in widths:
        for amplitude in incident_amplitudes:
            for horizon in candidate_horizons:
                rows = [
                    row for row in response_rows
                    if (
                        np.isclose(row["sigma_t_s"], sigma_t)
                        and np.isclose(
                            row["incident_amplitude_per_event_W_m2"],
                            amplitude,
                        )
                        and np.isclose(row["horizon_s"], horizon)
                    )
                ]
                matrix = _direct_matrix_summary(
                    rows,
                    thicknesses=screening_thicknesses,
                )
                heating_ended = all(row["heating_ended"] for row in rows)
                transient_duration = 7.0 * sigma_t
                monotonic_peak_response = all(
                    check["passed"]
                    for check in monotonicity_checks
                    if (
                        np.isclose(check["sigma_t_s"], sigma_t)
                        and np.isclose(check["horizon_s"], horizon)
                    )
                )
                candidate = {
                    "sigma_t_s": sigma_t,
                    "single_event_fwhm_s": 2.355 * sigma_t,
                    "screen_family_duration_s": transient_duration,
                    "incident_amplitude_per_event_W_m2": amplitude,
                    "horizon_s": horizon,
                    "heating_ended": heating_ended,
                    "pulse_is_transient_within_horizon": bool(
                        transient_duration <= 0.60 * horizon
                    ),
                    "monotonic_peak_response_across_amplitudes": (
                        monotonic_peak_response
                    ),
                    **matrix,
                }
                candidate["screen_gate_passed"] = bool(
                    heating_ended
                    and candidate["pulse_is_transient_within_horizon"]
                    and monotonic_peak_response
                    and matrix["all_cells_within_validity"]
                    and matrix["thinnest_design_sometimes_infeasible"]
                    and matrix["thickest_design_all_feasible"]
                    and matrix["interior_minimum_feasible_scenario_count"] > 0
                    and matrix["minimum_feasible_thickness_varies"]
                )
                candidates.append(candidate)

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            bool(candidate["screen_gate_passed"]),
            bool(candidate["all_cells_within_validity"]),
            bool(candidate["thinnest_design_sometimes_infeasible"]),
            bool(candidate["thickest_design_all_feasible"]),
            int(candidate["interior_minimum_feasible_scenario_count"]),
            bool(candidate["minimum_feasible_thickness_varies"]),
            -abs(float(candidate["feasible_fraction"]) - 0.50),
            -abs(
                float(candidate["maximum_hot_face_temperature_K"])
                - config.hot_face_temperature_limit
            ),
            -float(candidate["horizon_s"]),
        ),
        reverse=True,
    )
    passing = [candidate for candidate in ranked if candidate["screen_gate_passed"]]
    if config.validity.tps_property_model == "temperature_dependent_table":
        refined_records, refinement_attempts = _run_stage6a_refinement(
            config,
            response_rows,
            widths=widths,
            amplitudes=incident_amplitudes,
            horizons=candidate_horizons,
            thicknesses=screening_thicknesses,
        )
    else:
        refined_records = []
        refinement_attempts = [{
            "status": "not_applicable",
            "reason": (
                "Stage-6A promotion is gated on the temperature-dependent "
                "property model."
            ),
        }]
    stage6a_promotion = evaluate_stage6a_promotion(refined_records)
    if (
        config.validity.tps_property_model == "temperature_dependent_table"
        and refinement_attempts
        and not refined_records
    ):
        stage6a_promotion["verdict"] = "negative_result"
        stage6a_promotion["failed_to_bracket_or_refine"] = True
    if stage6a_promotion["promote_to_stage_6b"]:
        conclusion = (
            "At least one refined amplitude band passed all five Stage-6A "
            "conditions, including the dt/2 boundary check. Stage 6B is "
            "authorized; review this report before launching it."
        )
    elif config.validity.tps_property_model == "temperature_dependent_table":
        conclusion = (
            "The temperature-dependent Stage-6A screen did not pass all five "
            "promotion conditions. Record the failed conditions as a negative "
            "result and stop before Stage 6B, performance work, or dataset "
            "generation."
        )
    elif passing:
        conclusion = (
            "At least one direct incident/reradiation candidate passed the "
            "coarse constant-property screen. This is not the temperature-"
            "dependent Stage-6A promotion gate."
        )
    else:
        conclusion = (
            "No tested direct incident/reradiation candidate passed the reduced "
            "screen. Record this as a negative result for the tested load family; "
            "do not widen the amplitude search solely to force promotion."
        )
    result = {
        "study_id": config.study_id,
        "study_config_sha256": config.sha256,
        "screen_type": "direct_incident_heating_with_nonlinear_reradiation",
        "surface": config.surface_payload,
        "material_properties": config.property_provenance,
        "constant_temperature_independent_properties": (
            config.validity.tps_property_model == "constant_effective"
        ),
        "settings": {
            "pulse_widths_s": list(widths),
            "incident_amplitudes_per_event_W_m2": list(incident_amplitudes),
            "horizons_s": list(candidate_horizons),
            "screening_thicknesses_m": list(screening_thicknesses),
            "backing_thickness_m": config.backing_thickness,
            "dt_s": time_config.dt,
            "save_interval_s": save_interval,
            "maximum_horizon_s": maximum_horizon,
            "hot_face_limit_hierarchy": config.hot_face_limit_hierarchy,
        },
        "reference_solve_count": len(solve_rows),
        "candidate_count": len(candidates),
        "separate_solves_avoided_by_trajectory_reuse": (
            len(solve_rows) * (len(candidate_horizons) - 1)
        ),
        "passing_candidate_count": len(passing),
        "viable_incident_radiation_candidate_found": bool(passing),
        "coarse_viable_incident_radiation_candidate_found": bool(passing),
        "stage_6a_promotion": stage6a_promotion,
        "stage_6a_refinement_attempts": refinement_attempts,
        "stage_6a_refined_boundaries": refined_records,
        "monotonicity_checks": monotonicity_checks,
        "stage_6b_authorized": stage6a_promotion["promote_to_stage_6b"],
        "production_dataset_authorized": bool(
            config.authoritative
            and stage6a_promotion["promote_to_stage_6b"]
        ),
        "best_candidate": ranked[0] if ranked else None,
        "passing_candidates": passing,
        "candidates": candidates,
        "reference_solves": solve_rows,
        "wall_seconds": time.perf_counter() - start,
        "conclusion": conclusion,
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("incident_reference_solves.csv", solve_rows),
        ("incident_response_matrix.csv", response_rows),
    ):
        with (destination / filename).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    candidate_rows = [
        {
            key: value
            for key, value in candidate.items()
            if not isinstance(value, dict)
        }
        for candidate in candidates
    ]
    with (destination / "incident_candidates.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    (destination / "incident_screen_report.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result
