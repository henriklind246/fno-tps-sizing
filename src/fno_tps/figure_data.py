"""Artifact layer for the four-figure result set.

This module recomputes or derives exactly the data each figure needs and writes it
to `figure_data.npz` plus a `figure_manifest.json` of scalars, identifiers, and
provenance. It never imports matplotlib: re-styling a figure must not require
re-running the finite-volume solver or the surrogate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence
import csv
import json

import numpy as np

from fno_tps.config import StudyConfig, inference_compatibility_sha256
from fno_tps.data import DatasetBundle, dataset_matches_config, load_dataset
from fno_tps.model import TRANSFER_SOURCE_REVISION
from fno_tps.physics import TPSFVSolver
from fno_tps.problem import SimulationCase
from fno_tps.runtime import load_model_checkpoint, resolve_device
from fno_tps.sizing import load_scenarios, predict_case_delta


REPRESENTATION_ORDER = ("summary", "temporal_global", "temporal_local")
CASE_SELECTORS = ("critical", "worst-error")
#: How many of the most critical held-out cases the `worst-error` selector scores.
#: Every candidate costs a full surrogate rollout, so the pool is deliberately small.
WORST_ERROR_POOL = 12
#: Thickness values round-trip through CSV as text; compare them at this resolution.
THICKNESS_DECIMALS = 9
_STRING_COLUMNS = frozenset({
    "scenario_id",
    "fv_classification",
    "fv_invalid_reasons",
    "fv_design_violations",
    "surrogate_classification",
})
_BOOL_LITERALS = {"True": True, "False": False, "true": True, "false": False}
_ROLE_NAMES = {-2: "thinner-2", -1: "thinner", 0: "selected", 1: "thicker", 2: "thicker+2"}


def _thickness_key(value: float) -> float:
    return round(float(value), THICKNESS_DECIMALS)


def _role_name(offset: int) -> str:
    return _ROLE_NAMES.get(offset, f"offset{offset:+d}")


def _coerce(column: str, value: str) -> Any:
    if column in _STRING_COLUMNS:
        return value
    if value in _BOOL_LITERALS:
        return _BOOL_LITERALS[value]
    if value == "":
        return None
    return float(value)


def _read_sizing_run(path: str | Path, config: StudyConfig) -> dict[str, Any]:
    directory = Path(path)
    matrix_path = directory / "sizing_matrix.csv"
    result_path = directory / "sizing_result.json"
    for required in (matrix_path, result_path):
        if not required.exists():
            raise FileNotFoundError(
                f"{required} not found. Produce it first, for example:\n"
                f"  fno-tps size --config {config.source_path or 'conf/demo.yaml'} "
                f"--scenarios conf/scenarios/demo.yaml "
                f"--checkpoint runs/<representation>/fno_best.pt "
                f"--output {directory}"
            )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["study_config_sha256"] != config.sha256:
        raise ValueError(
            f"Sizing run {directory} was produced with study configuration "
            f"{result['study_config_sha256'][:12]}, but the supplied configuration "
            f"hashes to {config.sha256[:12]}. Re-run sizing or pass the matching config."
        )
    with matrix_path.open("r", newline="", encoding="utf-8") as handle:
        rows = [
            {column: _coerce(column, value) for column, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise ValueError(f"{matrix_path} contains no sizing cells.")
    return {
        "path": str(directory.resolve()),
        "representation": str(result["representation"]),
        "result": result,
        "rows": rows,
    }


def _load_sizing_runs(
    sizing_dirs: Sequence[str | Path],
    config: StudyConfig,
) -> list[dict[str, Any]]:
    runs = [_read_sizing_run(path, config) for path in sizing_dirs]
    seen: dict[str, str] = {}
    for run in runs:
        representation = run["representation"]
        if representation in seen:
            raise ValueError(
                f"Two sizing directories declare representation {representation!r}: "
                f"{seen[representation]} and {run['path']}."
            )
        seen[representation] = run["path"]
    return sorted(
        runs,
        key=lambda run: (
            REPRESENTATION_ORDER.index(run["representation"])
            if run["representation"] in REPRESENTATION_ORDER
            else len(REPRESENTATION_ORDER)
        ),
    )


def _select_primary(
    runs: list[dict[str, Any]],
    requested: str | None,
    checkpoint_representation: str | None,
) -> dict[str, Any]:
    wanted = requested or checkpoint_representation
    if wanted is None:
        if len(runs) == 1:
            return runs[0]
        raise ValueError(
            "Cannot tell which sizing run drives figures 2 and 3. Pass "
            "--primary-representation, or supply a --checkpoint."
        )
    for run in runs:
        if run["representation"] == wanted:
            return run
    available = ", ".join(run["representation"] for run in runs)
    raise ValueError(
        f"No sizing directory was supplied for representation {wanted!r}; "
        f"got {available}. Run `fno-tps size` for that representation first."
    )


def _critical_histories(
    solver: TPSFVSolver,
    temperatures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Peak bond and structural-interface temperature at each saved time."""
    bond = np.max(solver.bond_temperature_envelope(temperatures), axis=1)
    interface = np.max(solver.structural_interface_temperature(temperatures), axis=1)
    return bond, interface


def _criticality(
    bond: np.ndarray,
    interface: np.ndarray,
    config: StudyConfig,
) -> np.ndarray:
    return np.maximum(
        bond / config.bond_temperature_limit,
        interface / config.structural_temperature_limit,
    )


def _physics_contract(config: StudyConfig) -> dict[str, Any]:
    """Compact, renderer-ready description of the active FV physics."""
    tps = config.property_provenance["tps"]
    surface = config.surface_payload
    temperature_dependent = (
        tps.get("model") == "temperature_dependent_table"
    )
    return {
        "solver": (
            "global_nonlinear_enthalpy"
            if temperature_dependent
            else "linear_effective_property"
        ),
        "material_model": tps.get("model"),
        "material_name": tps.get("name"),
        "material_version": tps.get("version"),
        "temperature_dependent": temperature_dependent,
        "anisotropic_conductivity": bool(
            tps.get("anisotropic_conductivity", False)
        ),
        "temperature_dependent_quantities": (
            [
                "conductivity_x",
                "conductivity_y",
                "volumetric_heat_capacity",
                "enthalpy",
            ]
            if temperature_dependent
            else []
        ),
        "pressure_mode": "fixed_reference_condition",
        "pressure_Pa": float(
            tps.get(
                "pressure_Pa",
                config.validity.conductivity_reference_pressure,
            )
        ),
        "pressure_basis": tps.get(
            "pressure_basis",
            "Configured reference pressure; not a transient pressure state.",
        ),
        "surface_model": surface["model"],
        "radiation_coupling": surface["radiation_coupling"],
        "reradiation_enabled": bool(config.surface.reradiation_enabled),
        "emissivity": float(surface["emissivity"]),
        "radiation_sink_temperature_K": float(
            surface["radiation_sink_temperature"]
        ),
    }


def _figure1(
    config: StudyConfig,
    bundle: DatasetBundle,
    model,
    normalization: dict[str, float],
    device,
    batch_size: int,
    case_id: str | None,
    case_selector: str,
    arrays: dict[str, np.ndarray],
    notes: list[str],
) -> dict[str, Any]:
    if case_selector not in CASE_SELECTORS:
        raise ValueError(f"case_selector must be one of {CASE_SELECTORS}.")
    test_ids = [int(value) for value in bundle.splits["test"]]
    if not test_ids:
        raise ValueError(
            "The dataset has an empty test split, so no held-out case can be shown."
        )
    times = np.asarray(bundle.times, dtype=np.float64)

    def fv_absolute(case_index: int) -> np.ndarray:
        return config.initial_temperature + np.asarray(
            bundle.trajectories[case_index], dtype=np.float64
        )

    if case_id is not None:
        matches = [index for index in test_ids if bundle.cases[index].case_id == case_id]
        if not matches:
            raise ValueError(
                f"Case {case_id!r} is not in the held-out test split. "
                f"Available ids include: "
                f"{', '.join(bundle.cases[i].case_id for i in test_ids[:5])}…"
            )
        chosen = matches[0]
        selection_rule = f"explicit case_id={case_id!r}"
    else:
        scores = []
        for index in test_ids:
            solver = TPSFVSolver(config, bundle.cases[index])
            bond, interface = _critical_histories(solver, fv_absolute(index))
            scores.append(float(np.max(_criticality(bond, interface, config))))
        order = list(np.argsort(scores)[::-1])
        if case_selector == "critical":
            chosen = test_ids[int(order[0])]
            selection_rule = "highest FV criticality ratio in the test split"
        else:
            pool = [test_ids[int(position)] for position in order[:WORST_ERROR_POOL]]
            errors = []
            for index in pool:
                solver = TPSFVSolver(config, bundle.cases[index])
                predicted, _ = predict_case_delta(
                    model,
                    config,
                    bundle.cases[index],
                    solver,
                    times,
                    normalization,
                    device,
                    batch_size,
                )
                reference = solver.quantities_of_interest(fv_absolute(index), times)
                predicted_qoi = solver.quantities_of_interest(
                    config.initial_temperature + predicted,
                    times,
                )
                errors.append(
                    max(
                        abs(predicted_qoi["bond_max"] - reference["bond_max"]),
                        abs(
                            predicted_qoi["structural_interface_max"]
                            - reference["structural_interface_max"]
                        ),
                    )
                )
            chosen = pool[int(np.argmax(errors))]
            selection_rule = (
                "largest surrogate critical-peak error among the "
                f"{len(pool)} most critical held-out cases"
            )

    case = bundle.cases[chosen]
    solver = TPSFVSolver(config, case)
    fv_delta = np.asarray(bundle.trajectories[chosen], dtype=np.float64)
    fno_delta, _ = predict_case_delta(
        model,
        config,
        case,
        solver,
        times,
        normalization,
        device,
        batch_size,
    )
    fv_temperature = config.initial_temperature + fv_delta
    fno_temperature = config.initial_temperature + fno_delta

    fv_bond, fv_interface = _critical_histories(solver, fv_temperature)
    critical_index = int(np.argmax(_criticality(fv_bond, fv_interface, config)))
    critical_time = float(times[critical_index])

    fv_field = fv_delta[critical_index]
    fno_field = fno_delta[critical_index]
    error_field = np.abs(fno_field - fv_field)

    bond_slice = solver.grid.bond_slice
    bond_plane = fv_temperature[critical_index, bond_slice, :]
    bond_row, bond_column = np.unravel_index(int(np.argmax(bond_plane)), bond_plane.shape)
    bond_depth = float(solver.grid.x_centers[bond_slice.start + int(bond_row)])
    bond_y = float(solver.grid.y_centers[int(bond_column)])

    fv_interface_field = solver.structural_interface_temperature(fv_temperature)
    fno_interface_field = solver.structural_interface_temperature(fno_temperature)
    interface_column = int(np.argmax(fv_interface_field[critical_index]))
    interface_depth = float(solver.grid.x_faces[solver.grid.structural_face_index + 1])

    fv_qoi = solver.quantities_of_interest(fv_temperature, times)
    fno_qoi = solver.quantities_of_interest(fno_temperature, times)

    arrays["f1_x_faces"] = solver.grid.x_faces.astype(np.float64)
    arrays["f1_y_faces"] = solver.grid.y_faces.astype(np.float64)
    arrays["f1_fv"] = fv_field
    arrays["f1_fno"] = fno_field
    arrays["f1_abs_error"] = error_field

    property_min = config.validity.property_tables.minimum_temperature
    property_max = (
        config.validity.property_tables.maximum_temperature
        or config.hot_face_temperature_limit
    )
    property_temperature = np.linspace(
        property_min,
        property_max,
        256,
        dtype=np.float64,
    )
    tps_model = solver.region_properties.tps
    property_k_x = np.asarray(
        tps_model.conductivity(
            property_temperature,
            mode="accepted",
            direction="x",
        ),
        dtype=np.float64,
    )
    property_k_y = np.asarray(
        tps_model.conductivity(
            property_temperature,
            mode="accepted",
            direction="y",
        ),
        dtype=np.float64,
    )
    property_rho_c = np.asarray(
        tps_model.volumetric_heat_capacity(
            property_temperature,
            mode="accepted",
        ),
        dtype=np.float64,
    )
    arrays["f1_property_temperature"] = property_temperature
    arrays["f1_property_k_x"] = property_k_x
    arrays["f1_property_k_y"] = property_k_y
    arrays["f1_property_rho_c"] = property_rho_c
    arrays["f1_property_alpha_x"] = property_k_x / property_rho_c
    arrays["f1_property_alpha_y"] = property_k_y / property_rho_c

    tps_temperatures = fv_temperature[:, : solver.grid.nx_tps, :]
    realized_property_range = [
        float(np.min(tps_temperatures)),
        float(np.max(tps_temperatures)),
    ]
    vmax = float(max(np.max(fv_field), np.max(fno_field)))
    vmin = float(min(0.0, np.min(fv_field), np.min(fno_field)))
    notes.append(
        f"Figure 1 shows held-out case {case.case_id!r} at t = {critical_time:g} s, "
        f"chosen by {selection_rule}."
    )
    return {
        "case_id": case.case_id,
        "case_index": int(chosen),
        "split": "test",
        "case_selector": case_selector if case_id is None else "explicit",
        "selection_rule": selection_rule,
        "worst_error_pool": WORST_ERROR_POOL if case_selector == "worst-error" else None,
        "d_tps": float(case.d_tps),
        "event_count": case.event_count,
        "defect_count": case.defect_count,
        "initial_temperature": float(config.initial_temperature),
        "t_critical": critical_time,
        "t_critical_index": critical_index,
        "x_tps_bond": float(case.d_tps),
        "x_bond_backing": float(case.d_tps + config.bond_thickness),
        "backing_thickness": float(config.backing_thickness),
        "bond_thickness": float(config.bond_thickness),
        "property_temperature_range_K": [
            float(property_temperature[0]),
            float(property_temperature[-1]),
        ],
        "realized_tps_temperature_range_K": realized_property_range,
        "property_pressure_Pa": float(
            config.validity.conductivity_reference_pressure
        ),
        "delta_t_limits": [vmin, vmax],
        "error_max_K": float(np.max(error_field)),
        "bond_marker": {
            "x": bond_depth,
            "y": bond_y,
            "fv_K": float(fv_temperature[critical_index, bond_slice.start + int(bond_row), int(bond_column)]),
            "fno_K": float(fno_temperature[critical_index, bond_slice.start + int(bond_row), int(bond_column)]),
        },
        "structural_marker": {
            "x": interface_depth,
            "y": float(solver.grid.y_centers[interface_column]),
            "fv_K": float(fv_interface_field[critical_index, interface_column]),
            "fno_K": float(fno_interface_field[critical_index, interface_column]),
        },
        "bond_global_peak": {
            "t": fv_qoi["bond_peak_time"],
            "y": fv_qoi["bond_peak_y"],
            "fv_K": fv_qoi["bond_max"],
            "fno_K": fno_qoi["bond_max"],
        },
        "structural_global_peak": {
            "t": fv_qoi["structural_interface_peak_time"],
            "y": fv_qoi["structural_interface_peak_y"],
            "fv_K": fv_qoi["structural_interface_max"],
            "fno_K": fno_qoi["structural_interface_max"],
        },
        "field_rmse_K": float(np.sqrt(np.mean((fno_field - fv_field) ** 2))),
        "all_time_field_rmse_K": float(np.sqrt(np.mean((fno_delta - fv_delta) ** 2))),
        "bond_peak_error_K": abs(fno_qoi["bond_max"] - fv_qoi["bond_max"]),
        "structural_peak_error_K": abs(
            fno_qoi["structural_interface_max"] - fv_qoi["structural_interface_max"]
        ),
    }


def _resolve_neighborhood(
    config: StudyConfig,
    result: dict[str, Any],
) -> dict[str, Any]:
    candidates = [_thickness_key(value) for value in config.thickness_candidates]
    selected = result.get("fno_selected_thickness")
    source = "fno"
    if selected is None:
        selected = result.get("fv_selected_thickness")
        source = "fv" if selected is not None else None
    window = min(3, len(candidates))
    if selected is None:
        start = max(0, (len(candidates) - window) // 2)
        indices = list(range(start, start + window))
        return {
            "selected_thickness": None,
            "selection_source": None,
            "indices": indices,
            "roles": ["candidate"] * window,
            "clamped": True,
            "reason": (
                "Neither the surrogate nor the FV solver found a feasible design in "
                "the candidate grid, so figure 2 shows the three central thicknesses "
                "instead of a selection neighbourhood."
            ),
        }
    selected_key = _thickness_key(selected)
    if selected_key not in candidates:
        raise ValueError(
            f"Selected thickness {selected} is not one of the configured candidates."
        )
    selected_index = candidates.index(selected_key)
    start = min(max(selected_index - 1, 0), len(candidates) - window)
    indices = list(range(start, start + window))
    clamped = selected_index - 1 != start
    reason = None
    if clamped:
        edge = "thinnest" if selected_index == 0 else "thickest"
        reason = (
            f"The selected design is the {edge} candidate in the grid, so figure 2 "
            "has no neighbour on that side and shows the nearest three candidates "
            "instead."
        )
    return {
        "selected_thickness": float(selected_key),
        "selection_source": source,
        "indices": indices,
        "roles": [_role_name(index - selected_index) for index in indices],
        "clamped": clamped,
        "reason": reason,
    }


def _figure2(
    config: StudyConfig,
    primary: dict[str, Any],
    scenario_path: str | Path,
    scenario_id: str | None,
    model,
    normalization: dict[str, float],
    device,
    batch_size: int,
    arrays: dict[str, np.ndarray],
    notes: list[str],
) -> dict[str, Any]:
    result = primary["result"]
    scenario_set_id, _, scenarios, scenario_hash = load_scenarios(scenario_path)
    if scenario_hash != result["scenario_sha256"]:
        raise ValueError(
            f"Scenario file {scenario_path} hashes to {scenario_hash[:12]}, but the "
            f"sizing run at {primary['path']} used {result['scenario_sha256'][:12]}. "
            "Pass the scenario set that produced the sizing matrix."
        )
    by_id = {scenario["scenario_id"]: scenario for scenario in scenarios}
    wanted = scenario_id
    if wanted is None:
        wanted = (
            result["fv_limiting_scenarios"]["critical"]
            or result["fno_limiting_scenarios"]["critical"]
            or scenarios[0]["scenario_id"]
        )
    if wanted not in by_id:
        raise ValueError(
            f"Scenario {wanted!r} is not in {scenario_set_id!r}; "
            f"available: {', '.join(sorted(by_id))}."
        )
    scenario = by_id[wanted]

    neighborhood = _resolve_neighborhood(config, result)
    if neighborhood["reason"]:
        notes.append(f"Figure 2: {neighborhood['reason']}")
    thicknesses = [
        float(config.thickness_candidates[index]) for index in neighborhood["indices"]
    ]

    times = np.asarray(config.time.saved_times(), dtype=np.float64)
    bond_fv, bond_fno, struct_fv, struct_fno = [], [], [], []
    hot_face_fv = []
    incident_flux, reradiated_flux, net_flux = [], [], []
    nonlinear_iterations, nonlinear_residuals = [], []
    nonlinear_damped, nonlinear_backtracks = [], []
    hot_spot_y = []
    peaks: list[dict[str, Any]] = []
    for thickness in thicknesses:
        case = SimulationCase(
            case_id=f"{scenario['scenario_id']}@{thickness:g}",
            d_tps=thickness,
            heating_events=scenario["heating_events"],
            bond_defects=scenario["bond_defects"],
        )
        solver = TPSFVSolver(config, case)
        trajectory = solver.solve(save_times=times)
        predicted, _ = predict_case_delta(
            model,
            config,
            case,
            solver,
            times,
            normalization,
            device,
            batch_size,
        )
        fv_bond, fv_interface = _critical_histories(solver, trajectory.temperatures)
        fno_bond, fno_interface = _critical_histories(
            solver,
            config.initial_temperature + predicted,
        )
        bond_fv.append(fv_bond)
        bond_fno.append(fno_bond)
        struct_fv.append(fv_interface)
        struct_fno.append(fno_interface)
        surface_temperature = np.asarray(
            trajectory.surface_temperatures,
            dtype=np.float64,
        )
        hot_face_fv.append(np.max(surface_temperature, axis=1))
        hot_flat = int(np.argmax(surface_temperature))
        _, hot_column = np.unravel_index(
            hot_flat,
            surface_temperature.shape,
        )
        hot_spot_y.append(float(solver.grid.y_centers[hot_column]))
        incident = np.asarray(
            case.heat_flux(solver.grid.y_centers, times).T,
            dtype=np.float64,
        )
        reradiated = solver.radiative_heat_flux(surface_temperature)
        incident_trace = incident[:, hot_column]
        reradiated_trace = reradiated[:, hot_column]
        incident_flux.append(incident_trace)
        reradiated_flux.append(reradiated_trace)
        net_flux.append(incident_trace - reradiated_trace)

        iteration_counts = np.asarray(
            trajectory.nonlinear_iteration_counts,
            dtype=np.float64,
        )
        final_residuals = np.asarray(
            trajectory.nonlinear_final_residual_norms,
            dtype=np.float64,
        )
        damped_counts = np.asarray(
            trajectory.nonlinear_damped_iteration_counts,
            dtype=np.float64,
        )
        backtrack_counts = np.asarray(
            trajectory.nonlinear_backtrack_counts,
            dtype=np.float64,
        )
        if final_residuals.size == 0:
            final_residuals = np.zeros_like(iteration_counts)
        if damped_counts.size == 0:
            damped_counts = np.zeros_like(iteration_counts)
        if backtrack_counts.size == 0:
            backtrack_counts = np.zeros_like(iteration_counts)
        nonlinear_iterations.append(iteration_counts)
        nonlinear_residuals.append(final_residuals)
        nonlinear_damped.append(damped_counts)
        nonlinear_backtracks.append(backtrack_counts)
        if (
            config.validity.reject_cases_outside_validity
            and trajectory.maximum_hot_face_temperature
            > config.hot_face_temperature_limit + 1e-8
        ):
            notes.append(
                f"Figure 2: cell {case.case_id!r} reaches "
                f"{trajectory.maximum_hot_face_temperature:.6g} K on the hot face, "
                "outside the declared validity domain."
            )
        peaks.append({
            "d_tps": thickness,
            "fv_bond_max": float(np.max(fv_bond)),
            "fno_bond_max": float(np.max(fno_bond)),
            "fv_structural_max": float(np.max(fv_interface)),
            "fno_structural_max": float(np.max(fno_interface)),
            "fv_feasible": bool(
                np.max(fv_bond) <= config.bond_temperature_limit
                and np.max(fv_interface) <= config.structural_temperature_limit
            ),
            "fno_feasible": bool(
                np.max(fno_bond) <= config.bond_temperature_limit
                and np.max(fno_interface) <= config.structural_temperature_limit
            ),
            "hot_face_max_K": float(trajectory.maximum_hot_face_temperature),
            "hot_spot_y_m": hot_spot_y[-1],
            "reradiated_energy_fraction": float(
                trajectory.radiated_energy[-1]
                / max(trajectory.boundary_input_energy[-1], 1.0e-30)
            ),
            "step_driver": trajectory.step_driver,
            "max_nonlinear_iterations": int(
                trajectory.max_nonlinear_iterations
            ),
            "max_nonlinear_residual_K": float(
                np.max(final_residuals)
            ),
            "total_nonlinear_backtracks": int(
                np.sum(backtrack_counts)
            ),
        })

    diagnostic_index = (
        neighborhood["roles"].index("selected")
        if "selected" in neighborhood["roles"]
        else len(thicknesses) // 2
    )
    arrays["f2_times"] = times
    arrays["f2_thicknesses"] = np.asarray(thicknesses, dtype=np.float64)
    arrays["f2_bond_fv"] = np.stack(bond_fv)
    arrays["f2_bond_fno"] = np.stack(bond_fno)
    arrays["f2_struct_fv"] = np.stack(struct_fv)
    arrays["f2_struct_fno"] = np.stack(struct_fno)
    arrays["f2_hot_face_fv"] = np.stack(hot_face_fv)
    arrays["f2_incident_flux"] = np.stack(incident_flux)
    arrays["f2_reradiated_flux"] = np.stack(reradiated_flux)
    arrays["f2_net_flux"] = np.stack(net_flux)
    arrays["f2_solver_times"] = (
        np.arange(1, len(nonlinear_iterations[0]) + 1, dtype=np.float64)
        * config.time.dt
    )
    arrays["f2_nonlinear_iterations"] = np.stack(nonlinear_iterations)
    arrays["f2_nonlinear_residuals"] = np.stack(nonlinear_residuals)
    arrays["f2_nonlinear_damped_iterations"] = np.stack(
        nonlinear_damped
    )
    arrays["f2_nonlinear_backtracks"] = np.stack(nonlinear_backtracks)
    return {
        "scenario_set_id": scenario_set_id,
        "scenario_id": scenario["scenario_id"],
        "scenario_source": "explicit" if scenario_id else "fv_limiting_scenario",
        "representation": primary["representation"],
        "thicknesses": thicknesses,
        "roles": neighborhood["roles"],
        "selected_thickness": neighborhood["selected_thickness"],
        "selection_source": neighborhood["selection_source"],
        "clamped": neighborhood["clamped"],
        "reason": neighborhood["reason"],
        "diagnostic_thickness_index": diagnostic_index,
        "diagnostic_thickness": thicknesses[diagnostic_index],
        "diagnostic_role": neighborhood["roles"][diagnostic_index],
        "diagnostic_hot_spot_y_m": hot_spot_y[diagnostic_index],
        "peaks": peaks,
    }


def _figure3(
    config: StudyConfig,
    primary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    notes: list[str],
) -> dict[str, Any]:
    rows = primary["rows"]
    result = primary["result"]
    thicknesses = sorted({_thickness_key(row["d_tps"]) for row in rows})
    columns = {
        "fv_bond": "fv_bond_max",
        "fno_bond": "surrogate_bond_max",
        "fv_struct": "fv_structural_max",
        "fno_struct": "surrogate_structural_max",
    }
    collected: dict[str, list[float]] = {
        f"{name}_{statistic}": []
        for name in columns
        for statistic in ("worst", "lo", "hi")
    }
    limiting: list[str] = []
    violation: list[bool] = []
    scenario_counts: set[int] = set()
    for thickness in thicknesses:
        subset = [row for row in rows if _thickness_key(row["d_tps"]) == thickness]
        scenario_counts.add(len(subset))
        for name, column in columns.items():
            values = [float(row[column]) for row in subset]
            collected[f"{name}_worst"].append(max(values))
            collected[f"{name}_lo"].append(min(values))
            collected[f"{name}_hi"].append(max(values))
        limiting.append(
            max(subset, key=lambda row: row["fv_criticality_ratio"])["scenario_id"]
        )
        violation.append(any(not row["surrogate_within_validity"] for row in subset))

    arrays["f3_thickness"] = np.asarray(thicknesses, dtype=np.float64)
    for key, values in collected.items():
        arrays[f"f3_{key}"] = np.asarray(values, dtype=np.float64)
    arrays["f3_surrogate_validity_violation"] = np.asarray(violation, dtype=bool)

    fno_selected = result.get("fno_selected_thickness")
    fv_selected = result.get("fv_selected_thickness")
    if fno_selected is None:
        notes.append(
            "Figure 3: the surrogate found no feasible design in the candidate grid, "
            "so no minimum FNO-feasible marker is drawn."
        )
    if fv_selected is None:
        notes.append(
            "Figure 3: the FV solver found no feasible design in the candidate grid, "
            "so no minimum FV-feasible marker is drawn."
        )
    if any(violation):
        notes.append(
            "Figure 3: at least one thickness has a surrogate prediction above the "
            f"{config.hot_face_temperature_limit:g} K resolved hot-face "
            "validity ceiling; "
            "those points are drawn as open markers."
        )
    return {
        "representation": primary["representation"],
        "thicknesses": thicknesses,
        "scenario_count": sorted(scenario_counts),
        "limiting_scenarios": limiting,
        "fno_selected_thickness": fno_selected,
        "fv_selected_thickness": fv_selected,
        "surrogate_validity_violation": violation,
    }


def _figure4(
    runs: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    notes: list[str],
) -> dict[str, Any]:
    panels: list[dict[str, Any]] = []
    for run in runs:
        representation = run["representation"]
        rows = run["rows"]
        m_fv = np.asarray(
            [min(row["fv_bond_margin"], row["fv_structural_margin"]) for row in rows],
            dtype=np.float64,
        )
        m_fno = np.asarray(
            [
                min(row["surrogate_bond_margin"], row["surrogate_structural_margin"])
                for row in rows
            ],
            dtype=np.float64,
        )
        false_feasible = np.asarray([bool(row["false_feasible"]) for row in rows])
        false_infeasible = np.asarray([bool(row["false_infeasible"]) for row in rows])
        arrays[f"f4_{representation}_m_fv"] = m_fv
        arrays[f"f4_{representation}_m_fno"] = m_fno
        arrays[f"f4_{representation}_false_feasible"] = false_feasible
        arrays[f"f4_{representation}_false_infeasible"] = false_infeasible
        arrays[f"f4_{representation}_d_tps"] = np.asarray(
            [float(row["d_tps"]) for row in rows], dtype=np.float64
        )
        disagreements = int(false_feasible.sum() + false_infeasible.sum())
        reported = run["result"].get("margin_sign_disagreement_count")
        if reported is not None and int(reported) != disagreements:
            notes.append(
                f"Figure 4: {representation} matrix has {disagreements} sign "
                f"disagreements but its sizing_result.json reports {reported}. The "
                "CSV was used; the two artifacts are out of sync."
            )
        panels.append({
            "representation": representation,
            "path": run["path"],
            "cell_count": len(rows),
            "false_feasible": int(false_feasible.sum()),
            "false_infeasible": int(false_infeasible.sum()),
            "margin_mae_K": float(np.mean(np.abs(m_fno - m_fv))),
            "margin_sign_disagreement_count": disagreements,
        })
    missing = [
        name
        for name in REPRESENTATION_ORDER
        if name not in {run["representation"] for run in runs}
    ]
    if missing:
        notes.append(
            "Figure 4 is missing the "
            f"{', '.join(missing)} representation(s); the ablation is incomplete."
        )
    return {"panels": panels, "missing_representations": missing}


def build_figure_data(
    config: StudyConfig,
    output_dir: str | Path,
    *,
    sizing_dirs: Sequence[str | Path] = (),
    data_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    scenario_path: str | Path | None = None,
    case_id: str | None = None,
    case_selector: str = "critical",
    scenario_id: str | None = None,
    primary_representation: str | None = None,
    figures: Iterable[int] = (1, 2, 3, 4),
    device: str = "auto",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Assemble `figure_data.npz` and `figure_manifest.json` under `output_dir`."""
    requested = tuple(sorted({int(value) for value in figures}))
    unknown = [value for value in requested if value not in (1, 2, 3, 4)]
    if unknown:
        raise ValueError(f"Unknown figure numbers {unknown}; expected 1, 2, 3, or 4.")
    if not requested:
        raise ValueError("At least one figure must be requested.")

    needs_sizing = any(value in requested for value in (2, 3, 4))
    needs_model = any(value in requested for value in (1, 2))
    if needs_sizing and not sizing_dirs:
        raise ValueError(
            "Figures 2, 3, and 4 read the sizing matrix. Pass at least one --sizing "
            "directory produced by `fno-tps size`."
        )
    if needs_model and checkpoint_path is None:
        raise ValueError("Figures 1 and 2 require --checkpoint.")
    if 1 in requested and data_dir is None:
        raise ValueError("Figure 1 requires --data (the held-out case comes from it).")
    if 2 in requested and scenario_path is None:
        raise ValueError("Figure 2 requires --scenarios to rebuild the sizing scenario.")

    runs = _load_sizing_runs(sizing_dirs, config) if sizing_dirs else []
    arrays: dict[str, np.ndarray] = {}
    notes: list[str] = []
    model = None
    checkpoint: dict[str, Any] | None = None
    normalization: dict[str, float] = {}
    resolved_device = None
    if needs_model:
        resolved_device = resolve_device(device)
        model, checkpoint = load_model_checkpoint(checkpoint_path, resolved_device)
        if (
            inference_compatibility_sha256(checkpoint["study"])
            != config.inference_compatibility_sha256
        ):
            raise ValueError(
                "Checkpoint inference configuration does not match the figure "
                "configuration."
            )
        normalization = {
            key: float(value) for key, value in checkpoint["normalization"].items()
        }

    primary = (
        _select_primary(
            runs,
            primary_representation,
            model.config.representation if model is not None else None,
        )
        if needs_sizing
        else None
    )

    bundle = None
    if 1 in requested:
        bundle = load_dataset(data_dir)
        if not dataset_matches_config(bundle, config):
            raise ValueError(
                "Dataset physics configuration does not match the figure "
                "configuration."
            )
        for key, value in normalization.items():
            if not np.isclose(value, bundle.normalization[key]):
                raise ValueError(
                    "Checkpoint and dataset normalization disagree "
                    f"({key}: {value} vs {bundle.normalization[key]})."
                )

    manifest: dict[str, Any] = {
        "study_id": config.study_id,
        "study_config_sha256": config.sha256,
        "study_authoritative": config.authoritative,
        "transfer_source_revision": TRANSFER_SOURCE_REVISION,
        "generated_figures": list(requested),
        "limits": {
            "bond_temperature": config.bond_temperature_limit,
            "structural_interface_temperature": config.structural_temperature_limit,
        },
        "physics": _physics_contract(config),
        "validity": {
            "hot_face_limit_hierarchy": config.hot_face_limit_hierarchy,
            "reject_cases_outside_validity": (
                config.validity.reject_cases_outside_validity
            ),
        },
        "checkpoint": (
            {
                "path": str(Path(checkpoint_path).resolve()),
                "representation": model.config.representation,
                "epoch": int(checkpoint["epoch"]),
                "transfer_source_revision": checkpoint["transfer_source_revision"],
            }
            if model is not None
            else None
        ),
        "dataset": (
            {
                "path": str(Path(data_dir).resolve()),
                "case_count": len(bundle.cases),
                "test_case_count": int(len(bundle.splits["test"])),
            }
            if bundle is not None
            else None
        ),
        "sizing_runs": [
            {
                "representation": run["representation"],
                "path": run["path"],
                "scenario_set_id": run["result"]["scenario_set_id"],
                "scenario_sha256": run["result"]["scenario_sha256"],
                "fno_selected_thickness": run["result"].get("fno_selected_thickness"),
                "fv_selected_thickness": run["result"].get("fv_selected_thickness"),
            }
            for run in runs
        ],
        "primary_representation": primary["representation"] if primary else None,
        "figure1": None,
        "figure2": None,
        "figure3": None,
        "figure4": None,
    }

    if 1 in requested:
        manifest["figure1"] = _figure1(
            config,
            bundle,
            model,
            normalization,
            resolved_device,
            batch_size,
            case_id,
            case_selector,
            arrays,
            notes,
        )
    if 2 in requested:
        manifest["figure2"] = _figure2(
            config,
            primary,
            scenario_path,
            scenario_id,
            model,
            normalization,
            resolved_device,
            batch_size,
            arrays,
            notes,
        )
    if 3 in requested:
        manifest["figure3"] = _figure3(config, primary, arrays, notes)
    if 4 in requested:
        manifest["figure4"] = _figure4(runs, arrays, notes)

    manifest["notes"] = notes
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(destination / "figure_data.npz", **arrays)
    (destination / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_figure_data(
    directory: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read back a manifest and its arrays for rendering."""
    root = Path(directory)
    manifest_path = root / "figure_manifest.json"
    array_path = root / "figure_data.npz"
    for required in (manifest_path, array_path):
        if not required.exists():
            raise FileNotFoundError(
                f"{required} not found. Run `fno-tps figures` to build it first."
            )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(array_path) as handle:
        arrays = {key: handle[key] for key in handle.files}
    return manifest, arrays
