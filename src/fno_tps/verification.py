from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json

import numpy as np

from fno_tps.config import (
    BondConfig,
    Material,
    MeshConfig,
    PropertyTableConfig,
    StudyConfig,
    SurfaceConfig,
    TimeConfig,
)
from fno_tps.materials import (
    RegionProperties,
    TabulatedPropertyModel,
    build_region_properties,
    load_property_table,
)
from fno_tps.physics import TPSFVSolver, horizon_diagnostics
from fno_tps.problem import BondDefect, HeatingEvent, SimulationCase


@dataclass(frozen=True)
class ConvergenceResult:
    resolutions: tuple[float, ...]
    errors: tuple[float, ...]
    orders: tuple[float, ...]
    minimum_order: float


def _manufactured_config(
    base: StudyConfig,
    mesh: MeshConfig,
    dt: float,
    t_final: float,
) -> StudyConfig:
    bond = BondConfig(
        k0=1.0,
        k_min=1.0,
        rho_c=1.0,
        severity=base.bond.severity,
        y_center=(0.1, 0.9),
        sigma=(0.05, 0.2),
        max_defects=2,
    )
    return replace(
        base,
        lateral_length=1.0,
        bond_thickness=0.2,
        backing_thickness=0.4,
        thickness_candidates=(0.4,),
        tps=Material.constant(1.0, 1.0),
        backing=Material.constant(1.0, 1.0),
        bond=bond,
        surface=SurfaceConfig.from_dict(None),
        heating=replace(
            base.heating,
            interpretation="effective_net_inward_conductive_heat_flux",
        ),
        mesh=mesh,
        time=TimeConfig(
            dt=dt,
            t_final=t_final,
            save_stride=max(1, int(round(t_final / dt))),
            horizon_candidates=(t_final,),
        ),
    )


def _manufactured_case() -> SimulationCase:
    return SimulationCase(
        case_id="mms",
        d_tps=0.4,
        heating_events=(HeatingEvent(1.0, 0.5, 0.01, 0.2, 0.01),),
        bond_defects=(),
    )


def _exact_temperature(x: np.ndarray, y: np.ndarray, time: float) -> np.ndarray:
    return 300.0 + np.exp(-time) * (
        np.cos(np.pi * x) + 0.5 * np.cos(2.0 * np.pi * y)
    )


def _manufactured_source(x: np.ndarray, y: np.ndarray, time: float) -> np.ndarray:
    decay = np.exp(-time)
    x_term = np.cos(np.pi * x)
    y_term = 0.5 * np.cos(2.0 * np.pi * y)
    return decay * (
        -(x_term + y_term)
        + np.pi**2 * x_term
        + (2.0 * np.pi) ** 2 * y_term
    )


def _nonlinear_manufactured_config(
    base: StudyConfig,
    mesh: MeshConfig,
    dt: float,
    t_final: float,
) -> StudyConfig:
    table_path = (
        Path(__file__).resolve().parents[2]
        / "conf"
        / "materials"
        / "nonlinear_verification.yaml"
    )
    table_config = PropertyTableConfig(
        path=str(table_path),
        version="MMS-1",
        extrapolation="reject",
        interpolation="linear",
        reference_temperature=400.0,
    )
    material = Material(
        model="temperature_dependent_table",
        reference_k=1.0,
        reference_rho_c=1.0,
        property_table=table_config,
    )
    config = replace(
        _manufactured_config(base, mesh, dt, t_final),
        authoritative=False,
        initial_temperature=400.0,
        tps=material,
        backing=material,
        bond=replace(
            _manufactured_config(base, mesh, dt, t_final).bond,
            k0=1.0,
            k_min=1.0,
            rho_c=1.0,
        ),
        validity=replace(
            base.validity,
            tps_property_model="temperature_dependent_table",
            hot_face=replace(
                base.validity.hot_face,
                study_temperature_limit=500.0,
            ),
            property_tables=replace(
                base.validity.property_tables,
                minimum_temperature=250.0,
                maximum_temperature=500.0,
            ),
            conductivity_reference_pressure=101325.0,
        ),
        surface=replace(
            SurfaceConfig.from_dict(None),
            radiation_coupling="coupled_nonlinear",
        ),
        heating=replace(
            base.heating,
            interpretation="effective_net_inward_conductive_heat_flux",
            t_center=(0.0, t_final),
        ),
        solver=replace(
            base.solver,
            nonlinear=replace(
                base.solver.nonlinear,
                residual_temperature_tolerance=1.0e-11,
                update_temperature_tolerance=1.0e-11,
                update_relative_tolerance=1.0e-13,
            ),
        ),
    )
    config.validate()
    return config


def _nonlinear_region_properties(config: StudyConfig) -> RegionProperties:
    regions = build_region_properties(config)
    assert config.tps.property_table is not None
    table = load_property_table(
        config.resolve_property_table_path(config.tps.property_table.path)
    )
    bond = TabulatedPropertyModel(
        table,
        reference_temperature=400.0,
        extrapolation="reject",
    )
    return RegionProperties(
        tps=regions.tps,
        bond=bond,
        backing=regions.backing,
    )


def _nonlinear_exact_temperature(
    x: np.ndarray,
    y: np.ndarray,
    time: float,
) -> np.ndarray:
    return 400.0 + 50.0 * np.exp(-time) * (
        np.cos(np.pi * x) + 0.5 * np.cos(2.0 * np.pi * y)
    )


def _nonlinear_manufactured_source(
    x: np.ndarray,
    y: np.ndarray,
    time: float,
) -> np.ndarray:
    amplitude = 50.0 * np.exp(-time)
    cos_x = np.cos(np.pi * x)
    cos_y = np.cos(2.0 * np.pi * y)
    shape = cos_x + 0.5 * cos_y
    k = 1.0 + 4.0e-3 * amplitude * shape
    rho_c = 1.0 + 2.0e-3 * amplitude * shape
    return (
        rho_c * (-amplitude * shape)
        + k * amplitude * np.pi**2 * (cos_x + 2.0 * cos_y)
        - 4.0e-3
        * amplitude**2
        * np.pi**2
        * (
            np.sin(np.pi * x) ** 2
            + np.sin(2.0 * np.pi * y) ** 2
        )
    )


def _zero_flux_integral(y: np.ndarray, t0: float, t1: float) -> np.ndarray:
    del t0, t1
    return np.zeros_like(y, dtype=np.float64)


def _weighted_l2(solver: TPSFVSolver, error: np.ndarray) -> float:
    volume = np.broadcast_to(solver.volume, error.shape)
    return float(np.sqrt(np.sum(volume * error**2) / np.sum(volume)))


def _orders(scales: list[float], errors: list[float]) -> tuple[float, ...]:
    return tuple(
        float(np.log(errors[i] / errors[i + 1]) / np.log(scales[i] / scales[i + 1]))
        for i in range(len(errors) - 1)
    )


def spatial_convergence(
    base: StudyConfig,
    levels: tuple[int, ...] = (6, 10, 16),
) -> ConvergenceResult:
    errors: list[float] = []
    scales: list[float] = []
    final_time = 0.01
    dt = 2.5e-5
    case = _manufactured_case()
    for level in levels:
        mesh = MeshConfig(
            nx_tps=2 * level,
            nx_bond=level,
            nx_back=2 * level,
            ny=5 * level,
        )
        config = _manufactured_config(base, mesh, dt, final_time)
        solver = TPSFVSolver(
            config,
            case,
            source_fn=_manufactured_source,
            flux_integral_fn=_zero_flux_integral,
        )
        initial = _exact_temperature(solver.X, solver.Y, 0.0)
        trajectory = solver.solve(
            save_times=np.array([0.0, final_time]),
            initial_temperature=initial,
        )
        exact = _exact_temperature(solver.X, solver.Y, final_time)
        errors.append(_weighted_l2(solver, trajectory.temperatures[-1] - exact))
        scales.append(float(np.max(solver.grid.dx)))
    orders = _orders(scales, errors)
    return ConvergenceResult(tuple(scales), tuple(errors), orders, min(orders))


def temporal_convergence(
    base: StudyConfig,
    dt_levels: tuple[float, ...] = (0.004, 0.002, 0.001),
) -> ConvergenceResult:
    errors: list[float] = []
    final_time = 0.04
    case = _manufactured_case()
    mesh = MeshConfig(nx_tps=32, nx_bond=16, nx_back=32, ny=80)
    reference_dt = min(dt_levels) / 8.0
    reference_config = _manufactured_config(base, mesh, reference_dt, final_time)
    reference_solver = TPSFVSolver(
        reference_config,
        case,
        source_fn=_manufactured_source,
        flux_integral_fn=_zero_flux_integral,
    )
    reference_initial = _exact_temperature(reference_solver.X, reference_solver.Y, 0.0)
    reference = reference_solver.solve(
        save_times=np.array([0.0, final_time]),
        initial_temperature=reference_initial,
    ).temperatures[-1]
    for dt in dt_levels:
        config = _manufactured_config(base, mesh, dt, final_time)
        solver = TPSFVSolver(
            config,
            case,
            source_fn=_manufactured_source,
            flux_integral_fn=_zero_flux_integral,
        )
        initial = _exact_temperature(solver.X, solver.Y, 0.0)
        trajectory = solver.solve(
            save_times=np.array([0.0, final_time]),
            initial_temperature=initial,
        )
        errors.append(_weighted_l2(solver, trajectory.temperatures[-1] - reference))
    orders = _orders(list(dt_levels), errors)
    return ConvergenceResult(tuple(dt_levels), tuple(errors), orders, min(orders))


def nonlinear_spatial_convergence(
    base: StudyConfig,
    levels: tuple[int, ...] = (6, 10, 16),
) -> ConvergenceResult:
    """Spatial MMS gate for enthalpy storage and temperature-dependent fluxes."""
    errors: list[float] = []
    scales: list[float] = []
    final_time = 0.01
    dt = 2.5e-5
    case = _manufactured_case()
    for level in levels:
        mesh = MeshConfig(
            nx_tps=2 * level,
            nx_bond=level,
            nx_back=2 * level,
            ny=5 * level,
        )
        config = _nonlinear_manufactured_config(
            base,
            mesh,
            dt,
            final_time,
        )
        solver = TPSFVSolver(
            config,
            case,
            source_fn=_nonlinear_manufactured_source,
            flux_integral_fn=_zero_flux_integral,
            region_properties=_nonlinear_region_properties(config),
        )
        initial = _nonlinear_exact_temperature(solver.X, solver.Y, 0.0)
        trajectory = solver.solve(
            save_times=np.asarray([0.0, final_time]),
            initial_temperature=initial,
        )
        exact = _nonlinear_exact_temperature(
            solver.X,
            solver.Y,
            final_time,
        )
        errors.append(
            _weighted_l2(solver, trajectory.temperatures[-1] - exact)
        )
        scales.append(float(np.max(solver.grid.dx)))
        if trajectory.max_nonlinear_iterations > 6:
            raise RuntimeError(
                "Nonlinear MMS exceeded the six-iteration regression limit."
            )
        initial_k, initial_rho_c = solver.properties_from_temperature(initial)
        del initial_k
        energy_scale = float(np.sum(solver.volume * initial_rho_c))
        if (
            float(np.max(np.abs(trajectory.energy_residual)))
            / max(energy_scale, 1.0e-30)
            >= 1.0e-10
        ):
            raise RuntimeError("Nonlinear MMS enthalpy energy balance failed.")
    orders = _orders(scales, errors)
    return ConvergenceResult(tuple(scales), tuple(errors), orders, min(orders))


def nonlinear_temporal_convergence(
    base: StudyConfig,
    dt_levels: tuple[float, ...] = (0.004, 0.002, 0.001),
) -> ConvergenceResult:
    """Temporal MMS gate; this detects property lagging in the CN flux term."""
    final_time = 0.04
    mesh = MeshConfig(nx_tps=32, nx_bond=16, nx_back=32, ny=80)
    case = _manufactured_case()

    def solve_at(dt: float) -> np.ndarray:
        config = _nonlinear_manufactured_config(
            base,
            mesh,
            dt,
            final_time,
        )
        solver = TPSFVSolver(
            config,
            case,
            source_fn=_nonlinear_manufactured_source,
            flux_integral_fn=_zero_flux_integral,
            region_properties=_nonlinear_region_properties(config),
        )
        initial = _nonlinear_exact_temperature(solver.X, solver.Y, 0.0)
        trajectory = solver.solve(
            save_times=np.asarray([0.0, final_time]),
            initial_temperature=initial,
        )
        if trajectory.max_nonlinear_iterations > 6:
            raise RuntimeError(
                "Nonlinear MMS exceeded the six-iteration regression limit."
            )
        _, initial_rho_c = solver.properties_from_temperature(initial)
        energy_scale = float(np.sum(solver.volume * initial_rho_c))
        if (
            float(np.max(np.abs(trajectory.energy_residual)))
            / max(energy_scale, 1.0e-30)
            >= 1.0e-10
        ):
            raise RuntimeError("Nonlinear MMS enthalpy energy balance failed.")
        return trajectory.temperatures[-1]

    reference = solve_at(min(dt_levels) / 8.0)
    errors: list[float] = []
    for dt in dt_levels:
        config = _nonlinear_manufactured_config(
            base,
            mesh,
            dt,
            final_time,
        )
        solver = TPSFVSolver(
            config,
            case,
            source_fn=_nonlinear_manufactured_source,
            flux_integral_fn=_zero_flux_integral,
            region_properties=_nonlinear_region_properties(config),
        )
        initial = _nonlinear_exact_temperature(solver.X, solver.Y, 0.0)
        trajectory = solver.solve(
            save_times=np.asarray([0.0, final_time]),
            initial_temperature=initial,
        )
        errors.append(
            _weighted_l2(solver, trajectory.temperatures[-1] - reference)
        )
    orders = _orders(list(dt_levels), errors)
    return ConvergenceResult(tuple(dt_levels), tuple(errors), orders, min(orders))


def nonlinear_steady_conduction(
    base: StudyConfig,
    levels: tuple[int, ...] = (4, 6, 10),
) -> dict[str, object]:
    """Kirchhoff-transform benchmark with a verification-only rear Dirichlet BC."""
    from scipy.optimize import brentq, root

    rear_temperature = 350.0
    incident_flux = 20.0
    errors: list[float] = []
    scales: list[float] = []
    flux_errors: list[float] = []
    for level in levels:
        mesh = MeshConfig(
            nx_tps=2 * level,
            nx_bond=level,
            nx_back=2 * level,
            ny=2,
        )
        config = _nonlinear_manufactured_config(
            base,
            mesh,
            0.01,
            0.01,
        )
        properties = _nonlinear_region_properties(config)
        solver = TPSFVSolver(
            config,
            _manufactured_case(),
            flux_integral_fn=_zero_flux_integral,
            region_properties=properties,
            rear_temperature=rear_temperature,
        )
        model = properties.tps
        assert isinstance(model, TabulatedPropertyModel)
        total_length = float(solver.grid.x_faces[-1])
        rear_psi = float(model.conductivity_integral(rear_temperature))
        target_psi = rear_psi + incident_flux * (
            total_length - solver.grid.x_centers
        )

        def inverse_psi(value: float) -> float:
            return float(
                brentq(
                    lambda temperature: (
                        float(model.conductivity_integral(temperature))
                        - value
                    ),
                    model.table.temperature_min,
                    model.table.temperature_max,
                )
            )

        exact_1d = np.asarray([inverse_psi(value) for value in target_psi])
        exact = np.broadcast_to(
            exact_1d[:, None],
            (solver.grid.nx, solver.grid.ny),
        ).copy()

        def steady_residual(flat: np.ndarray) -> np.ndarray:
            temperature = flat.reshape(solver.grid.nx, solver.grid.ny)
            solver._update_property_state(temperature, mode="iterate")
            residual = solver._conduction_flux(temperature)
            residual = residual.copy()
            residual[0] += incident_flux * solver.grid.dy
            return residual.reshape(-1)

        solution = root(
            steady_residual,
            exact.reshape(-1),
            tol=1.0e-11,
        )
        if not solution.success:
            raise RuntimeError(
                "Kirchhoff steady-conduction reference solve failed: "
                f"{solution.message}"
            )
        numerical = solution.x.reshape(solver.grid.nx, solver.grid.ny)
        errors.append(_weighted_l2(solver, numerical - exact))
        scales.append(float(np.max(solver.grid.dx)))
        solver._update_property_state(numerical, mode="accepted")
        rear_outward_flux = (
            2.0
            * solver.k[-1]
            * (numerical[-1] - rear_temperature)
            / solver.grid.dx[-1]
        )
        flux_errors.append(
            float(
                np.max(
                    np.abs(rear_outward_flux - incident_flux)
                )
                / incident_flux
            )
        )
    orders = _orders(scales, errors)
    result = ConvergenceResult(
        tuple(scales),
        tuple(errors),
        orders,
        min(orders),
    )
    return {
        **asdict(result),
        "maximum_relative_flux_error": max(flux_errors),
        "passed": bool(
            result.minimum_order >= 1.8
            and max(flux_errors) < 1.0e-10
        ),
    }


def radiation_temporal_convergence(
    base: StudyConfig,
    dt_levels: tuple[float, ...] = (0.5, 0.25, 0.125),
    *,
    amplitude: float | None = None,
) -> ConvergenceResult:
    """Verify temporal order for the nonlinear incident/reradiation boundary."""
    if not base.surface.reradiation_enabled:
        raise ValueError(
            "radiation_temporal_convergence requires reradiation to be enabled."
        )
    final_time = 4.0
    mesh = MeshConfig(nx_tps=8, nx_bond=2, nx_back=4, ny=16)
    case = SimulationCase(
        case_id="radiation-temporal-convergence",
        d_tps=base.thickness_candidates[0],
        heating_events=(
            HeatingEvent(
                amplitude=(
                    min(4000.0, base.heating.amplitude[1])
                    if amplitude is None
                    else float(amplitude)
                ),
                y_center=0.5 * base.lateral_length,
                t_center=1.5,
                sigma_y=0.15 * base.lateral_length,
                sigma_t=0.5,
            ),
        ),
        bond_defects=(),
    )

    def config_at(dt: float) -> StudyConfig:
        return replace(
            base,
            mesh=mesh,
            time=TimeConfig(
                dt=dt,
                t_final=final_time,
                save_stride=int(round(final_time / dt)),
                horizon_candidates=(final_time,),
            ),
        )

    reference_dt = min(dt_levels) / 8.0
    reference_config = config_at(reference_dt)
    reference_solver = TPSFVSolver(reference_config, case)
    reference = reference_solver.solve(
        save_times=np.asarray([0.0, final_time])
    ).temperatures[-1]
    errors = []
    for dt in dt_levels:
        config = config_at(dt)
        solver = TPSFVSolver(config, case)
        trajectory = solver.solve(save_times=np.asarray([0.0, final_time]))
        errors.append(
            _weighted_l2(solver, trajectory.temperatures[-1] - reference)
        )
    orders = _orders(list(dt_levels), errors)
    return ConvergenceResult(tuple(dt_levels), tuple(errors), orders, min(orders))


def radiation_flux_level_temporal_convergence(
    base: StudyConfig,
) -> dict[str, object]:
    """Check temporal order at low, medium, and high incident-flux levels."""
    maximum = float(base.heating.amplitude[1])
    amplitudes = {
        "low": 0.25 * maximum,
        "medium": 0.625 * maximum,
        "high": maximum,
    }
    results = {
        name: radiation_temporal_convergence(
            base,
            amplitude=amplitude,
        )
        for name, amplitude in amplitudes.items()
    }
    return {
        "amplitudes_W_m2": amplitudes,
        "levels": {
            name: asdict(result)
            for name, result in results.items()
        },
        "minimum_order": min(
            result.minimum_order for result in results.values()
        ),
        "passed": all(
            result.minimum_order >= 1.8 for result in results.values()
        ),
    }


def constant_property_equivalence(
    base: StudyConfig,
) -> dict[str, float | int | bool]:
    """Compare the nonlinear enthalpy driver against the retained linear path."""
    from fno_tps.materials import TabulatedPropertyModel, load_property_table

    table_path = (
        Path(__file__).resolve().parents[2]
        / "conf"
        / "materials"
        / "tps_placeholder.yaml"
    )
    table = load_property_table(table_path)
    table_model = TabulatedPropertyModel(
        table,
        reference_temperature=300.0,
        extrapolation="reject",
    )
    reference_k = float(
        table_model.conductivity(300.0, direction="x")
    )
    reference_rho_c = float(
        table_model.volumetric_heat_capacity(300.0)
    )
    mesh = MeshConfig(nx_tps=4, nx_bond=1, nx_back=2, ny=4)
    time_config = TimeConfig(
        dt=0.25,
        t_final=1.0,
        save_stride=1,
        horizon_candidates=(1.0,),
    )
    linear = replace(
        base,
        authoritative=False,
        mesh=mesh,
        time=time_config,
        tps=Material.constant(reference_k, reference_rho_c),
        validity=replace(
            base.validity,
            tps_property_model="constant_effective",
        ),
        surface=SurfaceConfig.from_dict(None),
        heating=replace(
            base.heating,
            interpretation="effective_net_inward_conductive_heat_flux",
            t_center=(0.2, 0.8),
        ),
    )
    table_material = Material(
        model="temperature_dependent_table",
        reference_k=reference_k,
        reference_rho_c=reference_rho_c,
        property_table=PropertyTableConfig(
            path=str(table_path),
            version="PLACEHOLDER-0",
            extrapolation="reject",
            interpolation="linear",
            reference_temperature=300.0,
        ),
    )
    nonlinear = replace(
        linear,
        tps=table_material,
        validity=replace(
            linear.validity,
            tps_property_model="temperature_dependent_table",
        ),
        surface=replace(
            linear.surface,
            radiation_coupling="coupled_nonlinear",
        ),
    )
    case = SimulationCase(
        case_id="constant-property-equivalence",
        d_tps=base.thickness_candidates[0],
        heating_events=(
            HeatingEvent(
                amplitude=min(4000.0, base.heating.amplitude[1]),
                y_center=0.5 * base.lateral_length,
                t_center=0.5,
                sigma_y=0.15 * base.lateral_length,
                sigma_t=0.2,
            ),
        ),
        bond_defects=(),
    )
    linear_trajectory = TPSFVSolver(linear, case).solve()
    nonlinear_trajectory = TPSFVSolver(nonlinear, case).solve()
    difference = float(
        np.max(
            np.abs(
                linear_trajectory.temperatures
                - nonlinear_trajectory.temperatures
            )
        )
    )
    return {
        "maximum_temperature_difference_K": difference,
        "maximum_nonlinear_iterations": (
            nonlinear_trajectory.max_nonlinear_iterations
        ),
        "passed": bool(
            difference < 1.0e-12
            and nonlinear_trajectory.max_nonlinear_iterations == 1
        ),
    }


def radiation_coupling_equivalence(
    base: StudyConfig,
) -> dict[str, float | bool]:
    if not base.surface.reradiation_enabled:
        return {"applicable": False, "passed": True}
    common = replace(
        base,
        authoritative=False,
        mesh=MeshConfig(nx_tps=4, nx_bond=1, nx_back=2, ny=4),
        time=TimeConfig(
            dt=0.25,
            t_final=1.0,
            save_stride=1,
            horizon_candidates=(1.0,),
        ),
        heating=replace(base.heating, t_center=(0.2, 0.8)),
    )
    case = SimulationCase(
        case_id="radiation-coupling-equivalence",
        d_tps=base.thickness_candidates[0],
        heating_events=(
            HeatingEvent(
                amplitude=min(4000.0, base.heating.amplitude[1]),
                y_center=0.5 * base.lateral_length,
                t_center=0.5,
                sigma_y=0.15 * base.lateral_length,
                sigma_t=0.2,
            ),
        ),
        bond_defects=(),
    )
    boundary_config = replace(
        common,
        surface=replace(
            common.surface,
            radiation_coupling="boundary_response",
        ),
    )
    coupled_config = replace(
        common,
        surface=replace(
            common.surface,
            radiation_coupling="coupled_nonlinear",
        ),
    )
    boundary = TPSFVSolver(boundary_config, case).solve()
    coupled = TPSFVSolver(coupled_config, case).solve()
    cell_difference = float(
        np.max(np.abs(boundary.temperatures - coupled.temperatures))
    )
    surface_difference = float(
        np.max(
            np.abs(
                boundary.surface_temperatures
                - coupled.surface_temperatures
            )
        )
    )
    return {
        "applicable": True,
        "maximum_cell_temperature_difference_K": cell_difference,
        "maximum_surface_temperature_difference_K": surface_difference,
        "passed": bool(
            max(cell_difference, surface_difference) <= 1.0e-8
        ),
    }


def energy_balance_check(base: StudyConfig) -> dict[str, float | bool]:
    event = HeatingEvent(
        amplitude=min(1e5, base.heating.amplitude[1]),
        y_center=0.5 * base.lateral_length,
        t_center=5.0,
        sigma_y=0.15 * base.lateral_length,
        sigma_t=2.0,
    )
    case = SimulationCase(
        case_id="energy",
        d_tps=base.thickness_candidates[0],
        heating_events=(event,),
        bond_defects=(),
    )
    final = min(base.time.t_final, 20.0)
    final = round(final / base.time.dt) * base.time.dt
    solver = TPSFVSolver(base, case)
    trajectory = solver.solve(save_times=np.array([0.0, final]))
    initial_field = np.full(
        (solver.grid.nx, solver.grid.ny),
        base.initial_temperature,
        dtype=np.float64,
    )
    _, initial_rho_c = solver.properties_from_temperature(
        initial_field,
        mode="accepted",
    )
    energy_scale = float(np.sum(solver.volume * initial_rho_c))
    physical_input = abs(
        trajectory.expected_energy[-1] - trajectory.expected_energy[0]
    )
    scale = max(physical_input, energy_scale)
    relative = float(np.max(np.abs(trajectory.energy_residual)) / scale)
    stored_boundary = np.sum(trajectory.boundary_energy_increment, axis=1)
    stored_boundary_total = float(np.sum(stored_boundary))
    stored_boundary_error = float(abs(
        stored_boundary_total - trajectory.net_boundary_energy[-1]
    ))
    energy_tolerance = 1e-9 if trajectory.step_driver == "linear" else 1e-10
    interface_flux, left_trace, right_trace = solver.face_flux_and_traces(
        trajectory.temperatures,
        solver.grid.structural_face_index,
    )
    del interface_flux
    trace_jump = float(np.max(np.abs(left_trace - right_trace)))
    return {
        "relative_energy_residual": relative,
        "energy_scale_J_per_K": energy_scale,
        "stored_boundary_increment_error_J": stored_boundary_error,
        "boundary_input_energy_J": float(
            trajectory.boundary_input_energy[-1]
        ),
        "reradiated_energy_J": float(trajectory.radiated_energy[-1]),
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
        "maximum_hot_face_temperature_K": (
            trajectory.maximum_hot_face_temperature
        ),
        "maximum_nonlinear_iterations": int(
            np.max(trajectory.nonlinear_iteration_counts, initial=0)
        ),
        "interface_trace_jump": trace_jump,
        "minimum_temperature": trajectory.minimum_temperature,
        "passed": bool(
            relative < energy_tolerance
            and stored_boundary_error < 1e-10 * max(1.0, scale)
            and trace_jump < 1e-9
            and trajectory.minimum_temperature >= base.initial_temperature - 1e-8
        ),
    }


def physical_behavior_check(base: StudyConfig) -> dict[str, float | bool]:
    """Verify the non-MMS multilayer, defect, and adiabatic contracts."""
    event = HeatingEvent(
        amplitude=min(1e5, base.heating.amplitude[1]),
        y_center=0.5 * base.lateral_length,
        t_center=5.0,
        sigma_y=0.15 * base.lateral_length,
        sigma_t=2.0,
    )
    defect = BondDefect(
        severity=max(1.0, base.bond.severity[0]),
        y_center=0.5 * base.lateral_length,
        sigma=max(base.bond.sigma[0], 0.05 * base.lateral_length),
    )
    defective_case = SimulationCase(
        case_id="physical-behavior-defective",
        d_tps=base.thickness_candidates[0],
        heating_events=(event,),
        bond_defects=(defect,),
    )
    healthy_case = SimulationCase(
        case_id="physical-behavior-healthy",
        d_tps=base.thickness_candidates[0],
        heating_events=(event,),
        bond_defects=(),
    )
    final = min(base.time.t_final, 20.0)
    final = round(final / base.time.dt) * base.time.dt
    times = np.asarray(base.time.saved_times(final))
    defective_solver = TPSFVSolver(base, defective_case)
    healthy_solver = TPSFVSolver(base, healthy_case)
    defective = defective_solver.solve(save_times=times)
    healthy = healthy_solver.solve(save_times=times)

    trace_jumps = []
    for face_index in (
        defective_solver.grid.nx_tps - 1,
        defective_solver.grid.structural_face_index,
    ):
        _, left_trace, right_trace = defective_solver.face_flux_and_traces(
            defective.temperatures,
            face_index,
        )
        trace_jumps.append(float(np.max(np.abs(left_trace - right_trace))))

    row_sum = np.asarray(defective_solver.conduction.sum(axis=1)).ravel()
    symmetry = defective_solver.conduction - defective_solver.conduction.T
    symmetry_residual = (
        float(np.max(np.abs(symmetry.data))) if symmetry.nnz else 0.0
    )
    bond_row = defective_solver.grid.nx_tps
    conductivity_range = float(np.ptp(defective_solver.k[bond_row]))
    lateral_conductance_range = float(np.ptp(defective_solver.g_y[bond_row]))
    response_difference = float(
        np.max(np.abs(defective.temperatures - healthy.temperatures))
    )

    zero_case = SimulationCase(
        case_id="physical-behavior-zero-load",
        d_tps=healthy_case.d_tps,
        heating_events=(
            HeatingEvent(
                amplitude=1.0,
                y_center=0.5 * base.lateral_length,
                t_center=1.0e6,
                sigma_y=0.1 * base.lateral_length,
                sigma_t=1.0,
            ),
        ),
        bond_defects=(),
    )
    constant = TPSFVSolver(
        base,
        zero_case,
        flux_integral_fn=_zero_flux_integral,
    ).solve(save_times=np.asarray([0.0, final]))
    constant_state_drift = float(
        np.max(np.abs(constant.temperatures - base.initial_temperature))
    )
    maximum_trace_jump = max(trace_jumps)
    maximum_row_sum = float(np.max(np.abs(row_sum)))
    passed = bool(
        maximum_trace_jump < 1e-9
        and maximum_row_sum < 1e-10
        and symmetry_residual < 1e-12
        and conductivity_range > 0.0
        and lateral_conductance_range > 0.0
        and response_difference > 1e-8
        and constant_state_drift < 1e-9
        and defective.minimum_temperature >= base.initial_temperature - 1e-8
        and np.isfinite(defective.temperatures).all()
    )
    return {
        "maximum_interface_trace_jump_K": maximum_trace_jump,
        "maximum_conduction_row_sum": maximum_row_sum,
        "conduction_symmetry_residual": symmetry_residual,
        "bond_conductivity_range": conductivity_range,
        "bond_lateral_conductance_range": lateral_conductance_range,
        "defect_response_difference_K": response_difference,
        "adiabatic_constant_state_drift_K": constant_state_drift,
        "minimum_temperature_K": defective.minimum_temperature,
        "one_factorization_per_case": (
            defective.factorization_count == 1
            if defective.step_driver == "linear"
            else None
        ),
        "factorizations_per_step": (
            defective.factorization_count
            / max(1, len(defective.nonlinear_iteration_counts))
        ),
        "passed": passed,
    }


def select_horizon(
    base: StudyConfig,
    cases: list[SimulationCase],
) -> dict[str, object]:
    reports = []
    for horizon in base.time.horizon_candidates:
        candidate_reports = []
        for case in cases:
            solver = TPSFVSolver(base, case)
            trajectory = solver.solve(save_times=np.asarray(base.time.saved_times(horizon)))
            candidate_reports.append(
                horizon_diagnostics(
                    solver,
                    trajectory,
                    design_window=base.time.t_final,
                )
            )
        accepted = bool(candidate_reports) and all(
            bool(report["accepted"]) for report in candidate_reports
        )
        reports.append({
            "horizon": horizon,
            "accepted": accepted,
            "cases": candidate_reports,
        })
        if accepted:
            return {"selected_horizon": horizon, "reports": reports}
    return {"selected_horizon": None, "reports": reports}


def thin_bond_study(
    base: StudyConfig,
    thickness_factors: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125),
) -> dict[str, object]:
    if base.validity.tps_property_model != "constant_effective":
        raise ValueError(
            "thin_bond_study is defined only for the constant-property model."
        )
    event = HeatingEvent(
        amplitude=min(base.heating.amplitude[1], 1e5),
        y_center=0.5 * base.lateral_length,
        t_center=5.0,
        sigma_y=0.15 * base.lateral_length,
        sigma_t=2.0,
    )
    case = SimulationCase(
        case_id="thin-bond",
        d_tps=base.thickness_candidates[0],
        heating_events=(event,),
        bond_defects=(
            BondDefect(
                severity=max(1.0, base.bond.severity[0]),
                y_center=0.5 * base.lateral_length,
                sigma=max(base.bond.sigma[0], 0.05 * base.lateral_length),
            ),
        ),
    )
    times = np.asarray(base.time.saved_times(min(base.time.t_final, 20.0)))
    y_grid = TPSFVSolver(base, case).grid.y_centers
    rc_y = base.bond_thickness / case.bond_conductivity(y_grid, base)
    interface_solver = TPSFVSolver(
        base,
        case,
        bond_thickness=0.0,
        nx_bond=0,
        interface_resistance_y=rc_y,
    )
    interface_trajectory = interface_solver.solve(save_times=times)
    face_index = interface_solver.grid.nx_tps - 1
    flux_ref, left_ref, right_ref = interface_solver.face_flux_and_traces(
        interface_trajectory.temperatures,
        face_index,
    )

    rows = []
    for factor in thickness_factors:
        thickness = base.bond_thickness * factor
        conductivity = thickness / rc_y
        finite_solver = TPSFVSolver(
            base,
            case,
            bond_thickness=thickness,
            nx_bond=base.mesh.nx_bond,
            bond_conductivity_override=conductivity,
        )
        trajectory = finite_solver.solve(save_times=times)
        flux_left, _, tps_trace = finite_solver.face_flux_and_traces(
            trajectory.temperatures,
            finite_solver.grid.nx_tps - 1,
        )
        flux_right, _, back_trace = finite_solver.face_flux_and_traces(
            trajectory.temperatures,
            finite_solver.grid.structural_face_index,
        )
        rows.append({
            "bond_thickness": thickness,
            "tps_trace_error": float(np.sqrt(np.mean((tps_trace - left_ref) ** 2))),
            "back_trace_error": float(np.sqrt(np.mean((back_trace - right_ref) ** 2))),
            "transmitted_flux_error": float(np.sqrt(np.mean((flux_right - flux_ref) ** 2))),
            "bond_flux_mismatch": float(np.sqrt(np.mean((flux_left - flux_right) ** 2))),
        })
    keys = ("tps_trace_error", "back_trace_error", "transmitted_flux_error")
    improving = all(rows[-1][key] < rows[0][key] for key in keys)
    monotonic = all(
        rows[index + 1][key] < rows[index][key]
        for key in keys
        for index in range(len(rows) - 1)
    )
    orders = {
        key: [
            float(
                np.log(rows[index][key] / rows[index + 1][key])
                / np.log(
                    rows[index]["bond_thickness"]
                    / rows[index + 1]["bond_thickness"]
                )
            )
            for index in range(len(rows) - 1)
        ]
        for key in keys
    }
    return {
        "resistance_profile": {
            "minimum": float(np.min(rc_y)),
            "maximum": float(np.max(rc_y)),
            "spatially_varying": bool(np.ptp(rc_y) > 0.0),
        },
        "levels": rows,
        "orders": orders,
        "monotonic_convergence": monotonic,
        "finest_improves_over_coarsest": improving,
    }


def run_verification(
    config: StudyConfig,
    output_path: str | Path | None = None,
    *,
    include_slow: bool = True,
) -> dict[str, object]:
    report: dict[str, object] = {
        "energy": energy_balance_check(config),
        "physical_behavior": physical_behavior_check(config),
        "material_properties": config.property_provenance,
        "constant_property_equivalence": (
            constant_property_equivalence(config)
        ),
        "radiation_coupling_equivalence": (
            radiation_coupling_equivalence(config)
        ),
    }
    if include_slow:
        spatial = spatial_convergence(config)
        temporal = temporal_convergence(config)
        nonlinear_model = (
            config.validity.tps_property_model
            == "temperature_dependent_table"
        )
        thin = None if nonlinear_model else thin_bond_study(config)
        nonlinear_spatial = (
            nonlinear_spatial_convergence(config)
            if nonlinear_model
            else None
        )
        nonlinear_temporal = (
            nonlinear_temporal_convergence(config)
            if nonlinear_model
            else None
        )
        nonlinear_steady = (
            nonlinear_steady_conduction(config)
            if nonlinear_model
            else None
        )
        radiation_temporal = (
            radiation_temporal_convergence(config)
            if config.surface.reradiation_enabled
            else None
        )
        radiation_flux_levels = (
            radiation_flux_level_temporal_convergence(config)
            if config.surface.reradiation_enabled
            else None
        )
        report.update({
            "spatial": asdict(spatial),
            "temporal": asdict(temporal),
            "radiation_temporal": (
                None
                if radiation_temporal is None
                else asdict(radiation_temporal)
            ),
            "radiation_flux_level_temporal": radiation_flux_levels,
            "thin_bond": thin,
            "nonlinear_spatial": (
                None
                if nonlinear_spatial is None
                else asdict(nonlinear_spatial)
            ),
            "nonlinear_temporal": (
                None
                if nonlinear_temporal is None
                else asdict(nonlinear_temporal)
            ),
            "nonlinear_steady_conduction": nonlinear_steady,
            "passed": bool(
                report["energy"]["passed"]
                and report["physical_behavior"]["passed"]
                and report["constant_property_equivalence"]["passed"]
                and report["radiation_coupling_equivalence"]["passed"]
                and spatial.minimum_order >= 1.8
                and temporal.minimum_order >= 1.8
                and (
                    radiation_temporal is None
                    or radiation_temporal.minimum_order >= 1.8
                )
                and (
                    radiation_flux_levels is None
                    or radiation_flux_levels["passed"]
                )
                and (
                    thin is None
                    or thin["monotonic_convergence"]
                )
                and (
                    nonlinear_spatial is None
                    or nonlinear_spatial.minimum_order >= 1.8
                )
                and (
                    nonlinear_temporal is None
                    or nonlinear_temporal.minimum_order >= 1.8
                )
                and (
                    nonlinear_steady is None
                    or nonlinear_steady["passed"]
                )
            ),
        })
    else:
        report["passed"] = bool(
            report["energy"]["passed"]
            and report["physical_behavior"]["passed"]
            and report["constant_property_equivalence"]["passed"]
            and report["radiation_coupling_equivalence"]["passed"]
        )
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
