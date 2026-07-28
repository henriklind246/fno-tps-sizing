from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fno_tps.config import Material, PropertyTableConfig, load_study_config
from fno_tps.physics import TPSFVSolver
from fno_tps.problem import HeatingEvent, SimulationCase
from fno_tps.verification import (
    _manufactured_case,
    _nonlinear_exact_temperature,
    _nonlinear_manufactured_config,
    _nonlinear_manufactured_source,
    _nonlinear_region_properties,
    _zero_flux_integral,
    nonlinear_spatial_convergence,
    nonlinear_steady_conduction,
    nonlinear_temporal_convergence,
)


def _small_configs(demo_config):
    common = replace(
        demo_config,
        authoritative=False,
        mesh=replace(
            demo_config.mesh,
            nx_tps=4,
            nx_bond=1,
            nx_back=2,
            ny=4,
        ),
        time=replace(
            demo_config.time,
            dt=0.25,
            t_final=1.0,
            save_stride=1,
            horizon_candidates=(1.0,),
        ),
        heating=replace(demo_config.heating, t_center=(0.2, 0.8)),
    )
    table_material = Material(
        model="temperature_dependent_table",
        reference_k=5.96e-2,
        reference_rho_c=1.203e5,
        property_table=PropertyTableConfig(
            path="conf/materials/tps_placeholder.yaml",
            version="PLACEHOLDER-0",
            reference_temperature=300.0,
        ),
    )
    nonlinear = replace(
        common,
        tps=table_material,
        validity=replace(
            common.validity,
            tps_property_model="temperature_dependent_table",
        ),
        surface=replace(
            common.surface,
            radiation_coupling="coupled_nonlinear",
        ),
    )
    common.validate()
    nonlinear.validate()
    return common, nonlinear


def _case() -> SimulationCase:
    return SimulationCase(
        "nonlinear",
        0.003,
        (HeatingEvent(4000.0, 0.1, 0.5, 0.025, 0.2),),
        (),
    )


def test_vectorized_assembly_matches_loop_oracle(demo_config) -> None:
    solver = TPSFVSolver(demo_config, _case())
    actual = solver.conduction
    expected = solver._assemble_conduction_reference()
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_allclose(
        actual.data,
        expected.data,
        rtol=0.0,
        atol=1.0e-15 * np.max(np.abs(expected.data)),
    )


def test_li900_directional_conductivity_drives_distinct_fv_axes() -> None:
    config = load_study_config("conf/nonlinear-pilot.yaml")
    solver = TPSFVSolver(config, _case())
    tps = slice(0, config.mesh.nx_tps)
    assert np.all(solver.k_y[tps] > solver.k_x[tps])
    assert solver.k_x[0, 0] == pytest.approx(
        config.tps.reference_k
    )
    assert solver.k_y[0, 0] == pytest.approx(
        config.tps.reference_k_y
    )
    expected_lateral_conductance = (
        solver.grid.dx[0] * solver.k_y[0, 0] / solver.grid.dy
    )
    assert solver.g_y[0, 0] == pytest.approx(expected_lateral_conductance)
    assert solver.surface_conductance(np.asarray([300.0]))[0] == pytest.approx(
        2.0 * config.tps.reference_k / solver.grid.dx[0]
    )


def test_constant_table_matches_linear_driver(demo_config) -> None:
    linear_config, nonlinear_config = _small_configs(demo_config)
    linear = TPSFVSolver(linear_config, _case()).solve()
    nonlinear = TPSFVSolver(nonlinear_config, _case()).solve()
    assert nonlinear.step_driver == "nonlinear"
    assert nonlinear.max_nonlinear_iterations == 1
    assert np.max(
        np.abs(linear.temperatures - nonlinear.temperatures)
    ) < 1.0e-12


def test_adiabatic_uniform_state_is_exact_fixed_point(demo_config) -> None:
    _, config = _small_configs(demo_config)
    config = replace(
        config,
        time=replace(
            config.time,
            t_final=0.5,
            horizon_candidates=(0.5,),
        ),
    )
    solver = TPSFVSolver(
        config,
        _case(),
        flux_integral_fn=lambda y, t0, t1: np.zeros_like(y),
    )
    trajectory = solver.solve()
    assert np.max(
        np.abs(trajectory.temperatures - config.initial_temperature)
    ) < 1.0e-13
    assert trajectory.accepted_range_excursions == 0


def test_enthalpy_energy_balance_uses_stored_boundary_increment(
    demo_config,
) -> None:
    _, config = _small_configs(demo_config)
    trajectory = TPSFVSolver(config, _case()).solve()
    input_scale = max(abs(trajectory.net_boundary_energy[-1]), 1.0)
    assert (
        np.max(np.abs(trajectory.energy_residual)) / input_scale
        < 1.0e-10
    )
    np.testing.assert_allclose(
        np.sum(trajectory.boundary_energy_increment),
        trajectory.net_boundary_energy[-1],
        rtol=1.0e-12,
        atol=1.0e-10,
    )


def test_coupled_radiation_uses_global_nonlinear_driver(
    incident_config,
) -> None:
    config = replace(
        incident_config,
        surface=replace(
            incident_config.surface,
            radiation_coupling="coupled_nonlinear",
        ),
        heating=replace(incident_config.heating, t_center=(1.0, 3.0)),
    )
    config.validate()
    solver = TPSFVSolver(config, _case())
    trajectory = solver.solve()
    assert trajectory.step_driver == "nonlinear"
    assert trajectory.factorization_count > 1
    assert trajectory.max_nonlinear_iterations >= 1
    np.testing.assert_allclose(
        np.sum(trajectory.boundary_energy_increment),
        trajectory.net_boundary_energy[-1],
        rtol=1.0e-12,
        atol=1.0e-9,
    )
    surface = trajectory.surface_temperatures[-1]
    incident = _case().heat_flux(
        solver.grid.y_centers,
        trajectory.times[-1],
    )
    residual = (
        solver.surface_conductance(trajectory.temperatures[-1, 0])
        * (surface - trajectory.temperatures[-1, 0])
        + solver.radiative_heat_flux(surface)
        - incident
    )
    assert np.max(np.abs(residual)) < 1.0e-6


def test_radiation_coupling_equivalence(incident_config) -> None:
    common = replace(
        incident_config,
        authoritative=False,
        mesh=replace(
            incident_config.mesh,
            nx_tps=4,
            nx_bond=1,
            nx_back=2,
            ny=4,
        ),
        time=replace(
            incident_config.time,
            dt=0.25,
            t_final=1.0,
            save_stride=1,
            horizon_candidates=(1.0,),
        ),
        heating=replace(incident_config.heating, t_center=(0.2, 0.8)),
    )
    boundary_response = TPSFVSolver(common, _case()).solve()
    coupled = TPSFVSolver(
        replace(
            common,
            surface=replace(
                common.surface,
                radiation_coupling="coupled_nonlinear",
            ),
        ),
        _case(),
    ).solve()
    assert np.max(
        np.abs(
            boundary_response.temperatures
            - coupled.temperatures
        )
    ) < 1.0e-8
    assert np.max(
        np.abs(
            boundary_response.surface_temperatures
            - coupled.surface_temperatures
        )
    ) < 1.0e-8


def test_nonlinear_step_reports_nonconvergence(demo_config) -> None:
    config = _nonlinear_manufactured_config(
        demo_config,
        replace(
            demo_config.mesh,
            nx_tps=4,
            nx_bond=2,
            nx_back=4,
            ny=8,
        ),
        0.004,
        0.004,
    )
    config = replace(
        config,
        solver=replace(
            config.solver,
            nonlinear=replace(
                config.solver.nonlinear,
                max_iterations=1,
            ),
        ),
    )
    solver = TPSFVSolver(
        config,
        _manufactured_case(),
        source_fn=_nonlinear_manufactured_source,
        flux_integral_fn=_zero_flux_integral,
        region_properties=_nonlinear_region_properties(config),
    )
    initial = _nonlinear_exact_temperature(solver.X, solver.Y, 0.0)
    with pytest.raises(
        RuntimeError,
        match=r"step=0.*worst_cell",
    ):
        solver.solve(
            save_times=np.asarray([0.0, 0.004]),
            initial_temperature=initial,
        )


def test_boundary_jacobian_matches_constant_k_finite_difference(
    incident_config,
) -> None:
    config = replace(
        incident_config,
        surface=replace(
            incident_config.surface,
            radiation_coupling="coupled_nonlinear",
        ),
    )
    solver = TPSFVSolver(config, _case())
    cell = np.full(solver.grid.ny, 350.0)
    incident = np.full(solver.grid.ny, 4000.0)
    _, _, h_eff = solver._surface_state(
        cell,
        incident,
        mode="iterate",
    )
    epsilon = 1.0e-4
    plus, _, _ = solver._surface_state(
        cell + epsilon,
        incident,
        mode="iterate",
    )
    minus, _, _ = solver._surface_state(
        cell - epsilon,
        incident,
        mode="iterate",
    )
    finite_difference = (
        0.5
        * solver.dt
        * solver.grid.dy
        * (
            solver.radiative_heat_flux(plus)
            - solver.radiative_heat_flux(minus)
        )
        / (2.0 * epsilon)
    )
    expected = 0.5 * solver.dt * solver.grid.dy * h_eff
    np.testing.assert_allclose(
        finite_difference,
        expected,
        rtol=1.0e-7,
        atol=1.0e-10,
    )


def test_eliminated_surface_matches_explicit_augmented_system(
    incident_config,
) -> None:
    from scipy.optimize import root

    table_material = Material(
        model="temperature_dependent_table",
        reference_k=1.0,
        reference_rho_c=1.0,
        property_table=PropertyTableConfig(
            path="conf/materials/nonlinear_verification.yaml",
            version="MMS-1",
            reference_temperature=400.0,
        ),
    )
    config = replace(
        incident_config,
        authoritative=False,
        tps=table_material,
        validity=replace(
            incident_config.validity,
            tps_property_model="temperature_dependent_table",
            hot_face=replace(
                incident_config.validity.hot_face,
                study_temperature_limit=500.0,
            ),
            property_tables=replace(
                incident_config.validity.property_tables,
                minimum_temperature=250.0,
                maximum_temperature=500.0,
            ),
        ),
        surface=replace(
            incident_config.surface,
            radiation_coupling="coupled_nonlinear",
        ),
        mesh=replace(
            incident_config.mesh,
            nx_tps=4,
            nx_bond=1,
            nx_back=2,
            ny=2,
        ),
        time=replace(
            incident_config.time,
            dt=0.25,
            t_final=0.25,
            save_stride=1,
            horizon_candidates=(0.25,),
        ),
        solver=replace(
            incident_config.solver,
            nonlinear=replace(
                incident_config.solver.nonlinear,
                residual_temperature_tolerance=1.0e-7,
            ),
        ),
    )
    case = SimulationCase(
        "explicit-surface",
        0.003,
        (HeatingEvent(1.0, 0.1, 0.125, 0.05, 0.1),),
        (),
    )
    solver = TPSFVSolver(config, case)
    trajectory = solver.solve()
    old = trajectory.temperatures[0]
    production = trajectory.temperatures[-1]
    production_surface = trajectory.surface_temperatures[-1]
    incident_old = case.heat_flux(solver.grid.y_centers, 0.0)
    incident_next = case.heat_flux(solver.grid.y_centers, 0.25)
    old_surface = solver.surface_temperature(old[0], incident_old)
    old_radiation = solver.radiative_heat_flux(old_surface)
    incident_energy = solver._boundary_energy(0.0, 0.25)
    old_enthalpy = solver.enthalpy_from_temperature(old)
    solver._update_property_state(old, mode="accepted")
    old_flux = (
        solver.conduction @ old.reshape(-1)
    ).reshape(old.shape)

    def augmented_residual(unknown: np.ndarray) -> np.ndarray:
        cells = unknown[: old.size].reshape(old.shape)
        surface = unknown[old.size :]
        enthalpy = solver.enthalpy_from_temperature(cells, mode="iterate")
        solver._update_property_state(cells, mode="iterate")
        flux = (
            solver.conduction @ cells.reshape(-1)
        ).reshape(cells.shape)
        residual = (
            solver.volume * (enthalpy - old_enthalpy)
            - 0.5 * solver.dt * (flux + old_flux)
        )
        boundary = incident_energy - (
            0.5
            * solver.dt
            * solver.grid.dy
            * (old_radiation + solver.radiative_heat_flux(surface))
        )
        residual[0] -= boundary
        surface_residual = (
            solver.surface_conductance(cells[0], mode="iterate")
            * (surface - cells[0])
            + solver.radiative_heat_flux(surface)
            - incident_next
        )
        return np.concatenate((residual.reshape(-1), surface_residual))

    solution = root(
        augmented_residual,
        np.concatenate((production.reshape(-1), production_surface)),
        tol=1.0e-11,
    )
    assert solution.success
    explicit_cells = solution.x[: old.size].reshape(old.shape)
    explicit_surface = solution.x[old.size :]
    np.testing.assert_allclose(
        production,
        explicit_cells,
        rtol=0.0,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        production_surface,
        explicit_surface,
        rtol=0.0,
        atol=1.0e-9,
    )
    production_q_net = solver.surface_conductance(
        production[0],
        mode="iterate",
    ) * (production_surface - production[0])
    explicit_q_net = solver.surface_conductance(
        explicit_cells[0],
        mode="iterate",
    ) * (explicit_surface - explicit_cells[0])
    np.testing.assert_allclose(
        production_q_net,
        explicit_q_net,
        rtol=0.0,
        atol=1.0e-9 * max(1.0, float(np.max(incident_next))),
    )


@pytest.mark.slow
def test_nonlinear_mms_is_second_order(demo_config) -> None:
    assert nonlinear_spatial_convergence(demo_config).minimum_order >= 1.8
    assert nonlinear_temporal_convergence(demo_config).minimum_order >= 1.8
    assert nonlinear_steady_conduction(demo_config)["passed"]
