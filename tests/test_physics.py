from __future__ import annotations

import numpy as np
import pytest

from fno_tps.physics import TPSFVSolver, horizon_diagnostics
from fno_tps.verification import (
    energy_balance_check,
    physical_behavior_check,
    radiation_flux_level_temporal_convergence,
    radiation_temporal_convergence,
    spatial_convergence,
    temporal_convergence,
    thin_bond_study,
)


def test_solver_conserves_exact_boundary_energy(demo_config, three_cases):
    solver = TPSFVSolver(demo_config, three_cases[0])
    trajectory = solver.solve(save_times=np.asarray(demo_config.time.saved_times()))
    injected = trajectory.expected_energy[-1] - trajectory.expected_energy[0]
    residual = np.max(np.abs(trajectory.energy_residual))
    assert residual / max(abs(injected), 1.0) < 1e-10
    assert trajectory.factorization_count == 1
    assert trajectory.minimum_temperature >= demo_config.initial_temperature - 1e-8
    assert trajectory.maximum_hot_face_temperature >= demo_config.initial_temperature
    assert trajectory.maximum_hot_face_temperature >= np.max(
        trajectory.temperatures[:, 0, :]
    )


def test_incident_reradiation_surface_and_energy_balance(incident_config):
    from fno_tps.problem import HeatingEvent, SimulationCase

    event = HeatingEvent(4000.0, 0.10, 1.5, 0.025, 0.5)
    case = SimulationCase("incident", 0.003, (event,), ())
    solver = TPSFVSolver(incident_config, case)
    trajectory = solver.solve(
        save_times=np.asarray(incident_config.time.saved_times())
    )
    scale = max(abs(trajectory.net_boundary_energy[-1]), 1.0)
    assert np.max(np.abs(trajectory.energy_residual)) / scale < 1e-9
    assert trajectory.boundary_input_energy[-1] > 0.0
    assert 0.0 < trajectory.radiated_energy[-1] < trajectory.boundary_input_energy[-1]
    assert trajectory.net_boundary_energy[-1] == pytest.approx(
        trajectory.boundary_input_energy[-1] - trajectory.radiated_energy[-1],
        rel=1e-12,
        abs=1e-10,
    )
    assert trajectory.maximum_hot_face_temperature >= np.max(
        trajectory.temperatures[:, 0, :]
    )
    incident = case.heat_flux(
        solver.grid.y_centers,
        trajectory.times[-1],
    )
    surface = trajectory.surface_temperatures[-1]
    residual = (
        solver.surface_heat_transfer_coefficient
        * (surface - trajectory.temperatures[-1, 0])
        + solver.radiative_heat_flux(surface)
        - incident
    )
    assert np.max(np.abs(residual)) < 1e-6
    assert np.max(trajectory.nonlinear_iteration_counts) > 0
    assert trajectory.factorization_count == 1


def test_reradiation_reduces_inward_energy_for_same_applied_flux(
    incident_config,
):
    from dataclasses import replace

    from fno_tps.config import SurfaceConfig
    from fno_tps.problem import HeatingEvent, SimulationCase

    event = HeatingEvent(4000.0, 0.10, 1.5, 0.025, 0.5)
    case = SimulationCase("comparison", 0.003, (event,), ())
    radiating = TPSFVSolver(incident_config, case).solve(
        save_times=np.asarray(incident_config.time.saved_times())
    )
    net_config = replace(
        incident_config,
        surface=SurfaceConfig.from_dict(None),
        heating=replace(
            incident_config.heating,
            interpretation="effective_net_inward_conductive_heat_flux",
        ),
    )
    prescribed = TPSFVSolver(net_config, case).solve(
        save_times=np.asarray(net_config.time.saved_times())
    )
    assert radiating.net_boundary_energy[-1] < prescribed.net_boundary_energy[-1]
    assert np.max(radiating.temperatures) < np.max(prescribed.temperatures)


@pytest.mark.slow
def test_incident_radiation_boundary_is_second_order_in_time():
    from fno_tps.config import load_study_config

    result = radiation_temporal_convergence(
        load_study_config("conf/incident-radiation-demo.yaml")
    )
    assert result.minimum_order >= 1.8


@pytest.mark.slow
def test_low_medium_high_flux_temporal_convergence():
    from fno_tps.config import load_study_config

    result = radiation_flux_level_temporal_convergence(
        load_study_config("conf/nonlinear-pilot.yaml")
    )
    assert result["passed"]


def test_local_bond_conductivity_changes_lateral_conductance(
    demo_config,
    heating_event,
):
    from fno_tps.problem import BondDefect, SimulationCase

    case = SimulationCase(
        "defect",
        0.02,
        (heating_event,),
        (BondDefect(2.0, 0.1, 0.015),),
    )
    solver = TPSFVSolver(demo_config, case)
    bond_row = demo_config.mesh.nx_tps
    assert np.ptp(solver.k[bond_row]) > 0.0
    assert np.ptp(solver.g_y[bond_row]) > 0.0


def test_interface_reconstruction_has_no_jump_for_finite_bond(
    demo_config,
    three_cases,
):
    solver = TPSFVSolver(demo_config, three_cases[0])
    trajectory = solver.solve(save_times=np.asarray(demo_config.time.saved_times()))
    _, left, right = solver.face_flux_and_traces(
        trajectory.temperatures,
        solver.grid.structural_face_index,
    )
    np.testing.assert_allclose(left, right, atol=1e-10)


def test_bond_qoi_includes_reconstructed_interface_temperatures(
    demo_config,
    three_cases,
):
    solver = TPSFVSolver(demo_config, three_cases[0])
    temperatures = np.full(
        (2, solver.grid.nx, solver.grid.ny),
        demo_config.initial_temperature,
    )
    temperatures[:, solver.grid.nx_tps - 1, :] = 500.0
    temperatures[:, solver.grid.bond_slice, :] = 400.0
    temperatures[:, solver.grid.backing_slice, :] = 350.0
    envelope = solver.bond_temperature_envelope(temperatures)
    _, _, hot_side_trace = solver.face_flux_and_traces(
        temperatures,
        solver.grid.nx_tps - 1,
    )
    qoi = solver.quantities_of_interest(
        temperatures,
        np.asarray([0.0, 1.0]),
    )

    assert np.max(envelope) == pytest.approx(np.max(hot_side_trace))
    assert qoi["bond_max"] == pytest.approx(np.max(hot_side_trace))
    assert qoi["bond_max"] > np.max(
        temperatures[:, solver.grid.bond_slice, :]
    )


def test_adiabatic_horizon_accepts_endpoint_control(demo_config, three_cases):
    solver = TPSFVSolver(demo_config, three_cases[0])
    trajectory = solver.solve(save_times=np.asarray(demo_config.time.saved_times()))
    report = horizon_diagnostics(
        solver,
        trajectory,
        design_window=demo_config.time.t_final,
        final_steps=2,
    )
    assert report["accepted"]
    assert report["design_window_covered"]
    assert report["bond_control"] in {"endpoint", "interior_peak"}
    assert report["structural_interface_control"] in {
        "endpoint",
        "interior_peak",
    }


def test_energy_verification_gate_passes():
    from fno_tps.config import load_study_config

    assert energy_balance_check(load_study_config("conf/demo.yaml"))["passed"]


def test_physical_behavior_gate_passes():
    from fno_tps.config import load_study_config

    assert physical_behavior_check(load_study_config("conf/demo.yaml"))["passed"]


@pytest.mark.slow
def test_manufactured_solution_is_second_order():
    from fno_tps.config import load_study_config

    config = load_study_config("conf/demo.yaml")
    assert spatial_convergence(config).minimum_order >= 1.8
    assert temporal_convergence(config).minimum_order >= 1.8


@pytest.mark.slow
def test_thin_bond_limit_improves():
    from fno_tps.config import load_study_config

    report = thin_bond_study(load_study_config("conf/demo.yaml"))
    assert report["resistance_profile"]["spatially_varying"]
    assert report["monotonic_convergence"]
    assert report["finest_improves_over_coarsest"]
