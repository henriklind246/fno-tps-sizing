from __future__ import annotations

from dataclasses import replace

import numpy as np

from fno_tps.acceptance import assess_trajectory
from fno_tps.physics import TPSFVSolver
from fno_tps.problem import HeatingEvent, SimulationCase


def test_hot_face_limit_violation_is_invalid_not_infeasible(
    demo_config,
) -> None:
    case = SimulationCase(
        "hot-face-limit",
        demo_config.thickness_candidates[0],
        (
            HeatingEvent(
                amplitude=1.0e8,
                y_center=0.10,
                t_center=1.5,
                sigma_y=0.025,
                sigma_t=0.5,
            ),
        ),
        (),
    )
    solver = TPSFVSolver(demo_config, case)
    trajectory = solver.solve()
    qoi = solver.quantities_of_interest(
        trajectory.temperatures,
        trajectory.times,
    )
    result = assess_trajectory(
        demo_config,
        trajectory,
        bond_max=qoi["bond_max"],
        structural_interface_max=qoi["structural_interface_max"],
    )
    assert result.classification == "invalid"
    assert not result.valid
    assert result.feasible is False
    assert "hot_face_temperature_limit" in result.invalid_reasons


def test_energy_residual_is_an_acceptance_gate(
    demo_config,
    three_cases,
) -> None:
    trajectory = TPSFVSolver(demo_config, three_cases[0]).solve()
    corrupted = replace(
        trajectory,
        energy_residual=np.full_like(trajectory.energy_residual, 1.0),
    )
    result = assess_trajectory(demo_config, corrupted)
    assert result.classification == "invalid"
    assert "energy_balance_residual" in result.invalid_reasons


def test_nonlinear_trajectory_exposes_acceptance_diagnostics(
    incident_config,
) -> None:
    coupled = replace(
        incident_config,
        surface=replace(
            incident_config.surface,
            radiation_coupling="coupled_nonlinear",
        ),
    )
    case = SimulationCase(
        "diagnostics",
        coupled.thickness_candidates[0],
        (HeatingEvent(4000.0, 0.10, 1.5, 0.025, 0.5),),
        (),
    )
    trajectory = TPSFVSolver(coupled, case).solve()
    assert trajectory.solver_converged
    assert trajectory.nonlinear_final_residual_norms.shape == (
        len(trajectory.energy_times) - 1,
    )
    assert trajectory.nonlinear_damped_iteration_counts.shape == (
        len(trajectory.energy_times) - 1,
    )
    assert trajectory.nonlinear_backtrack_counts.shape == (
        len(trajectory.energy_times) - 1,
    )
    assert np.isfinite(trajectory.property_query_temperature_range).all()
