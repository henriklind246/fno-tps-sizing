from __future__ import annotations

from pathlib import Path

import pytest

import fno_tps.screening as screening_module
from fno_tps.screening import (
    _run_stage6a_refinement,
    evaluate_stage6a_promotion,
    _amplitude_interval,
    _matrix_at_amplitude,
    _screen_scenarios,
    run_incident_radiation_screen,
    run_pulse_width_screen,
)


def _response_rows() -> list[dict[str, float | str]]:
    return [
        {
            "scenario_id": "a",
            "d_tps": 0.003,
            "hot_rise_per_amplitude_K_per_W_m2": 0.10,
            "bond_rise_per_amplitude_K_per_W_m2": 0.20,
            "structural_rise_per_amplitude_K_per_W_m2": 0.10,
        },
        {
            "scenario_id": "a",
            "d_tps": 0.024,
            "hot_rise_per_amplitude_K_per_W_m2": 0.10,
            "bond_rise_per_amplitude_K_per_W_m2": 0.05,
            "structural_rise_per_amplitude_K_per_W_m2": 0.04,
        },
        {
            "scenario_id": "b",
            "d_tps": 0.003,
            "hot_rise_per_amplitude_K_per_W_m2": 0.08,
            "bond_rise_per_amplitude_K_per_W_m2": 0.10,
            "structural_rise_per_amplitude_K_per_W_m2": 0.08,
        },
        {
            "scenario_id": "b",
            "d_tps": 0.024,
            "hot_rise_per_amplitude_K_per_W_m2": 0.08,
            "bond_rise_per_amplitude_K_per_W_m2": 0.04,
            "structural_rise_per_amplitude_K_per_W_m2": 0.03,
        },
    ]


def test_amplitude_interval_combines_validity_and_design_bounds():
    interval = _amplitude_interval(
        _response_rows(),
        initial_temperature=300.0,
        hot_face_limit=500.0,
        bond_limit=450.0,
        structural_limit=400.0,
        thinnest=0.003,
        thickest=0.024,
    )
    assert interval["hot_face_amplitude_upper_bound_W_m2"] == pytest.approx(2000.0)
    assert interval["thin_failure_amplitude_lower_bound_W_m2"] == pytest.approx(750.0)
    assert interval["thick_feasible_amplitude_upper_bound_W_m2"] == pytest.approx(
        2500.0
    )
    assert interval["amplitude_interval_width_W_m2"] == pytest.approx(1250.0)
    assert interval["linear_interval_viable"]
    assert interval["comfortable_interval"]
    assert interval["recommended_amplitude_W_m2"] == pytest.approx(1375.0)


def test_recommended_amplitude_matrix_has_thin_failure_and_thick_feasibility():
    rows = _response_rows()
    rows.extend([
        {
            "scenario_id": "a",
            "d_tps": 0.012,
            "hot_rise_per_amplitude_K_per_W_m2": 0.10,
            "bond_rise_per_amplitude_K_per_W_m2": 0.08,
            "structural_rise_per_amplitude_K_per_W_m2": 0.06,
        },
        {
            "scenario_id": "b",
            "d_tps": 0.012,
            "hot_rise_per_amplitude_K_per_W_m2": 0.08,
            "bond_rise_per_amplitude_K_per_W_m2": 0.06,
            "structural_rise_per_amplitude_K_per_W_m2": 0.05,
        },
    ])
    matrix = _matrix_at_amplitude(
        rows,
        1375.0,
        initial_temperature=300.0,
        hot_face_limit=500.0,
        bond_limit=450.0,
        structural_limit=400.0,
        thicknesses=(0.003, 0.012, 0.024),
    )
    assert matrix["all_cells_within_validity"]
    assert matrix["thinnest_design_sometimes_infeasible"]
    assert matrix["thickest_design_all_feasible"]


def test_screen_scenarios_hold_spatial_design_fixed_across_width(demo_config):
    scenarios = _screen_scenarios(demo_config, 60.0, 1000.0)
    assert [scenario.case_id for scenario in scenarios] == [
        "screen-centered-single-healthy",
        "screen-localized-single-defect",
        "screen-moving-triple-two-defects",
    ]
    assert [scenario.event_count for scenario in scenarios] == [1, 1, 3]
    assert [scenario.defect_count for scenario in scenarios] == [0, 1, 2]
    assert [event.t_center for event in scenarios[-1].heating_events] == [
        180.0,
        210.0,
        240.0,
    ]
    assert {
        event.amplitude
        for scenario in scenarios
        for event in scenario.heating_events
    } == {1000.0}


def test_pulse_screen_reuses_long_trajectories_and_writes_reports(
    tmp_path,
    demo_config,
):
    output = tmp_path / "screen"
    report = run_pulse_width_screen(
        demo_config,
        output,
        pulse_widths=(0.5,),
        horizons=(3.0, 4.0),
        thicknesses=(0.003, 0.012, 0.024),
        reference_amplitude=1000.0,
    )
    assert report["reference_solve_count"] == 9
    assert report["expected_reference_solve_count"] == 9
    assert report["candidate_horizon_count"] == 2
    assert report["separate_solves_avoided_by_trajectory_reuse"] == 9
    assert len(report["reference_solves"]) == 9
    assert Path(output / "pulse_reference_solves.csv").exists()
    assert Path(output / "pulse_response_matrix.csv").exists()
    assert Path(output / "pulse_amplitude_intervals.csv").exists()
    assert Path(output / "pulse_screen_report.json").exists()


def test_linear_screen_rejects_nonlinear_surface_model(
    tmp_path,
    incident_config,
):
    with pytest.raises(ValueError, match="incident-screen"):
        run_pulse_width_screen(
            incident_config,
            tmp_path / "invalid-linear",
            pulse_widths=(0.5,),
            horizons=(4.0,),
            thicknesses=(0.003, 0.012, 0.024),
        )


def test_incident_screen_runs_direct_amplitudes_and_writes_reports(
    tmp_path,
    incident_config,
):
    output = tmp_path / "incident"
    report = run_incident_radiation_screen(
        incident_config,
        output,
        pulse_widths=(0.5,),
        amplitudes=(1000.0,),
        horizons=(4.0,),
        thicknesses=(0.003, 0.012, 0.024),
    )
    assert report["reference_solve_count"] == 9
    assert report["candidate_count"] == 1
    assert report["constant_temperature_independent_properties"]
    assert report["monotonicity_checks"]
    assert all(row["passed"] for row in report["monotonicity_checks"])
    assert Path(output / "incident_reference_solves.csv").exists()
    assert Path(output / "incident_response_matrix.csv").exists()
    assert Path(output / "incident_candidates.csv").exists()
    assert Path(output / "incident_screen_report.json").exists()


def test_stage6a_promotion_uses_refined_boundaries_and_dt_half() -> None:
    thin = {
        "bond_max_K": 451.0,
        "structural_interface_max_K": 390.0,
        "hot_face_max_K": 480.0,
        "accepted_range_excursions": 0,
    }
    thick = {
        "bond_max_K": 430.0,
        "structural_interface_max_K": 380.0,
        "hot_face_max_K": 470.0,
        "accepted_range_excursions": 0,
    }
    interior = {
        "thin_feasible": False,
        "thick_feasible": True,
        "thin": {
            "hot_face_max_K": 475.0,
            "bond_max_K": 452.0,
            "structural_interface_max_K": 391.0,
        },
        "thick": {
            "hot_face_max_K": 465.0,
            "bond_max_K": 429.0,
            "structural_interface_max_K": 379.0,
        },
    }
    result = evaluate_stage6a_promotion([
        {
            "configuration_id": "candidate",
            "thin_at_A_min": thin,
            "thick_at_A_min": thick,
            "bond_limit_K": 450.0,
            "structural_limit_K": 400.0,
            "hot_face_limit_K": 500.0,
            "A_min_dt": 1000.0,
            "A_max_dt": 1200.0,
            "A_min_half_dt": 1010.0,
            "A_max_half_dt": 1210.0,
            "interior_dt": interior,
            "interior_half_dt": interior,
        }
    ])
    assert result["promote_to_stage_6b"]
    assert result["verdict"] == "promote"


def test_stage6a_refines_bracketed_boundaries(
    demo_config,
    monkeypatch,
) -> None:
    scenario_id = "screen-centered-single-healthy"
    rows = []
    for amplitude in (1000.0, 2000.0):
        for thickness in (0.003, 0.024):
            threshold = 1200.0 if thickness == 0.003 else 1600.0
            rows.append({
                "sigma_t_s": 0.5,
                "horizon_s": 4.0,
                "scenario_id": scenario_id,
                "d_tps": thickness,
                "incident_amplitude_per_event_W_m2": amplitude,
                "feasible": amplitude < threshold,
                "within_validity": True,
            })

    def fake_result(
        config,
        *,
        amplitude,
        thickness,
        dt,
        **kwargs,
    ):
        del config, kwargs
        threshold = (
            (1210.0 if dt < demo_config.time.dt else 1200.0)
            if thickness == 0.003
            else (1610.0 if dt < demo_config.time.dt else 1600.0)
        )
        feasible = amplitude < threshold
        return {
            "amplitude_W_m2": amplitude,
            "d_tps": thickness,
            "hot_face_max_K": 470.0,
            "bond_max_K": 440.0 if feasible else 451.0,
            "structural_interface_max_K": 390.0,
            "accepted_range_excursions": 0,
            "within_validity": True,
            "feasible": feasible,
        }

    monkeypatch.setattr(
        screening_module,
        "_incident_design_result",
        fake_result,
    )
    records, attempts = _run_stage6a_refinement(
        demo_config,
        rows,
        widths=(0.5,),
        amplitudes=(1000.0, 2000.0),
        horizons=(4.0,),
        thicknesses=(0.003, 0.024),
    )
    assert len(records) == 1
    assert attempts[0]["status"] == "refined"
    assert evaluate_stage6a_promotion(records)["promote_to_stage_6b"]
