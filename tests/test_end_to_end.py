from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from fno_tps.data import generate_dataset
from fno_tps.evaluation import evaluate_checkpoint
from fno_tps.pilot import _pilot_scenarios, run_pilot_matrix
from fno_tps.runtime import load_model_checkpoint
from fno_tps.sizing import run_full_sizing
from fno_tps.training import train_model


def test_one_epoch_checkpoint_evaluation_and_sizing(
    tmp_path,
    demo_config,
    three_cases,
    heating_event,
):
    data_dir = tmp_path / "data"
    bundle = generate_dataset(demo_config, data_dir, cases=three_cases)
    del bundle
    report = train_model(
        demo_config,
        data_dir,
        tmp_path / "run",
        "summary",
        epochs=1,
        device="cpu",
        batch_size=2,
    )
    model_a, checkpoint = load_model_checkpoint(report["best_checkpoint"])
    model_b, _ = load_model_checkpoint(report["best_checkpoint"])
    for left, right in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(left, right)
    assert checkpoint["epoch"] == 0
    assert checkpoint["selection_metrics"]["selection_key"] == (
        report["checkpoint_selection"]["selection_key"]
    )
    assert checkpoint["model_config"]["hard_initial_condition"]
    assert report["initial_condition"]["t0_in_training_targets"] is False
    assert "tps_thickness_loss_weighting" in report
    assert report["checkpoint_archive"]["checkpoint_count"] == 1
    assert report["checkpoint_archive"]["model_only"]
    _, archived_checkpoint = load_model_checkpoint(
        report["pareto_frontier"][0]["checkpoint"]
    )
    assert archived_checkpoint["resume_capable"] is False
    assert "optimizer_state" not in archived_checkpoint
    assert checkpoint["resume_capable"] is True
    assert set(report["checkpoint_candidates"]) == {
        "lowest_field_rmse",
        "lowest_critical_peak_error",
        "lowest_absolute_margin_tail",
        "best_continuous_composite",
        "latest",
    }
    assert len(report["pareto_frontier"]) == 1

    evaluation = evaluate_checkpoint(
        demo_config,
        data_dir,
        report["best_checkpoint"],
        tmp_path / "evaluation.json",
        split="test",
        device="cpu",
    )
    assert evaluation["case_count"] == 1
    assert evaluation["normalization_source"] == "checkpoint"
    assert evaluation["false_feasible_count"] >= 0
    assert evaluation["false_infeasible_count"] >= 0
    assert evaluation["margin_error_p95_K"] >= 0.0
    assert evaluation["optimistic_margin_error_p99_K"] >= 0.0
    assert evaluation["safety_buffer_K"] == 0.0
    assert evaluation["t0_max_abs_delta_T_K"] < 1.0e-5
    record = evaluation["records"][0]
    assert record["false_feasible"] == (
        record["predicted_feasible"] and not record["true_feasible"]
    )
    assert record["false_infeasible"] == (
        not record["predicted_feasible"] and record["true_feasible"]
    )
    assert record["optimistic_margin_error_K"] == max(
        0.0,
        record["predicted_minimum_margin_K"]
        - record["true_minimum_margin_K"],
    )
    assert Path(tmp_path / "evaluation.json").exists()

    calibrated = evaluate_checkpoint(
        demo_config,
        data_dir,
        report["best_checkpoint"],
        split="test",
        device="cpu",
        calibration_split="val",
        calibration_quantile=95.0,
    )
    assert calibrated["safety_buffer_source"] == "calibration_split"
    assert calibrated["safety_calibration"]["split"] == "val"
    assert calibrated["safety_buffer_K"] >= 0.0
    for calibrated_record in calibrated["records"]:
        predicted_margin = calibrated_record["predicted_minimum_margin_K"]
        if predicted_margin < 0.0:
            expected_classification = "surrogate_infeasible"
        elif predicted_margin > calibrated["safety_buffer_K"]:
            expected_classification = "surrogate_feasible"
        else:
            expected_classification = "near_boundary_fv_required"
        assert (
            calibrated_record["buffered_classification"]
            == expected_classification
        )

    all_evaluation = evaluate_checkpoint(
        demo_config,
        data_dir,
        report["best_checkpoint"],
        split="all",
        device="cpu",
    )
    assert all_evaluation["case_count"] == 3

    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        yaml.safe_dump({
            "scenario_set_id": "test",
            "authoritative": False,
            "scenarios": [{
                "scenario_id": "one",
                "heating_events": [{
                    "amplitude": heating_event.amplitude,
                    "y_center": heating_event.y_center,
                    "t_center": heating_event.t_center,
                    "sigma_y": heating_event.sigma_y,
                    "sigma_t": heating_event.sigma_t,
                }],
                "bond_defects": [],
            }],
        }),
        encoding="utf-8",
    )
    sizing = run_full_sizing(
        demo_config,
        scenario_path,
        report["best_checkpoint"],
        tmp_path / "sizing",
        allow_demo=True,
        device="cpu",
        batch_size=4,
    )
    assert len(sizing["rows"]) == len(demo_config.thickness_candidates)
    assert sizing["scenario_ranking_accuracy"] is None
    assert "fno_selection_neighborhood" in sizing
    assert "field_rmse_K" in sizing["rows"][0]
    assert "critical_max_peak_error_K" in sizing["rows"][0]
    assert "optimistic_margin_error_K" in sizing["rows"][0]
    assert sizing["safety_buffer_K"] == 0.0
    assert Path(tmp_path / "sizing" / "sizing_matrix.csv").exists()
    assert Path(tmp_path / "sizing" / "sizing_result.json").exists()

    invalid_scenario_path = tmp_path / "invalid-scenarios.yaml"
    invalid_scenario_path.write_text(
        yaml.safe_dump({
            "scenario_set_id": "invalid-test",
            "authoritative": False,
            "scenarios": [{
                "scenario_id": "too-hot",
                "heating_events": [{
                    "amplitude": 1.0e8,
                    "y_center": heating_event.y_center,
                    "t_center": heating_event.t_center,
                    "sigma_y": heating_event.sigma_y,
                    "sigma_t": heating_event.sigma_t,
                }],
                "bond_defects": [],
            }],
        }),
        encoding="utf-8",
    )
    invalid_sizing = run_full_sizing(
        demo_config,
        invalid_scenario_path,
        report["best_checkpoint"],
        tmp_path / "invalid-sizing",
        allow_demo=True,
        device="cpu",
        batch_size=4,
    )
    assert invalid_sizing["invalid_fv_cell_count"] > 0
    assert {
        row["fv_classification"] for row in invalid_sizing["rows"]
    } == {"invalid"}


def test_pilot_matrix_writes_machine_readable_diagnostics(
    tmp_path,
    demo_config,
):
    report = run_pilot_matrix(
        demo_config,
        tmp_path / "pilot",
        scenario_count=1,
        backing_thicknesses=(0.002, 0.004),
    )
    assert report["simulation_count"] == 2 * len(
        demo_config.thickness_candidates
    )
    assert report["backing_thicknesses_m"] == [0.002, 0.004]
    assert report["hot_face_within_validity"]
    assert report["all_cells_within_validity"]
    assert report["invalid_cell_count"] == 0
    assert all(report["scenario_validity_complete"].values())
    assert not report["authority_review_ready"]
    assert Path(tmp_path / "pilot" / "pilot_matrix.csv").exists()
    assert Path(tmp_path / "pilot" / "pilot_report.json").exists()


def test_pilot_envelope_contains_aligned_maximum_case(demo_config):
    scenarios = _pilot_scenarios(demo_config, 5)
    assert [scenario.case_id for scenario in scenarios] == [
        "pilot-nominal-single-healthy",
        "pilot-localized-single-defect",
        "pilot-moving-double-defect",
        "pilot-distributed-triple-healthy",
        "pilot-aligned-triple-severe",
    ]
    aligned = scenarios[-1]
    assert aligned.event_count == 3
    assert aligned.defect_count == 2
    assert {
        event.amplitude for event in aligned.heating_events
    } == {demo_config.heating.amplitude[1]}
    assert len({
        event.y_center for event in aligned.heating_events
    }) == 1
    assert {
        defect.severity for defect in aligned.bond_defects
    } == {demo_config.bond.severity[1]}
