from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from fno_tps.data import generate_dataset
from fno_tps.figure_data import (
    _coerce,
    _figure3,
    _figure4,
    _physics_contract,
    _resolve_neighborhood,
    _thickness_key,
    build_figure_data,
    load_figure_data,
)
from fno_tps.sizing import run_full_sizing
from fno_tps.training import train_model


def _sizing_row(scenario_id, d_tps, fv_bond, fno_bond, fv_struct, fno_struct, **overrides):
    row = {
        "scenario_id": scenario_id,
        "d_tps": d_tps,
        "fv_bond_max": fv_bond,
        "surrogate_bond_max": fno_bond,
        "fv_structural_max": fv_struct,
        "surrogate_structural_max": fno_struct,
        "fv_bond_margin": 450.0 - fv_bond,
        "surrogate_bond_margin": 450.0 - fno_bond,
        "fv_structural_margin": 400.0 - fv_struct,
        "surrogate_structural_margin": 400.0 - fno_struct,
        "fv_criticality_ratio": max(fv_bond / 450.0, fv_struct / 400.0),
        "surrogate_criticality_ratio": max(fno_bond / 450.0, fno_struct / 400.0),
        "surrogate_within_validity": True,
        "false_feasible": (
            min(450.0 - fno_bond, 400.0 - fno_struct) >= 0.0
            and min(450.0 - fv_bond, 400.0 - fv_struct) < 0.0
        ),
        "false_infeasible": (
            min(450.0 - fno_bond, 400.0 - fno_struct) < 0.0
            and min(450.0 - fv_bond, 400.0 - fv_struct) >= 0.0
        ),
    }
    row.update(overrides)
    return row


def test_csv_cells_round_trip_to_python_types():
    assert _coerce("scenario_id", "0012") == "0012"
    assert _coerce("false_feasible", "True") is True
    assert _coerce("surrogate_within_validity", "False") is False
    assert _coerce("d_tps", "0.005") == 0.005
    assert _coerce("bond_peak_time_error", "") is None


def test_nonlinear_figure_contract_distinguishes_fixed_pressure_from_state():
    from fno_tps.config import load_study_config

    contract = _physics_contract(
        load_study_config("conf/nonlinear-pilot.yaml")
    )
    assert contract["solver"] == "global_nonlinear_enthalpy"
    assert contract["temperature_dependent"] is True
    assert contract["anisotropic_conductivity"] is True
    assert contract["pressure_mode"] == "fixed_reference_condition"
    assert contract["pressure_Pa"] == pytest.approx(101330.0)
    assert contract["reradiation_enabled"] is True
    assert contract["radiation_coupling"] == "coupled_nonlinear"


def test_figure1_display_interpolation_refines_without_extrapolation():
    from fno_tps.figures import _interpolate_cell_field

    x_faces = np.array([0.0, 1.0, 3.0])
    y_faces = np.array([0.0, 2.0, 4.0])
    field = np.array([[1.0, 3.0], [5.0, 7.0]])

    same_x, same_y, same_field = _interpolate_cell_field(
        x_faces,
        y_faces,
        field,
        factor=1,
    )
    np.testing.assert_allclose(same_x, x_faces)
    np.testing.assert_allclose(same_y, y_faces)
    np.testing.assert_allclose(same_field, field)

    refined_x, refined_y, refined = _interpolate_cell_field(
        x_faces,
        y_faces,
        field,
        factor=4,
    )
    assert refined_x.shape == (9,)
    assert refined_y.shape == (9,)
    assert refined.shape == (8, 8)
    assert float(np.min(refined)) >= float(np.min(field))
    assert float(np.max(refined)) <= float(np.max(field))
    assert np.unique(refined).size > field.size


def test_thickness_keys_survive_a_csv_text_round_trip():
    for value in (0.003, 0.012, 1.0 / 3.0):
        assert _thickness_key(float(repr(value))) == _thickness_key(value)
    assert _thickness_key(0.005) != _thickness_key(0.007)


def test_neighborhood_is_the_selection_and_its_two_neighbours(demo_config):
    candidates = demo_config.thickness_candidates
    resolved = _resolve_neighborhood(
        demo_config,
        {
            "fno_selected_thickness": candidates[3],
            "fv_selected_thickness": candidates[2],
        },
    )
    assert resolved["indices"] == [2, 3, 4]
    assert resolved["roles"] == ["thinner", "selected", "thicker"]
    assert resolved["selection_source"] == "fno"
    assert resolved["clamped"] is False
    assert resolved["reason"] is None


def test_neighborhood_clamps_at_the_thinnest_candidate(demo_config):
    resolved = _resolve_neighborhood(
        demo_config,
        {
            "fno_selected_thickness": None,
            "fv_selected_thickness": demo_config.thickness_candidates[0],
        },
    )
    assert resolved["indices"] == [0, 1, 2]
    assert resolved["roles"] == ["selected", "thicker", "thicker+2"]
    assert resolved["selection_source"] == "fv"
    assert resolved["clamped"] is True
    assert "thinnest" in resolved["reason"]


def test_neighborhood_reports_when_no_design_is_feasible(demo_config):
    resolved = _resolve_neighborhood(
        demo_config,
        {"fno_selected_thickness": None, "fv_selected_thickness": None},
    )
    assert resolved["selected_thickness"] is None
    assert resolved["selection_source"] is None
    assert resolved["roles"] == ["candidate"] * 3
    assert resolved["clamped"] is True
    assert "found a feasible design" in resolved["reason"]


def test_figure3_reduces_scenarios_to_worst_case_and_envelope(demo_config):
    rows = [
        _sizing_row("mild", 0.003, 430.0, 431.0, 380.0, 381.0),
        _sizing_row("severe", 0.003, 460.0, 459.0, 410.0, 409.0),
        _sizing_row("mild", 0.005, 410.0, 409.0, 360.0, 361.0),
        _sizing_row(
            "severe", 0.005, 440.0, 441.0, 390.0, 391.0,
            surrogate_within_validity=False,
        ),
    ]
    primary = {
        "representation": "temporal_local",
        "rows": rows,
        "result": {
            "fno_selected_thickness": 0.005,
            "fv_selected_thickness": 0.005,
        },
    }
    arrays: dict[str, np.ndarray] = {}
    notes: list[str] = []
    section = _figure3(demo_config, primary, arrays, notes)

    np.testing.assert_allclose(arrays["f3_thickness"], [0.003, 0.005])
    np.testing.assert_allclose(arrays["f3_fv_bond_worst"], [460.0, 440.0])
    np.testing.assert_allclose(arrays["f3_fv_bond_lo"], [430.0, 410.0])
    np.testing.assert_allclose(arrays["f3_fv_bond_hi"], [460.0, 440.0])
    np.testing.assert_allclose(arrays["f3_fno_struct_worst"], [409.0, 391.0])
    assert list(arrays["f3_surrogate_validity_violation"]) == [False, True]
    assert section["limiting_scenarios"] == ["severe", "severe"]
    assert section["scenario_count"] == [2]
    assert any("validity ceiling" in note for note in notes)


def test_figure3_notes_a_missing_feasible_design(demo_config):
    primary = {
        "representation": "summary",
        "rows": [_sizing_row("only", 0.003, 500.0, 500.0, 500.0, 500.0)],
        "result": {"fno_selected_thickness": None, "fv_selected_thickness": None},
    }
    notes: list[str] = []
    _figure3(demo_config, primary, {}, notes)
    assert any("surrogate found no feasible design" in note for note in notes)
    assert any("FV solver found no feasible design" in note for note in notes)


def test_figure4_margin_is_the_controlling_constraint():
    rows = [
        # Structural controls for FV; the surrogate calls the same cell feasible.
        _sizing_row("a", 0.003, 440.0, 438.0, 405.0, 398.0),
        # Both agree the cell is comfortably feasible.
        _sizing_row("b", 0.005, 400.0, 402.0, 350.0, 351.0),
    ]
    run = {
        "representation": "temporal_local",
        "path": "/tmp/sizing",
        "rows": rows,
        "result": {"margin_sign_disagreement_count": 1},
    }
    arrays: dict[str, np.ndarray] = {}
    notes: list[str] = []
    section = _figure4([run], arrays, notes)

    np.testing.assert_allclose(arrays["f4_temporal_local_m_fv"], [-5.0, 50.0])
    np.testing.assert_allclose(arrays["f4_temporal_local_m_fno"], [2.0, 48.0])
    assert list(arrays["f4_temporal_local_false_feasible"]) == [True, False]
    assert list(arrays["f4_temporal_local_false_infeasible"]) == [False, False]
    panel = section["panels"][0]
    assert panel["false_feasible"] == 1
    assert panel["margin_sign_disagreement_count"] == 1
    assert section["missing_representations"] == ["summary", "temporal_global"]
    assert not any("out of sync" in note for note in notes)


def test_figure4_flags_a_matrix_that_disagrees_with_its_report():
    run = {
        "representation": "summary",
        "path": "/tmp/sizing",
        "rows": [_sizing_row("a", 0.003, 440.0, 438.0, 405.0, 398.0)],
        "result": {"margin_sign_disagreement_count": 0},
    }
    notes: list[str] = []
    _figure4([run], {}, notes)
    assert any("out of sync" in note for note in notes)


def test_sizing_backed_figures_require_a_sizing_directory(tmp_path, demo_config):
    with pytest.raises(ValueError, match="--sizing"):
        build_figure_data(demo_config, tmp_path / "figures", figures=(3,))


def test_a_missing_sizing_directory_names_the_command(tmp_path, demo_config):
    with pytest.raises(FileNotFoundError, match="fno-tps size"):
        build_figure_data(
            demo_config,
            tmp_path / "figures",
            sizing_dirs=[tmp_path / "absent"],
            figures=(3,),
        )


_LIMITS = {"bond_temperature": 450.0, "structural_interface_temperature": 400.0}


def test_margin_scatter_renders_the_quadrants_when_cells_straddle_zero(tmp_path):
    pytest.importorskip("matplotlib")
    from fno_tps.figures import render_figure4

    manifest = {
        "limits": _LIMITS,
        "figure4": {
            "panels": [{
                "representation": "summary",
                "path": str(tmp_path),
                "cell_count": 4,
                "false_feasible": 1,
                "false_infeasible": 1,
                "margin_mae_K": 4.0,
                "margin_sign_disagreement_count": 2,
            }],
            "missing_representations": ["temporal_global", "temporal_local"],
        },
    }
    arrays = {
        "f4_summary_m_fv": np.array([-5.0, 3.0, 8.0, -2.0]),
        "f4_summary_m_fno": np.array([2.0, -1.0, 7.0, -3.0]),
        "f4_summary_false_feasible": np.array([True, False, False, False]),
        "f4_summary_false_infeasible": np.array([False, True, False, False]),
        "f4_summary_d_tps": np.array([0.003, 0.005, 0.007, 0.009]),
    }
    written = render_figure4(manifest, arrays, tmp_path)
    assert len(written) == 2
    for path in written:
        assert Path(path).stat().st_size > 4096


def test_sizing_curves_render_without_a_surrogate_selection(tmp_path):
    pytest.importorskip("matplotlib")
    from fno_tps.figures import render_figure3

    manifest = {
        "limits": _LIMITS,
        "figure3": {
            "representation": "summary",
            "thicknesses": [0.003, 0.005],
            "scenario_count": [2],
            "limiting_scenarios": ["severe", "severe"],
            "fno_selected_thickness": None,
            "fv_selected_thickness": 0.005,
            "surrogate_validity_violation": [False, True],
        },
    }
    arrays = {
        "f3_thickness": np.array([0.003, 0.005]),
        "f3_surrogate_validity_violation": np.array([False, True]),
    }
    for key, values in (
        ("fv_bond", (460.0, 440.0)),
        ("fno_bond", (459.0, 441.0)),
        ("fv_struct", (410.0, 390.0)),
        ("fno_struct", (409.0, 391.0)),
    ):
        for statistic, offset in (("worst", 0.0), ("hi", 0.0), ("lo", -8.0)):
            arrays[f"f3_{key}_{statistic}"] = np.array(values) + offset
    written = render_figure3(manifest, arrays, tmp_path)
    assert len(written) == 2
    for path in written:
        assert Path(path).stat().st_size > 4096


@pytest.mark.slow
def test_figure_artifacts_and_rendering_end_to_end(
    tmp_path,
    demo_config,
    three_cases,
    heating_event,
):
    data_dir = tmp_path / "data"
    generate_dataset(demo_config, data_dir, cases=three_cases)
    training = train_model(
        demo_config,
        data_dir,
        tmp_path / "run",
        "summary",
        epochs=1,
        device="cpu",
        batch_size=2,
    )
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        yaml.safe_dump({
            "scenario_set_id": "figure-test",
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
    sizing_dir = tmp_path / "sizing"
    run_full_sizing(
        demo_config,
        scenario_path,
        training["best_checkpoint"],
        sizing_dir,
        allow_demo=True,
        device="cpu",
        batch_size=4,
    )

    figure_dir = tmp_path / "figures"
    manifest = build_figure_data(
        demo_config,
        figure_dir,
        sizing_dirs=[sizing_dir],
        data_dir=data_dir,
        checkpoint_path=training["best_checkpoint"],
        scenario_path=scenario_path,
        device="cpu",
        batch_size=4,
    )

    assert manifest["study_config_sha256"] == demo_config.sha256
    assert manifest["primary_representation"] == "summary"
    assert manifest["checkpoint"]["representation"] == "summary"
    assert manifest["dataset"]["test_case_count"] >= 1
    assert manifest["generated_figures"] == [1, 2, 3, 4]

    reloaded, arrays = load_figure_data(figure_dir)
    assert reloaded == manifest

    nx, ny = demo_config.mesh.nx, demo_config.mesh.ny
    assert arrays["f1_fv"].shape == (nx, ny)
    assert arrays["f1_fno"].shape == (nx, ny)
    assert arrays["f1_abs_error"].shape == (nx, ny)
    assert arrays["f1_x_faces"].shape == (nx + 1,)
    assert arrays["f1_y_faces"].shape == (ny + 1,)
    for key in (
        "f1_property_temperature",
        "f1_property_k_x",
        "f1_property_k_y",
        "f1_property_rho_c",
        "f1_property_alpha_x",
        "f1_property_alpha_y",
    ):
        assert arrays[key].shape == (256,)
    np.testing.assert_allclose(
        arrays["f1_abs_error"],
        np.abs(arrays["f1_fno"] - arrays["f1_fv"]),
    )

    times = len(demo_config.time.saved_times())
    for key in ("f2_bond_fv", "f2_bond_fno", "f2_struct_fv", "f2_struct_fno"):
        assert arrays[key].shape == (3, times)
    for key in (
        "f2_hot_face_fv",
        "f2_incident_flux",
        "f2_reradiated_flux",
        "f2_net_flux",
    ):
        assert arrays[key].shape == (3, times)
    np.testing.assert_allclose(
        arrays["f2_net_flux"],
        arrays["f2_incident_flux"] - arrays["f2_reradiated_flux"],
    )
    assert arrays["f2_nonlinear_iterations"].shape[0] == 3
    assert arrays["f2_nonlinear_residuals"].shape == arrays[
        "f2_nonlinear_iterations"
    ].shape
    assert arrays["f2_times"].shape == (times,)
    assert len(manifest["figure2"]["roles"]) == 3
    assert manifest["figure2"]["scenario_id"] == "one"

    candidates = len(demo_config.thickness_candidates)
    assert arrays["f3_thickness"].shape == (candidates,)
    for key in ("f3_fv_bond_worst", "f3_fno_struct_worst"):
        assert arrays[key].shape == (candidates,)
    assert np.all(arrays["f3_fv_bond_hi"] >= arrays["f3_fv_bond_lo"])

    assert arrays["f4_summary_m_fv"].shape == (candidates,)
    assert arrays["f4_summary_m_fno"].shape == (candidates,)
    assert manifest["figure4"]["panels"][0]["cell_count"] == candidates

    pytest.importorskip("matplotlib")
    from fno_tps.figures import render_from_directory

    written = render_from_directory(figure_dir)
    assert sorted(written) == [1, 2, 3, 4]
    for paths in written.values():
        assert len(paths) == 2
        for path in paths:
            assert Path(path).stat().st_size > 4096
