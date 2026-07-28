from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fno_tps.config import load_study_config
from fno_tps.problem import BondDefect, SimulationCase, TPSProblem


def test_demo_config_is_non_authoritative_and_uses_limit_hierarchy():
    config = load_study_config("conf/demo.yaml")
    assert not config.authoritative
    assert config.study_id == "tps_fno_li900_rtv560_al_demo_v1"
    assert config.tps.k == pytest.approx(5.96e-2)
    assert config.tps.rho_c == pytest.approx(1.203e5)
    assert config.bond_temperature_limit == pytest.approx(450.0)
    assert config.validity.tps_property_model == "constant_effective"
    assert config.validity.property_tables.minimum_temperature == pytest.approx(
        300.0
    )
    assert config.validity.hot_face.study_temperature_limit == pytest.approx(
        1400.0
    )
    assert config.hot_face_temperature_limit == pytest.approx(1400.0)
    assert config.validity.max_property_relative_deviation == pytest.approx(0.20)
    assert config.validity.reject_cases_outside_validity
    assert config.heating.interpretation == (
        "effective_net_inward_conductive_heat_flux"
    )
    assert config.heating.amplitude == pytest.approx((2000.0, 5000.0))
    assert config.training.temporal_samples == 128
    assert len(config.thickness_candidates) == 9
    assert config.recent_window == pytest.approx(0.1 * config.time.t_final)
    config.require_authoritative(allow_demo=True)


def test_incident_radiation_config_keeps_constant_material_properties():
    config = load_study_config("conf/incident-radiation-demo.yaml")
    assert config.surface.model == "incident_heat_flux_with_reradiation"
    assert config.surface.reradiation_enabled
    assert config.surface.emissivity_model == "constant"
    assert config.surface.emissivity == pytest.approx(0.77)
    assert config.surface.radiation_sink_temperature == pytest.approx(300.0)
    assert config.heating.interpretation == "incident_aerothermal_heat_flux"
    assert config.validity.tps_property_model == "constant_effective"
    assert config.tps.k == pytest.approx(5.96e-2)
    assert config.tps.rho_c == pytest.approx(1.203e5)


def test_resolved_configuration_hashes_are_frozen():
    assert load_study_config("conf/demo.yaml").sha256 == (
        "61fefd9205838e8a0309f1ba8ee0aafa74353f26ee782e434dfdd13de15cdeab"
    )
    assert load_study_config("conf/incident-radiation-demo.yaml").sha256 == (
        "927691242b8be43712a2c114f9780bd57432dd29abbae3fe57892b079b965b47"
    )


def test_surface_and_heating_interpretations_must_match():
    config = load_study_config("conf/incident-radiation-demo.yaml")
    invalid = replace(
        config,
        heating=replace(
            config.heating,
            interpretation="effective_net_inward_conductive_heat_flux",
        ),
    )
    with pytest.raises(ValueError, match="requires heating.interpretation"):
        invalid.validate()


def test_production_template_rejects_unresolved_values():
    with pytest.raises(ValueError, match="missing required"):
        load_study_config("conf/production.template.yaml")


def test_authoritative_config_requires_out_of_validity_rejection():
    config = load_study_config("conf/demo.yaml")
    invalid = replace(
        config,
        authoritative=True,
        validity=replace(
            config.validity,
            reject_cases_outside_validity=False,
        ),
    )
    with pytest.raises(ValueError, match="must reject"):
        invalid.validate()


def test_nonlinear_pilot_resolves_all_hot_face_limit_sources():
    config = load_study_config("conf/nonlinear-pilot.yaml")
    assert config.authoritative
    assert config.surface.emissivity_authoritative
    assert config.surface.emissivity_quantity == "total_hemispherical"
    assert config.surface.emissivity == pytest.approx(0.76)
    assert config.tps.reference_k_y > config.tps.reference_k
    assert config.property_provenance["tps"]["anisotropic_conductivity"]
    hierarchy = config.hot_face_limit_hierarchy
    assert hierarchy["study_temperature_limit_K"] == pytest.approx(1400.0)
    assert hierarchy["material_multiple_use_limit_K"] == pytest.approx(1590.0)
    assert hierarchy["property_table_temperature_limit_K"] == pytest.approx(
        1533.33
    )
    assert hierarchy["emissivity_temperature_limit_K"] == pytest.approx(
        1400.0
    )
    assert hierarchy["resolved_hot_face_temperature_limit_K"] == pytest.approx(
        1400.0
    )


def test_authoritative_incident_study_requires_surface_provenance():
    config = load_study_config("conf/nonlinear-pilot.yaml")
    with pytest.raises(ValueError, match="authoritative emissivity"):
        replace(
            config,
            surface=replace(
                config.surface,
                emissivity_authoritative=False,
            ),
        ).validate()
    with pytest.raises(ValueError, match="total hemispherical"):
        replace(
            config,
            surface=replace(
                config.surface,
                emissivity_quantity="spectral",
            ),
        ).validate()


def test_resolved_config_capture_is_round_trip_loadable():
    config = load_study_config("conf/demo.yaml")
    resolved = type(config).from_dict(config.as_dict())
    assert resolved == replace(config, source_path=None)
    assert resolved.sha256 == config.sha256


def test_dotted_override_is_validated():
    config = load_study_config(
        "conf/demo.yaml",
        ["training.epochs=2", "training.width=16"],
    )
    assert config.training.epochs == 2
    assert config.training.width == 16
    with pytest.raises(KeyError):
        load_study_config("conf/demo.yaml", ["training.unknown=1"])


def test_exact_gaussian_integral_matches_quadrature(heating_event):
    y = np.linspace(0.01, 0.19, 7)
    time = np.linspace(0.2, 2.7, 20001)
    numerical = np.trapezoid(heating_event.values(y, time), time, axis=-1)
    exact = heating_event.integral(y, time[0], time[-1])
    np.testing.assert_allclose(exact, numerical, rtol=1e-8, atol=1e-8)


def test_bond_field_is_positive_and_healthy_without_defects(demo_config, heating_event):
    y = np.linspace(0.0, demo_config.lateral_length, 20)
    healthy = SimulationCase("healthy", 0.02, (heating_event,), ())
    np.testing.assert_allclose(
        healthy.bond_conductivity(y, demo_config),
        demo_config.bond.k0,
    )
    degraded = SimulationCase(
        "degraded",
        0.02,
        (heating_event,),
        (BondDefect(2.0, 0.1, 0.02),),
    )
    field = degraded.bond_conductivity(y, demo_config)
    assert np.all(field >= demo_config.bond.k_min)
    assert np.min(field) < demo_config.bond.k0


def test_linear_bond_degradation_reaches_declared_floor(
    demo_config,
    heating_event,
):
    case = SimulationCase(
        "severe",
        0.003,
        (heating_event,),
        (BondDefect(0.90, 0.10, 0.01),),
    )
    field = case.bond_conductivity(np.asarray([0.10]), demo_config)
    assert field[0] == pytest.approx(demo_config.bond.k_min)


def test_stage_a_sampling_is_balanced_and_capped():
    config = load_study_config("conf/demo.yaml")
    cases = TPSProblem(config).sample_cases()
    assert len(cases) == 108 * len(config.thickness_candidates)
    counts = {}
    for case in cases:
        key = (case.d_tps, case.event_count, case.defect_count)
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {12}
