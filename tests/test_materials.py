from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv

import numpy as np
import pytest
import yaml

from fno_tps.config import Material, PropertyTableConfig
from fno_tps.materials import (
    PropertyRangeError,
    TabulatedPropertyModel,
    load_property_table,
)


def _write_table(path: Path, **overrides: object) -> Path:
    values: dict[str, object] = {
        "name": "test material",
        "version": "TEST-1",
        "source": "synthetic verification data",
        "authoritative": True,
        "interpolation": "linear",
        "density": 10.0,
        "pressure_Pa": 101325.0,
        "pressure_basis": "synthetic fixed-pressure test",
        "temperature": [300.0, 400.0, 500.0],
        "conductivity": [1.0, 2.0, 4.0],
        "specific_heat": [2.0, 4.0, 8.0],
    }
    values.update(overrides)
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({
            "temperature": [300.0, 300.0],
            "conductivity": [1.0, 2.0],
            "specific_heat": [2.0, 4.0],
        }, "strictly increasing"),
        ({"conductivity": [1.0, 0.0]}, "conductivity"),
        ({"specific_heat": [1.0, -1.0]}, "specific_heat"),
        ({"pressure_Pa": 0.0}, "pressure_Pa"),
        ({"pressure_basis": ""}, "pressure_basis"),
        ({"interpolation": "pchip"}, "linear"),
    ],
)
def test_property_table_validation(
    tmp_path: Path,
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        load_property_table(_write_table(tmp_path / "table.yaml", **overrides))


def test_content_sha256_changes_with_raw_file(tmp_path: Path) -> None:
    path = _write_table(tmp_path / "table.yaml")
    first = load_property_table(path).content_sha256
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert load_property_table(path).content_sha256 != first


def test_reject_and_clamp_policies_have_separate_counts(tmp_path: Path) -> None:
    table = load_property_table(
        _write_table(
            tmp_path / "table.yaml",
            specific_heat=[2.0, 4.0, 6.0],
        )
    )
    rejecting = TabulatedPropertyModel(table, extrapolation="reject")
    rejecting.conductivity(np.asarray([250.0]), mode="iterate")
    assert rejecting.iteration_range_clamps == 1
    assert rejecting.accepted_range_excursions == 0
    with pytest.raises(PropertyRangeError):
        rejecting.conductivity(np.asarray([250.0]), mode="accepted")
    assert rejecting.accepted_range_excursions == 1

    clamping = TabulatedPropertyModel(table, extrapolation="clamp")
    assert clamping.conductivity(250.0) == pytest.approx(1.0)
    assert clamping.accepted_range_excursions == 1


def test_enthalpy_is_exact_for_piecewise_linear_cp(tmp_path: Path) -> None:
    table = load_property_table(
        _write_table(
            tmp_path / "table.yaml",
            temperature=[300.0, 400.0],
            conductivity=[1.0, 1.0],
            specific_heat=[2.0, 6.0],
            density=10.0,
        )
    )
    model = TabulatedPropertyModel(table, reference_temperature=300.0)
    expected = 10.0 * (2.0 * 50.0 + 0.5 * 0.04 * 50.0**2)
    assert model.enthalpy(350.0) == pytest.approx(expected)


def test_enthalpy_derivative_matches_volumetric_heat_capacity(
    tmp_path: Path,
) -> None:
    table = load_property_table(
        _write_table(
            tmp_path / "table.yaml",
            specific_heat=[2.0, 4.0, 6.0],
        )
    )
    model = TabulatedPropertyModel(table, reference_temperature=300.0)
    epsilon = 1.0e-2
    for temperature in (350.0, 400.0, 450.0):
        derivative = (
            model.enthalpy(temperature + epsilon)
            - model.enthalpy(temperature - epsilon)
        ) / (2.0 * epsilon)
        expected = model.volumetric_heat_capacity(temperature)
        assert derivative == pytest.approx(expected, rel=1.0e-8)


def test_reference_temperature_defaults_and_is_range_checked(
    tmp_path: Path,
) -> None:
    table = load_property_table(_write_table(tmp_path / "table.yaml"))
    assert TabulatedPropertyModel(table).reference_temperature == 300.0
    with pytest.raises(ValueError, match="reference_temperature"):
        TabulatedPropertyModel(table, reference_temperature=250.0)


def test_clamped_enthalpy_is_c1_at_endpoints(tmp_path: Path) -> None:
    table = load_property_table(_write_table(tmp_path / "table.yaml"))
    model = TabulatedPropertyModel(table, extrapolation="clamp")
    epsilon = 1.0e-5
    for endpoint in (table.temperature_min, table.temperature_max):
        left = (
            model.enthalpy(endpoint) - model.enthalpy(endpoint - epsilon)
        ) / epsilon
        right = (
            model.enthalpy(endpoint + epsilon) - model.enthalpy(endpoint)
        ) / epsilon
        assert left == pytest.approx(right, rel=1.0e-7)


def test_placeholder_authority_pressure_and_sink_validation(
    demo_config,
) -> None:
    material = Material(
        model="temperature_dependent_table",
        reference_k=5.96e-2,
        reference_rho_c=1.203e5,
        property_table=PropertyTableConfig(
            path="conf/materials/tps_placeholder.yaml",
            version="PLACEHOLDER-0",
            reference_temperature=300.0,
        ),
    )
    validity = replace(
        demo_config.validity,
        tps_property_model="temperature_dependent_table",
        conductivity_reference_pressure=101325.0000001,
    )
    config = replace(
        demo_config,
        authoritative=False,
        tps=material,
        validity=validity,
        surface=replace(
            demo_config.surface,
            radiation_coupling="coupled_nonlinear",
            radiation_sink_temperature=3.0,
        ),
        heating=replace(demo_config.heating, t_center=(1.0, 3.0)),
    )
    config.validate()
    assert (
        config.property_provenance["tps"]["pressure_basis"]
        == "PLACEHOLDER — not a measured condition"
    )
    with pytest.raises(ValueError, match="non-authoritative"):
        replace(config, authoritative=True).validate()
    with pytest.raises(ValueError, match="pressure_Pa"):
        replace(
            config,
            validity=replace(
                validity,
                conductivity_reference_pressure=50000.0,
            ),
        ).validate()


def test_authoritative_li900_table_and_raw_csvs_are_consistent() -> None:
    table_path = Path("conf/materials/tps_li900_rcg_authoritative.yaml")
    table = load_property_table(table_path)
    assert table.authoritative
    assert table.is_anisotropic
    assert table.density == pytest.approx(144.0)
    assert table.density_uncertainty == pytest.approx(2.11)
    assert table.pressure_Pa == pytest.approx(101330.0)
    assert table.temperature_min == pytest.approx(116.667)
    assert table.temperature_max == pytest.approx(1533.33)

    model = TabulatedPropertyModel(
        table,
        reference_temperature=300.0,
        extrapolation="reject",
    )
    assert model.conductivity(300.0, direction="x") == pytest.approx(
        0.05124798686711595
    )
    assert model.conductivity(300.0, direction="y") == pytest.approx(
        0.07194798398709752
    )
    assert model.volumetric_heat_capacity(300.0) == pytest.approx(
        102045.8386193291
    )

    for curve, csv_path in (
        (
            table.conductivity_x,
            Path("conf/materials/li900_k_through_thickness_tpsx.csv"),
        ),
        (
            table.conductivity_y,
            Path("conf/materials/li900_k_in_plane_tpsx.csv"),
        ),
    ):
        with csv_path.open(newline="", encoding="utf-8") as stream:
            selected = [
                row
                for row in csv.DictReader(stream)
                if float(row["pressure_Pa"]) == 101330.0
            ]
        np.testing.assert_allclose(
            curve.temperature,
            [float(row["temperature_K"]) for row in selected],
        )
        np.testing.assert_allclose(
            curve.values,
            [float(row["conductivity_W_mK"]) for row in selected],
        )
