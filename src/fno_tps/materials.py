from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import yaml

from fno_tps.config import Material, StudyConfig


Array = np.ndarray
EvalMode = Literal["iterate", "accepted"]


class PropertyRangeError(ValueError):
    """Raised when an accepted state lies outside a tabulated property range."""


@dataclass(frozen=True)
class PropertyCurve:
    temperature: Array
    values: Array
    uncertainty: Array | None = None
    label: str = ""
    units: str = ""
    source_url: str = ""
    uncertainty_basis: str = ""

    @property
    def temperature_min(self) -> float:
        return float(self.temperature[0])

    @property
    def temperature_max(self) -> float:
        return float(self.temperature[-1])

    def provenance(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "units": self.units,
            "source_url": self.source_url,
            "temperature_range_K": [
                self.temperature_min,
                self.temperature_max,
            ],
            "knot_count": int(len(self.temperature)),
        }
        if self.uncertainty is not None:
            payload["uncertainty_basis"] = self.uncertainty_basis
            payload["reported_uncertainty_range"] = [
                float(np.min(self.uncertainty)),
                float(np.max(self.uncertainty)),
            ]
        return payload


@dataclass(frozen=True)
class PropertyTable:
    name: str
    version: str
    source: str
    authoritative: bool
    content_sha256: str
    density: float
    density_uncertainty: float | None
    pressure_Pa: float
    pressure_basis: str
    conductivity_x: PropertyCurve
    conductivity_y: PropertyCurve
    specific_heat_curve: PropertyCurve
    material_system: str = ""

    @property
    def temperature_min(self) -> float:
        return max(
            self.conductivity_x.temperature_min,
            self.conductivity_y.temperature_min,
            self.specific_heat_curve.temperature_min,
        )

    @property
    def temperature_max(self) -> float:
        return min(
            self.conductivity_x.temperature_max,
            self.conductivity_y.temperature_max,
            self.specific_heat_curve.temperature_max,
        )

    @property
    def is_anisotropic(self) -> bool:
        return not (
            np.array_equal(
                self.conductivity_x.temperature,
                self.conductivity_y.temperature,
            )
            and np.array_equal(
                self.conductivity_x.values,
                self.conductivity_y.values,
            )
        )

    # Compatibility accessors for the original scalar-table contract.
    @property
    def temperature(self) -> Array:
        return self.specific_heat_curve.temperature

    @property
    def conductivity(self) -> Array:
        return self.conductivity_x.values

    @property
    def specific_heat(self) -> Array:
        return self.specific_heat_curve.values

    def provenance(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "authoritative": self.authoritative,
            "content_sha256": self.content_sha256,
            "material_system": self.material_system,
            "density_kg_m3": self.density,
            "pressure_Pa": self.pressure_Pa,
            "pressure_basis": self.pressure_basis,
            "temperature_range_K": [
                self.temperature_min,
                self.temperature_max,
            ],
            "anisotropic_conductivity": self.is_anisotropic,
            "conductivity_x": self.conductivity_x.provenance(),
            "conductivity_y": self.conductivity_y.provenance(),
            "specific_heat": self.specific_heat_curve.provenance(),
        }
        if self.density_uncertainty is not None:
            payload["density_uncertainty_kg_m3"] = self.density_uncertainty
        return payload


def _validated_curve(
    payload: Any,
    *,
    source: Path,
    field: str,
    default_label: str,
    default_units: str,
) -> PropertyCurve:
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: {field} must be a mapping.")
    try:
        temperature = np.asarray(payload["temperature"], dtype=np.float64)
        values = np.asarray(
            payload["values"] if "values" in payload else payload["value"],
            dtype=np.float64,
        )
    except KeyError as exc:
        raise ValueError(
            f"{source}: {field} is missing required field {exc.args[0]!r}."
        ) from exc
    uncertainty_raw = payload.get("uncertainty")
    uncertainty = (
        None
        if uncertainty_raw is None
        else np.asarray(uncertainty_raw, dtype=np.float64)
    )
    if (
        temperature.ndim != 1
        or values.ndim != 1
        or len(temperature) < 2
        or values.shape != temperature.shape
        or (
            uncertainty is not None
            and (
                uncertainty.ndim != 1
                or uncertainty.shape != temperature.shape
            )
        )
    ):
        raise ValueError(
            f"{source}: {field} temperature, values, and optional uncertainty "
            "must be equal-length 1D arrays with at least two points."
        )
    if not (
        np.isfinite(temperature).all()
        and np.isfinite(values).all()
        and (
            uncertainty is None
            or np.isfinite(uncertainty).all()
        )
    ):
        raise ValueError(f"{source}: all {field} values must be finite.")
    if np.any(np.diff(temperature) <= 0.0):
        raise ValueError(
            f"{source}: {field} temperatures must be strictly increasing."
        )
    if np.any(values <= 0.0):
        raise ValueError(f"{source}: {field} values must be positive.")
    if uncertainty is not None and np.any(uncertainty < 0.0):
        raise ValueError(f"{source}: {field} uncertainty must be non-negative.")
    return PropertyCurve(
        temperature=temperature,
        values=values,
        uncertainty=uncertainty,
        label=str(payload.get("label", default_label)),
        units=str(payload.get("units", default_units)),
        source_url=str(payload.get("source_url", "")),
        uncertainty_basis=str(payload.get("uncertainty_basis", "")),
    )


def load_property_table(path: str | Path) -> PropertyTable:
    source = Path(path).resolve()
    raw = source.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a YAML mapping.")
    required_metadata = ("name", "version", "source", "authoritative")
    missing_metadata = [key for key in required_metadata if key not in payload]
    if missing_metadata:
        raise ValueError(
            f"{source}: missing required metadata fields {missing_metadata}."
        )
    for key in ("name", "version", "source"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"{source}: {key} must be a non-empty string.")
    if not isinstance(payload["authoritative"], bool):
        raise ValueError(f"{source}: authoritative must be a boolean.")

    interpolation = str(payload.get("interpolation", "linear"))
    if interpolation != "linear":
        raise ValueError(
            f"{source}: only interpolation='linear' is supported, got "
            f"{interpolation!r}."
        )
    pressure_basis = payload.get("pressure_basis")
    if not isinstance(pressure_basis, str) or not pressure_basis.strip():
        raise ValueError(f"{source}: pressure_basis must be a non-empty string.")

    try:
        density = float(payload["density"])
        pressure = float(payload["pressure_Pa"])
    except KeyError as exc:
        raise ValueError(
            f"{source}: missing required field {exc.args[0]!r}."
        ) from exc
    if not np.isfinite(density) or not np.isfinite(pressure):
        raise ValueError(f"{source}: density and pressure must be finite.")
    if density <= 0.0:
        raise ValueError(f"{source}: density must be positive.")
    if pressure <= 0.0:
        raise ValueError(f"{source}: pressure_Pa must be positive.")

    conductivity_payload = payload.get("conductivity")
    if isinstance(conductivity_payload, dict):
        if "x" not in conductivity_payload or "y" not in conductivity_payload:
            raise ValueError(
                f"{source}: directional conductivity requires both x and y."
            )
        conductivity_x = _validated_curve(
            conductivity_payload["x"],
            source=source,
            field="conductivity.x",
            default_label="through-thickness",
            default_units="W/m-K",
        )
        conductivity_y = _validated_curve(
            conductivity_payload["y"],
            source=source,
            field="conductivity.y",
            default_label="in-plane",
            default_units="W/m-K",
        )
        specific_heat_curve = _validated_curve(
            payload.get("specific_heat"),
            source=source,
            field="specific_heat",
            default_label="specific heat",
            default_units="J/kg-K",
        )
    else:
        try:
            legacy_temperature = payload["temperature"]
            legacy_conductivity = payload["conductivity"]
            legacy_specific_heat = payload["specific_heat"]
        except KeyError as exc:
            raise ValueError(
                f"{source}: missing required field {exc.args[0]!r}."
            ) from exc
        conductivity_x = _validated_curve(
            {
                "temperature": legacy_temperature,
                "values": legacy_conductivity,
                "label": "isotropic",
                "units": "W/m-K",
            },
            source=source,
            field="conductivity",
            default_label="isotropic",
            default_units="W/m-K",
        )
        conductivity_y = conductivity_x
        specific_heat_curve = _validated_curve(
            {
                "temperature": legacy_temperature,
                "values": legacy_specific_heat,
                "label": "specific heat",
                "units": "J/kg-K",
            },
            source=source,
            field="specific_heat",
            default_label="specific heat",
            default_units="J/kg-K",
        )

    density_uncertainty_raw = payload.get("density_uncertainty")
    density_uncertainty = (
        None
        if density_uncertainty_raw is None
        else float(density_uncertainty_raw)
    )
    if density_uncertainty is not None and (
        not np.isfinite(density_uncertainty)
        or density_uncertainty < 0.0
    ):
        raise ValueError(
            f"{source}: density_uncertainty must be finite and non-negative."
        )

    return PropertyTable(
        name=str(payload.get("name", source.stem)),
        version=str(payload.get("version", "")),
        source=str(payload.get("source", "")),
        authoritative=bool(payload.get("authoritative", False)),
        content_sha256=sha256(raw).hexdigest(),
        density=density,
        density_uncertainty=density_uncertainty,
        pressure_Pa=pressure,
        pressure_basis=pressure_basis,
        conductivity_x=conductivity_x,
        conductivity_y=conductivity_y,
        specific_heat_curve=specific_heat_curve,
        material_system=str(payload.get("material_system", "")),
    )


class PropertyModel(Protocol):
    is_temperature_dependent: bool
    reference_temperature: float

    def conductivity(
        self,
        temperature: Array | float,
        mode: EvalMode = "accepted",
        direction: Literal["x", "y"] = "x",
    ) -> Array: ...

    def volumetric_heat_capacity(
        self,
        temperature: Array | float,
        mode: EvalMode = "accepted",
    ) -> Array: ...

    def enthalpy(
        self,
        temperature: Array | float,
        mode: EvalMode = "accepted",
    ) -> Array: ...

    def provenance(self) -> dict[str, Any]: ...


@dataclass
class ConstantPropertyModel:
    k: float
    rho_c: float
    reference_temperature: float = 0.0
    is_temperature_dependent: bool = False
    query_temperature_min: float = float("inf")
    query_temperature_max: float = float("-inf")

    def _observe(self, temperature: Array | float) -> Array:
        values = np.asarray(temperature, dtype=np.float64)
        if values.size:
            self.query_temperature_min = min(
                self.query_temperature_min,
                float(np.min(values)),
            )
            self.query_temperature_max = max(
                self.query_temperature_max,
                float(np.max(values)),
            )
        return values

    def conductivity(
        self,
        temperature: Array | float,
        mode: EvalMode = "accepted",
        direction: Literal["x", "y"] = "x",
    ) -> Array:
        del mode, direction
        return np.full_like(self._observe(temperature), self.k)

    def volumetric_heat_capacity(
        self,
        temperature: Array | float,
        mode: EvalMode = "accepted",
    ) -> Array:
        del mode
        return np.full_like(self._observe(temperature), self.rho_c)

    def enthalpy(
        self,
        temperature: Array | float,
        mode: EvalMode = "accepted",
    ) -> Array:
        del mode
        values = self._observe(temperature)
        return self.rho_c * (values - self.reference_temperature)

    def provenance(self) -> dict[str, Any]:
        return {
            "model": "constant_effective",
            "reference_k_W_mK": self.k,
            "reference_rho_c_J_m3K": self.rho_c,
            "reference_temperature_K": self.reference_temperature,
        }


class TabulatedPropertyModel:
    is_temperature_dependent = True

    def __init__(
        self,
        table: PropertyTable,
        *,
        reference_temperature: float | None = None,
        extrapolation: str = "reject",
    ):
        if extrapolation not in {"reject", "clamp"}:
            raise ValueError("extrapolation must be 'reject' or 'clamp'.")
        self.table = table
        self.reference_temperature = (
            table.temperature_min
            if reference_temperature is None
            else float(reference_temperature)
        )
        if not (
            table.temperature_min
            <= self.reference_temperature
            <= table.temperature_max
        ):
            raise ValueError(
                "reference_temperature must lie inside the property-table range."
            )
        self.extrapolation = extrapolation
        self.iteration_range_clamps = 0
        self.accepted_range_excursions = 0
        self.query_temperature_min = float("inf")
        self.query_temperature_max = float("-inf")
        self._node_enthalpy = self._integral_nodes(
            table.specific_heat_curve.temperature,
            table.specific_heat_curve.values,
        )
        self._reference_enthalpy = float(
            self._piecewise_integral(
                np.asarray(self.reference_temperature),
                table.specific_heat_curve.temperature,
                table.specific_heat_curve.values,
                self._node_enthalpy,
            )
        )

    def _bounded(self, temperature: Array | float, mode: EvalMode) -> Array:
        if mode not in {"iterate", "accepted"}:
            raise ValueError(f"Unsupported property evaluation mode {mode!r}.")
        values = np.asarray(temperature, dtype=np.float64)
        if values.size:
            self.query_temperature_min = min(
                self.query_temperature_min,
                float(np.min(values)),
            )
            self.query_temperature_max = max(
                self.query_temperature_max,
                float(np.max(values)),
            )
        range_tolerance = max(
            1.0e-8,
            64.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                abs(self.table.temperature_min),
                abs(self.table.temperature_max),
            ),
        )
        outside = (
            (values < self.table.temperature_min - range_tolerance)
            | (values > self.table.temperature_max + range_tolerance)
        )
        count = int(np.count_nonzero(outside))
        if count:
            if mode == "iterate":
                self.iteration_range_clamps += count
            else:
                self.accepted_range_excursions += count
                if self.extrapolation == "reject":
                    observed = (float(np.min(values)), float(np.max(values)))
                    raise PropertyRangeError(
                        "Accepted temperature range "
                        f"[{observed[0]:.6g}, {observed[1]:.6g}] K lies outside "
                        f"table range [{self.table.temperature_min:.6g}, "
                        f"{self.table.temperature_max:.6g}] K."
                    )
        return np.clip(
            values,
            self.table.temperature_min,
            self.table.temperature_max,
        )

    def conductivity(
        self,
        temperature: Array | float,
        mode: EvalMode = "accepted",
        direction: Literal["x", "y"] = "x",
    ) -> Array:
        bounded = self._bounded(temperature, mode)
        if direction == "x":
            curve = self.table.conductivity_x
        elif direction == "y":
            curve = self.table.conductivity_y
        else:
            raise ValueError("direction must be 'x' or 'y'.")
        return np.interp(
            bounded,
            curve.temperature,
            curve.values,
        )

    def volumetric_heat_capacity(
        self,
        temperature: Array | float,
        mode: EvalMode = "accepted",
    ) -> Array:
        bounded = self._bounded(temperature, mode)
        return self.table.density * np.interp(
            bounded,
            self.table.specific_heat_curve.temperature,
            self.table.specific_heat_curve.values,
        )

    @staticmethod
    def _integral_nodes(temperature: Array, values: Array) -> Array:
        increments = (
            0.5
            * (values[:-1] + values[1:])
            * np.diff(temperature)
        )
        return np.concatenate(([0.0], np.cumsum(increments)))

    def _piecewise_integral(
        self,
        temperature: Array,
        knots: Array,
        values: Array,
        node_integral: Array,
    ) -> Array:
        flat = np.asarray(temperature, dtype=np.float64).reshape(-1)
        indices = np.searchsorted(
            knots,
            flat,
            side="right",
        ) - 1
        indices = np.clip(indices, 0, len(knots) - 2)
        t0 = knots[indices]
        dt = knots[indices + 1] - t0
        s = (flat - t0) / dt
        result = node_integral[indices] + dt * (
            values[indices] * s
            + 0.5 * (values[indices + 1] - values[indices]) * s * s
        )
        return result.reshape(np.asarray(temperature).shape)

    def enthalpy(
        self,
        temperature: Array | float,
        mode: EvalMode = "accepted",
    ) -> Array:
        values = np.asarray(temperature, dtype=np.float64)
        bounded = self._bounded(values, mode)
        integral = self._piecewise_integral(
            bounded,
            self.table.specific_heat_curve.temperature,
            self.table.specific_heat_curve.values,
            self._node_enthalpy,
        )
        # Under clamp, the property itself is held at the endpoint. Its exact
        # enthalpy continuation is therefore linear and remains C1.
        below = values < self.table.temperature_min
        above = values > self.table.temperature_max
        if np.any(below):
            integral = np.asarray(integral).copy()
            endpoint_cp = float(
                np.interp(
                    self.table.temperature_min,
                    self.table.specific_heat_curve.temperature,
                    self.table.specific_heat_curve.values,
                )
            )
            integral[below] += (
                endpoint_cp
                * (values[below] - self.table.temperature_min)
            )
        if np.any(above):
            integral = np.asarray(integral).copy()
            endpoint_cp = float(
                np.interp(
                    self.table.temperature_max,
                    self.table.specific_heat_curve.temperature,
                    self.table.specific_heat_curve.values,
                )
            )
            integral[above] += (
                endpoint_cp
                * (values[above] - self.table.temperature_max)
            )
        return self.table.density * (integral - self._reference_enthalpy)

    def conductivity_integral(
        self,
        temperature: Array | float,
        mode: EvalMode = "accepted",
        direction: Literal["x", "y"] = "x",
    ) -> Array:
        values = np.asarray(temperature, dtype=np.float64)
        bounded = self._bounded(values, mode)
        if direction == "x":
            curve = self.table.conductivity_x
        elif direction == "y":
            curve = self.table.conductivity_y
        else:
            raise ValueError("direction must be 'x' or 'y'.")
        nodes = self._integral_nodes(curve.temperature, curve.values)
        integral = self._piecewise_integral(
            bounded,
            curve.temperature,
            curve.values,
            nodes,
        )
        reference = float(
            self._piecewise_integral(
                np.asarray(self.reference_temperature),
                curve.temperature,
                curve.values,
                nodes,
            )
        )
        return integral - reference

    def provenance(self) -> dict[str, Any]:
        return {
            "model": "temperature_dependent_table",
            **self.table.provenance(),
            "extrapolation": self.extrapolation,
            "interpolation": "linear",
            "reference_temperature_K": self.reference_temperature,
            "reference_conductivity_x_W_mK": float(
                self.conductivity(
                    self.reference_temperature,
                    direction="x",
                )
            ),
            "reference_conductivity_y_W_mK": float(
                self.conductivity(
                    self.reference_temperature,
                    direction="y",
                )
            ),
            "reference_volumetric_heat_capacity_J_m3K": float(
                self.volumetric_heat_capacity(self.reference_temperature)
            ),
        }


@dataclass(frozen=True)
class RegionProperties:
    tps: PropertyModel
    bond: PropertyModel
    backing: PropertyModel

    @property
    def is_temperature_dependent(self) -> bool:
        return any(
            model.is_temperature_dependent
            for model in (self.tps, self.bond, self.backing)
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "tps": self.tps.provenance(),
            "bond": self.bond.provenance(),
            "backing": self.backing.provenance(),
        }

    @property
    def iteration_range_clamps(self) -> int:
        return sum(
            int(getattr(model, "iteration_range_clamps", 0))
            for model in (self.tps, self.bond, self.backing)
        )

    @property
    def accepted_range_excursions(self) -> int:
        return sum(
            int(getattr(model, "accepted_range_excursions", 0))
            for model in (self.tps, self.bond, self.backing)
        )

    @property
    def query_temperature_range(self) -> tuple[float, float]:
        minima = [
            float(getattr(model, "query_temperature_min", float("inf")))
            for model in (self.tps, self.bond, self.backing)
        ]
        maxima = [
            float(getattr(model, "query_temperature_max", float("-inf")))
            for model in (self.tps, self.bond, self.backing)
        ]
        minimum = min(minima)
        maximum = max(maxima)
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            return (float("nan"), float("nan"))
        return (minimum, maximum)


def _model_from_material(config: StudyConfig, material: Material) -> PropertyModel:
    if material.model == "constant_effective":
        return ConstantPropertyModel(
            float(material.k),
            float(material.rho_c),
        )
    if material.property_table is None:
        raise ValueError("A table-backed material requires property_table.")
    table_path = config.resolve_property_table_path(material.property_table.path)
    table = load_property_table(table_path)
    return TabulatedPropertyModel(
        table,
        reference_temperature=material.property_table.reference_temperature,
        extrapolation=material.property_table.extrapolation,
    )


def build_region_properties(config: StudyConfig) -> RegionProperties:
    return RegionProperties(
        tps=_model_from_material(config, config.tps),
        bond=ConstantPropertyModel(config.bond.k0, config.bond.rho_c),
        backing=_model_from_material(config, config.backing),
    )
