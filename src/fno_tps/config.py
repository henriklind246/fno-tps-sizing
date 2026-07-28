from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import math

import yaml


ALLOWED_TPS_PROPERTY_MODELS = frozenset(
    {"constant_effective", "temperature_dependent_table"}
)


@dataclass(frozen=True)
class PropertyTableConfig:
    path: str
    version: str
    content_sha256: str | None = None
    extrapolation: str = "reject"
    interpolation: str = "linear"
    reference_temperature: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PropertyTableConfig":
        out = cls(
            path=str(value["path"]),
            version=str(value["version"]),
            content_sha256=(
                None
                if value.get("content_sha256") is None
                else str(value["content_sha256"])
            ),
            extrapolation=str(value.get("extrapolation", "reject")),
            interpolation=str(value.get("interpolation", "linear")),
            reference_temperature=(
                None
                if value.get("reference_temperature") is None
                else float(value["reference_temperature"])
            ),
        )
        if out.extrapolation not in {"reject", "clamp"}:
            raise ValueError("property_table.extrapolation must be reject or clamp.")
        if out.interpolation != "linear":
            raise ValueError(
                "Only property_table.interpolation='linear' is supported."
            )
        return out


@dataclass(frozen=True)
class Material:
    k: float | None = None
    rho_c: float | None = None
    model: str = "constant_effective"
    reference_k: float | None = None
    reference_k_y: float | None = None
    reference_rho_c: float | None = None
    property_table: PropertyTableConfig | None = None

    def __post_init__(self) -> None:
        if self.model not in ALLOWED_TPS_PROPERTY_MODELS:
            raise ValueError(f"Unsupported material model {self.model!r}.")
        reference_k = self.k if self.reference_k is None else self.reference_k
        reference_rho_c = (
            self.rho_c if self.reference_rho_c is None else self.reference_rho_c
        )
        if reference_k is None or reference_rho_c is None:
            raise ValueError("Material reference_k and reference_rho_c are required.")
        if reference_k <= 0.0 or reference_rho_c <= 0.0:
            raise ValueError("Material reference properties must be positive.")
        object.__setattr__(self, "reference_k", float(reference_k))
        reference_k_y = (
            reference_k
            if self.reference_k_y is None
            else self.reference_k_y
        )
        if reference_k_y <= 0.0:
            raise ValueError("Material reference_k_y must be positive.")
        object.__setattr__(self, "reference_k_y", float(reference_k_y))
        object.__setattr__(self, "reference_rho_c", float(reference_rho_c))
        if self.model == "constant_effective":
            if self.k is None or self.rho_c is None:
                raise ValueError("Constant materials require k and rho_c.")
            if self.k <= 0.0 or self.rho_c <= 0.0:
                raise ValueError("Material k and rho_c must be positive.")
            if self.property_table is not None:
                raise ValueError("Constant materials cannot define property_table.")
        elif self.property_table is None:
            raise ValueError(
                "temperature_dependent_table materials require property_table."
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Material":
        model = str(value.get("model", "constant_effective"))
        k = None if value.get("k") is None else float(value["k"])
        rho_c = None if value.get("rho_c") is None else float(value["rho_c"])
        return cls(
            k=k,
            rho_c=rho_c,
            model=model,
            reference_k=float(value.get("reference_k", k))
            if value.get("reference_k", k) is not None
            else None,
            reference_k_y=(
                None
                if value.get("reference_k_y") is None
                else float(value["reference_k_y"])
            ),
            reference_rho_c=float(value.get("reference_rho_c", rho_c))
            if value.get("reference_rho_c", rho_c) is not None
            else None,
            property_table=(
                None
                if value.get("property_table") is None
                else PropertyTableConfig.from_dict(value["property_table"])
            ),
        )

    @classmethod
    def constant(cls, k: float, rho_c: float) -> "Material":
        return cls(k=float(k), rho_c=float(rho_c))

    def as_dict(self) -> dict[str, Any]:
        if (
            self.model == "constant_effective"
            and self.property_table is None
            and self.reference_k == self.k
            and self.reference_k_y == self.k
            and self.reference_rho_c == self.rho_c
        ):
            return {"k": self.k, "rho_c": self.rho_c}
        payload: dict[str, Any] = {
            "model": self.model,
            "reference_k": self.reference_k,
            "reference_k_y": self.reference_k_y,
            "reference_rho_c": self.reference_rho_c,
        }
        if self.k is not None:
            payload["k"] = self.k
        if self.rho_c is not None:
            payload["rho_c"] = self.rho_c
        if self.property_table is not None:
            payload["property_table"] = asdict(self.property_table)
        return payload


@dataclass(frozen=True)
class HotFaceValidityConfig:
    study_temperature_limit: float
    material_multiple_use_limit: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HotFaceValidityConfig":
        out = cls(
            study_temperature_limit=float(value["study_temperature_limit"]),
            material_multiple_use_limit=(
                None
                if value.get("material_multiple_use_limit") is None
                else float(value["material_multiple_use_limit"])
            ),
        )
        if out.study_temperature_limit <= 0.0:
            raise ValueError(
                "validity.hot_face.study_temperature_limit must be positive."
            )
        if (
            out.material_multiple_use_limit is not None
            and out.material_multiple_use_limit <= 0.0
        ):
            raise ValueError(
                "validity.hot_face.material_multiple_use_limit must be positive."
            )
        return out

    @property
    def configured_temperature_limit(self) -> float:
        values = [self.study_temperature_limit]
        if self.material_multiple_use_limit is not None:
            values.append(self.material_multiple_use_limit)
        return min(values)


@dataclass(frozen=True)
class PropertyTableValidityConfig:
    allow_extrapolation: bool
    minimum_temperature: float
    maximum_temperature: float | None = None

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
    ) -> "PropertyTableValidityConfig":
        out = cls(
            allow_extrapolation=bool(value.get("allow_extrapolation", False)),
            minimum_temperature=float(value["minimum_temperature"]),
            maximum_temperature=(
                None
                if value.get("maximum_temperature") is None
                else float(value["maximum_temperature"])
            ),
        )
        if out.minimum_temperature <= 0.0:
            raise ValueError(
                "validity.property_tables.minimum_temperature must be positive."
            )
        if (
            out.maximum_temperature is not None
            and out.maximum_temperature <= out.minimum_temperature
        ):
            raise ValueError(
                "validity.property_tables.maximum_temperature must exceed "
                "minimum_temperature."
            )
        return out


@dataclass(frozen=True)
class ValidityConfig:
    tps_property_model: str
    hot_face: HotFaceValidityConfig
    property_tables: PropertyTableValidityConfig
    max_property_relative_deviation: float
    conductivity_reference_pressure: float
    reject_cases_outside_validity: bool
    reject_on_property_extrapolation: bool = True
    reject_on_nonlinear_failure: bool = True
    reject_on_nonphysical_temperature: bool = True
    reject_on_energy_residual: bool = True
    minimum_physical_temperature: float = 0.0
    max_relative_energy_residual: float = 1.0e-7

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ValidityConfig":
        # Read the pre-hierarchy shape so archived benchmark configurations and
        # manifests remain loadable. Newly resolved configurations always emit
        # the explicit hot-face/property-table hierarchy.
        hot_face_payload = value.get("hot_face")
        if hot_face_payload is None:
            hot_face_payload = {
                "study_temperature_limit": value["max_hot_face_temperature"],
                "material_multiple_use_limit": None,
            }
        table_payload = value.get("property_tables")
        if table_payload is None:
            table_payload = {
                "allow_extrapolation": False,
                "minimum_temperature": value["tps_property_temperature_min"],
                "maximum_temperature": value.get("max_hot_face_temperature"),
            }
        out = cls(
            tps_property_model=str(value["tps_property_model"]),
            hot_face=HotFaceValidityConfig.from_dict(hot_face_payload),
            property_tables=PropertyTableValidityConfig.from_dict(
                table_payload
            ),
            max_property_relative_deviation=float(
                value["max_property_relative_deviation"]
            ),
            conductivity_reference_pressure=float(
                value["conductivity_reference_pressure"]
            ),
            reject_cases_outside_validity=bool(
                value["reject_cases_outside_validity"]
            ),
            reject_on_property_extrapolation=bool(
                value.get("reject_on_property_extrapolation", True)
            ),
            reject_on_nonlinear_failure=bool(
                value.get("reject_on_nonlinear_failure", True)
            ),
            reject_on_nonphysical_temperature=bool(
                value.get("reject_on_nonphysical_temperature", True)
            ),
            reject_on_energy_residual=bool(
                value.get("reject_on_energy_residual", True)
            ),
            minimum_physical_temperature=float(
                value.get("minimum_physical_temperature", 0.0)
            ),
            max_relative_energy_residual=float(
                value.get("max_relative_energy_residual", 1.0e-7)
            ),
        )
        if out.tps_property_model not in ALLOWED_TPS_PROPERTY_MODELS:
            raise ValueError(
                "validity.tps_property_model must be one of "
                f"{sorted(ALLOWED_TPS_PROPERTY_MODELS)}."
            )
        if not 0.0 < out.max_property_relative_deviation < 1.0:
            raise ValueError(
                "validity.max_property_relative_deviation must lie in (0, 1)."
            )
        if out.conductivity_reference_pressure <= 0.0:
            raise ValueError(
                "validity.conductivity_reference_pressure must be positive."
            )
        if out.minimum_physical_temperature < 0.0:
            raise ValueError(
                "validity.minimum_physical_temperature cannot be negative."
            )
        if out.max_relative_energy_residual <= 0.0:
            raise ValueError(
                "validity.max_relative_energy_residual must be positive."
            )
        if (
            not out.property_tables.allow_extrapolation
            and not out.reject_on_property_extrapolation
        ):
            raise ValueError(
                "Forbidden property extrapolation requires "
                "validity.reject_on_property_extrapolation=true."
            )
        return out

    @property
    def tps_property_temperature_min(self) -> float:
        """Compatibility alias for readers of pre-hierarchy manifests."""
        return self.property_tables.minimum_temperature

    @property
    def max_hot_face_temperature(self) -> float:
        """Compatibility alias; new code should use StudyConfig's resolved limit."""
        return self.hot_face.configured_temperature_limit

    def as_dict(self) -> dict[str, Any]:
        return {
            "tps_property_model": self.tps_property_model,
            "hot_face": asdict(self.hot_face),
            "property_tables": asdict(self.property_tables),
            "max_property_relative_deviation": (
                self.max_property_relative_deviation
            ),
            "conductivity_reference_pressure": (
                self.conductivity_reference_pressure
            ),
            "reject_cases_outside_validity": (
                self.reject_cases_outside_validity
            ),
            "reject_on_property_extrapolation": (
                self.reject_on_property_extrapolation
            ),
            "reject_on_nonlinear_failure": self.reject_on_nonlinear_failure,
            "reject_on_nonphysical_temperature": (
                self.reject_on_nonphysical_temperature
            ),
            "reject_on_energy_residual": self.reject_on_energy_residual,
            "minimum_physical_temperature": self.minimum_physical_temperature,
            "max_relative_energy_residual": (
                self.max_relative_energy_residual
            ),
        }


@dataclass(frozen=True)
class SurfaceConfig:
    model: str
    emissivity_model: str
    emissivity: float
    coating: str | None
    condition: str | None
    emissivity_quantity: str
    emissivity_basis: str
    emissivity_source: str
    emissivity_source_url: str
    emissivity_authoritative: bool
    radiation_sink_temperature: float
    stefan_boltzmann_constant: float
    nonlinear_absolute_flux_tolerance: float
    nonlinear_relative_tolerance: float
    nonlinear_max_iterations: int
    nonlinear_relaxation: float
    radiation_coupling: str
    emissivity_temperature_min: float | None
    emissivity_temperature_max: float | None

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "SurfaceConfig":
        payload = {} if value is None else dict(value)
        model = str(payload.get("model", "prescribed_net_heat_flux"))
        defaults = {
            "emissivity_model": "constant",
            "emissivity": 0.0,
            "coating": None,
            "condition": None,
            "emissivity_quantity": "unspecified",
            "emissivity_basis": "",
            "emissivity_source": "",
            "emissivity_source_url": "",
            "emissivity_authoritative": False,
            "radiation_sink_temperature": 300.0,
            "stefan_boltzmann_constant": 5.670374419e-8,
            "nonlinear_absolute_flux_tolerance": 1.0e-7,
            "nonlinear_relative_tolerance": 1.0e-10,
            "nonlinear_max_iterations": 100,
            "nonlinear_relaxation": 0.80,
            "radiation_coupling": "boundary_response",
            "emissivity_temperature_min": None,
            "emissivity_temperature_max": None,
        }
        defaults.update(payload)
        out = cls(
            model=model,
            emissivity_model=str(defaults["emissivity_model"]),
            emissivity=float(defaults["emissivity"]),
            coating=(
                None
                if defaults["coating"] is None
                else str(defaults["coating"])
            ),
            condition=(
                None
                if defaults["condition"] is None
                else str(defaults["condition"])
            ),
            emissivity_quantity=str(defaults["emissivity_quantity"]),
            emissivity_basis=str(defaults["emissivity_basis"]),
            emissivity_source=str(defaults["emissivity_source"]),
            emissivity_source_url=str(defaults["emissivity_source_url"]),
            emissivity_authoritative=bool(
                defaults["emissivity_authoritative"]
            ),
            radiation_sink_temperature=float(
                defaults["radiation_sink_temperature"]
            ),
            stefan_boltzmann_constant=float(
                defaults["stefan_boltzmann_constant"]
            ),
            nonlinear_absolute_flux_tolerance=float(
                defaults["nonlinear_absolute_flux_tolerance"]
            ),
            nonlinear_relative_tolerance=float(
                defaults["nonlinear_relative_tolerance"]
            ),
            nonlinear_max_iterations=int(defaults["nonlinear_max_iterations"]),
            nonlinear_relaxation=float(defaults["nonlinear_relaxation"]),
            radiation_coupling=str(defaults["radiation_coupling"]),
            emissivity_temperature_min=(
                None
                if defaults["emissivity_temperature_min"] is None
                else float(defaults["emissivity_temperature_min"])
            ),
            emissivity_temperature_max=(
                None
                if defaults["emissivity_temperature_max"] is None
                else float(defaults["emissivity_temperature_max"])
            ),
        )
        if out.model not in {
            "prescribed_net_heat_flux",
            "incident_heat_flux_with_reradiation",
        }:
            raise ValueError(f"Unsupported surface.model={out.model!r}.")
        if out.emissivity_model != "constant":
            raise ValueError(
                "The current implementation requires surface.emissivity_model="
                "'constant'; temperature-dependent emissivity is out of scope."
            )
        if not 0.0 <= out.emissivity <= 1.0:
            raise ValueError("surface.emissivity must lie in [0, 1].")
        if out.emissivity_authoritative and (
            not out.emissivity_source.strip()
            or not out.emissivity_source_url.strip()
            or not out.emissivity_basis.strip()
        ):
            raise ValueError(
                "Authoritative emissivity requires source, source URL, and basis."
            )
        if out.radiation_sink_temperature <= 0.0:
            raise ValueError(
                "surface.radiation_sink_temperature must be positive."
            )
        if out.stefan_boltzmann_constant <= 0.0:
            raise ValueError(
                "surface.stefan_boltzmann_constant must be positive."
            )
        if (
            out.nonlinear_absolute_flux_tolerance <= 0.0
            or out.nonlinear_relative_tolerance <= 0.0
            or out.nonlinear_max_iterations < 1
            or not 0.0 < out.nonlinear_relaxation <= 1.0
        ):
            raise ValueError("Invalid nonlinear surface-solver controls.")
        if out.radiation_coupling not in {
            "boundary_response",
            "coupled_nonlinear",
        }:
            raise ValueError(
                "surface.radiation_coupling must be boundary_response or "
                "coupled_nonlinear."
            )
        if (
            out.model == "incident_heat_flux_with_reradiation"
            and out.emissivity <= 0.0
        ):
            raise ValueError(
                "Incident heating with reradiation requires positive emissivity."
            )
        if out.model == "prescribed_net_heat_flux" and out.emissivity != 0.0:
            raise ValueError(
                "prescribed_net_heat_flux requires surface.emissivity=0 because "
                "the prescribed load is already net of unresolved surface losses."
            )
        if (
            out.emissivity_temperature_min is not None
            and out.emissivity_temperature_min <= 0.0
        ):
            raise ValueError("surface.emissivity_temperature_min must be positive.")
        if (
            out.emissivity_temperature_max is not None
            and (
                out.emissivity_temperature_min is None
                or out.emissivity_temperature_max
                <= out.emissivity_temperature_min
            )
        ):
            raise ValueError(
                "surface.emissivity_temperature_max requires and must exceed "
                "surface.emissivity_temperature_min."
            )
        return out

    @property
    def reradiation_enabled(self) -> bool:
        return self.model == "incident_heat_flux_with_reradiation"

    def as_dict(self, *, preserve_legacy_shape: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if preserve_legacy_shape and self.radiation_coupling == "boundary_response":
            payload.pop("radiation_coupling")
        return payload


@dataclass(frozen=True)
class NonlinearSolverConfig:
    max_iterations: int = 25
    residual_temperature_tolerance: float = 1.0e-8
    update_temperature_tolerance: float = 1.0e-8
    update_relative_tolerance: float = 1.0e-10
    initial_damping: float = 1.0
    minimum_damping: float = 0.015625
    max_temperature_step: float = 250.0
    armijo_c: float = 1.0e-4
    max_backtracks: int = 6

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any] | None,
    ) -> "NonlinearSolverConfig":
        payload = {} if value is None else dict(value)
        defaults = asdict(cls())
        defaults.update(payload)
        out = cls(
            max_iterations=int(defaults["max_iterations"]),
            residual_temperature_tolerance=float(
                defaults["residual_temperature_tolerance"]
            ),
            update_temperature_tolerance=float(
                defaults["update_temperature_tolerance"]
            ),
            update_relative_tolerance=float(
                defaults["update_relative_tolerance"]
            ),
            initial_damping=float(defaults["initial_damping"]),
            minimum_damping=float(defaults["minimum_damping"]),
            max_temperature_step=float(defaults["max_temperature_step"]),
            armijo_c=float(defaults["armijo_c"]),
            max_backtracks=int(defaults["max_backtracks"]),
        )
        if (
            out.max_iterations < 1
            or out.max_backtracks < 0
            or min(
                out.residual_temperature_tolerance,
                out.update_temperature_tolerance,
                out.update_relative_tolerance,
                out.initial_damping,
                out.minimum_damping,
                out.max_temperature_step,
                out.armijo_c,
            )
            <= 0.0
            or out.initial_damping > 1.0
            or out.minimum_damping > out.initial_damping
        ):
            raise ValueError("Invalid solver.nonlinear controls.")
        return out


@dataclass(frozen=True)
class SolverConfig:
    nonlinear: NonlinearSolverConfig

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "SolverConfig":
        payload = {} if value is None else dict(value)
        return cls(
            nonlinear=NonlinearSolverConfig.from_dict(payload.get("nonlinear"))
        )


@dataclass(frozen=True)
class MeshConfig:
    nx_tps: int
    nx_bond: int
    nx_back: int
    ny: int

    @property
    def nx(self) -> int:
        return self.nx_tps + self.nx_bond + self.nx_back

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MeshConfig":
        out = cls(**{name: int(value[name]) for name in ("nx_tps", "nx_bond", "nx_back", "ny")})
        if min(out.nx_tps, out.nx_back, out.ny) < 2 or out.nx_bond < 1:
            raise ValueError("TPS/back/ny require at least 2 cells and bond at least 1.")
        return out


@dataclass(frozen=True)
class TimeConfig:
    dt: float
    t_final: float
    save_stride: int
    horizon_candidates: tuple[float, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TimeConfig":
        out = cls(
            dt=float(value["dt"]),
            t_final=float(value["t_final"]),
            save_stride=int(value["save_stride"]),
            horizon_candidates=tuple(float(v) for v in value.get("horizon_candidates", [value["t_final"]])),
        )
        if out.dt <= 0.0 or out.t_final <= 0.0 or out.save_stride < 1:
            raise ValueError("dt, t_final, and save_stride must be positive.")
        out.assert_aligned(out.t_final)
        for horizon in out.horizon_candidates:
            out.assert_aligned(horizon)
        return out

    def assert_aligned(self, time: float) -> None:
        steps = float(time) / self.dt
        if abs(steps - round(steps)) > 1e-10:
            raise ValueError(f"time={time} is not aligned with dt={self.dt}.")

    def saved_times(self, final_time: float | None = None) -> list[float]:
        horizon = self.t_final if final_time is None else float(final_time)
        self.assert_aligned(horizon)
        n_steps = int(round(horizon / self.dt))
        indices = list(range(0, n_steps + 1, self.save_stride))
        if indices[-1] != n_steps:
            indices.append(n_steps)
        return [index * self.dt for index in indices]


@dataclass(frozen=True)
class HeatingBounds:
    amplitude: tuple[float, float]
    y_center: tuple[float, float]
    t_center: tuple[float, float]
    sigma_y: tuple[float, float]
    sigma_t: tuple[float, float]
    max_events: int = 3
    interpretation: str = "effective_net_inward_conductive_heat_flux"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HeatingBounds":
        out = cls(
            amplitude=_pair(value["amplitude"]),
            y_center=_pair(value["y_center"]),
            t_center=_pair(value["t_center"]),
            sigma_y=_pair(value["sigma_y"]),
            sigma_t=_pair(value["sigma_t"]),
            max_events=int(value.get("max_events", 3)),
            interpretation=str(
                value.get(
                    "interpretation",
                    "effective_net_inward_conductive_heat_flux",
                )
            ),
        )
        for name in ("amplitude", "sigma_y", "sigma_t"):
            lo, hi = getattr(out, name)
            if lo <= 0.0 or hi < lo:
                raise ValueError(f"Invalid positive range for heating.{name}.")
        if out.max_events != 3:
            raise ValueError("The core study requires max_events=3.")
        if out.interpretation not in {
            "effective_net_inward_conductive_heat_flux",
            "incident_aerothermal_heat_flux",
        }:
            raise ValueError(
                "heating.interpretation must be either an effective net inward "
                "conductive flux or an incident aerothermal flux."
            )
        return out


@dataclass(frozen=True)
class BondConfig:
    k0: float
    k_min: float
    rho_c: float
    severity: tuple[float, float]
    y_center: tuple[float, float]
    sigma: tuple[float, float]
    max_defects: int = 2

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BondConfig":
        out = cls(
            k0=float(value["k0"]),
            k_min=float(value["k_min"]),
            rho_c=float(value["rho_c"]),
            severity=_pair(value["severity"]),
            y_center=_pair(value["y_center"]),
            sigma=_pair(value["sigma"]),
            max_defects=int(value.get("max_defects", 2)),
        )
        if not (0.0 < out.k_min <= out.k0) or out.rho_c <= 0.0:
            raise ValueError("Bond requires 0 < k_min <= k0 and positive rho_c.")
        if out.severity[0] < 0.0 or out.severity[1] < out.severity[0]:
            raise ValueError("Bond severity range is invalid.")
        if out.sigma[0] <= 0.0 or out.sigma[1] < out.sigma[0]:
            raise ValueError("Bond sigma range is invalid.")
        if out.max_defects != 2:
            raise ValueError("The core study requires max_defects=2.")
        return out


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    patience: int
    targets_per_case: int
    seed: int
    modes1: int
    modes2: int
    width: int
    layers: int
    cond_hidden: int
    temporal_hidden: int
    forcing_embed_dim: int
    temporal_samples: int
    dropout: float
    bond_loss_weight: float
    interface_loss_weight: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainingConfig":
        defaults = {
            "batch_size": 32,
            "epochs": 40,
            "learning_rate": 2e-3,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
            "patience": 10,
            "targets_per_case": 4,
            "seed": 42,
            "modes1": 6,
            "modes2": 6,
            "width": 32,
            "layers": 4,
            "cond_hidden": 128,
            "temporal_hidden": 64,
            "forcing_embed_dim": 16,
            "temporal_samples": 128,
            "dropout": 0.0,
            "bond_loss_weight": 0.0,
            "interface_loss_weight": 0.0,
        }
        defaults.update(value)
        out = cls(**{
            name: (int(defaults[name]) if name in {
                "batch_size", "epochs", "patience", "targets_per_case", "seed",
                "modes1", "modes2", "width", "layers", "cond_hidden",
                "temporal_hidden", "forcing_embed_dim",
                "temporal_samples",
            } else float(defaults[name]))
            for name in cls.__dataclass_fields__
        })
        if min(
            out.batch_size,
            out.epochs,
            out.targets_per_case,
            out.width,
            out.layers,
            out.temporal_samples,
        ) < 1:
            raise ValueError("Training counts must be positive.")
        return out


@dataclass(frozen=True)
class StudyConfig:
    study_id: str
    authoritative: bool
    seed: int
    initial_temperature: float
    lateral_length: float
    bond_thickness: float
    backing_thickness: float
    thickness_candidates: tuple[float, ...]
    bond_temperature_limit: float
    structural_temperature_limit: float
    validity: ValidityConfig
    surface: SurfaceConfig
    recent_window_fraction: float
    cases_per_stratum: int
    max_stage_a_cases: int
    tps: Material
    backing: Material
    bond: BondConfig
    heating: HeatingBounds
    mesh: MeshConfig
    time: TimeConfig
    training: TrainingConfig
    solver: SolverConfig
    source_path: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any], source_path: str | None = None) -> "StudyConfig":
        required = (
            "study_id", "authoritative", "seed", "initial_temperature",
            "lateral_length", "bond_thickness", "backing_thickness",
            "thickness_candidates", "limits", "materials", "bond", "heating",
            "mesh", "time", "training", "validity",
        )
        missing = [key for key in required if key not in value or value[key] is None]
        if missing:
            raise ValueError(f"Study configuration is missing required values: {missing}")
        out = cls(
            study_id=str(value["study_id"]),
            authoritative=bool(value["authoritative"]),
            seed=int(value["seed"]),
            initial_temperature=float(value["initial_temperature"]),
            lateral_length=float(value["lateral_length"]),
            bond_thickness=float(value["bond_thickness"]),
            backing_thickness=float(value["backing_thickness"]),
            thickness_candidates=tuple(sorted(float(v) for v in value["thickness_candidates"])),
            bond_temperature_limit=float(value["limits"]["bond_temperature"]),
            structural_temperature_limit=float(value["limits"]["structural_interface_temperature"]),
            validity=ValidityConfig.from_dict(value["validity"]),
            surface=SurfaceConfig.from_dict(value.get("surface")),
            recent_window_fraction=float(value.get("recent_window_fraction", 0.10)),
            cases_per_stratum=int(value.get("cases_per_stratum", 12)),
            max_stage_a_cases=int(value.get("max_stage_a_cases", 1500)),
            tps=Material.from_dict(value["materials"]["tps"]),
            backing=Material.from_dict(value["materials"]["backing"]),
            bond=BondConfig.from_dict(value["bond"]),
            heating=HeatingBounds.from_dict(value["heating"]),
            mesh=MeshConfig.from_dict(value["mesh"]),
            time=TimeConfig.from_dict(value["time"]),
            training=TrainingConfig.from_dict(value["training"]),
            solver=SolverConfig.from_dict(value.get("solver")),
            source_path=source_path,
        )
        out.validate()
        return out

    def validate(self) -> None:
        if not self.study_id.strip():
            raise ValueError("study_id must be non-empty.")
        if min(self.lateral_length, self.bond_thickness, self.backing_thickness) <= 0.0:
            raise ValueError("All fixed geometry lengths must be positive.")
        if not self.thickness_candidates or self.thickness_candidates[0] <= 0.0:
            raise ValueError("At least one positive TPS thickness is required.")
        if len(set(self.thickness_candidates)) != len(self.thickness_candidates):
            raise ValueError("TPS thickness candidates must be unique.")
        if self.recent_window_fraction != 0.10:
            raise ValueError("The baseline requires recent_window_fraction=0.10.")
        if (
            self.validity.property_tables.minimum_temperature
            > self.initial_temperature
        ):
            raise ValueError(
                "The initial temperature is below the declared TPS property range."
            )
        if (
            self.validity.hot_face.configured_temperature_limit
            <= self.initial_temperature
        ):
            raise ValueError(
                "The configured hot-face study/material limit must exceed the "
                "initial temperature."
            )
        if (
            self.authoritative
            and self.validity.tps_property_model == "constant_effective"
            and self.validity.max_property_relative_deviation > 0.20
        ):
            raise ValueError(
                "Authoritative constant-effective studies require a maximum "
                "property deviation no greater than 20%."
            )
        if (
            self.authoritative
            and not self.validity.reject_cases_outside_validity
        ):
            raise ValueError(
                "Authoritative studies must reject cases outside the validity domain."
            )
        if (
            self.authoritative
            and self.surface.reradiation_enabled
            and not self.surface.emissivity_authoritative
        ):
            raise ValueError(
                "Authoritative incident-heating studies require an authoritative "
                "emissivity definition."
            )
        if self.authoritative and self.surface.reradiation_enabled:
            if self.surface.emissivity_quantity != "total_hemispherical":
                raise ValueError(
                    "Authoritative reradiation requires total hemispherical "
                    "emissivity."
                )
            if (
                self.surface.coating is None
                or not self.surface.coating.strip()
                or self.surface.condition is None
                or not self.surface.condition.strip()
            ):
                raise ValueError(
                    "Authoritative reradiation requires coating and surface "
                    "condition identifiers."
                )
        if self.cases_per_stratum < 1 or self.max_stage_a_cases < 9 * len(self.thickness_candidates):
            raise ValueError("Stage-A case budget cannot cover every (thickness, M, J) stratum.")
        if self.heating.y_center[0] < 0.0 or self.heating.y_center[1] > self.lateral_length:
            raise ValueError("Heating y-center bounds must lie within the lateral domain.")
        if self.bond.y_center[0] < 0.0 or self.bond.y_center[1] > self.lateral_length:
            raise ValueError("Bond-defect y-center bounds must lie within the lateral domain.")
        if self.heating.t_center[0] < 0.0 or self.heating.t_center[1] > self.time.t_final:
            raise ValueError("Heating time-center bounds must lie within the simulation horizon.")
        expected_interpretation = (
            "incident_aerothermal_heat_flux"
            if self.surface.reradiation_enabled
            else "effective_net_inward_conductive_heat_flux"
        )
        if self.heating.interpretation != expected_interpretation:
            raise ValueError(
                f"surface.model={self.surface.model!r} requires "
                f"heating.interpretation={expected_interpretation!r}."
            )
        if (
            self.surface.reradiation_enabled
            and self.surface.radiation_sink_temperature
            > self.hot_face_temperature_limit
        ):
            raise ValueError(
                "The radiation sink temperature cannot exceed the declared "
                "resolved hot-face validity limit."
            )
        if self.surface.reradiation_enabled:
            if (
                self.surface.emissivity_temperature_min is not None
                and self.surface.emissivity_temperature_min
                > self.initial_temperature
            ):
                raise ValueError(
                    "The emissivity model must cover initial_temperature."
                )
            if (
                self.surface.emissivity_temperature_max is not None
                and self.surface.emissivity_temperature_max
                < self.validity.hot_face.study_temperature_limit
            ):
                raise ValueError(
                    "The emissivity model must cover the configured hot-face "
                    "study temperature limit."
                )
        if self.validity.tps_property_model != self.tps.model:
            raise ValueError(
                "validity.tps_property_model must match materials.tps.model."
            )
        if (
            self.tps.model == "temperature_dependent_table"
            and self.surface.radiation_coupling == "boundary_response"
        ):
            raise ValueError(
                "temperature_dependent_table is incompatible with "
                "surface.radiation_coupling='boundary_response'; use "
                "'coupled_nonlinear'."
            )
        self._validate_property_table(self.tps, "tps")
        self._validate_property_table(self.backing, "backing")

    def resolve_property_table_path(self, path: str | Path) -> Path:
        raw = Path(path)
        if raw.is_absolute():
            return raw.resolve()
        if self.source_path is not None:
            adjacent = Path(self.source_path).resolve().parent / raw
            if adjacent.exists():
                return adjacent.resolve()
        if raw.exists():
            return raw.resolve()
        if self.source_path is not None:
            return (Path(self.source_path).resolve().parent / raw).resolve()
        return raw.resolve()

    def _validate_property_table(self, material: Material, region: str) -> None:
        if material.model != "temperature_dependent_table":
            return
        from fno_tps.materials import TabulatedPropertyModel, load_property_table

        assert material.property_table is not None
        table_config = material.property_table
        table = load_property_table(
            self.resolve_property_table_path(table_config.path)
        )
        if table.version != table_config.version:
            raise ValueError(
                f"{region} property-table version {table.version!r} does not "
                f"match configured version {table_config.version!r}."
            )
        if (
            table_config.content_sha256 is not None
            and table.content_sha256 != table_config.content_sha256
        ):
            raise ValueError(
                f"{region} property-table content_sha256 does not match the file."
            )
        reference_temperature = (
            table.temperature_min
            if table_config.reference_temperature is None
            else table_config.reference_temperature
        )
        if not (
            table.temperature_min
            <= reference_temperature
            <= table.temperature_max
        ):
            raise ValueError(
                f"{region} property-table reference_temperature must lie in "
                "the table range."
            )
        model = TabulatedPropertyModel(
            table,
            reference_temperature=reference_temperature,
            extrapolation=table_config.extrapolation,
        )
        expected_reference_k_x = float(
            model.conductivity(reference_temperature, direction="x")
        )
        expected_reference_k_y = float(
            model.conductivity(reference_temperature, direction="y")
        )
        expected_reference_rho_c = float(
            model.volumetric_heat_capacity(reference_temperature)
        )
        for label, configured, expected in (
            ("reference_k", material.reference_k, expected_reference_k_x),
            ("reference_k_y", material.reference_k_y, expected_reference_k_y),
            (
                "reference_rho_c",
                material.reference_rho_c,
                expected_reference_rho_c,
            ),
        ):
            if not math.isclose(
                float(configured),
                expected,
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"{region} {label}={configured:.12g} does not match "
                    f"the table value {expected:.12g} at "
                    f"{reference_temperature:.9g} K."
                )
        if table_config.extrapolation == "reject":
            if table.temperature_min > self.initial_temperature:
                raise ValueError(
                    f"{region} property table must cover initial_temperature."
                )
            if region == "tps":
                if (
                    table.temperature_min
                    > self.validity.property_tables.minimum_temperature
                ):
                    raise ValueError(
                        "TPS property table must cover the declared minimum "
                        "property temperature."
                    )
                if (
                    table.temperature_max
                    < self.validity.hot_face.study_temperature_limit
                ):
                    raise ValueError(
                        "TPS property table must conservatively cover "
                        "validity.hot_face.study_temperature_limit even though TPS "
                        "properties are evaluated at cell centres, not at the "
                        "reconstructed surface."
                    )
                declared_maximum = (
                    self.validity.property_tables.maximum_temperature
                )
                if (
                    declared_maximum is not None
                    and table.temperature_max < declared_maximum
                ):
                    raise ValueError(
                        "TPS property table does not cover "
                        "validity.property_tables.maximum_temperature."
                    )
            expected_extrapolation = (
                "clamp"
                if self.validity.property_tables.allow_extrapolation
                else "reject"
            )
            if table_config.extrapolation != expected_extrapolation:
                raise ValueError(
                    f"{region} property_table.extrapolation must be "
                    f"{expected_extrapolation!r} to match "
                    "validity.property_tables.allow_extrapolation."
                )
        if region == "tps":
            pressure_tolerance = max(
                1.0,
                1.0e-6 * self.validity.conductivity_reference_pressure,
            )
            if (
                abs(
                    table.pressure_Pa
                    - self.validity.conductivity_reference_pressure
                )
                > pressure_tolerance
            ):
                raise ValueError(
                    "TPS property-table pressure_Pa does not match "
                    "validity.conductivity_reference_pressure within tolerance."
                )
        if self.authoritative and not table.authoritative:
            raise ValueError(
                f"Authoritative study cannot use non-authoritative {region} "
                "property data."
            )

    @property
    def recent_window(self) -> float:
        return self.recent_window_fraction * self.time.t_final

    @property
    def q_ref(self) -> float:
        return self.heating.max_events * self.heating.amplitude[1]

    @property
    def max_total_thickness(self) -> float:
        return max(self.thickness_candidates) + self.bond_thickness + self.backing_thickness

    @property
    def k_ref(self) -> float:
        return max(
            float(self.tps.reference_k),
            float(self.tps.reference_k_y),
            float(self.backing.reference_k),
            float(self.backing.reference_k_y),
            self.bond.k0,
        )

    @property
    def rho_c_ref(self) -> float:
        return max(
            float(self.tps.reference_rho_c),
            float(self.backing.reference_rho_c),
            self.bond.rho_c,
        )

    @property
    def hot_face_temperature_limit(self) -> float:
        """Resolve the study, material, emissivity, and TPS-data ceilings."""
        limits = [
            self.validity.hot_face.study_temperature_limit,
        ]
        material_limit = (
            self.validity.hot_face.material_multiple_use_limit
        )
        if material_limit is not None:
            limits.append(material_limit)
        if self.surface.emissivity_temperature_max is not None:
            limits.append(self.surface.emissivity_temperature_max)
        if (
            self.tps.model == "temperature_dependent_table"
            and self.tps.property_table is not None
        ):
            from fno_tps.materials import load_property_table

            table = load_property_table(
                self.resolve_property_table_path(self.tps.property_table.path)
            )
            limits.append(table.temperature_max)
        declared_maximum = self.validity.property_tables.maximum_temperature
        if declared_maximum is not None:
            limits.append(declared_maximum)
        return float(min(limits))

    @property
    def hot_face_limit_hierarchy(self) -> dict[str, float | None]:
        table_maximum: float | None = None
        if (
            self.tps.model == "temperature_dependent_table"
            and self.tps.property_table is not None
        ):
            from fno_tps.materials import load_property_table

            table_maximum = load_property_table(
                self.resolve_property_table_path(self.tps.property_table.path)
            ).temperature_max
        return {
            "study_temperature_limit_K": (
                self.validity.hot_face.study_temperature_limit
            ),
            "material_multiple_use_limit_K": (
                self.validity.hot_face.material_multiple_use_limit
            ),
            "emissivity_temperature_limit_K": (
                self.surface.emissivity_temperature_max
            ),
            "property_table_temperature_limit_K": table_maximum,
            "resolved_hot_face_temperature_limit_K": (
                self.hot_face_temperature_limit
            ),
        }

    @property
    def surface_payload(self) -> dict[str, Any]:
        return asdict(self.surface)

    @property
    def property_provenance(self) -> dict[str, Any]:
        from fno_tps.materials import build_region_properties

        return build_region_properties(self).provenance()

    def structural_interface_regions_are_constant_property(self) -> bool:
        return (
            self.backing.model == "constant_effective"
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "study_id": self.study_id,
            "authoritative": self.authoritative,
            "seed": self.seed,
            "initial_temperature": self.initial_temperature,
            "lateral_length": self.lateral_length,
            "bond_thickness": self.bond_thickness,
            "backing_thickness": self.backing_thickness,
            "thickness_candidates": self.thickness_candidates,
            "limits": {
                "bond_temperature": self.bond_temperature_limit,
                "structural_interface_temperature": (
                    self.structural_temperature_limit
                ),
            },
            "validity": self.validity.as_dict(),
            "recent_window_fraction": self.recent_window_fraction,
            "cases_per_stratum": self.cases_per_stratum,
            "max_stage_a_cases": self.max_stage_a_cases,
            "materials": {
                "tps": self.tps.as_dict(),
                "backing": self.backing.as_dict(),
            },
            "bond": asdict(self.bond),
            "heating": asdict(self.heating),
            "mesh": asdict(self.mesh),
            "time": asdict(self.time),
            "training": asdict(self.training),
        }
        # Preserve the pre-reradiation hash and checkpoint compatibility for the
        # unchanged prescribed-net-flux model. Non-default surface physics is
        # always explicit in the resolved configuration and provenance.
        if self.surface != SurfaceConfig.from_dict(None):
            payload["surface"] = self.surface.as_dict(
                preserve_legacy_shape=True
            )
        if self.solver != SolverConfig.from_dict(None):
            payload["solver"] = asdict(self.solver)
        return payload

    @property
    def sha256(self) -> str:
        return sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def require_authoritative(self, allow_demo: bool = False) -> None:
        if not self.authoritative and not allow_demo:
            raise ValueError(
                f"Study {self.study_id!r} is non-production. Pass allow_demo=True "
                "only for an explicitly illustrative run."
            )


def _pair(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        if set(value) != {"min", "max"}:
            raise ValueError(
                f"Range mappings require exactly min and max keys, got {value!r}."
            )
        value = (value["min"], value["max"])
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"Expected a two-element range or {{min, max}} mapping, got {value!r}."
        )
    return float(value[0]), float(value[1])


def _apply_override(payload: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Invalid override {expression!r}; expected dotted.path=value.")
    dotted, raw_value = expression.split("=", 1)
    parts = dotted.split(".")
    cursor: Any = payload
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(f"Unknown configuration path {dotted!r}.")
        cursor = cursor[part]
    if not isinstance(cursor, dict) or parts[-1] not in cursor:
        raise KeyError(f"Unknown configuration path {dotted!r}.")
    cursor[parts[-1]] = yaml.safe_load(raw_value)


def load_study_config(
    path: str | Path,
    overrides: list[str] | tuple[str, ...] | None = None,
) -> StudyConfig:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a YAML mapping.")
    for expression in overrides or ():
        _apply_override(value, expression)
    return StudyConfig.from_dict(value, source_path=str(source))
