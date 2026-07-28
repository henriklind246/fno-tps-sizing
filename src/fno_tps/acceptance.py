from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from fno_tps.config import StudyConfig
from fno_tps.physics import Trajectory


@dataclass(frozen=True)
class TrajectoryAcceptance:
    valid: bool
    feasible: bool | None
    classification: str
    invalid_reasons: tuple[str, ...]
    design_violations: tuple[str, ...]
    relative_energy_residual: float
    resolved_hot_face_temperature_limit_K: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def relative_energy_residual(trajectory: Trajectory) -> float:
    """Return a conservative dimensionless whole-trajectory energy residual."""
    stored_change = trajectory.internal_energy - trajectory.internal_energy[0]
    expected_change = trajectory.expected_energy - trajectory.expected_energy[0]
    scale = max(
        1.0,
        float(np.max(np.abs(stored_change), initial=0.0)),
        float(np.max(np.abs(expected_change), initial=0.0)),
        float(np.max(np.abs(trajectory.boundary_input_energy), initial=0.0)),
        float(np.max(np.abs(trajectory.radiated_energy), initial=0.0)),
    )
    return float(
        np.max(np.abs(trajectory.energy_residual), initial=0.0) / scale
    )


def assess_trajectory(
    config: StudyConfig,
    trajectory: Trajectory,
    *,
    bond_max: float | None = None,
    structural_interface_max: float | None = None,
) -> TrajectoryAcceptance:
    """Apply validity gates before evaluating the two design constraints."""
    invalid_reasons: list[str] = []
    energy_residual = relative_energy_residual(trajectory)
    hot_limit = config.hot_face_temperature_limit

    if config.validity.reject_on_nonlinear_failure and not trajectory.solver_converged:
        invalid_reasons.append("nonlinear_solver_failure")
    if (
        config.validity.reject_on_property_extrapolation
        and trajectory.accepted_range_excursions > 0
    ):
        invalid_reasons.append("property_table_extrapolation")
    if (
        config.validity.reject_on_nonphysical_temperature
        and trajectory.minimum_temperature
        <= config.validity.minimum_physical_temperature
    ):
        invalid_reasons.append("nonphysical_temperature")
    if trajectory.maximum_hot_face_temperature > hot_limit + 1.0e-8:
        invalid_reasons.append("hot_face_temperature_limit")
    if (
        config.validity.reject_on_energy_residual
        and energy_residual > config.validity.max_relative_energy_residual
    ):
        invalid_reasons.append("energy_balance_residual")

    design_violations: list[str] = []
    qoi_available = (
        bond_max is not None and structural_interface_max is not None
    )
    if qoi_available:
        if float(bond_max) > config.bond_temperature_limit + 1.0e-8:
            design_violations.append("bond_temperature")
        if (
            float(structural_interface_max)
            > config.structural_temperature_limit + 1.0e-8
        ):
            design_violations.append("structural_interface_temperature")

    valid = not invalid_reasons
    feasible = None if not qoi_available else valid and not design_violations
    classification = (
        "invalid"
        if not valid
        else (
            "valid"
            if feasible is None
            else ("feasible" if feasible else "infeasible")
        )
    )
    return TrajectoryAcceptance(
        valid=valid,
        feasible=feasible,
        classification=classification,
        invalid_reasons=tuple(invalid_reasons),
        design_violations=tuple(design_violations),
        relative_energy_residual=energy_residual,
        resolved_hot_face_temperature_limit_K=hot_limit,
    )
