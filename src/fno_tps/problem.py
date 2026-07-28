from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erf, sqrt
from typing import Any, Iterable

import numpy as np

from fno_tps.config import StudyConfig


@dataclass(frozen=True)
class HeatingEvent:
    amplitude: float
    y_center: float
    t_center: float
    sigma_y: float
    sigma_t: float

    def __post_init__(self) -> None:
        if self.amplitude <= 0.0 or self.sigma_y <= 0.0 or self.sigma_t <= 0.0:
            raise ValueError("Heating amplitude and widths must be positive.")

    def values(self, y: np.ndarray, t: float | np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64)
        y_term = np.exp(-((y_arr - self.y_center) / self.sigma_y) ** 2)
        t_term = np.exp(-((t_arr - self.t_center) / self.sigma_t) ** 2)
        if t_arr.ndim == 0:
            return self.amplitude * y_term * float(t_term)
        return self.amplitude * y_term[..., None] * t_term

    def integral(self, y: np.ndarray, t0: float, t1: float) -> np.ndarray:
        if t1 < t0:
            raise ValueError("Heating integral requires t1 >= t0.")
        temporal = 0.5 * sqrt(np.pi) * self.sigma_t * (
            erf((t1 - self.t_center) / self.sigma_t)
            - erf((t0 - self.t_center) / self.sigma_t)
        )
        lateral = np.exp(-((np.asarray(y) - self.y_center) / self.sigma_y) ** 2)
        return self.amplitude * lateral * temporal

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HeatingEvent":
        return cls(**{name: float(value[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class BondDefect:
    severity: float
    y_center: float
    sigma: float

    def __post_init__(self) -> None:
        if self.severity < 0.0 or self.sigma <= 0.0:
            raise ValueError("Bond severity must be nonnegative and sigma positive.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BondDefect":
        return cls(**{name: float(value[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class SimulationCase:
    case_id: str
    d_tps: float
    heating_events: tuple[HeatingEvent, ...]
    bond_defects: tuple[BondDefect, ...]

    def __post_init__(self) -> None:
        if self.d_tps <= 0.0:
            raise ValueError("TPS thickness must be positive.")
        if not 1 <= len(self.heating_events) <= 3:
            raise ValueError("Each case requires one to three heating events.")
        if len(self.bond_defects) > 2:
            raise ValueError("Each case permits zero to two bond defects.")

    @property
    def event_count(self) -> int:
        return len(self.heating_events)

    @property
    def defect_count(self) -> int:
        return len(self.bond_defects)

    def heat_flux(self, y: np.ndarray, t: float | np.ndarray) -> np.ndarray:
        result = None
        for event in self.heating_events:
            values = event.values(y, t)
            result = values if result is None else result + values
        return np.asarray(result, dtype=np.float64)

    def heat_integral(self, y: np.ndarray, t0: float, t1: float) -> np.ndarray:
        result = np.zeros_like(np.asarray(y), dtype=np.float64)
        for event in self.heating_events:
            result += event.integral(y, t0, t1)
        return result

    def heat_peak(self, y: np.ndarray, t_end: float, temporal_samples: int = 256) -> np.ndarray:
        if t_end <= 0.0:
            return self.heat_flux(y, 0.0)
        candidates = np.linspace(0.0, t_end, temporal_samples, dtype=np.float64)
        centers = [
            min(max(event.t_center, 0.0), t_end)
            for event in self.heating_events
            if 0.0 <= event.t_center <= t_end
        ]
        if centers:
            candidates = np.unique(np.concatenate([candidates, np.asarray(centers)]))
        return np.max(self.heat_flux(y, candidates), axis=-1)

    def bond_conductivity(self, y: np.ndarray, config: StudyConfig) -> np.ndarray:
        degradation = np.zeros_like(np.asarray(y), dtype=np.float64)
        for defect in self.bond_defects:
            degradation += defect.severity * np.exp(
                -((np.asarray(y) - defect.y_center) / defect.sigma) ** 2
            )
        return np.maximum(
            config.bond.k_min,
            config.bond.k0 * (1.0 - degradation),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "d_tps": self.d_tps,
            "heating_events": [asdict(event) for event in self.heating_events],
            "bond_defects": [asdict(defect) for defect in self.bond_defects],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SimulationCase":
        return cls(
            case_id=str(value["case_id"]),
            d_tps=float(value["d_tps"]),
            heating_events=tuple(HeatingEvent.from_dict(v) for v in value["heating_events"]),
            bond_defects=tuple(BondDefect.from_dict(v) for v in value.get("bond_defects", [])),
        )


class TPSProblem:
    """Single TPS adapter owning case sampling and deterministic field construction."""

    def __init__(self, config: StudyConfig):
        self.config = config

    def sample_cases(
        self,
        cases_per_stratum: int | None = None,
        max_cases: int | None = None,
    ) -> list[SimulationCase]:
        repeats = self.config.cases_per_stratum if cases_per_stratum is None else int(cases_per_stratum)
        cap = self.config.max_stage_a_cases if max_cases is None else int(max_cases)
        strata = len(self.config.thickness_candidates) * 3 * 3
        repeats = min(repeats, cap // strata)
        if repeats < 1:
            raise ValueError("Case cap cannot cover every balanced stratum.")
        rng = np.random.default_rng(self.config.seed)
        cases: list[SimulationCase] = []
        index = 0
        for d_tps in self.config.thickness_candidates:
            for event_count in (1, 2, 3):
                for defect_count in (0, 1, 2):
                    for _ in range(repeats):
                        cases.append(
                            self.sample_case(
                                rng,
                                d_tps=d_tps,
                                event_count=event_count,
                                defect_count=defect_count,
                                case_id=f"train-{index:06d}",
                            )
                        )
                        index += 1
        rng.shuffle(cases)
        return cases

    def sample_case(
        self,
        rng: np.random.Generator,
        *,
        d_tps: float,
        event_count: int,
        defect_count: int,
        case_id: str,
    ) -> SimulationCase:
        if d_tps not in self.config.thickness_candidates:
            raise ValueError("Sampled TPS thickness must be a configured candidate.")
        if event_count not in (1, 2, 3) or defect_count not in (0, 1, 2):
            raise ValueError("Sampled event/defect counts must be in the core strata.")
        return SimulationCase(
            case_id=case_id,
            d_tps=d_tps,
            heating_events=tuple(
                self._sample_event(rng) for _ in range(event_count)
            ),
            bond_defects=tuple(
                self._sample_defect(rng) for _ in range(defect_count)
            ),
        )

    def _sample_event(self, rng: np.random.Generator) -> HeatingEvent:
        bounds = self.config.heating
        return HeatingEvent(
            amplitude=float(rng.uniform(*bounds.amplitude)),
            y_center=float(rng.uniform(*bounds.y_center)),
            t_center=float(rng.uniform(*bounds.t_center)),
            sigma_y=_log_uniform(rng, *bounds.sigma_y),
            sigma_t=_log_uniform(rng, *bounds.sigma_t),
        )

    def _sample_defect(self, rng: np.random.Generator) -> BondDefect:
        bounds = self.config.bond
        return BondDefect(
            severity=float(rng.uniform(*bounds.severity)),
            y_center=float(rng.uniform(*bounds.y_center)),
            sigma=_log_uniform(rng, *bounds.sigma),
        )


def cases_from_iterable(values: Iterable[dict[str, Any]]) -> list[SimulationCase]:
    return [SimulationCase.from_dict(value) for value in values]


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))
