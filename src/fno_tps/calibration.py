from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np
import torch

from fno_tps.acceptance import assess_trajectory
from fno_tps.config import StudyConfig, inference_compatibility_sha256
from fno_tps.data import TPSInputBuilder, generate_dataset, load_dataset
from fno_tps.evaluation import evaluate_checkpoint
from fno_tps.physics import TPSFVSolver
from fno_tps.problem import BondDefect, HeatingEvent, SimulationCase, TPSProblem
from fno_tps.runtime import load_model_checkpoint, resolve_device


DEFAULT_CALIBRATION_SEED = 20_260_729
CALIBRATION_ROLE_ORDER = (
    "bond_boundary",
    "structural_boundary",
    "feasible_boundary",
    "infeasible_boundary",
    "nearest_boundary",
)


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_fingerprint(case: SimulationCase) -> str:
    payload = case.as_dict()
    payload.pop("case_id", None)
    return _json_sha256(payload)


def _excluded_case_fingerprints(
    data_dirs: Iterable[str | Path],
) -> tuple[set[str], list[str]]:
    fingerprints: set[str] = set()
    resolved_paths: list[str] = []
    for data_dir in data_dirs:
        root = Path(data_dir).resolve()
        bundle = load_dataset(root)
        resolved_paths.append(str(root))
        fingerprints.update(case_fingerprint(case) for case in bundle.cases)
    return fingerprints, resolved_paths


def _enriched_candidate(
    problem: TPSProblem,
    rng: np.random.Generator,
    *,
    d_tps: float,
    index: int,
    case_id: str,
) -> tuple[SimulationCase, str]:
    event_count = 1 + index % 3
    defect_count = (index // 3) % 3
    case = problem.sample_case(
        rng,
        d_tps=d_tps,
        event_count=event_count,
        defect_count=defect_count,
        case_id=case_id,
    )
    family = ("random", "aligned_sweep", "aligned_high")[index % 3]
    if family == "random":
        return case, family

    heating = problem.config.heating
    bond = problem.config.bond
    fraction = (
        0.82 + 0.18 * rng.random()
        if family == "aligned_high"
        else (0.05 + 0.90 * ((index // 3) % 12) / 11.0)
    )
    anchor_y = float(rng.uniform(*heating.y_center))
    anchor_t = float(rng.uniform(*heating.t_center))
    sigma_y = float(np.exp(rng.uniform(
        np.log(heating.sigma_y[0]),
        np.log(heating.sigma_y[1]),
    )))
    sigma_t = float(np.exp(rng.uniform(
        np.log(heating.sigma_t[0]),
        np.log(heating.sigma_t[1]),
    )))
    amplitude = float(
        heating.amplitude[0]
        + fraction * (heating.amplitude[1] - heating.amplitude[0])
    )
    events = tuple(
        HeatingEvent(
            amplitude=float(np.clip(
                amplitude * rng.uniform(0.94, 1.06),
                *heating.amplitude,
            )),
            y_center=float(np.clip(
                anchor_y + rng.normal(0.0, 0.05 * sigma_y),
                *heating.y_center,
            )),
            t_center=float(np.clip(
                anchor_t + rng.normal(0.0, 0.05 * sigma_t),
                *heating.t_center,
            )),
            sigma_y=sigma_y,
            sigma_t=sigma_t,
        )
        for _ in range(event_count)
    )
    defects = tuple(
        BondDefect(
            severity=float(rng.uniform(
                max(bond.severity[0], 0.75 * bond.severity[1]),
                bond.severity[1],
            )),
            y_center=float(np.clip(
                anchor_y + rng.normal(0.0, 0.05 * sigma_y),
                *bond.y_center,
            )),
            sigma=float(np.exp(rng.uniform(
                np.log(bond.sigma[0]),
                np.log(bond.sigma[1]),
            ))),
        )
        for _ in range(defect_count)
    )
    return replace(
        case,
        heating_events=events,
        bond_defects=defects,
    ), family


def _candidate_pool(
    config: StudyConfig,
    *,
    candidate_pool_per_thickness: int,
    seed: int,
    excluded_fingerprints: set[str],
) -> tuple[list[SimulationCase], list[dict[str, Any]]]:
    if candidate_pool_per_thickness < len(CALIBRATION_ROLE_ORDER):
        raise ValueError(
            "candidate_pool_per_thickness must cover every calibration role."
        )
    problem = TPSProblem(config)
    rng = np.random.default_rng(seed)
    cases: list[SimulationCase] = []
    metadata: list[dict[str, Any]] = []
    seen = set(excluded_fingerprints)
    for thickness_index, d_tps in enumerate(config.thickness_candidates):
        accepted = 0
        attempts = 0
        while accepted < candidate_pool_per_thickness:
            if attempts >= 100 * candidate_pool_per_thickness:
                raise RuntimeError(
                    f"Unable to build a unique calibration pool for {d_tps:g} m."
                )
            candidate, family = _enriched_candidate(
                problem,
                rng,
                d_tps=d_tps,
                index=attempts,
                case_id=(
                    f"calibration-candidate-{thickness_index:02d}-"
                    f"{attempts:05d}"
                ),
            )
            attempts += 1
            fingerprint = case_fingerprint(candidate)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            cases.append(candidate)
            metadata.append({
                "candidate_index": len(cases) - 1,
                "case_id": candidate.case_id,
                "case_fingerprint_sha256": fingerprint,
                "d_tps": candidate.d_tps,
                "event_count": candidate.event_count,
                "defect_count": candidate.defect_count,
                "sampling_family": family,
            })
            accepted += 1
    return cases, metadata


@torch.no_grad()
def _screen_candidates(
    config: StudyConfig,
    cases: list[SimulationCase],
    metadata: list[dict[str, Any]],
    checkpoint_path: str | Path,
    *,
    device: str,
    batch_size: int,
    screening_time_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if screening_time_count < 3:
        raise ValueError("screening_time_count must be at least 3.")
    resolved_device = resolve_device(device)
    model, checkpoint = load_model_checkpoint(checkpoint_path, resolved_device)
    if (
        inference_compatibility_sha256(checkpoint["study"])
        != config.inference_compatibility_sha256
    ):
        raise ValueError(
            "Screening checkpoint inference configuration does not match."
        )
    normalization = {
        key: float(value)
        for key, value in checkpoint["normalization"].items()
    }
    builder = TPSInputBuilder(config, normalization)
    times = np.linspace(
        0.0,
        config.time.t_final,
        screening_time_count,
        dtype=np.float64,
    )
    solvers = [TPSFVSolver(config, case) for case in cases]
    prediction = np.empty(
        (
            len(cases),
            len(times),
            config.mesh.nx,
            config.mesh.ny,
        ),
        dtype=np.float32,
    )
    flat_count = len(cases) * len(times)
    model.eval()
    for start in range(0, flat_count, batch_size):
        descriptors = []
        for flat_index in range(start, min(start + batch_size, flat_count)):
            case_index, time_index = divmod(flat_index, len(times))
            case = cases[case_index]
            solver = solvers[case_index]
            descriptors.append(builder.build(
                model.config.representation,
                case,
                solver.grid.x_centers,
                solver.grid.dx,
                solver.grid.y_centers,
                solver.bond_conductivity,
                float(times[time_index]),
            ))
        spatial = torch.from_numpy(np.stack([
            item["spatial"] for item in descriptors
        ])).to(resolved_device)
        conditioning = torch.from_numpy(np.stack([
            item["cond_static"] for item in descriptors
        ])).to(resolved_device)
        forcing_array = np.stack([
            item["forcing_seq"] for item in descriptors
        ])
        forcing = (
            None
            if forcing_array.size == 0
            else torch.from_numpy(forcing_array).to(resolved_device)
        )
        output = model(spatial, conditioning, forcing)
        physical = (
            output.detach().cpu().numpy()[..., 0]
            * normalization["delta_t_std"]
            + normalization["delta_t_mean"]
        )
        for offset, flat_index in enumerate(
            range(start, min(start + batch_size, flat_count))
        ):
            case_index, time_index = divmod(flat_index, len(times))
            prediction[case_index, time_index] = physical[offset]

    screened: list[dict[str, Any]] = []
    for index, (case, solver, row) in enumerate(
        zip(cases, solvers, metadata)
    ):
        absolute_temperature = (
            config.initial_temperature + prediction[index]
        )
        qoi = solver.quantities_of_interest(absolute_temperature, times)
        bond_margin = (
            config.bond_temperature_limit - qoi["bond_max"]
        )
        structural_margin = (
            config.structural_temperature_limit
            - qoi["structural_interface_max"]
        )
        screened.append({
            **row,
            "predicted_bond_margin_K": float(bond_margin),
            "predicted_structural_margin_K": float(structural_margin),
            "predicted_minimum_margin_K": float(min(
                bond_margin,
                structural_margin,
            )),
            "predicted_governing_limit": (
                "bond" if bond_margin <= structural_margin else "structural"
            ),
            "predicted_hot_face_max_K": float(
                np.max(absolute_temperature[:, 0, :])
            ),
        })
    return screened, {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "representation": model.config.representation,
        "device": str(resolved_device),
        "screening_time_count": len(times),
        "screening_times_seconds": times.tolist(),
    }


def _role_score(row: dict[str, Any], role: str) -> float:
    bond_margin = float(row["predicted_bond_margin_K"])
    structural_margin = float(row["predicted_structural_margin_K"])
    minimum_margin = min(bond_margin, structural_margin)
    if role == "bond_boundary":
        return abs(bond_margin)
    if role == "structural_boundary":
        return abs(structural_margin)
    if role == "feasible_boundary":
        return abs(minimum_margin - 5.0)
    if role == "infeasible_boundary":
        return abs(minimum_margin + 5.0)
    return abs(minimum_margin)


def _select_screened_cases(
    config: StudyConfig,
    cases: list[SimulationCase],
    screened: list[dict[str, Any]],
    *,
    cases_per_thickness: int,
    hot_face_guard_K: float,
) -> tuple[list[list[SimulationCase]], list[dict[str, Any]]]:
    if cases_per_thickness < 1:
        raise ValueError("cases_per_thickness must be positive.")
    if hot_face_guard_K < 0.0:
        raise ValueError("hot_face_guard_K must be nonnegative.")
    case_alternatives: list[list[SimulationCase]] = []
    selected_rows: list[dict[str, Any]] = []
    for d_tps in config.thickness_candidates:
        thickness_rows = [
            row for row in screened
            if np.isclose(float(row["d_tps"]), d_tps)
        ]
        guarded = [
            row for row in thickness_rows
            if float(row["predicted_hot_face_max_K"])
            <= config.hot_face_temperature_limit - hot_face_guard_K
            and row["sampling_family"] != "aligned_high"
        ]
        eligible = (
            guarded
            if len(guarded) >= cases_per_thickness
            else thickness_rows
        )
        closest_bond_distance = min(
            abs(float(row["predicted_bond_margin_K"]))
            for row in eligible
        )
        roles = list(CALIBRATION_ROLE_ORDER)
        if closest_bond_distance > 25.0:
            roles[0] = "nearest_boundary_extra"
        primary_used: set[int] = set()
        slots: list[tuple[str, str, dict[str, Any]]] = []
        for slot in range(cases_per_thickness):
            role = (
                roles[slot]
                if slot < len(roles)
                else f"nearest_boundary_{slot + 1}"
            )
            score_role = (
                role
                if role in CALIBRATION_ROLE_ORDER
                else "nearest_boundary"
            )
            best = min(
                (
                    row for row in eligible
                    if int(row["candidate_index"]) not in primary_used
                ),
                key=lambda row: (
                    _role_score(row, score_role),
                    row["case_fingerprint_sha256"],
                ),
            )
            candidate_index = int(best["candidate_index"])
            primary_used.add(candidate_index)
            slots.append((role, score_role, best))

        fallback_used = set(primary_used)
        for role, score_role, best in slots:
            ranked_fallbacks = sorted(
                (
                    row for row in eligible
                    if int(row["candidate_index"]) not in fallback_used
                ),
                key=lambda row: (
                    _role_score(row, score_role),
                    row["case_fingerprint_sha256"],
                ),
            )
            alternatives = [best]
            for fallback in ranked_fallbacks[:10]:
                alternatives.append(fallback)
                fallback_used.add(int(fallback["candidate_index"]))
            case_alternatives.append([
                cases[int(row["candidate_index"])]
                for row in alternatives
            ])
            selected_rows.append({
                **best,
                "calibration_role": role,
                "screening_role_score": _role_score(best, score_role),
                "screened_alternative_count": len(alternatives),
                "screening_hot_face_guard_satisfied": (
                    best in guarded
                ),
            })
    return case_alternatives, selected_rows


def _truth_coverage(
    config: StudyConfig,
    data_dir: str | Path,
    selected_rows: list[dict[str, Any]],
    *,
    boundary_band_K: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle = load_dataset(data_dir)
    records: list[dict[str, Any]] = []
    for case_index, (case, selected) in enumerate(
        zip(bundle.cases, selected_rows)
    ):
        solver = TPSFVSolver(config, case)
        absolute_temperature = (
            config.initial_temperature
            + np.asarray(bundle.trajectories[case_index])
        )
        qoi = solver.quantities_of_interest(
            absolute_temperature,
            bundle.times,
        )
        bond_margin = (
            config.bond_temperature_limit - qoi["bond_max"]
        )
        structural_margin = (
            config.structural_temperature_limit
            - qoi["structural_interface_max"]
        )
        minimum_margin = min(bond_margin, structural_margin)
        records.append({
            **selected,
            "case_index": case_index,
            "dataset_case_id": case.case_id,
            "true_bond_margin_K": float(bond_margin),
            "true_structural_margin_K": float(structural_margin),
            "true_minimum_margin_K": float(minimum_margin),
            "true_governing_limit": (
                "bond" if bond_margin <= structural_margin else "structural"
            ),
            "true_feasible": bool(minimum_margin >= 0.0),
            "true_distance_to_limit_K": float(abs(minimum_margin)),
            "inside_boundary_band": bool(
                abs(minimum_margin) <= boundary_band_K
            ),
        })

    by_thickness: list[dict[str, Any]] = []
    for d_tps in config.thickness_candidates:
        subset = [
            row for row in records
            if np.isclose(float(row["d_tps"]), d_tps)
        ]
        by_thickness.append({
            "d_tps": float(d_tps),
            "case_count": len(subset),
            "near_boundary_case_count": sum(
                int(row["inside_boundary_band"]) for row in subset
            ),
            "bond_governing_case_count": sum(
                int(row["true_governing_limit"] == "bond")
                for row in subset
            ),
            "structural_governing_case_count": sum(
                int(row["true_governing_limit"] == "structural")
                for row in subset
            ),
            "minimum_true_distance_to_limit_K": min(
                float(row["true_distance_to_limit_K"])
                for row in subset
            ),
        })
    coverage = {
        "case_count": len(records),
        "boundary_band_K": float(boundary_band_K),
        "near_boundary_case_count": sum(
            int(row["inside_boundary_band"]) for row in records
        ),
        "feasible_case_count": sum(
            int(row["true_feasible"]) for row in records
        ),
        "infeasible_case_count": sum(
            int(not row["true_feasible"]) for row in records
        ),
        "bond_governing_case_count": sum(
            int(row["true_governing_limit"] == "bond")
            for row in records
        ),
        "structural_governing_case_count": sum(
            int(row["true_governing_limit"] == "structural")
            for row in records
        ),
        "thickness_strata": by_thickness,
    }
    return records, coverage


def generate_boundary_calibration_dataset(
    config: StudyConfig,
    output_dir: str | Path,
    screening_checkpoint: str | Path,
    *,
    cases_per_thickness: int = 5,
    candidate_pool_per_thickness: int = 72,
    seed: int = DEFAULT_CALIBRATION_SEED,
    excluded_data_dirs: Iterable[str | Path] = (),
    device: str = "auto",
    batch_size: int = 64,
    screening_time_count: int = 25,
    boundary_band_K: float = 10.0,
    hot_face_guard_K: float = 20.0,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    screening_path = destination / "calibration_screening_candidates.json"
    excluded_fingerprints, excluded_paths = _excluded_case_fingerprints(
        excluded_data_dirs
    )
    candidates, metadata = _candidate_pool(
        config,
        candidate_pool_per_thickness=candidate_pool_per_thickness,
        seed=seed,
        excluded_fingerprints=excluded_fingerprints,
    )
    cached_screening: dict[str, Any] | None = None
    if screening_path.is_file():
        candidate_cache = json.loads(
            screening_path.read_text(encoding="utf-8")
        )
        cached_rows = candidate_cache.get("candidates", [])
        cached_metadata = candidate_cache.get("screening", {})
        expected_fingerprints = [
            row["case_fingerprint_sha256"] for row in metadata
        ]
        cached_fingerprints = [
            row.get("case_fingerprint_sha256") for row in cached_rows
        ]
        if (
            cached_metadata.get("checkpoint_sha256")
            == _file_sha256(screening_checkpoint)
            and int(cached_metadata.get("screening_time_count", -1))
            == screening_time_count
            and cached_fingerprints == expected_fingerprints
        ):
            cached_screening = candidate_cache
    if cached_screening is None:
        screened, screening = _screen_candidates(
            config,
            candidates,
            metadata,
            screening_checkpoint,
            device=device,
            batch_size=batch_size,
            screening_time_count=screening_time_count,
        )
        screening_path.write_text(
            json.dumps({
                "screening": screening,
                "candidate_count": len(screened),
                "candidates": screened,
            }, indent=2),
            encoding="utf-8",
        )
    else:
        screened = list(cached_screening["candidates"])
        screening = dict(cached_screening["screening"])
    case_alternatives, selected_rows = _select_screened_cases(
        config,
        candidates,
        screened,
        cases_per_thickness=cases_per_thickness,
        hot_face_guard_K=hot_face_guard_K,
    )
    selected_fingerprints = {
        case_fingerprint(alternatives[0])
        for alternatives in case_alternatives
    }
    overlap = selected_fingerprints & excluded_fingerprints
    if overlap:
        raise RuntimeError(
            "Calibration cohort overlaps an excluded dataset."
        )
    bundle = generate_dataset(
        config,
        destination,
        case_alternatives=case_alternatives,
    )
    screened_by_fingerprint = {
        row["case_fingerprint_sha256"]: row for row in screened
    }
    actual_selected_rows: list[dict[str, Any]] = []
    for slot, (case, primary) in enumerate(
        zip(bundle.cases, selected_rows)
    ):
        fingerprint = case_fingerprint(case)
        actual = screened_by_fingerprint[fingerprint]
        alternatives = case_alternatives[slot]
        alternative_fingerprints = [
            case_fingerprint(candidate) for candidate in alternatives
        ]
        actual_selected_rows.append({
            **actual,
            "calibration_role": primary["calibration_role"],
            "primary_candidate_case_id": primary["case_id"],
            "selected_alternative_rank": (
                alternative_fingerprints.index(fingerprint)
            ),
            "screened_alternative_count": len(alternatives),
            "screening_hot_face_guard_satisfied": (
                float(actual["predicted_hot_face_max_K"])
                <= config.hot_face_temperature_limit - hot_face_guard_K
            ),
        })
    truth_records, coverage = _truth_coverage(
        config,
        destination,
        actual_selected_rows,
        boundary_band_K=boundary_band_K,
    )
    design = {
        "dataset_role": "dedicated_checkpoint_calibration",
        "seed": int(seed),
        "study_config_sha256": config.sha256,
        "stratification": "equal_case_count_by_tps_thickness",
        "cases_per_thickness": int(cases_per_thickness),
        "candidate_pool_per_thickness": int(
            candidate_pool_per_thickness
        ),
        "selection_method": (
            "new deterministic in-domain cases screened by the supplied "
            "surrogate, then verified by the nonlinear FV solver"
        ),
        "screening": screening,
        "screening_candidates_file": str(screening_path.resolve()),
        "screening_hot_face_guard_K": float(hot_face_guard_K),
        "excluded_dataset_paths": excluded_paths,
        "excluded_case_fingerprint_count": len(excluded_fingerprints),
        "verified_overlap_count": 0,
        "truth_coverage": coverage,
        "records": truth_records,
    }
    (destination / "calibration_design.json").write_text(
        json.dumps(design, indent=2),
        encoding="utf-8",
    )
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "dataset_role": "dedicated_checkpoint_calibration",
        "calibration_design_file": "calibration_design.json",
        "calibration_seed": int(seed),
        "calibration_screening_checkpoint_sha256": screening[
            "checkpoint_sha256"
        ],
        "calibration_excluded_dataset_paths": excluded_paths,
        "calibration_verified_overlap_count": 0,
        "calibration_truth_coverage": coverage,
    })
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "output": str(destination.resolve()),
        "case_count": len(case_alternatives),
        "screening": screening,
        "screening_candidates": str(screening_path.resolve()),
        "truth_coverage": coverage,
        "calibration_design": str(
            (destination / "calibration_design.json").resolve()
        ),
        "manifest": str(manifest_path.resolve()),
    }


def constrained_calibration_metrics(
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    false_feasible = int(evaluation["false_feasible_count"])
    absolute_tail = float(evaluation["margin_error_p95_K"])
    critical_peak = float(evaluation["critical_mean_peak_error_K"])
    field_error = float(evaluation["field_rmse_K"])
    epoch = int(evaluation["checkpoint_epoch"])
    selection_key = (
        0 if false_feasible == 0 else 1,
        false_feasible,
        absolute_tail,
        critical_peak,
        field_error,
        epoch,
    )
    return {
        "strategy": (
            "lexicographic(zero_false_feasible_first, "
            "false_feasible_count, margin_error_p95_K, "
            "critical_mean_peak_error_K, field_rmse_K, epoch)"
        ),
        "false_feasible_constraint_satisfied": false_feasible == 0,
        "false_feasible_count": false_feasible,
        "false_infeasible_count": int(
            evaluation["false_infeasible_count"]
        ),
        "margin_error_p95_K": absolute_tail,
        "critical_mean_peak_error_K": critical_peak,
        "field_rmse_K": field_error,
        "optimistic_margin_error_p95_K": float(
            evaluation["optimistic_margin_error_p95_K"]
        ),
        "selection_key": list(selection_key),
    }


def _apply_buffer(
    records: list[dict[str, Any]],
    safety_buffer_K: float,
) -> dict[str, Any]:
    false_feasible = 0
    false_infeasible = 0
    abstentions = 0
    classifications: list[dict[str, Any]] = []
    for record in records:
        predicted_margin = float(record["predicted_minimum_margin_K"])
        true_feasible = bool(record["true_feasible"])
        if predicted_margin < 0.0:
            classification = "surrogate_infeasible"
        elif predicted_margin > safety_buffer_K:
            classification = "surrogate_feasible"
        else:
            classification = "near_boundary_fv_required"
        buffered_false_feasible = (
            classification == "surrogate_feasible"
            and not true_feasible
        )
        buffered_false_infeasible = (
            classification == "surrogate_infeasible"
            and true_feasible
        )
        abstention = classification == "near_boundary_fv_required"
        false_feasible += int(buffered_false_feasible)
        false_infeasible += int(buffered_false_infeasible)
        abstentions += int(abstention)
        classifications.append({
            "case_index": int(record["case_index"]),
            "classification": classification,
            "buffered_false_feasible": buffered_false_feasible,
            "buffered_false_infeasible": buffered_false_infeasible,
            "safety_abstention": abstention,
        })
    return {
        "buffered_false_feasible_count": false_feasible,
        "buffered_false_infeasible_count": false_infeasible,
        "safety_abstention_count": abstentions,
        "classifications": classifications,
    }


def evaluate_calibration_checkpoints(
    config: StudyConfig,
    data_dir: str | Path,
    checkpoint_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    expected_epochs: Iterable[int] = (),
    device: str = "auto",
    safety_buffer_quantile: float = 100.0,
    near_boundary_limit_K: float = 10.0,
) -> dict[str, Any]:
    if not 0.0 <= safety_buffer_quantile <= 100.0:
        raise ValueError(
            "safety_buffer_quantile must lie in [0, 100]."
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    calibration_bundle = load_dataset(data_dir)
    if calibration_bundle.manifest.get("dataset_role") != (
        "dedicated_checkpoint_calibration"
    ):
        raise ValueError(
            "Checkpoint calibration requires a dataset explicitly marked "
            "dedicated_checkpoint_calibration."
        )
    unique_paths = []
    seen_paths: set[Path] = set()
    for checkpoint_path in checkpoint_paths:
        path = Path(checkpoint_path).resolve()
        if path in seen_paths:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        seen_paths.add(path)
        unique_paths.append(path)
    if not unique_paths:
        raise ValueError("At least one checkpoint is required.")

    evaluated: list[dict[str, Any]] = []
    evaluation_reports: dict[int, dict[str, Any]] = {}
    for checkpoint_path in unique_paths:
        _, checkpoint = load_model_checkpoint(checkpoint_path)
        epoch = int(checkpoint["epoch"])
        report_path = (
            destination / f"checkpoint_epoch_{epoch:04d}.json"
        )
        evaluation = evaluate_checkpoint(
            config,
            data_dir,
            checkpoint_path,
            report_path,
            split="all",
            device=device,
            near_boundary_limit_K=near_boundary_limit_K,
        )
        metrics = constrained_calibration_metrics(evaluation)
        evaluated.append({
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "checkpoint_epoch": epoch,
            "evaluation_report": str(report_path.resolve()),
            **metrics,
        })
        evaluation_reports[epoch] = evaluation

    evaluated.sort(key=lambda row: tuple(row["selection_key"]))
    selected = evaluated[0]
    selected_evaluation = evaluation_reports[
        int(selected["checkpoint_epoch"])
    ]
    optimistic_errors = [
        float(record["optimistic_margin_error_K"])
        for record in selected_evaluation["records"]
    ]
    safety_buffer_K = float(np.percentile(
        optimistic_errors,
        safety_buffer_quantile,
    ))
    buffered = _apply_buffer(
        selected_evaluation["records"],
        safety_buffer_K,
    )
    expected = sorted({int(epoch) for epoch in expected_epochs})
    evaluated_epochs = sorted(
        int(row["checkpoint_epoch"]) for row in evaluated
    )
    missing_epochs = [
        epoch for epoch in expected if epoch not in evaluated_epochs
    ]
    dataset_manifest_path = Path(data_dir) / "manifest.json"
    policy_status = (
        "frozen"
        if not missing_epochs
        else "provisional_missing_checkpoint_candidates"
    )
    policy = {
        "schema_version": 1,
        "status": policy_status,
        "checkpoint": selected["checkpoint"],
        "checkpoint_epoch": selected["checkpoint_epoch"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "calibration_dataset": str(Path(data_dir).resolve()),
        "calibration_dataset_manifest_sha256": _file_sha256(
            dataset_manifest_path
        ),
        "selection_rule": selected["strategy"],
        "selection_metrics": {
            key: selected[key]
            for key in (
                "false_feasible_constraint_satisfied",
                "false_feasible_count",
                "false_infeasible_count",
                "margin_error_p95_K",
                "critical_mean_peak_error_K",
                "field_rmse_K",
                "optimistic_margin_error_p95_K",
                "selection_key",
            )
        },
        "safety_buffer": {
            "method": "one_sided_optimistic_margin_error_quantile",
            "quantile": float(safety_buffer_quantile),
            "value_K": safety_buffer_K,
            "calibration_case_count": len(optimistic_errors),
            "zero_buffered_false_feasible_verified": (
                buffered["buffered_false_feasible_count"] == 0
            ),
        },
        "abstention_zone": {
            "predicted_minimum_margin_lower_K": 0.0,
            "predicted_minimum_margin_upper_K": safety_buffer_K,
            "lower_inclusive": True,
            "upper_inclusive": True,
            "classification": "near_boundary_fv_required",
        },
        "calibration_outcomes": {
            key: buffered[key]
            for key in (
                "buffered_false_feasible_count",
                "buffered_false_infeasible_count",
                "safety_abstention_count",
            )
        },
        "expected_checkpoint_epochs": expected,
        "evaluated_checkpoint_epochs": evaluated_epochs,
        "missing_checkpoint_epochs": missing_epochs,
        "final_untouched_test_evaluated": False,
    }
    policy_path = destination / "safety_policy.json"
    policy_path.write_text(
        json.dumps(policy, indent=2),
        encoding="utf-8",
    )
    report = {
        "status": policy_status,
        "calibration_dataset": str(Path(data_dir).resolve()),
        "dataset_role": calibration_bundle.manifest["dataset_role"],
        "selection_rule": selected["strategy"],
        "ranked_checkpoints": evaluated,
        "selected_checkpoint": selected,
        "safety_policy": str(policy_path.resolve()),
        "safety_buffer_K": safety_buffer_K,
        "buffered_calibration_outcomes": {
            key: buffered[key]
            for key in (
                "buffered_false_feasible_count",
                "buffered_false_infeasible_count",
                "safety_abstention_count",
            )
        },
        "expected_checkpoint_epochs": expected,
        "evaluated_checkpoint_epochs": evaluated_epochs,
        "missing_checkpoint_epochs": missing_epochs,
        "final_untouched_test_evaluated": False,
    }
    report_path = destination / "calibration_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return {
        **report,
        "calibration_report": str(report_path.resolve()),
    }
