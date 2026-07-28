from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fno_tps.acceptance import assess_trajectory
from fno_tps.config import StudyConfig
from fno_tps.model import TRANSFER_SOURCE_REVISION
from fno_tps.physics import TPSFVSolver
from fno_tps.problem import SimulationCase, TPSProblem
from fno_tps.runtime import capture_git_state


SplitName = Literal["train", "val", "test"]
Representation = Literal["summary", "temporal_global", "temporal_local"]
DATASET_SCHEMA_VERSION = 3


@dataclass
class DatasetBundle:
    root: Path
    trajectories: np.ndarray
    times: np.ndarray
    y_centers: np.ndarray
    x_centers: np.ndarray
    dx: np.ndarray
    bond_conductivity: np.ndarray
    max_hot_face_temperature: np.ndarray
    cases: list[SimulationCase]
    splits: dict[str, np.ndarray]
    normalization: dict[str, float]
    manifest: dict[str, Any]


def stratified_simulation_split(
    cases: list[SimulationCase],
    seed: int,
) -> dict[str, np.ndarray]:
    groups: dict[tuple[float, int, int], list[int]] = {}
    for index, case in enumerate(cases):
        key = (case.d_tps, case.event_count, case.defect_count)
        groups.setdefault(key, []).append(index)
    rng = np.random.default_rng(seed)
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        indices = np.asarray(groups[key], dtype=np.int64)
        rng.shuffle(indices)
        count = len(indices)
        if count >= 3:
            n_train = max(1, int(np.floor(0.70 * count)))
            n_val = max(1, int(round(0.15 * count)))
            if n_train + n_val >= count:
                n_train = count - 2
                n_val = 1
        elif count == 2:
            n_train, n_val = 1, 0
        else:
            n_train, n_val = 1, 0
        splits["train"].extend(indices[:n_train].tolist())
        splits["val"].extend(indices[n_train:n_train + n_val].tolist())
        splits["test"].extend(indices[n_train + n_val:].tolist())
    return {
        name: np.asarray(sorted(indices), dtype=np.int64)
        for name, indices in splits.items()
    }


def compute_training_normalization(
    trajectories: np.ndarray,
    train_ids: np.ndarray,
) -> dict[str, float]:
    if len(train_ids) == 0:
        raise ValueError("At least one training simulation is required.")
    count = 0
    total = 0.0
    total_square = 0.0
    for case_id in train_ids:
        values = np.asarray(trajectories[int(case_id)], dtype=np.float64)
        count += values.size
        total += float(np.sum(values))
        total_square += float(np.sum(values * values))
    mean = total / count
    variance = max(total_square / count - mean * mean, 0.0)
    standard_deviation = max(float(np.sqrt(variance)), 1e-8)
    return {"delta_t_mean": float(mean), "delta_t_std": standard_deviation}


def generate_dataset(
    config: StudyConfig,
    output_dir: str | Path,
    *,
    cases: list[SimulationCase] | None = None,
    cases_per_stratum: int | None = None,
    max_cases: int | None = None,
) -> DatasetBundle:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    problem = TPSProblem(config)
    auto_generated = cases is None
    candidate_cases = (
        problem.sample_cases(cases_per_stratum, max_cases)
        if auto_generated
        else list(cases)
    )
    if not candidate_cases:
        raise ValueError("Dataset generation requires at least one case.")
    times = np.asarray(config.time.saved_times(), dtype=np.float64)
    first_solver = TPSFVSolver(config, candidate_cases[0])
    nx, ny = first_solver.grid.nx, first_solver.grid.ny
    trajectories = np.empty(
        (len(candidate_cases), len(times), nx, ny),
        dtype=np.float32,
    )
    x_centers = np.empty((len(candidate_cases), nx), dtype=np.float32)
    dx = np.empty_like(x_centers)
    bond_k = np.empty((len(candidate_cases), ny), dtype=np.float32)
    max_hot_face = np.empty(len(candidate_cases), dtype=np.float32)
    generated_cases: list[SimulationCase] = []
    replacement_rng = np.random.default_rng(config.seed + 7919)
    rejected_case_count = 0
    rejected_by_stratum: dict[str, int] = {}
    invalid_retained_count = 0
    retained_accepted_range_excursions = 0
    retained_iteration_range_clamps = 0
    observed_cell_temperature_min = float("inf")
    observed_cell_temperature_max = float("-inf")
    maximum_relative_energy_residual = 0.0
    trajectory_diagnostics: list[dict[str, Any]] = []

    for index, initial_case in enumerate(candidate_cases):
        case = initial_case
        attempts = 0
        while True:
            solver = TPSFVSolver(config, case)
            trajectory = solver.solve(save_times=times)
            hot_face = trajectory.maximum_hot_face_temperature
            acceptance = assess_trajectory(config, trajectory)
            valid = acceptance.valid
            if valid or not config.validity.reject_cases_outside_validity:
                invalid_retained_count += int(not valid)
                break
            rejected_case_count += 1
            stratum = (
                f"d={case.d_tps:g}|M={case.event_count}|J={case.defect_count}"
            )
            rejected_by_stratum[stratum] = (
                rejected_by_stratum.get(stratum, 0) + 1
            )
            attempts += 1
            if not auto_generated:
                raise ValueError(
                    f"Case {case.case_id!r} is invalid: "
                    f"{', '.join(acceptance.invalid_reasons)}. Hot face "
                    f"{hot_face:.6g} K; resolved limit "
                    f"{config.hot_face_temperature_limit:.6g} K; relative "
                    f"energy residual {acceptance.relative_energy_residual:.6g}."
                )
            if attempts >= 1000:
                raise RuntimeError(
                    f"Unable to sample a valid replacement for stratum {stratum} "
                    "after 1000 attempts. Revise the declared study domain."
                )
            case = problem.sample_case(
                replacement_rng,
                d_tps=initial_case.d_tps,
                event_count=initial_case.event_count,
                defect_count=initial_case.defect_count,
                case_id=f"replacement-{index:06d}-{attempts:04d}",
            )
        case = replace(case, case_id=f"train-{index:06d}")
        retained_accepted_range_excursions += (
            trajectory.accepted_range_excursions
        )
        retained_iteration_range_clamps += trajectory.iteration_range_clamps
        observed_cell_temperature_min = min(
            observed_cell_temperature_min,
            trajectory.observed_temperature_range[0],
        )
        observed_cell_temperature_max = max(
            observed_cell_temperature_max,
            trajectory.observed_temperature_range[1],
        )
        maximum_relative_energy_residual = max(
            maximum_relative_energy_residual,
            acceptance.relative_energy_residual,
        )
        trajectory_diagnostics.append({
            "case_id": case.case_id,
            "classification": acceptance.classification,
            "invalid_reasons": list(acceptance.invalid_reasons),
            "solver_converged": trajectory.solver_converged,
            "nonlinear_iterations_per_timestep": (
                trajectory.nonlinear_iteration_counts.tolist()
            ),
            "nonlinear_final_residual_norms_K": (
                trajectory.nonlinear_final_residual_norms.tolist()
            ),
            "nonlinear_damped_iterations_per_timestep": (
                trajectory.nonlinear_damped_iteration_counts.tolist()
            ),
            "nonlinear_backtracks_per_timestep": (
                trajectory.nonlinear_backtrack_counts.tolist()
            ),
            "failed_nonlinear_iterations": (
                trajectory.failed_nonlinear_iterations
            ),
            "property_query_temperature_range_K": list(
                trajectory.property_query_temperature_range
            ),
            "maximum_hot_face_temperature_K": hot_face,
            "incident_energy_J": float(trajectory.boundary_input_energy[-1]),
            "reradiated_energy_J": float(trajectory.radiated_energy[-1]),
            "stored_energy_change_J": float(
                trajectory.internal_energy[-1]
                - trajectory.internal_energy[0]
            ),
            "energy_balance_residual_J": float(
                trajectory.energy_residual[-1]
            ),
            "maximum_relative_energy_residual": (
                acceptance.relative_energy_residual
            ),
        })
        generated_cases.append(case)
        trajectories[index] = (
            trajectory.temperatures - config.initial_temperature
        ).astype(np.float32)
        x_centers[index] = solver.grid.x_centers.astype(np.float32)
        dx[index] = solver.grid.dx.astype(np.float32)
        bond_k[index] = solver.bond_conductivity.astype(np.float32)
        max_hot_face[index] = hot_face

    splits = stratified_simulation_split(generated_cases, config.seed + 17)
    normalization = compute_training_normalization(trajectories, splits["train"])
    study_payload = config.as_dict()
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "study_id": config.study_id,
        "authoritative": config.authoritative,
        "code_provenance": capture_git_state(),
        "study_config": study_payload,
        "study_config_sha256": config.sha256,
        "transfer_source_revision": TRANSFER_SOURCE_REVISION,
        "case_count": len(generated_cases),
        "shapes": {
            "trajectories": list(trajectories.shape),
            "x_centers": list(x_centers.shape),
            "dx": list(dx.shape),
            "bond_conductivity": list(bond_k.shape),
            "max_hot_face_temperature": list(max_hot_face.shape),
        },
        "dtypes": {
            "trajectories": "float32",
            "geometry": "float32",
            "solver": "float64",
        },
        "time": {
            "dt": config.time.dt,
            "save_stride": config.time.save_stride,
            "t_final": config.time.t_final,
        },
        "normalization": normalization,
        "recent_window": config.recent_window,
        "q_ref": config.q_ref,
        "heating_interpretation": config.heating.interpretation,
        "surface": config.surface_payload,
        "material_properties": config.property_provenance,
        "spatial_channels": {
            "version": 3,
            "names": [
                "logical_x",
                "normalized_y",
                "normalized_physical_x",
                "normalized_dx",
                "reference_conductivity_x",
                "reference_conductivity_y",
                "reference_volumetric_heat_capacity",
                "tps_mask",
                "bond_mask",
                "backing_mask",
            ],
            "material_law_contract": (
                "Fixed material laws for the complete study. Reference "
                "properties are evaluated at configured reference temperatures; "
                "no target-temperature properties are supplied."
            ),
        },
        "reference_property_normalization": {
            "conductivity_W_mK": config.k_ref,
            "volumetric_heat_capacity_J_m3K": config.rho_c_ref,
        },
        "bond_conductivity_model": (
            "linear_gaussian_degradation_with_positive_floor"
        ),
        "validity": {
            **config.as_dict()["validity"],
            "hot_face_limit_hierarchy": config.hot_face_limit_hierarchy,
            "hot_face_evaluation_grid": "all_internal_solver_steps",
            "rejected_case_count": rejected_case_count,
            "rejected_by_stratum": rejected_by_stratum,
            "invalid_retained_count": invalid_retained_count,
            "observed_max_hot_face_temperature": float(
                np.max(max_hot_face)
            ),
            "accepted_range_excursions": (
                retained_accepted_range_excursions
            ),
            "iteration_range_clamps_diagnostic_only": (
                retained_iteration_range_clamps
            ),
            "observed_cell_temperature_range_K": [
                observed_cell_temperature_min,
                observed_cell_temperature_max,
            ],
            "maximum_relative_energy_residual": (
                maximum_relative_energy_residual
            ),
        },
        "temporal_samples": config.training.temporal_samples,
        "split_seed": config.seed + 17,
        "split_policy": "simulation_id_stratified_by_thickness_event_count_defect_count",
        "temperature_storage": "delta_T_from_uniform_T0",
        "units": "SI",
    }

    np.save(destination / "trajectories.npy", trajectories)
    np.save(destination / "times.npy", times.astype(np.float32))
    np.save(destination / "y_centers.npy", first_solver.grid.y_centers.astype(np.float32))
    np.save(destination / "x_centers.npy", x_centers)
    np.save(destination / "dx.npy", dx)
    np.save(destination / "bond_conductivity.npy", bond_k)
    np.save(destination / "max_hot_face_temperature.npy", max_hot_face)
    np.savez(destination / "splits.npz", **splits)
    (destination / "cases.jsonl").write_text(
        "".join(json.dumps(case.as_dict(), sort_keys=True) + "\n" for case in generated_cases),
        encoding="utf-8",
    )
    (destination / "trajectory_diagnostics.jsonl").write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in trajectory_diagnostics
        ),
        encoding="utf-8",
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return load_dataset(destination)


def load_dataset(path: str | Path) -> DatasetBundle:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest["schema_version"]) != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported dataset schema {manifest['schema_version']}; "
            f"expected {DATASET_SCHEMA_VERSION}."
        )
    cases = [
        SimulationCase.from_dict(json.loads(line))
        for line in (root / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split_file = np.load(root / "splits.npz")
    max_hot_face = np.load(root / "max_hot_face_temperature.npy")
    validity = manifest["validity"]
    hierarchy = validity["hot_face"]
    hot_face_limit = min(
        value
        for value in (
            hierarchy["study_temperature_limit"],
            hierarchy.get("material_multiple_use_limit"),
            validity.get("property_tables", {}).get("maximum_temperature"),
            validity.get("hot_face_limit_hierarchy", {}).get(
                "emissivity_temperature_limit_K"
            ),
            validity.get("hot_face_limit_hierarchy", {}).get(
                "property_table_temperature_limit_K"
            ),
        )
        if value is not None
    )
    if (
        validity["reject_cases_outside_validity"]
        and np.any(max_hot_face > float(hot_face_limit) + 1e-6)
    ):
        raise ValueError(
            "Dataset contains a hot-face temperature outside its declared "
            "validity domain."
        )
    return DatasetBundle(
        root=root,
        trajectories=np.load(root / "trajectories.npy", mmap_mode="r"),
        times=np.load(root / "times.npy"),
        y_centers=np.load(root / "y_centers.npy"),
        x_centers=np.load(root / "x_centers.npy", mmap_mode="r"),
        dx=np.load(root / "dx.npy", mmap_mode="r"),
        bond_conductivity=np.load(root / "bond_conductivity.npy", mmap_mode="r"),
        max_hot_face_temperature=max_hot_face,
        cases=cases,
        splits={name: split_file[name] for name in ("train", "val", "test")},
        normalization={
            key: float(value)
            for key, value in manifest["normalization"].items()
        },
        manifest=manifest,
    )


class TPSInputBuilder:
    def __init__(
        self,
        config: StudyConfig,
        normalization: dict[str, float],
        temporal_samples: int | None = None,
    ):
        self.config = config
        self.normalization = normalization
        self.temporal_samples = (
            config.training.temporal_samples
            if temporal_samples is None
            else int(temporal_samples)
        )
        self.logical_x = (
            (np.arange(config.mesh.nx, dtype=np.float32) + 0.5) / config.mesh.nx
        )

    def spatial_base(
        self,
        case: SimulationCase,
        x_centers: np.ndarray,
        dx: np.ndarray,
        y_centers: np.ndarray,
        bond_conductivity: np.ndarray,
    ) -> np.ndarray:
        """Build static inputs.

        Channels 4–6 are reference material descriptors. Under a table model
        they are not the instantaneous conductivity or heat capacity.
        """
        nx, ny = self.config.mesh.nx, self.config.mesh.ny
        xi = np.broadcast_to(self.logical_x[:, None], (nx, ny))
        y_norm_1d = y_centers / self.config.lateral_length
        y_norm = np.broadcast_to(y_norm_1d[None, :], (nx, ny))
        x_physical = np.broadcast_to(
            (x_centers / self.config.max_total_thickness)[:, None],
            (nx, ny),
        )
        dx_field = np.broadcast_to(
            (dx / self.config.max_total_thickness)[:, None],
            (nx, ny),
        )
        reference_conductivity_x = np.empty((nx, ny), dtype=np.float32)
        reference_conductivity_y = np.empty_like(reference_conductivity_x)
        reference_rho_c = np.empty_like(reference_conductivity_x)
        tps_mask = np.zeros_like(reference_conductivity_x)
        bond_mask = np.zeros_like(reference_conductivity_x)
        backing_mask = np.zeros_like(reference_conductivity_x)
        tps_end = self.config.mesh.nx_tps
        bond_end = tps_end + self.config.mesh.nx_bond
        reference_conductivity_x[:tps_end] = self.config.tps.reference_k
        reference_conductivity_x[tps_end:bond_end] = bond_conductivity
        reference_conductivity_x[bond_end:] = self.config.backing.reference_k
        reference_conductivity_y[:tps_end] = self.config.tps.reference_k_y
        reference_conductivity_y[tps_end:bond_end] = bond_conductivity
        reference_conductivity_y[bond_end:] = (
            self.config.backing.reference_k_y
        )
        reference_rho_c[:tps_end] = self.config.tps.reference_rho_c
        reference_rho_c[tps_end:bond_end] = self.config.bond.rho_c
        reference_rho_c[bond_end:] = self.config.backing.reference_rho_c
        tps_mask[:tps_end] = 1.0
        bond_mask[tps_end:bond_end] = 1.0
        backing_mask[bond_end:] = 1.0
        return np.stack([
            xi,
            y_norm,
            x_physical,
            dx_field,
            reference_conductivity_x / self.config.k_ref,
            reference_conductivity_y / self.config.k_ref,
            reference_rho_c / self.config.rho_c_ref,
            tps_mask,
            bond_mask,
            backing_mask,
        ], axis=-1).astype(np.float32)

    def condition(self, case: SimulationCase, target_time: float) -> np.ndarray:
        lo, hi = self.config.thickness_candidates[0], self.config.thickness_candidates[-1]
        thickness = 0.0 if hi == lo else (case.d_tps - lo) / (hi - lo)
        return np.asarray(
            [thickness, target_time / self.config.time.t_final],
            dtype=np.float32,
        )

    def forcing_summaries(
        self,
        case: SimulationCase,
        y_centers: np.ndarray,
        target_time: float,
    ) -> np.ndarray:
        q_ref = self.config.q_ref
        current = case.heat_flux(y_centers, target_time) / q_ref
        if target_time > 0.0:
            cumulative = case.heat_integral(y_centers, 0.0, target_time) / (
                q_ref * target_time
            )
        else:
            cumulative = np.zeros_like(current)
        start = max(0.0, target_time - self.config.recent_window)
        recent = case.heat_integral(y_centers, start, target_time) / (
            q_ref * self.config.recent_window
        )
        peak = case.heat_peak(y_centers, target_time) / q_ref
        return np.stack([current, cumulative, recent, peak], axis=-1).astype(np.float32)

    def forcing_sequence(
        self,
        case: SimulationCase,
        y_centers: np.ndarray,
        target_time: float,
    ) -> np.ndarray:
        relative = np.linspace(0.0, 1.0, self.temporal_samples, dtype=np.float32)
        physical_times = relative.astype(np.float64) * float(target_time)
        flux = case.heat_flux(y_centers, physical_times) / self.config.q_ref
        relative_field = np.broadcast_to(relative[None, :], flux.shape)
        return np.stack([relative_field, flux], axis=-1).astype(np.float32)

    def build(
        self,
        representation: Representation,
        case: SimulationCase,
        x_centers: np.ndarray,
        dx: np.ndarray,
        y_centers: np.ndarray,
        bond_conductivity: np.ndarray,
        target_time: float,
        target_delta_temperature: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        base = self.spatial_base(
            case,
            x_centers,
            dx,
            y_centers,
            bond_conductivity,
        )
        if representation == "summary":
            summaries = self.forcing_summaries(case, y_centers, target_time)
            broadcast = np.broadcast_to(
                summaries[None, :, :],
                (self.config.mesh.nx, self.config.mesh.ny, 4),
            )
            spatial = np.concatenate([base, broadcast], axis=-1).astype(np.float32)
            forcing = np.zeros((0, 0, 0), dtype=np.float32)
        elif representation in ("temporal_global", "temporal_local"):
            spatial = base
            forcing = self.forcing_sequence(case, y_centers, target_time)
        else:
            raise ValueError(f"Unknown representation {representation!r}.")
        item = {
            "spatial": spatial,
            "cond_static": self.condition(case, target_time),
            "forcing_seq": forcing,
        }
        if target_delta_temperature is not None:
            mean = self.normalization["delta_t_mean"]
            standard_deviation = self.normalization["delta_t_std"]
            target = (target_delta_temperature - mean) / standard_deviation
            item["target"] = target[..., None].astype(np.float32)
        return item


class CausalTargetDataset(Dataset):
    def __init__(
        self,
        config: StudyConfig,
        bundle: DatasetBundle,
        split: SplitName,
        representation: Representation,
    ):
        self.config = config
        self.bundle = bundle
        self.split = split
        self.representation = representation
        self.case_ids = np.asarray(bundle.splits[split], dtype=np.int64)
        self.builder = TPSInputBuilder(config, bundle.normalization)
        self.epoch = 0
        self.samples: list[tuple[int, int]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self.split != "train":
            self.samples = [
                (int(case_id), target_index)
                for case_id in self.case_ids
                for target_index in range(len(self.bundle.times))
            ]
            return
        rng = np.random.default_rng(self.config.training.seed + self.epoch)
        nonzero = np.flatnonzero(self.bundle.times > 0.0)
        samples: list[tuple[int, int]] = []
        for case_id in self.case_ids:
            # The initial condition is an exact, physically meaningful target:
            # delta T(x, y, t=0) = 0. Include it once per case every epoch so
            # the model cannot improve the transient fit while leaving the
            # initial state unconstrained.
            samples.append((int(case_id), 0))
            case = self.bundle.cases[int(case_id)]
            event_mask = np.zeros(len(self.bundle.times), dtype=bool)
            for event in case.heating_events:
                event_mask |= np.abs(self.bundle.times - event.t_center) <= 2.0 * event.sigma_t
            event_indices = np.flatnonzero(event_mask & (self.bundle.times > 0.0))
            delayed_start = max(
                event.t_center + 3.0 * event.sigma_t
                for event in case.heating_events
            )
            delayed_indices = np.flatnonzero(self.bundle.times >= delayed_start)
            pools = [
                nonzero,
                event_indices if len(event_indices) else nonzero,
                delayed_indices if len(delayed_indices) else nonzero,
            ]
            categories = rng.choice(
                3,
                size=self.config.training.targets_per_case,
                p=(0.50, 0.25, 0.25),
            )
            for category in categories:
                samples.append((
                    int(case_id),
                    int(rng.choice(pools[int(category)])),
                ))
        rng.shuffle(samples)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        case_id, target_index = self.samples[index]
        case = self.bundle.cases[case_id]
        item = self.builder.build(
            self.representation,
            case,
            np.asarray(self.bundle.x_centers[case_id]),
            np.asarray(self.bundle.dx[case_id]),
            self.bundle.y_centers,
            np.asarray(self.bundle.bond_conductivity[case_id]),
            float(self.bundle.times[target_index]),
            np.asarray(self.bundle.trajectories[case_id, target_index]),
        )
        tensors = {key: torch.from_numpy(value) for key, value in item.items()}
        tensors["case_index"] = torch.tensor(case_id, dtype=torch.int64)
        tensors["target_index"] = torch.tensor(target_index, dtype=torch.int64)
        return tensors


def create_dataloaders(
    config: StudyConfig,
    bundle: DatasetBundle,
    representation: Representation,
    *,
    batch_size: int | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    size = config.training.batch_size if batch_size is None else int(batch_size)
    datasets = [
        CausalTargetDataset(config, bundle, split, representation)
        for split in ("train", "val", "test")
    ]
    # persistent_workers must stay off: `set_epoch` reshuffles the train
    # sampling in the parent, and only re-forked workers observe it.
    return tuple(
        DataLoader(
            dataset,
            batch_size=size,
            shuffle=(dataset.split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for dataset in datasets
    )
