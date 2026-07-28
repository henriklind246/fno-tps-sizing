from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fno_tps.data import (
    CausalTargetDataset,
    TPSInputBuilder,
    generate_dataset,
)


def test_dataset_contracts_and_simulation_splits(
    tmp_path,
    demo_config,
    three_cases,
):
    bundle = generate_dataset(demo_config, tmp_path / "data", cases=three_cases)
    assert np.max(bundle.max_hot_face_temperature) <= 1400.0
    assert set(bundle.manifest["code_provenance"]) == {
        "git_commit",
        "git_status",
    }
    assert bundle.manifest["validity"]["invalid_retained_count"] == 0
    assert bundle.manifest["spatial_channels"]["version"] == 3
    assert bundle.manifest["spatial_channels"]["names"][-3:] == [
        "tps_mask",
        "bond_mask",
        "backing_mask",
    ]
    assert (
        "target-temperature properties are supplied"
        in bundle.manifest["spatial_channels"]["material_law_contract"]
    )
    assert (tmp_path / "data" / "trajectory_diagnostics.jsonl").exists()
    split_sets = {name: set(values.tolist()) for name, values in bundle.splits.items()}
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    assert set.union(*split_sets.values()) == {0, 1, 2}
    for representation, channels in (
        ("summary", 14),
        ("temporal_global", 10),
        ("temporal_local", 10),
    ):
        train = CausalTargetDataset(demo_config, bundle, "train", representation)
        val = CausalTargetDataset(demo_config, bundle, "val", representation)
        assert all(target_index > 0 for _, target_index in train.samples)
        assert any(target_index == 0 for _, target_index in val.samples)
        item = train[0]
        assert item["spatial"].shape[-1] == channels
        if representation == "summary":
            assert item["forcing_seq"].shape == (0, 0, 0)
        else:
            assert item["forcing_seq"].shape == (
                demo_config.mesh.ny,
                128,
                2,
            )


def test_target_sampling_is_deterministic_by_epoch(
    tmp_path,
    demo_config,
    three_cases,
):
    bundle = generate_dataset(demo_config, tmp_path / "data", cases=three_cases)
    dataset = CausalTargetDataset(demo_config, bundle, "train", "summary")
    first = list(dataset.samples)
    dataset.set_epoch(0)
    assert dataset.samples == first
    dataset.set_epoch(1)
    assert dataset.samples != first


def test_cumulative_summary_uses_elapsed_window(demo_config, heating_event):
    from fno_tps.problem import SimulationCase

    case = SimulationCase("summary-window", 0.02, (heating_event,), ())
    builder = TPSInputBuilder(
        demo_config,
        {"delta_t_mean": 0.0, "delta_t_std": 1.0},
    )
    y = np.asarray([heating_event.y_center])
    target_time = 2.0
    summaries = builder.forcing_summaries(case, y, target_time)
    expected = case.heat_integral(y, 0.0, target_time) / (
        demo_config.q_ref * target_time
    )
    np.testing.assert_allclose(summaries[:, 1], expected, rtol=1e-6)
    np.testing.assert_array_equal(
        builder.forcing_summaries(case, y, 0.0)[:, 1],
        np.zeros(1),
    )


def test_causal_builder_never_queries_after_target(demo_config):
    class RecordingCase:
        d_tps = demo_config.thickness_candidates[0]
        calls: list[np.ndarray | float] = []

        def heat_flux(self, y, t):
            values = np.asarray(t)
            self.calls.append(values.copy())
            shape = (len(y),) if values.ndim == 0 else (len(y), len(values))
            return np.zeros(shape)

        def heat_integral(self, y, t0, t1):
            self.calls.append(float(t1))
            return np.zeros(len(y))

        def heat_peak(self, y, t):
            self.calls.append(float(t))
            return np.zeros(len(y))

    case = RecordingCase()
    builder = TPSInputBuilder(
        demo_config,
        {"delta_t_mean": 0.0, "delta_t_std": 1.0},
    )
    y = (np.arange(demo_config.mesh.ny) + 0.5) * (
        demo_config.lateral_length / demo_config.mesh.ny
    )
    builder.forcing_summaries(case, y, 2.0)
    builder.forcing_sequence(case, y, 2.0)
    for call in case.calls:
        assert np.max(np.asarray(call)) <= 2.0


def test_explicit_out_of_validity_case_is_rejected(
    tmp_path,
    demo_config,
):
    from fno_tps.problem import HeatingEvent, SimulationCase

    invalid = SimulationCase(
        "invalid-hot-face",
        demo_config.thickness_candidates[0],
        (
            HeatingEvent(
                amplitude=1.0e8,
                y_center=0.10,
                t_center=1.5,
                sigma_y=0.025,
                sigma_t=0.5,
            ),
        ),
        (),
    )
    with pytest.raises(ValueError, match="hot_face_temperature_limit"):
        generate_dataset(demo_config, tmp_path / "invalid", cases=[invalid])


def test_balanced_generation_resamples_invalid_cases(
    tmp_path,
    demo_config,
):
    from fno_tps.config import HeatingBounds

    config = replace(
        demo_config,
        heating=HeatingBounds(
            amplitude=(2.0e4, 2.0e5),
            y_center=(0.08, 0.12),
            t_center=(1.0, 2.0),
            sigma_y=(0.02, 0.03),
            sigma_t=(0.4, 0.6),
            max_events=3,
            interpretation="effective_net_inward_conductive_heat_flux",
        ),
    )
    bundle = generate_dataset(
        config,
        tmp_path / "resampled",
        cases_per_stratum=1,
        max_cases=9 * len(config.thickness_candidates),
    )
    assert bundle.manifest["validity"]["rejected_case_count"] > 0
    assert bundle.manifest["validity"]["invalid_retained_count"] == 0
    assert np.max(bundle.max_hot_face_temperature) <= 1400.0
    counts = {}
    for case in bundle.cases:
        key = (case.d_tps, case.event_count, case.defect_count)
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {1}
