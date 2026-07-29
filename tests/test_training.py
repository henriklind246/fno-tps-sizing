from __future__ import annotations

from dataclasses import replace

import torch

from fno_tps.training import (
    _checkpoint_selection_metrics,
    _meaningfully_improved,
    _pareto_frontier,
    _thickness_loss_weighting_report,
    _thickness_loss_weights,
    _training_loss,
)


def _validation(
    *,
    field: float = 5.0,
    mean: float,
    p95: float,
    false_feasible: int = 0,
    false_infeasible: int = 0,
) -> dict[str, float | int]:
    return {
        "field_rmse_K": field,
        "critical_mean_peak_error_K": mean,
        "margin_error_p95_K": p95,
        "optimistic_margin_error_p95_K": p95,
        "false_feasible_count": false_feasible,
        "false_infeasible_count": false_infeasible,
        "feasibility_disagreement_count": (
            false_feasible + false_infeasible
        ),
    }


def test_training_candidate_selection_does_not_use_sign_counts_as_constraints():
    conservative = _checkpoint_selection_metrics(
        _validation(mean=4.0, p95=8.0),
    )
    more_accurate = _checkpoint_selection_metrics(
        _validation(mean=1.0, p95=2.0, false_feasible=1),
    )
    assert tuple(more_accurate["selection_key"]) < tuple(
        conservative["selection_key"]
    )
    assert more_accurate["false_feasible_count"] == 1


def test_training_candidate_selection_uses_absolute_tail_plus_peak_error():
    low_tail = _checkpoint_selection_metrics(
        _validation(mean=3.0, p95=4.0),
    )
    high_tail = _checkpoint_selection_metrics(
        _validation(mean=2.0, p95=9.0),
    )
    assert low_tail["checkpoint_selection_score_K"] == 7.0
    assert tuple(low_tail["selection_key"]) < tuple(high_tail["selection_key"])


def test_meaningful_improvement_ignores_numerical_noise():
    assert _meaningfully_improved(9.0, 10.0)
    assert not _meaningfully_improved(9.9995, 10.0)


def test_pareto_frontier_retains_non_dominated_candidates():
    records = [
        {
            "epoch": 0,
            "field_rmse_K": 3.0,
            "critical_mean_peak_error_K": 3.0,
            "margin_error_p95_K": 3.0,
            "optimistic_margin_error_p95_K": 3.0,
        },
        {
            "epoch": 1,
            "field_rmse_K": 2.0,
            "critical_mean_peak_error_K": 4.0,
            "margin_error_p95_K": 2.0,
            "optimistic_margin_error_p95_K": 2.0,
        },
        {
            "epoch": 2,
            "field_rmse_K": 4.0,
            "critical_mean_peak_error_K": 4.0,
            "margin_error_p95_K": 4.0,
            "optimistic_margin_error_p95_K": 4.0,
        },
    ]
    assert [row["epoch"] for row in _pareto_frontier(records)] == [0, 1]


def test_inverse_thickness_weights_are_monotone_and_mean_normalized(
    demo_config,
):
    config = replace(
        demo_config,
        training=replace(
            demo_config.training,
            thickness_loss_weight_power=0.5,
        ),
    )
    thicknesses = torch.tensor(config.thickness_candidates)
    weights = _thickness_loss_weights(thicknesses, config)
    assert torch.all(weights[:-1] > weights[1:])
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
    torch.testing.assert_close(
        weights[0] / weights[-1],
        torch.tensor(
            (config.thickness_candidates[-1]
             / config.thickness_candidates[0]) ** 0.5
        ),
    )
    report = _thickness_loss_weighting_report(config)
    assert report["enabled"]
    assert report["maximum_weight"] > 1.0
    assert report["minimum_weight"] < 1.0


def test_zero_power_disables_thickness_weighting(demo_config):
    thicknesses = torch.tensor(demo_config.thickness_candidates)
    weights = _thickness_loss_weights(thicknesses, demo_config)
    torch.testing.assert_close(weights, torch.ones_like(weights))


def test_training_loss_applies_one_weight_per_case(demo_config):
    prediction = torch.tensor([
        [[[1.0]]],
        [[[2.0]]],
    ])
    target = torch.zeros_like(prediction)
    spatial = torch.ones(2, 1, 1, 10)
    weights = torch.tensor([2.0, 0.5])
    loss = _training_loss(
        prediction,
        target,
        spatial,
        demo_config,
        weights,
    )
    # Mean of weighted per-case MSE: (2*1 + 0.5*4) / 2.
    torch.testing.assert_close(loss, torch.tensor(2.0))
