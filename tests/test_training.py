from __future__ import annotations

from fno_tps.training import (
    _checkpoint_selection_metrics,
    _meaningfully_improved,
    _pareto_frontier,
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
        "optimistic_margin_error_p95_K": p95,
        "false_feasible_count": false_feasible,
        "false_infeasible_count": false_infeasible,
        "feasibility_disagreement_count": (
            false_feasible + false_infeasible
        ),
    }


def test_checkpoint_selection_does_not_use_sign_counts_as_rank_constraints():
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


def test_checkpoint_selection_uses_mean_plus_optimistic_tail():
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
            "optimistic_margin_error_p95_K": 3.0,
        },
        {
            "epoch": 1,
            "field_rmse_K": 2.0,
            "critical_mean_peak_error_K": 4.0,
            "optimistic_margin_error_p95_K": 2.0,
        },
        {
            "epoch": 2,
            "field_rmse_K": 4.0,
            "critical_mean_peak_error_K": 4.0,
            "optimistic_margin_error_p95_K": 4.0,
        },
    ]
    assert [row["epoch"] for row in _pareto_frontier(records)] == [0, 1]
