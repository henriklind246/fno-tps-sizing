from __future__ import annotations

from fno_tps.training import _checkpoint_selection_metrics


def _validation(
    *,
    mean: float,
    p95: float,
    false_feasible: int = 0,
    false_infeasible: int = 0,
) -> dict[str, float | int]:
    return {
        "critical_mean_peak_error_K": mean,
        "margin_error_p95_K": p95,
        "false_feasible_count": false_feasible,
        "false_infeasible_count": false_infeasible,
        "feasibility_disagreement_count": (
            false_feasible + false_infeasible
        ),
    }


def test_checkpoint_selection_prioritizes_safety_constraints():
    safe = _checkpoint_selection_metrics(
        _validation(mean=4.0, p95=8.0),
    )
    unsafe = _checkpoint_selection_metrics(
        _validation(mean=1.0, p95=2.0, false_feasible=1),
    )
    assert tuple(safe["selection_key"]) < tuple(unsafe["selection_key"])


def test_checkpoint_selection_uses_mean_plus_tail_after_safety():
    low_tail = _checkpoint_selection_metrics(
        _validation(mean=3.0, p95=4.0),
    )
    high_tail = _checkpoint_selection_metrics(
        _validation(mean=2.0, p95=9.0),
    )
    assert low_tail["checkpoint_selection_score_K"] == 7.0
    assert tuple(low_tail["selection_key"]) < tuple(high_tail["selection_key"])
