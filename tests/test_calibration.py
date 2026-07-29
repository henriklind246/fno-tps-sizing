from __future__ import annotations

from fno_tps.calibration import (
    _apply_buffer,
    _candidate_pool,
    case_fingerprint,
    constrained_calibration_metrics,
)
from fno_tps.problem import TPSProblem


def _evaluation(
    *,
    epoch: int,
    false_feasible: int,
    margin_p95: float,
    critical: float,
    field: float = 5.0,
) -> dict[str, float | int]:
    return {
        "checkpoint_epoch": epoch,
        "false_feasible_count": false_feasible,
        "false_infeasible_count": 0,
        "margin_error_p95_K": margin_p95,
        "critical_mean_peak_error_K": critical,
        "field_rmse_K": field,
        "optimistic_margin_error_p95_K": 0.0,
    }


def test_constrained_selection_prioritizes_zero_false_feasible():
    accurate_but_unsafe = constrained_calibration_metrics(_evaluation(
        epoch=1,
        false_feasible=1,
        margin_p95=1.0,
        critical=1.0,
    ))
    conservative = constrained_calibration_metrics(_evaluation(
        epoch=2,
        false_feasible=0,
        margin_p95=10.0,
        critical=10.0,
    ))
    assert tuple(conservative["selection_key"]) < tuple(
        accurate_but_unsafe["selection_key"]
    )


def test_constrained_selection_uses_absolute_tail_then_critical_peak():
    low_tail = constrained_calibration_metrics(_evaluation(
        epoch=1,
        false_feasible=0,
        margin_p95=2.0,
        critical=9.0,
    ))
    high_tail = constrained_calibration_metrics(_evaluation(
        epoch=2,
        false_feasible=0,
        margin_p95=3.0,
        critical=1.0,
    ))
    assert tuple(low_tail["selection_key"]) < tuple(
        high_tail["selection_key"]
    )

    low_critical = constrained_calibration_metrics(_evaluation(
        epoch=3,
        false_feasible=0,
        margin_p95=2.0,
        critical=2.0,
    ))
    assert tuple(low_critical["selection_key"]) < tuple(
        low_tail["selection_key"]
    )


def test_buffer_routes_nonnegative_near_limit_cases_to_fv():
    records = [
        {
            "case_index": 0,
            "predicted_minimum_margin_K": 4.0,
            "true_feasible": False,
        },
        {
            "case_index": 1,
            "predicted_minimum_margin_K": 6.0,
            "true_feasible": True,
        },
        {
            "case_index": 2,
            "predicted_minimum_margin_K": -2.0,
            "true_feasible": False,
        },
    ]
    result = _apply_buffer(records, 4.0)
    assert result["buffered_false_feasible_count"] == 0
    assert result["buffered_false_infeasible_count"] == 0
    assert result["safety_abstention_count"] == 1
    assert [
        row["classification"] for row in result["classifications"]
    ] == [
        "near_boundary_fv_required",
        "surrogate_feasible",
        "surrogate_infeasible",
    ]


def test_candidate_pool_is_thickness_balanced_and_excludes_known_case(
    demo_config,
):
    known = TPSProblem(demo_config).sample_cases(
        cases_per_stratum=1,
    )[0]
    cases, metadata = _candidate_pool(
        demo_config,
        candidate_pool_per_thickness=5,
        seed=20260729,
        excluded_fingerprints={case_fingerprint(known)},
    )
    expected = 5 * len(demo_config.thickness_candidates)
    assert len(cases) == expected
    assert len(metadata) == expected
    for d_tps in demo_config.thickness_candidates:
        assert sum(case.d_tps == d_tps for case in cases) == 5
