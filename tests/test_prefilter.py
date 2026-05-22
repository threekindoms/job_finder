from src.matching.prefilter import (
    aggregate_prefilter_results,
    select_prefilter_shortlist,
)
from src.models import LocalPrefilterResult


def make_result(
    job_link: str,
    score: int,
    should_advance: bool,
    reason: str,
) -> LocalPrefilterResult:
    return LocalPrefilterResult(
        job_link=job_link,
        local_score=score,
        should_advance=should_advance,
        short_reason=reason,
    )


def test_aggregate_prefilter_results_uses_two_model_agreement():
    results = aggregate_prefilter_results(
        {
            "llama": [
                make_result("https://example.com/jobs/strong", 92, True, "Strong fit"),
                make_result("https://example.com/jobs/weak", 62, False, "Unclear"),
            ],
            "mistral": [
                make_result("https://example.com/jobs/strong", 88, True, "Good fit"),
                make_result("https://example.com/jobs/weak", 71, False, "Still weak"),
            ],
        }
    )

    assert [(str(result.job_link), result.should_advance) for result in results] == [
        ("https://example.com/jobs/strong", True),
        ("https://example.com/jobs/weak", False),
    ]
    assert results[0].short_reason == "Strong fit"


def test_aggregate_prefilter_results_uses_third_model_to_break_tie():
    results = aggregate_prefilter_results(
        {
            "llama": [make_result("https://example.com/jobs/partial", 62, False, "Unclear")],
            "mistral": [
                make_result("https://example.com/jobs/partial", 71, True, "Possible fit")
            ],
            "deepseek": [
                make_result("https://example.com/jobs/partial", 68, True, "Likely fit")
            ],
        }
    )

    assert len(results) == 1
    assert results[0].should_advance is True
    assert results[0].short_reason == "Possible fit"


def test_aggregate_prefilter_results_rejects_unresolved_two_model_tie():
    try:
        aggregate_prefilter_results(
            {
                "llama": [
                    make_result("https://example.com/jobs/partial", 62, False, "Unclear")
                ],
                "mistral": [
                    make_result("https://example.com/jobs/partial", 71, True, "Possible fit")
                ],
            }
        )
    except ValueError as exc:
        assert "third model result is required" in str(exc)
    else:
        raise AssertionError("expected unresolved disagreement to fail")


def test_aggregate_prefilter_results_keeps_highest_score_for_auditability():
    results = aggregate_prefilter_results(
        {
            "llama": [make_result("https://example.com/jobs/weak", 20, False, "No evidence")],
            "deepseek": [
                make_result("https://example.com/jobs/weak", 45, False, "Missing Kubernetes")
            ],
        }
    )

    assert len(results) == 1
    assert results[0].local_score == 45
    assert results[0].should_advance is False
    assert results[0].short_reason == "Missing Kubernetes"


def test_select_prefilter_shortlist_uses_should_advance_or_threshold():
    shortlist = select_prefilter_shortlist(
        [
            make_result("https://example.com/jobs/advanced", 40, True, "Any yes vote"),
            make_result("https://example.com/jobs/high-score", 75, False, "High score"),
            make_result("https://example.com/jobs/rejected", 35, False, "No fit"),
        ],
        score_threshold=70,
    )

    assert [str(result.job_link) for result in shortlist] == [
        "https://example.com/jobs/high-score",
        "https://example.com/jobs/advanced",
    ]


def test_select_prefilter_shortlist_sorts_deterministically():
    shortlist = select_prefilter_shortlist(
        [
            make_result("https://example.com/jobs/b", 80, True, "Fit"),
            make_result("https://example.com/jobs/a", 80, True, "Fit"),
        ],
        score_threshold=70,
    )

    assert [str(result.job_link) for result in shortlist] == [
        "https://example.com/jobs/a",
        "https://example.com/jobs/b",
    ]
