from collections import defaultdict
from collections.abc import Iterable

from src.models import LocalPrefilterResult


def aggregate_prefilter_results(
    results_by_model: dict[str, Iterable[LocalPrefilterResult]],
) -> list[LocalPrefilterResult]:
    """Aggregate model votes into one consensus result per job."""
    grouped: dict[str, list[LocalPrefilterResult]] = defaultdict(list)
    for results in results_by_model.values():
        for result in results:
            grouped[str(result.job_link)].append(result)

    aggregated: list[LocalPrefilterResult] = []
    for job_link in sorted(grouped):
        job_results = grouped[job_link]
        votes = [result.should_advance for result in job_results]
        if len(votes) == 2 and votes[0] != votes[1]:
            raise ValueError(
                f"unresolved prefilter disagreement for {job_link}; "
                "a third model result is required"
            )

        should_advance = sum(votes) > len(votes) / 2
        representative_pool = [
            result for result in job_results if result.should_advance == should_advance
        ]
        representative = max(
            representative_pool,
            key=lambda result: (result.local_score, result.short_reason),
        )
        aggregated.append(
            LocalPrefilterResult(
                job_link=job_link,
                local_score=max(result.local_score for result in job_results),
                should_advance=should_advance,
                short_reason=representative.short_reason,
            )
        )

    return aggregated


def select_prefilter_shortlist(
    aggregated_results: Iterable[LocalPrefilterResult],
    score_threshold: int,
) -> list[LocalPrefilterResult]:
    """Select deterministic shortlist candidates from aggregated local results."""
    eligible = [
        result
        for result in aggregated_results
        if result.should_advance or result.local_score >= score_threshold
    ]
    return sorted(
        eligible,
        key=lambda result: (-result.local_score, str(result.job_link)),
    )
