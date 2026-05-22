from collections.abc import Iterable

from src.models import LocalPrefilterResult, MatchResult


def cap_prefilter_shortlist(
    results: Iterable[LocalPrefilterResult],
    max_jobs: int,
) -> list[LocalPrefilterResult]:
    """Return the top capped local-prefilter results for detailed scoring."""
    ranked = sorted(
        results,
        key=lambda result: (-result.local_score, str(result.job_link)),
    )
    return ranked[:max_jobs]


def partition_ranked_matches(
    matches: Iterable[MatchResult],
    top_n: int,
) -> tuple[list[MatchResult], list[dict[str, str | int]]]:
    """Split scored matches into detailed top-N and lightweight remaining rows."""
    ranked = sorted(
        matches,
        key=lambda result: (-result.overall_score, str(result.job_link)),
    )
    top_matches = ranked[:top_n]
    remaining = [
        {
            "job_link": str(result.job_link),
            "overall_score": result.overall_score,
        }
        for result in ranked[top_n:]
    ]
    return top_matches, remaining
