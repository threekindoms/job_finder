from collections.abc import Callable, Iterable

from src.config import Settings
from src.models import CandidateProfile, JobPosting, MatchResult


def resolve_shortlist_scoring_target(settings: Settings) -> tuple[str, str]:
    """Return the configured provider/model pair for detailed scoring."""
    provider = settings.shortlist_scoring_provider
    model = settings.shortlist_scoring_model
    if not model:
        raise ValueError("shortlist scoring model is not configured")

    return provider, model


def score_shortlisted_jobs(
    candidate_profile: CandidateProfile,
    jobs: Iterable[JobPosting],
    scorer: Callable[[CandidateProfile, JobPosting], MatchResult | dict],
) -> list[MatchResult]:
    """Run detailed scoring over shortlisted jobs and validate results."""
    results = []
    for job in jobs:
        try:
            results.append(MatchResult.model_validate(scorer(candidate_profile, job)))
        except Exception as exc:
            company_part = f" @ {job.company}" if job.company else ""
            print(f"  SCORE_FAILED [{job.title}{company_part}]: {exc}")
    return sorted(
        results,
        key=lambda result: (-result.overall_score, str(result.job_link)),
    )
