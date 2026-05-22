import json
from pathlib import Path

import pytest

from src.config import Settings
from src.matching.ranking import cap_prefilter_shortlist, partition_ranked_matches
from src.matching.scoring import resolve_shortlist_scoring_target, score_shortlisted_jobs
from src.models import CandidateProfile, ConfidenceLevel, JobPosting, LocalPrefilterResult, MatchResult


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_json(name: str):
    return json.loads((FIXTURES_DIR / name).read_text())


def build_candidate() -> CandidateProfile:
    return CandidateProfile.model_validate(load_json("candidate_profile.json"))


def build_jobs() -> list[JobPosting]:
    return [JobPosting.model_validate(job) for job in load_json("jobs.json")[:2]]


def make_match(job: JobPosting, score: int) -> MatchResult:
    return MatchResult(
        job_link=job.link,
        overall_score=score,
        required_qualifications_score=48 if score >= 90 else 38,
        preferred_qualifications_score=16 if score >= 90 else 12,
        experience_level_score=8 if score >= 90 else 7,
        domain_fit_score=8 if score >= 90 else 7,
        location_or_availability_score=0 if score >= 90 else 1,
        strengths=["Fit"],
        gaps=[],
        other_considerations=[],
        resume_improvement_suggestions=[],
        confidence=ConfidenceLevel.HIGH,
    )


def test_resolve_shortlist_scoring_target_uses_local_model():
    provider, model = resolve_shortlist_scoring_target(
        Settings(shortlist_scoring_provider="ollama", shortlist_scoring_model="llama3.1:8b")
    )

    assert (provider, model) == ("ollama", "llama3.1:8b")


def test_resolve_shortlist_scoring_target_uses_openai_model():
    provider, model = resolve_shortlist_scoring_target(
        Settings(shortlist_scoring_provider="openai", shortlist_scoring_model="gpt-4o-mini")
    )

    assert (provider, model) == ("openai", "gpt-4o-mini")


def test_resolve_shortlist_scoring_target_rejects_missing_model():
    with pytest.raises(ValueError, match="shortlist scoring model is not configured"):
        resolve_shortlist_scoring_target(Settings(shortlist_scoring_provider="openai"))


def test_score_shortlisted_jobs_validates_and_sorts_results():
    jobs = build_jobs()
    scores = {
        str(jobs[0].link): 88,
        str(jobs[1].link): 94,
    }

    results = score_shortlisted_jobs(
        build_candidate(),
        jobs,
        scorer=lambda _candidate, job: make_match(job, scores[str(job.link)]),
    )

    assert [result.overall_score for result in results] == [80, 65]
    assert [str(result.job_link) for result in results] == [
        str(jobs[1].link),
        str(jobs[0].link),
    ]


def test_cap_prefilter_shortlist_enforces_limit_and_ordering():
    shortlist = cap_prefilter_shortlist(
        [
            LocalPrefilterResult(
                job_link="https://example.com/jobs/b",
                local_score=90,
                should_advance=True,
                short_reason="Fit",
            ),
            LocalPrefilterResult(
                job_link="https://example.com/jobs/a",
                local_score=90,
                should_advance=True,
                short_reason="Fit",
            ),
            LocalPrefilterResult(
                job_link="https://example.com/jobs/c",
                local_score=75,
                should_advance=True,
                short_reason="Fit",
            ),
        ],
        max_jobs=2,
    )

    assert [str(result.job_link) for result in shortlist] == [
        "https://example.com/jobs/a",
        "https://example.com/jobs/b",
    ]


def test_partition_ranked_matches_splits_top_n_and_remaining_rows():
    jobs = build_jobs()
    third_job = JobPosting(
        title="Platform Engineer",
        link="https://example.com/jobs/platform-engineer",
        description="Platform work",
        requirements=["Kubernetes"],
        optional_requirements=[],
    )
    matches = [
        make_match(jobs[0], 88),
        make_match(jobs[1], 94),
        make_match(third_job, 72),
    ]

    top_matches, remaining = partition_ranked_matches(matches, top_n=2)

    assert [result.overall_score for result in top_matches] == [80, 65]
    assert remaining == [
        {
            "job_link": "https://example.com/jobs/platform-engineer",
            "overall_score": 65,
        }
    ]
