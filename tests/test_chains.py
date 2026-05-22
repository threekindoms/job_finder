from langchain_core.runnables import RunnableLambda

from src.chains import (
    build_candidate_profile_chain,
    build_job_posting_chain,
    build_local_prefilter_chain,
    build_shortlist_scoring_chain,
)
from src.models import CandidateProfile, ConfidenceLevel, JobPosting, LocalPrefilterResult, MatchResult


class FakeChatModel:
    def __init__(self, result):
        self.result = result
        self.schemas = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return RunnableLambda(lambda _: self.result)


def test_build_candidate_profile_chain_uses_structured_candidate_schema():
    model = FakeChatModel(
        CandidateProfile(
            professional_summary="Backend engineer",
            skills=["Python"],
            work_experience=[],
            education=[],
            location=None,
        )
    )

    result = build_candidate_profile_chain(model).invoke({"resume_text": "Resume text"})

    assert model.schemas == [CandidateProfile]
    assert result.skills == ["Python"]


def test_build_job_posting_chain_uses_structured_job_schema():
    model = FakeChatModel(
        JobPosting(
            title="Backend Engineer",
            link="https://example.com/jobs/backend-engineer",
            description="Build backend services.",
            requirements=["Python"],
            optional_requirements=[],
        )
    )

    result = build_job_posting_chain(model).invoke({"raw_job_text": "Job text"})

    assert model.schemas == [JobPosting]
    assert result.title == "Backend Engineer"


def test_build_local_prefilter_chain_uses_structured_prefilter_schema():
    model = FakeChatModel(
        LocalPrefilterResult(
            job_link="https://example.com/jobs/backend-engineer",
            local_score=92,
            should_advance=True,
            short_reason="Strong fit",
        )
    )

    result = build_local_prefilter_chain(model).invoke(
        {
            "candidate_profile": "Candidate",
            "job_posting": "Job",
        }
    )

    assert model.schemas == [LocalPrefilterResult]
    assert result.should_advance is True


def test_build_shortlist_scoring_chain_uses_structured_match_schema():
    model = FakeChatModel(
        MatchResult(
            job_link="https://example.com/jobs/backend-engineer",
            overall_score=92,
            required_qualifications_score=48,
            preferred_qualifications_score=16,
            experience_level_score=8,
            domain_fit_score=8,
            location_or_availability_score=0,
            strengths=["Strong fit"],
            gaps=[],
            other_considerations=[],
            resume_improvement_suggestions=[],
            confidence=ConfidenceLevel.HIGH,
        )
    )

    result = build_shortlist_scoring_chain(model).invoke(
        {
            "candidate_profile": "Candidate",
            "job_posting": "Job",
        }
    )

    assert model.schemas == [MatchResult]
    assert result.overall_score == 80
