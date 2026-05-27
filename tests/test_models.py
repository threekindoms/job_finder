import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.models import (
    CandidateProfile,
    ConfidenceLevel,
    JobPosting,
    LocalPrefilterResult,
    MatchResult,
    RunReport,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_json(name: str):
    return json.loads((FIXTURES_DIR / name).read_text())


def test_candidate_profile_fixture_matches_schema():
    profile = CandidateProfile.model_validate(load_json("candidate_profile.json"))

    assert profile.skills == ["Java", "Python", "Go"]
    assert profile.location == "San Francisco, CA"


def test_job_fixtures_match_schema():
    jobs = [JobPosting.model_validate(job) for job in load_json("jobs.json")]

    assert len(jobs) == 5
    assert jobs[0].title == "Backend Software Engineer"


def test_job_posting_normalizes_common_local_model_keys():
    job = JobPosting.model_validate(
        {
            "Title": "Backend Engineer",
            "Link": "https://example.com/jobs/backend-engineer",
            "Description": "Build backend services.",
            "Basic Qualifications": ["Python"],
            "Preferred Qualifications": ["AWS"],
        }
    )

    assert job.title == "Backend Engineer"
    assert job.requirements == ["Python"]
    assert job.optional_requirements == ["AWS"]


def test_job_posting_flattens_grouped_requirements():
    job = JobPosting.model_validate(
        {
            "title": "Backend Engineer",
            "link": "https://example.com/jobs/backend-engineer",
            "description": "Build backend services.",
            "requirements": [
                {"name": "Basic Qualifications", "items": ["Python"]},
                {"name": "Preferred Qualifications", "items": ["AWS"]},
            ],
            "optional_requirements": [],
        }
    )

    assert job.requirements == ["Python"]
    assert job.optional_requirements == ["AWS"]


def test_job_posting_coerces_null_list_fields_to_empty_list():
    job = JobPosting.model_validate(
        {
            "title": "Backend Engineer",
            "link": "https://example.com/jobs/backend-engineer",
            "description": "Build backend services.",
            "requirements": None,
            "optional_requirements": None,
        }
    )

    assert job.requirements == []
    assert job.optional_requirements == []


def test_malformed_job_fixture_fails_validation():
    with pytest.raises(ValidationError):
        JobPosting.model_validate(load_json("malformed_job.json"))


def test_prefilter_result_enforces_score_bounds():
    result = LocalPrefilterResult(
        job_link="https://example.com/jobs/backend-engineer",
        local_score=88,
        should_advance=True,
        short_reason="Strong backend overlap.",
    )

    assert result.local_score == 88


def test_prefilter_result_normalizes_fractional_scores():
    result = LocalPrefilterResult(
        job_link="https://example.com/jobs/backend",
        local_score=0.88,
        should_advance=True,
        short_reason="Strong fit.",
    )

    assert result.local_score == 88

    with pytest.raises(ValidationError):
        LocalPrefilterResult(
            job_link="https://example.com/jobs/backend-engineer",
            local_score=101,
            should_advance=True,
            short_reason="Invalid score.",
        )


def test_match_result_normalizes_weighted_dimension_bounds():
    match = MatchResult(
        job_link="https://example.com/jobs/backend-engineer",
        overall_score=100,
        required_qualifications_score=53,
        preferred_qualifications_score=22,
        experience_level_score=9,
        domain_fit_score=15,
        location_or_availability_score=1,
        strengths=["Strong programming-language match."],
        gaps=[],
        other_considerations=[],
        resume_improvement_suggestions=[],
        confidence=ConfidenceLevel.HIGH,
    )

    assert match.confidence is ConfidenceLevel.HIGH

    normalized = MatchResult(
        job_link="https://example.com/jobs/backend-engineer",
        overall_score=92,
        required_qualifications_score=61,
        preferred_qualifications_score=25,
        experience_level_score=20,
        domain_fit_score=30,
        location_or_availability_score=7,
        strengths=[],
        gaps=[],
        other_considerations=[],
        resume_improvement_suggestions=[],
        confidence=ConfidenceLevel.HIGH,
    )

    assert normalized.required_qualifications_score == 53
    assert normalized.preferred_qualifications_score == 22
    assert normalized.experience_level_score == 9
    assert normalized.domain_fit_score == 15
    assert normalized.location_or_availability_score == 1
    assert normalized.overall_score == 100


def test_match_result_normalizes_singleton_text_lists():
    match = MatchResult(
        job_link="https://example.com/jobs/backend-engineer",
        overall_score=92,
        required_qualifications_score=40,
        preferred_qualifications_score=10,
        experience_level_score=15,
        domain_fit_score=15,
        location_or_availability_score=0,
        strengths="Strong Python fit.",
        gaps="Limited Terraform evidence.",
        other_considerations="Remote role.",
        resume_improvement_suggestions="Add deployment examples.",
        confidence=ConfidenceLevel.HIGH,
    )

    assert match.strengths == ["Strong Python fit."]
    assert match.gaps == ["Limited Terraform evidence."]
    assert match.other_considerations == ["Remote role."]
    assert match.resume_improvement_suggestions == ["Add deployment examples."]


def test_run_report_accepts_nested_models():
    jobs = [JobPosting.model_validate(job) for job in load_json("jobs.json")[:1]]
    report = RunReport(searched_jobs=jobs)

    assert report.searched_jobs[0].title == "Backend Software Engineer"
    assert report.top_matches == []


def test_settings_from_env_parses_defaults_and_lists(monkeypatch):
    monkeypatch.setenv("LOCAL_PREFILTER_MODELS", "qwen2.5:7b, llama3.1:8b")
    monkeypatch.setenv("MAX_PAID_LLM_JOBS", "12")
    monkeypatch.setenv("SHORTLIST_SCORING_PROVIDER", "openai")

    settings = Settings.from_env()

    assert settings.local_prefilter_models == ["qwen2.5:7b", "llama3.1:8b"]
    assert settings.max_paid_llm_jobs == 12
    assert settings.shortlist_scoring_provider == "openai"


def test_settings_from_env_resolves_relative_paths_against_pythonpath(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.delenv("RUNS_DIR", raising=False)
    monkeypatch.delenv("STORAGE_DB_PATH", raising=False)
    monkeypatch.delenv("LINKEDIN_COOKIE_FILE", raising=False)

    settings = Settings.from_env()

    assert settings.runs_dir == tmp_path / "runs"
    assert settings.storage_db_path == tmp_path / "storage" / "store.db"
    assert settings.linkedin_cookie_file == tmp_path / "config" / "linkedin_cookies.json"


def test_settings_from_env_absolute_paths_are_not_relocated(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("RUNS_DIR", "/var/runs")
    monkeypatch.setenv("STORAGE_DB_PATH", "/var/db/store.db")

    settings = Settings.from_env()

    assert settings.runs_dir == Path("/var/runs")
    assert settings.storage_db_path == Path("/var/db/store.db")


def test_settings_from_env_relative_paths_stay_relative_without_pythonpath(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("RUNS_DIR", raising=False)

    settings = Settings.from_env()

    assert not settings.runs_dir.is_absolute()
    assert settings.runs_dir == Path("runs")
