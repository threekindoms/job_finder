import json
from pathlib import Path

from src.models import ConfidenceLevel, JobPosting, MatchResult, RunReport, UsageSummary
from src.reporting import build_artifacts, build_summary
from src.storage import create_run_dir, write_run_artifacts


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_jobs() -> list[JobPosting]:
    return [
        JobPosting.model_validate(job)
        for job in json.loads((FIXTURES_DIR / "jobs.json").read_text())
    ]


def build_report() -> RunReport:
    jobs = load_jobs()[:2]
    return RunReport(
        searched_jobs=jobs,
        top_matches=[
            MatchResult(
                job_link=jobs[0].link,
                overall_score=92,
                required_qualifications_score=40,
                preferred_qualifications_score=10,
                experience_level_score=15,
                domain_fit_score=15,
                location_or_availability_score=0,
                strengths=["Strong backend overlap."],
                gaps=[],
                other_considerations=[],
                resume_improvement_suggestions=[],
                confidence=ConfidenceLevel.HIGH,
            )
        ],
        remaining_ranked_jobs=[
            {
                "job_link": str(jobs[1].link),
                "overall_score": 74,
            }
        ],
        usage=UsageSummary(
            local_prefilter_models=["llama3.1:8b"],
            shortlist_scoring_provider="openai",
            shortlist_scoring_model="gpt-4.1-mini",
            searched_job_count=2,
            prefiltered_job_count=2,
            shortlist_job_count=1,
            scored_job_count=1,
            prompt_tokens=100,
            completion_tokens=40,
            estimated_cost_usd=0.02,
        ),
    )


def test_build_summary_counts_report_sections():
    summary = build_summary(build_report())

    assert summary == {
        "searched_job_count": 2,
        "top_match_count": 1,
        "remaining_ranked_job_count": 1,
        "usage": {
            "completion_tokens": 40,
            "estimated_cost_usd": 0.02,
            "local_prefilter_models": ["llama3.1:8b"],
            "prefiltered_job_count": 2,
            "prompt_tokens": 100,
            "scored_job_count": 1,
            "searched_job_count": 2,
            "shortlist_job_count": 1,
            "shortlist_scoring_model": "gpt-4.1-mini",
            "shortlist_scoring_provider": "openai",
        },
    }


def test_build_artifacts_serializes_models():
    artifacts = build_artifacts(build_report())

    assert artifacts["jobs"][0]["title"] == "Backend Software Engineer"
    assert artifacts["top_matches"][0]["confidence"] == "high"
    remaining = artifacts["remaining_ranked_jobs"][0]
    assert remaining["overall_score"] == 74
    assert remaining["title"] == "Software Engineer II"
    assert remaining["company"] in (None, "")


def test_write_run_artifacts_creates_expected_files(tmp_path):
    run_dir = create_run_dir(tmp_path, timestamp="20260515T120000")
    paths = write_run_artifacts(run_dir, build_report())

    assert run_dir == tmp_path / "20260515T120000"
    assert sorted(path.name for path in paths.values()) == [
        "jobs.json",
        "remaining_ranked_jobs.json",
        "summary.json",
        "top_matches.json",
        "usage.json",
    ]
    assert json.loads(paths["summary"].read_text()) == {
        "remaining_ranked_job_count": 1,
        "searched_job_count": 2,
        "top_match_count": 1,
        "usage": {
            "completion_tokens": 40,
            "estimated_cost_usd": 0.02,
            "local_prefilter_models": ["llama3.1:8b"],
            "prefiltered_job_count": 2,
            "prompt_tokens": 100,
            "scored_job_count": 1,
            "searched_job_count": 2,
            "shortlist_job_count": 1,
            "shortlist_scoring_model": "gpt-4.1-mini",
            "shortlist_scoring_provider": "openai",
        },
    }
    assert json.loads(paths["usage"].read_text())["estimated_cost_usd"] == 0.02
