import json
from pathlib import Path

import pytest

from src.cli import PipelineDependencies, build_parser, run_workflow
from src.config import Settings
from src.models import ConfidenceLevel, LocalPrefilterResult, MatchResult


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_json(name: str):
    return json.loads((FIXTURES_DIR / name).read_text())


def make_dependencies() -> PipelineDependencies:
    jobs = load_json("jobs.json")[:3]

    def prefilter_runner(_candidate, loaded_jobs, _models):
        return {
            "llama3.1:8b": [
                LocalPrefilterResult(
                    job_link=loaded_jobs[0].link,
                    local_score=92,
                    should_advance=True,
                    short_reason="Strong fit",
                ),
                LocalPrefilterResult(
                    job_link=loaded_jobs[1].link,
                    local_score=74,
                    should_advance=True,
                    short_reason="Possible fit",
                ),
                LocalPrefilterResult(
                    job_link=loaded_jobs[2].link,
                    local_score=20,
                    should_advance=False,
                    short_reason="Missing Kubernetes",
                ),
            ]
        }

    def scorer(_candidate, job):
        score = 94 if "backend" in str(job.link) else 78
        return MatchResult(
            job_link=job.link,
            overall_score=score,
            required_qualifications_score=48 if score == 94 else 38,
            preferred_qualifications_score=16 if score == 94 else 12,
            experience_level_score=8 if score == 94 else 7,
            domain_fit_score=8 if score == 94 else 7,
            location_or_availability_score=0 if score == 94 else 1,
            strengths=["Fit"],
            gaps=[],
            other_considerations=[],
            resume_improvement_suggestions=[],
            confidence=ConfidenceLevel.HIGH,
        )

    return PipelineDependencies(
        candidate_extractor=lambda _text: load_json("candidate_profile.json"),
        job_loader=lambda _links: jobs,
        prefilter_runner=prefilter_runner,
        scorer=scorer,
    )


def test_run_workflow_end_to_end_writes_artifacts(tmp_path):
    report, run_dir = run_workflow(
        resume_path=FIXTURES_DIR / "resume.txt",
        manual_links_path=FIXTURES_DIR / "job_links.txt",
        dependencies=make_dependencies(),
        settings=Settings(
            local_prefilter_models=["llama3.1:8b"],
            max_paid_llm_jobs=2,
            runs_dir=tmp_path,
        ),
        top_n=1,
        timestamp="20260515T120000",
    )

    assert len(report.searched_jobs) == 3
    assert len(report.top_matches) == 1
    assert report.remaining_ranked_jobs == [
        {
            "job_link": "https://example.com/jobs/software-engineer-ii",
            "overall_score": 65,
        }
    ]
    assert run_dir == tmp_path / "20260515T120000"
    assert (run_dir / "summary.json").exists()
    assert report.usage is not None
    assert report.usage.shortlist_job_count == 2
    assert (run_dir / "usage.json").exists()


def test_run_workflow_surfaces_missing_manual_links_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_workflow(
            resume_path=FIXTURES_DIR / "resume.txt",
            manual_links_path=tmp_path / "missing.txt",
            dependencies=make_dependencies(),
            settings=Settings(runs_dir=tmp_path),
        )


def test_build_parser_accepts_search_url_argument():
    args = build_parser().parse_args(
        [
            "--resume",
            "resume.txt",
            "--search-url",
            "https://www.linkedin.com/jobs/search/",
        ]
    )

    assert args.search_url == "https://www.linkedin.com/jobs/search/"


def test_build_parser_accepts_jobs_file_argument():
    args = build_parser().parse_args(
        [
            "--resume",
            "resume.txt",
            "--manual-links",
            "links.txt",
            "--jobs-file",
            "jobs.json",
        ]
    )

    assert args.jobs_file == "jobs.json"


def test_build_parser_accepts_candidate_profile_file_argument():
    args = build_parser().parse_args(
        [
            "--resume",
            "resume.txt",
            "--manual-links",
            "links.txt",
            "--jobs-file",
            "jobs.json",
            "--candidate-profile-file",
            "candidate.json",
        ]
    )

    assert args.candidate_profile_file == "candidate.json"


def test_build_parser_accepts_direct_job_urls():
    args = build_parser().parse_args(
        [
            "--resume",
            "resume.txt",
            "--job-url",
            "https://example.com/jobs/backend",
            "--job-url",
            "https://example.com/jobs/frontend",
        ]
    )

    assert args.job_url == [
        "https://example.com/jobs/backend",
        "https://example.com/jobs/frontend",
    ]


def test_build_parser_accepts_scrape_only_flag():
    args = build_parser().parse_args(
        [
            "--search-url",
            "https://www.linkedin.com/jobs/search/",
            "--scrape-only",
        ]
    )

    assert args.scrape_only is True
    assert args.search_url == "https://www.linkedin.com/jobs/search/"


def test_build_parser_scrape_only_defaults_to_false():
    args = build_parser().parse_args(
        ["--resume", "resume.txt", "--manual-links", "links.txt"]
    )

    assert args.scrape_only is False
