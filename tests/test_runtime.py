import json
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.config import Settings
from src.models import CandidateProfile, JobPosting, LocalPrefilterResult
from src.runtime import build_ollama_dependencies, load_saved_job_records


def test_load_saved_job_records_reads_json(tmp_path):
    path = tmp_path / "jobs.json"
    payload = [{"title": "Engineer"}]
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_saved_job_records(path) == payload


def test_build_ollama_dependencies_can_use_saved_candidate_profile(tmp_path, monkeypatch):
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text("[]", encoding="utf-8")
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "professional_summary": "Engineer",
                "skills": ["Python"],
                "work_experience": [],
                "education": [],
                "location": None,
            }
        ),
        encoding="utf-8",
    )

    constructed_models = []

    class FakeModel:
        def __init__(self, *args, **kwargs):
            constructed_models.append(kwargs["model"])

        def __call__(self, _messages, **_kwargs):
            return AIMessage(content=json.dumps(
                {"local_score": 80, "should_advance": True, "short_reason": "ok"}
            ))

        def with_structured_output(self, *_args, **_kwargs):
            return RunnableLambda(lambda _payload: None)

    monkeypatch.setattr("src.runtime.ChatOllama", FakeModel)

    dependencies = build_ollama_dependencies(
        Settings(
            local_prefilter_models=["llama3.1:8b"],
            shortlist_scoring_model="llama3.1:8b",
        ),
        jobs_path,
        candidate_profile_file=candidate_path,
    )

    initial_model_count = len(constructed_models)
    assert dependencies.candidate_extractor("unused").skills == ["Python"]
    assert len(constructed_models) == initial_model_count


def test_build_ollama_dependencies_can_load_public_job_urls(monkeypatch):
    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, _messages, **_kwargs):
            return AIMessage(content=json.dumps({
                "title": "Backend Engineer",
                "description": "Build Python services.",
                "requirements": ["Python"],
                "optional_requirements": [],
            }))

        def with_structured_output(self, *_args, **_kwargs):
            return RunnableLambda(lambda _payload: None)

    monkeypatch.setattr("src.runtime.ChatOllama", FakeModel)
    monkeypatch.setattr(
        "src.runtime.fetch_public_job_page_text",
        lambda url: f"Source URL: {url}\nTitle: Backend Engineer",
    )

    dependencies = build_ollama_dependencies(
        Settings(
            local_prefilter_models=["llama3.1:8b"],
            shortlist_scoring_model="llama3.1:8b",
        ),
        jobs_file=None,
    )

    jobs = dependencies.job_loader(["https://example.com/jobs/backend"])

    assert jobs[0]["title"] == "Backend Engineer"
    assert str(jobs[0]["link"]) == "https://example.com/jobs/backend"


def test_prefilter_runner_uses_third_model_only_for_disagreements(tmp_path, monkeypatch):
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text("[]", encoding="utf-8")
    calls = {"m1": 0, "m2": 0, "m3": 0}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.model = kwargs["model"]

        def __call__(self, messages, **_kwargs):
            calls[self.model] += 1
            is_disputed = "disputed" in str(messages)
            votes = {"m1": True, "m2": not is_disputed, "m3": True}
            job_link = (
                "https://example.com/jobs/disputed"
                if is_disputed
                else "https://example.com/jobs/agreed"
            )
            return AIMessage(content=json.dumps({
                "job_link": job_link,
                "local_score": 80,
                "should_advance": votes[self.model],
                "short_reason": self.model,
            }))

        def with_structured_output(self, *_args, **_kwargs):
            return RunnableLambda(lambda _payload: None)

    monkeypatch.setattr("src.runtime.ChatOllama", FakeModel)

    dependencies = build_ollama_dependencies(
        Settings(
            local_prefilter_models=["m1", "m2", "m3"],
            shortlist_scoring_model="m1",
        ),
        jobs_path,
    )
    jobs = [
        JobPosting(
            title="Agreed",
            link="https://example.com/jobs/agreed",
            description="agreed",
        ),
        JobPosting(
            title="Disputed",
            link="https://example.com/jobs/disputed",
            description="disputed",
        ),
    ]

    results = dependencies.prefilter_runner(
        CandidateProfile(
            professional_summary="Engineer",
            skills=[],
            work_experience=[],
            education=[],
            location=None,
        ),
        jobs,
        ["m1", "m2", "m3"],
    )

    assert [str(result.job_link) for result in results["m3"]] == [
        "https://example.com/jobs/disputed"
    ]
    assert calls == {"m1": 2, "m2": 2, "m3": 1}
