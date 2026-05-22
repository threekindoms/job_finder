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


def _make_candidate() -> CandidateProfile:
    return CandidateProfile(
        professional_summary="Engineer",
        skills=[],
        work_experience=[],
        education=[],
        location=None,
    )


def _make_job(title: str, slug: str) -> JobPosting:
    return JobPosting(
        title=title,
        link=f"https://example.com/jobs/{slug}",
        description=slug,
    )


def test_prefilter_runner_routes_one_primary_failure_to_tiebreaker(tmp_path, monkeypatch):
    """#3: if one primary fails a job, tiebreaker is called for that job."""
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text("[]", encoding="utf-8")
    calls: dict[str, int] = {"m1": 0, "m2": 0, "m3": 0}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.model = kwargs["model"]

        def __call__(self, messages, **_kwargs):
            calls[self.model] += 1
            if self.model == "m2":
                raise RuntimeError("m2 simulated failure")
            return AIMessage(content=json.dumps({
                "job_link": "https://example.com/jobs/job",
                "local_score": 80,
                "should_advance": True,
                "short_reason": self.model,
            }))

        def with_structured_output(self, *_a, **_k):
            return RunnableLambda(lambda _: None)

    monkeypatch.setattr("src.runtime.ChatOllama", FakeModel)
    dependencies = build_ollama_dependencies(
        Settings(local_prefilter_models=["m1", "m2", "m3"], shortlist_scoring_model="m1"),
        jobs_path,
    )

    results = dependencies.prefilter_runner(
        _make_candidate(),
        [_make_job("Job", "job")],
        ["m1", "m2", "m3"],
    )

    # m1 succeeded, m2 failed → synthetic opposing vote → m3 (tiebreaker) must run
    assert calls["m3"] == 1
    # m3 said True → 2-1 True (m1 True, synthetic False, m3 True)
    assert len(results["m3"]) == 1
    assert results["m3"][0].should_advance is True
    # synthetic opposing vote was injected
    assert "_failed_primary" in results
    assert results["_failed_primary"][0].should_advance is False


def test_prefilter_runner_tiebreaker_is_decisive_when_one_primary_fails(tmp_path, monkeypatch):
    """#3: tiebreaker's vote decides the outcome when one primary failed."""
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text("[]", encoding="utf-8")

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.model = kwargs["model"]

        def __call__(self, messages, **_kwargs):
            if self.model == "m1":
                raise RuntimeError("m1 simulated failure")
            # m2 says True; m3 (tiebreaker) says False
            vote = self.model != "m3"
            return AIMessage(content=json.dumps({
                "job_link": "https://example.com/jobs/job",
                "local_score": 70,
                "should_advance": vote,
                "short_reason": self.model,
            }))

        def with_structured_output(self, *_a, **_k):
            return RunnableLambda(lambda _: None)

    monkeypatch.setattr("src.runtime.ChatOllama", FakeModel)
    dependencies = build_ollama_dependencies(
        Settings(local_prefilter_models=["m1", "m2", "m3"], shortlist_scoring_model="m1"),
        jobs_path,
    )

    results = dependencies.prefilter_runner(
        _make_candidate(),
        [_make_job("Job", "job")],
        ["m1", "m2", "m3"],
    )

    # m1 failed → synthetic True (opposing m2's True → synthetic False)
    # m2 True, synthetic False, m3 False → 2-1 False
    from src.matching.prefilter import aggregate_prefilter_results
    aggregated = aggregate_prefilter_results(results)
    assert len(aggregated) == 1
    assert aggregated[0].should_advance is False


def test_prefilter_runner_tiebreaker_failure_passes_disputed_job(tmp_path, monkeypatch):
    """#4: if both primaries disagree and the tiebreaker fails, job passes (recall-first)."""
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text("[]", encoding="utf-8")

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.model = kwargs["model"]

        def __call__(self, messages, **_kwargs):
            if self.model == "m3":
                raise RuntimeError("m3 simulated failure")
            vote = self.model == "m1"
            return AIMessage(content=json.dumps({
                "job_link": "https://example.com/jobs/job",
                "local_score": 65,
                "should_advance": vote,
                "short_reason": self.model,
            }))

        def with_structured_output(self, *_a, **_k):
            return RunnableLambda(lambda _: None)

    monkeypatch.setattr("src.runtime.ChatOllama", FakeModel)
    dependencies = build_ollama_dependencies(
        Settings(local_prefilter_models=["m1", "m2", "m3"], shortlist_scoring_model="m1"),
        jobs_path,
    )

    results = dependencies.prefilter_runner(
        _make_candidate(),
        [_make_job("Job", "job")],
        ["m1", "m2", "m3"],
    )

    # m1 True, m2 False, m3 fails → fallback pass injected
    assert "_tiebreaker_fallback" in results
    assert results["_tiebreaker_fallback"][0].should_advance is True

    from src.matching.prefilter import aggregate_prefilter_results
    aggregated = aggregate_prefilter_results(results)
    assert len(aggregated) == 1
    assert aggregated[0].should_advance is True


def test_prefilter_runner_routes_both_primary_failures_to_tiebreaker(tmp_path, monkeypatch):
    """#5: if both primaries fail, tiebreaker is called and its result is authoritative."""
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text("[]", encoding="utf-8")
    calls: dict[str, int] = {"m1": 0, "m2": 0, "m3": 0}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.model = kwargs["model"]

        def __call__(self, messages, **_kwargs):
            calls[self.model] += 1
            if self.model in ("m1", "m2"):
                raise RuntimeError(f"{self.model} simulated failure")
            return AIMessage(content=json.dumps({
                "job_link": "https://example.com/jobs/job",
                "local_score": 75,
                "should_advance": False,
                "short_reason": "tiebreaker says no",
            }))

        def with_structured_output(self, *_a, **_k):
            return RunnableLambda(lambda _: None)

    monkeypatch.setattr("src.runtime.ChatOllama", FakeModel)
    dependencies = build_ollama_dependencies(
        Settings(local_prefilter_models=["m1", "m2", "m3"], shortlist_scoring_model="m1"),
        jobs_path,
    )

    results = dependencies.prefilter_runner(
        _make_candidate(),
        [_make_job("Job", "job")],
        ["m1", "m2", "m3"],
    )

    # m3 must have been called
    assert calls["m3"] == 1
    # tiebreaker said False → only result for this job
    assert len(results["m3"]) == 1
    assert results["m3"][0].should_advance is False

    from src.matching.prefilter import aggregate_prefilter_results
    aggregated = aggregate_prefilter_results(results)
    assert len(aggregated) == 1
    assert aggregated[0].should_advance is False
