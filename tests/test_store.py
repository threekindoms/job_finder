import json
from datetime import datetime, timedelta, timezone

import pytest

from src.models import CandidateProfile, ConfidenceLevel, JobPosting, LocalPrefilterResult, MatchResult
from src.store.cache import (
    _JOB_TTL_DAYS,
    get_candidate,
    get_job,
    get_match,
    get_prefilter,
    put_candidate,
    put_job,
    put_match,
    put_prefilter,
    resume_hash,
    scoring_prompt_version,
)
from src.store.db import open_db


@pytest.fixture
def db():
    conn = open_db(":memory:")
    yield conn
    conn.close()


def make_candidate() -> CandidateProfile:
    return CandidateProfile(
        professional_summary="Backend engineer with Python and Go.",
        skills=["Python", "Go"],
        work_experience=[],
        education=["B.S. Computer Science"],
        location="Seattle, WA",
    )


def make_job() -> JobPosting:
    return JobPosting(
        title="Backend Engineer",
        company="Acme",
        link="https://example.com/jobs/backend-engineer",
        description="Build services.",
        requirements=["Python"],
        optional_requirements=["Go"],
    )


def make_match(job: JobPosting) -> MatchResult:
    return MatchResult(
        job_link=job.link,
        overall_score=80,
        required_qualifications_score=48,
        preferred_qualifications_score=16,
        experience_level_score=8,
        domain_fit_score=8,
        location_or_availability_score=0,
        strengths=["Strong Python fit."],
        gaps=[],
        other_considerations=[],
        resume_improvement_suggestions=[],
        confidence=ConfidenceLevel.HIGH,
    )


# ── schema ────────────────────────────────────────────────────────────────────

def test_open_db_creates_tables(db):
    tables = {
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"candidate_profiles", "job_postings", "prefilter_results", "match_results"} <= tables


# ── CandidateProfile ──────────────────────────────────────────────────────────

def test_candidate_cache_miss_returns_none(db):
    assert get_candidate(db, "nonexistent") is None


def test_candidate_round_trip(db):
    profile = make_candidate()
    r_hash = resume_hash("My resume text")
    put_candidate(db, r_hash, profile)
    cached = get_candidate(db, r_hash)
    assert cached is not None
    assert cached.skills == profile.skills
    assert cached.location == profile.location


def test_candidate_put_replace(db):
    r_hash = resume_hash("resume")
    put_candidate(db, r_hash, make_candidate())
    updated = make_candidate()
    updated.location = "New York, NY"
    put_candidate(db, r_hash, updated)
    cached = get_candidate(db, r_hash)
    assert cached is not None
    assert cached.location == "New York, NY"


def test_resume_hash_is_deterministic():
    text = "Same resume text"
    assert resume_hash(text) == resume_hash(text)


def test_resume_hash_differs_for_different_text():
    assert resume_hash("resume A") != resume_hash("resume B")


# ── JobPosting ────────────────────────────────────────────────────────────────

def test_job_cache_miss_returns_none(db):
    assert get_job(db, "https://example.com/jobs/missing") is None


def test_job_round_trip(db):
    job = make_job()
    put_job(db, job)
    cached = get_job(db, str(job.link))
    assert cached is not None
    assert cached.title == job.title
    assert cached.company == job.company
    assert str(cached.link) == str(job.link)


def test_job_ttl_expiry(db):
    job = make_job()
    put_job(db, job)
    # Backdate the fetched_at timestamp past the TTL
    stale_ts = (
        datetime.now(timezone.utc) - timedelta(days=_JOB_TTL_DAYS + 1)
    ).isoformat()
    db.execute(
        "UPDATE job_postings SET fetched_at = ? WHERE url = ?",
        (stale_ts, str(job.link)),
    )
    db.commit()
    assert get_job(db, str(job.link)) is None


def test_job_within_ttl_is_returned(db):
    job = make_job()
    put_job(db, job)
    recent_ts = (
        datetime.now(timezone.utc) - timedelta(days=_JOB_TTL_DAYS - 1)
    ).isoformat()
    db.execute(
        "UPDATE job_postings SET fetched_at = ? WHERE url = ?",
        (recent_ts, str(job.link)),
    )
    db.commit()
    assert get_job(db, str(job.link)) is not None


# ── LocalPrefilterResult ──────────────────────────────────────────────────────

def test_prefilter_cache_miss_returns_none(db):
    assert get_prefilter(db, "h", "https://example.com/jobs/x", "model") is None


def test_prefilter_round_trip(db):
    job = make_job()
    result = LocalPrefilterResult(
        job_link=job.link,
        local_score=78,
        should_advance=True,
        short_reason="Strong Python match.",
    )
    put_prefilter(db, "candidate_hash", str(job.link), "llama3.1:8b", result)
    cached = get_prefilter(db, "candidate_hash", str(job.link), "llama3.1:8b")
    assert cached is not None
    assert cached.local_score == 78
    assert cached.should_advance is True


def test_prefilter_different_model_is_cache_miss(db):
    job = make_job()
    result = LocalPrefilterResult(
        job_link=job.link, local_score=70, should_advance=True, short_reason="Fit."
    )
    put_prefilter(db, "h", str(job.link), "model_A", result)
    assert get_prefilter(db, "h", str(job.link), "model_B") is None


def test_prefilter_different_candidate_is_cache_miss(db):
    job = make_job()
    result = LocalPrefilterResult(
        job_link=job.link, local_score=70, should_advance=True, short_reason="Fit."
    )
    put_prefilter(db, "hash_A", str(job.link), "model", result)
    assert get_prefilter(db, "hash_B", str(job.link), "model") is None


# ── MatchResult ───────────────────────────────────────────────────────────────

def test_match_cache_miss_returns_none(db):
    assert get_match(db, "h", "https://example.com/jobs/x", "model", "v1") is None


def test_match_round_trip(db):
    job = make_job()
    result = make_match(job)
    put_match(db, "candidate_hash", str(job.link), "llama3.1:8b", "v1", result)
    cached = get_match(db, "candidate_hash", str(job.link), "llama3.1:8b", "v1")
    assert cached is not None
    assert cached.overall_score == result.overall_score
    assert cached.confidence is ConfidenceLevel.HIGH


def test_match_different_prompt_version_is_cache_miss(db):
    job = make_job()
    result = make_match(job)
    put_match(db, "h", str(job.link), "model", "version_A", result)
    assert get_match(db, "h", str(job.link), "model", "version_B") is None


def test_match_different_model_is_cache_miss(db):
    job = make_job()
    result = make_match(job)
    put_match(db, "h", str(job.link), "model_A", "v1", result)
    assert get_match(db, "h", str(job.link), "model_B", "v1") is None


def test_match_different_candidate_hash_is_cache_miss(db):
    job = make_job()
    result = make_match(job)
    put_match(db, "hash_A", str(job.link), "model", "v1", result)
    assert get_match(db, "hash_B", str(job.link), "model", "v1") is None


# ── scoring_prompt_version ────────────────────────────────────────────────────

def test_prompt_version_is_deterministic():
    v1 = scoring_prompt_version("Score the candidate.")
    v2 = scoring_prompt_version("Score the candidate.")
    assert v1 == v2


def test_prompt_version_changes_on_prompt_change():
    v1 = scoring_prompt_version("Original prompt.")
    v2 = scoring_prompt_version("Modified prompt.")
    assert v1 != v2


def test_prompt_version_changes_on_weight_change(monkeypatch):
    import src.store.cache as cache_mod
    original_maxima = dict(cache_mod.SCORE_MAXIMA)
    v1 = scoring_prompt_version("prompt")

    monkeypatch.setitem(cache_mod.SCORE_MAXIMA, "required_qualifications_score", 50)
    v2 = scoring_prompt_version("prompt")
    assert v1 != v2
