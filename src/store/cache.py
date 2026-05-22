import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone

from src.models import SCORE_MAXIMA, CandidateProfile, JobPosting, LocalPrefilterResult, MatchResult

logger = logging.getLogger(__name__)

_JOB_TTL_DAYS = 7


def resume_hash(resume_text: str) -> str:
    """SHA-256 of resume text bytes — stable cache key for a candidate."""
    return hashlib.sha256(resume_text.encode("utf-8")).hexdigest()


def scoring_prompt_version(system_prompt: str) -> str:
    """Short hash of scoring system prompt + current weight maxima.

    Changing the prompt text or any weight value produces a new version,
    automatically invalidating cached match_results without manual clearing.
    """
    data = system_prompt + json.dumps(SCORE_MAXIMA, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── CandidateProfile ──────────────────────────────────────────────────────────

def get_candidate(conn: sqlite3.Connection, r_hash: str) -> CandidateProfile | None:
    row = conn.execute(
        "SELECT profile_json FROM candidate_profiles WHERE resume_hash = ?", (r_hash,)
    ).fetchone()
    if row is None:
        logger.debug("candidate cache miss (hash=%s)", r_hash[:12])
        return None
    logger.debug("candidate cache hit (hash=%s)", r_hash[:12])
    return CandidateProfile.model_validate_json(row[0])


def put_candidate(conn: sqlite3.Connection, r_hash: str, profile: CandidateProfile) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO candidate_profiles (resume_hash, extracted_at, profile_json)"
        " VALUES (?, ?, ?)",
        (r_hash, _now_iso(), profile.model_dump_json()),
    )
    conn.commit()
    logger.debug("candidate written (hash=%s)", r_hash[:12])


# ── JobPosting ────────────────────────────────────────────────────────────────

def get_job(conn: sqlite3.Connection, url: str) -> JobPosting | None:
    row = conn.execute(
        "SELECT posting_json, fetched_at FROM job_postings WHERE url = ?", (url,)
    ).fetchone()
    if row is None:
        logger.debug("job cache miss: %s", url)
        return None
    try:
        fetched = datetime.fromisoformat(row[1])
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - fetched > timedelta(days=_JOB_TTL_DAYS):
            logger.debug("job cache expired: %s", url)
            return None
    except Exception:
        logger.debug("job cache TTL parse error, treating as miss: %s", url)
        return None
    logger.debug("job cache hit: %s", url)
    return JobPosting.model_validate_json(row[0])


def put_job(conn: sqlite3.Connection, job: JobPosting) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO job_postings (url, fetched_at, posting_json)"
        " VALUES (?, ?, ?)",
        (str(job.link), _now_iso(), job.model_dump_json()),
    )
    conn.commit()
    logger.debug("job written: %s", job.link)


# ── LocalPrefilterResult ──────────────────────────────────────────────────────

def get_prefilter(
    conn: sqlite3.Connection,
    candidate_hash: str,
    job_url: str,
    model_name: str,
) -> LocalPrefilterResult | None:
    row = conn.execute(
        "SELECT result_json FROM prefilter_results"
        " WHERE candidate_hash=? AND job_url=? AND model_name=?",
        (candidate_hash, job_url, model_name),
    ).fetchone()
    if row is None:
        logger.debug("prefilter cache miss: %s [model=%s]", job_url, model_name)
        return None
    logger.debug("prefilter cache hit: %s [model=%s]", job_url, model_name)
    return LocalPrefilterResult.model_validate_json(row[0])


def put_prefilter(
    conn: sqlite3.Connection,
    candidate_hash: str,
    job_url: str,
    model_name: str,
    result: LocalPrefilterResult,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO prefilter_results"
        " (candidate_hash, job_url, model_name, filtered_at, result_json)"
        " VALUES (?, ?, ?, ?, ?)",
        (candidate_hash, job_url, model_name, _now_iso(), result.model_dump_json()),
    )
    conn.commit()
    logger.debug(
        "prefilter written: %s [model=%s advance=%s score=%d]",
        job_url, model_name, result.should_advance, result.local_score,
    )


# ── MatchResult ───────────────────────────────────────────────────────────────

def get_match(
    conn: sqlite3.Connection,
    candidate_hash: str,
    job_url: str,
    scoring_model: str,
    prompt_version: str,
) -> MatchResult | None:
    row = conn.execute(
        "SELECT result_json FROM match_results"
        " WHERE candidate_hash=? AND job_url=? AND scoring_model=? AND prompt_version=?",
        (candidate_hash, job_url, scoring_model, prompt_version),
    ).fetchone()
    if row is None:
        logger.debug("match cache miss: %s [model=%s prompt=%s]", job_url, scoring_model, prompt_version)
        return None
    logger.debug("match cache hit: %s [model=%s prompt=%s]", job_url, scoring_model, prompt_version)
    return MatchResult.model_validate_json(row[0])


def put_match(
    conn: sqlite3.Connection,
    candidate_hash: str,
    job_url: str,
    scoring_model: str,
    prompt_version: str,
    result: MatchResult,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO match_results"
        " (candidate_hash, job_url, scoring_model, prompt_version, scored_at, result_json)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (candidate_hash, job_url, scoring_model, prompt_version, _now_iso(), result.model_dump_json()),
    )
    conn.commit()
    logger.debug("match written: %s [model=%s score=%d]", job_url, scoring_model, result.overall_score)
