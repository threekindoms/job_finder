import re
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from src.models import JobPosting

_LEGAL_SUFFIXES = re.compile(
    r"\b(llc|inc|corp|ltd|co|limited|group|technologies|technology|tech"
    r"|solutions|services|international|global|holdings|plc|gmbh|ag)\b",
    re.IGNORECASE,
)
_NON_WORD = re.compile(r"[^\w\s]")


def _normalize_company(name: str) -> str:
    text = _NON_WORD.sub(" ", name.lower())
    text = _LEGAL_SUFFIXES.sub(" ", text)
    return " ".join(text.split())


def company_matches(job_company: str, ignore_name: str) -> bool:
    """Return True if job_company fuzzy-matches ignore_name.

    Strips punctuation and common corporate suffixes before comparing, then
    checks whether either normalized name is contained in the other.
    """
    if not job_company or not ignore_name:
        return False
    a = _normalize_company(job_company)
    b = _normalize_company(ignore_name)
    if not a or not b:
        return False
    return a == b or a in b or b in a


class JobPostingExtractor(Protocol):
    def invoke(self, raw_job_text: str) -> JobPosting | dict:
        ...


def extract_job_posting(
    raw_job_text: str,
    extractor: JobPostingExtractor,
) -> JobPosting:
    """Run an injectable structured extractor and validate its output."""
    extracted = extractor.invoke(raw_job_text)
    return JobPosting.model_validate(extracted)


def normalize_job_records(
    records: Iterable[Mapping[str, Any]],
    ignored_links: Iterable[str] | None = None,
    ignored_companies: Iterable[str] | None = None,
) -> tuple[list[JobPosting], list[JobPosting]]:
    """Validate, deduplicate, and filter raw job records.

    Returns (accepted_jobs, company_excluded_jobs).
    """
    ignored_links_set = set(ignored_links or [])
    ignore_co_list = list(ignored_companies or [])
    jobs: list[JobPosting] = []
    company_excluded: list[JobPosting] = []
    seen_links: set[str] = set()

    for record in records:
        job = JobPosting.model_validate(record)
        link = str(job.link)
        if link in ignored_links_set or link in seen_links:
            continue
        if ignore_co_list and job.company and any(
            company_matches(job.company, name) for name in ignore_co_list
        ):
            company_excluded.append(job)
            continue

        seen_links.add(link)
        jobs.append(job)

    return jobs, company_excluded
