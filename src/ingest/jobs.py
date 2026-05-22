from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from src.models import JobPosting


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
) -> list[JobPosting]:
    """Validate, deduplicate, and filter raw job records."""
    ignored = set(ignored_links or [])
    jobs: list[JobPosting] = []
    seen_links: set[str] = set()

    for record in records:
        job = JobPosting.model_validate(record)
        link = str(job.link)
        if link in ignored or link in seen_links:
            continue

        seen_links.add(link)
        jobs.append(job)

    return jobs
