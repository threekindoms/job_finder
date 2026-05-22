from typing import Any

from src.models import RunReport


def build_summary(report: RunReport) -> dict[str, Any]:
    """Build a compact deterministic summary for run artifacts."""
    summary = {
        "searched_job_count": len(report.searched_jobs),
        "top_match_count": len(report.top_matches),
        "remaining_ranked_job_count": len(report.remaining_ranked_jobs),
    }
    if report.usage is not None:
        summary["usage"] = report.usage.model_dump(mode="json")
    return summary


def build_artifacts(report: RunReport) -> dict[str, Any]:
    """Convert a run report into JSON-serializable artifact payloads."""
    jobs_by_link = {str(job.link): job for job in report.searched_jobs}

    def _enrich_match(match) -> dict[str, Any]:
        d = match.model_dump(mode="json")
        job = jobs_by_link.get(str(match.job_link))
        if job is not None:
            d["title"] = job.title
            d["company"] = job.company
        return d

    return {
        "candidate_profile": (
            None if report.candidate_profile is None
            else report.candidate_profile.model_dump(mode="json")
        ),
        "jobs": [job.model_dump(mode="json") for job in report.searched_jobs],
        "top_matches": [_enrich_match(match) for match in report.top_matches],
        "remaining_ranked_jobs": [
            {
                **row,
                **({
                    "title": jobs_by_link[row["job_link"]].title,
                    "company": jobs_by_link[row["job_link"]].company,
                } if row.get("job_link") in jobs_by_link else {}),
            }
            for row in report.remaining_ranked_jobs
        ],
        "summary": build_summary(report),
        "usage": None if report.usage is None else report.usage.model_dump(mode="json"),
    }
