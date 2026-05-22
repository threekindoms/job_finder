import argparse
import tomllib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from src.config import Settings
from src.ingest.jobs import normalize_job_records
from src.ingest.manual_links import load_job_links
from src.ingest.resume import load_resume_text
from src.matching.prefilter import aggregate_prefilter_results, select_prefilter_shortlist
from src.matching.ranking import cap_prefilter_shortlist, partition_ranked_matches
from src.matching.scoring import score_shortlisted_jobs
from src.models import (
    CandidateProfile,
    JobPosting,
    LocalPrefilterResult,
    MatchResult,
    RunReport,
    UsageSummary,
)
from src.storage import create_run_dir, write_run_artifacts


class PipelineDependencies:
    def __init__(
        self,
        candidate_extractor: Callable[[str], CandidateProfile | dict],
        job_loader: Callable[[list[str]], Iterable[dict[str, Any]]],
        prefilter_runner: Callable[
            [CandidateProfile, list[JobPosting], list[str]],
            dict[str, Iterable[LocalPrefilterResult]],
        ],
        scorer: Callable[[CandidateProfile, JobPosting], MatchResult | dict],
    ):
        self.candidate_extractor = candidate_extractor
        self.job_loader = job_loader
        self.prefilter_runner = prefilter_runner
        self.scorer = scorer


def run_workflow(
    resume_path: str | Path,
    manual_links_path: str | Path | None,
    dependencies: PipelineDependencies,
    settings: Settings,
    job_urls: Iterable[str] | None = None,
    ignore_links: Iterable[str] | None = None,
    top_n: int = 10,
    timestamp: str | None = None,
) -> tuple[RunReport, Path]:
    """Run the full local workflow with injectable external dependencies."""
    print(f"Resume: {resume_path}")
    resume_text = load_resume_text(resume_path)
    candidate = CandidateProfile.model_validate(dependencies.candidate_extractor(resume_text))

    if manual_links_path is not None:
        links = load_job_links(manual_links_path)
    else:
        links = list(dict.fromkeys(job_urls or []))
        if not links:
            raise ValueError("at least one job URL is required")

    print(f"Fetching {len(links)} job(s)...")
    jobs = normalize_job_records(
        dependencies.job_loader(links),
        ignored_links=ignore_links,
    )

    print(f"Running prefilter on {len(jobs)} job(s)...")
    model_results = dependencies.prefilter_runner(
        candidate,
        jobs,
        settings.local_prefilter_models,
    )
    aggregated = aggregate_prefilter_results(model_results)
    eligible = select_prefilter_shortlist(
        aggregated,
        score_threshold=settings.skip_paid_llm_if_local_score_below,
    )
    capped = cap_prefilter_shortlist(eligible, max_jobs=settings.max_paid_llm_jobs)
    jobs_by_link = {str(job.link): job for job in jobs}
    shortlisted_jobs = [jobs_by_link[str(result.job_link)] for result in capped]
    print(f"Shortlisted {len(shortlisted_jobs)} of {len(jobs)} job(s) for scoring.")

    print(f"Scoring {len(shortlisted_jobs)} job(s)...")
    scored = score_shortlisted_jobs(candidate, shortlisted_jobs, dependencies.scorer)
    top_matches, remaining_ranked_jobs = partition_ranked_matches(scored, top_n=top_n)
    report = RunReport(
        candidate_profile=candidate,
        searched_jobs=jobs,
        top_matches=top_matches,
        remaining_ranked_jobs=remaining_ranked_jobs,
        usage=UsageSummary(
            local_prefilter_models=settings.local_prefilter_models,
            shortlist_scoring_provider=settings.shortlist_scoring_provider,
            shortlist_scoring_model=settings.shortlist_scoring_model,
            searched_job_count=len(jobs),
            prefiltered_job_count=len(aggregated),
            shortlist_job_count=len(shortlisted_jobs),
            scored_job_count=len(scored),
        ),
    )

    print("Writing run artifacts...")
    run_dir = create_run_dir(settings.runs_dir, timestamp=timestamp)
    write_run_artifacts(run_dir, report)
    return report, run_dir


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Load run parameters from a TOML config file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    for key in ("job_url", "ignore_link"):
        if key in raw and isinstance(raw[key], str):
            raw[key] = [raw[key]]
    return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local job-matching workflow.")
    parser.add_argument("--config", help="Path to a TOML run config file.")
    parser.add_argument("--resume", help="Path to a resume file (PDF, DOCX, or TXT).")
    parser.add_argument(
        "--manual-links",
        help="Path to a newline-delimited local job-link file.",
    )
    parser.add_argument(
        "--job-url",
        action="append",
        default=[],
        help="Direct public job URL to load from the web; repeat for multiple values.",
    )
    parser.add_argument(
        "--search-url",
        help="LinkedIn search URL; scrapes all pages automatically.",
    )
    parser.add_argument(
        "--scrape-only",
        action="store_true",
        default=False,
        help="Scrape job URLs from --search-url, save to a file, and exit without matching.",
    )
    parser.add_argument(
        "--jobs-file",
        help="Path to saved structured job records JSON for executable manual-link runs.",
    )
    parser.add_argument(
        "--candidate-profile-file",
        help="Optional saved CandidateProfile JSON to bypass resume extraction.",
    )
    parser.add_argument(
        "--ignore-link",
        action="append",
        default=[],
        help="Job link to ignore; repeat for multiple values.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Number of top matches.")
    return parser


def main() -> None:
    # Pre-parse only --config so its values can become parser defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    pre_args, _ = pre.parse_known_args()

    parser = build_parser()
    if pre_args.config:
        parser.set_defaults(**load_run_config(pre_args.config))

    args = parser.parse_args()

    if args.scrape_only:
        if not args.search_url:
            parser.error("--scrape-only requires --search-url")
        if args.manual_links or args.job_url:
            parser.error("--scrape-only cannot be combined with --manual-links or --job-url")

        import tempfile

        from dotenv import load_dotenv

        from src.ingest.scraping import LinkedInSearchScraper

        load_dotenv()
        settings = Settings.from_env()

        print(f"Scraping: {args.search_url}")
        scraper = LinkedInSearchScraper(
            cookie_file=settings.linkedin_cookie_file,
            headless=settings.linkedin_headless,
            max_pages=settings.linkedin_max_pages,
        )
        job_urls = scraper.collect_job_links(args.search_url)
        print(f"Collected {len(job_urls)} job URL(s).")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_job_links.txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("\n".join(job_urls) + "\n")
            tmp_path = f.name

        print(f"Saved to: {tmp_path}")
        print(f"Use with: --manual-links {tmp_path}")
        return

    if not args.resume:
        parser.error("--resume is required (or set resume in the config file)")

    if args.search_url and (args.manual_links or args.job_url):
        parser.error("--search-url cannot be combined with --manual-links or --job-url")
    if not args.search_url and not args.manual_links and not args.job_url:
        parser.error("provide one of: --search-url, --manual-links, or --job-url")

    from dotenv import load_dotenv
    from src.runtime import build_ollama_dependencies

    load_dotenv()
    settings = Settings.from_env()

    manual_links_path: str | None = args.manual_links or None
    job_urls: list[str] = list(args.job_url)

    if args.search_url:
        import tempfile

        from src.ingest.scraping import LinkedInSearchScraper

        print(f"Scraping: {args.search_url}")
        scraper = LinkedInSearchScraper(
            cookie_file=settings.linkedin_cookie_file,
            headless=settings.linkedin_headless,
            max_pages=settings.linkedin_max_pages,
        )
        job_urls = scraper.collect_job_links(args.search_url)
        print(f"Collected {len(job_urls)} job URL(s).")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_job_links.txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("\n".join(job_urls) + "\n")
            scraped_links_path = f.name
        print(f"Saved scraped links to: {scraped_links_path}")

    from src.store import open_db, backup_db

    store_conn = open_db(settings.storage_db_path)

    dependencies = build_ollama_dependencies(
        settings,
        args.jobs_file,
        candidate_profile_file=args.candidate_profile_file,
        store_conn=store_conn,
    )
    _report, run_dir = run_workflow(
        resume_path=args.resume,
        manual_links_path=manual_links_path,
        dependencies=dependencies,
        settings=settings,
        job_urls=job_urls,
        ignore_links=args.ignore_link,
        top_n=args.top_n,
    )

    try:
        dest = backup_db(store_conn, settings.backup_db_path)
        print(f"Store backed up to: {dest}")
    except Exception as exc:
        print(f"Store backup failed (run still succeeded): {exc}")
    finally:
        store_conn.close()

    print(run_dir)


if __name__ == "__main__":
    main()
