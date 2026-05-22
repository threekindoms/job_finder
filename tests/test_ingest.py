import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.ingest.jobs import company_matches, extract_job_posting, normalize_job_records
from src.ingest.manual_links import load_job_links
from src.ingest.resume import extract_candidate_profile, load_resume_text
from src.ingest.scraping import LinkedInSearchScraper


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_json(name: str):
    return json.loads((FIXTURES_DIR / name).read_text())


def test_load_job_links_deduplicates_and_ignores_blank_lines():
    links = load_job_links(FIXTURES_DIR / "job_links.txt")

    assert links == [
        "https://example.com/jobs/backend-engineer",
        "https://example.com/jobs/software-engineer-ii",
    ]


def test_load_job_links_rejects_empty_files(tmp_path):
    path = tmp_path / "empty_links.txt"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no valid job URLs"):
        load_job_links(path)


def test_load_job_links_rejects_invalid_urls(tmp_path):
    path = tmp_path / "invalid_links.txt"
    path.write_text("not-a-url\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_job_links(path)


def test_normalize_job_records_deduplicates_and_filters_ignored_jobs():
    jobs, _ = normalize_job_records(
        load_json("jobs.json"),
        ignored_links=["https://example.com/jobs/ignored-job"],
    )

    assert [job.title for job in jobs] == [
        "Backend Software Engineer",
        "Software Engineer II",
        "Platform Engineer",
    ]


def _job_record(title: str, company: str, slug: str) -> dict:
    return {
        "title": title,
        "company": company,
        "link": f"https://example.com/jobs/{slug}",
        "description": slug,
    }


def test_company_matches_exact():
    assert company_matches("Google", "Google") is True


def test_company_matches_strips_legal_suffix():
    assert company_matches("Google LLC", "Google") is True
    assert company_matches("Google", "Google, Inc.") is True


def test_company_matches_case_insensitive():
    assert company_matches("AMAZON", "amazon") is True


def test_company_matches_contained_name():
    assert company_matches("Amazon Web Services", "Amazon") is True
    assert company_matches("Meta", "Meta Platforms") is True


def test_company_matches_rejects_unrelated():
    assert company_matches("Microsoft", "Google") is False


def test_company_matches_empty_strings():
    assert company_matches("", "Google") is False
    assert company_matches("Google", "") is False


def test_normalize_job_records_excludes_ignored_company():
    records = [
        _job_record("SWE", "Google LLC", "g"),
        _job_record("SWE", "Amazon", "a"),
        _job_record("SWE", "Acme Corp", "acme"),
    ]
    jobs, excluded = normalize_job_records(records, ignored_companies=["Google", "Amazon"])

    assert [job.company for job in jobs] == ["Acme Corp"]
    assert len(excluded) == 2
    assert {job.company for job in excluded} == {"Google LLC", "Amazon"}


def test_normalize_job_records_keeps_jobs_with_no_company():
    records = [
        _job_record("SWE", "", "no-company"),
        _job_record("SWE", "Google LLC", "google"),
    ]
    jobs, excluded = normalize_job_records(records, ignored_companies=["Google"])

    assert len(jobs) == 1
    assert jobs[0].title == "SWE"
    assert str(jobs[0].link) == "https://example.com/jobs/no-company"
    assert len(excluded) == 1
    assert excluded[0].company == "Google LLC"


def test_normalize_job_records_rejects_malformed_jobs():
    with pytest.raises(ValidationError):
        normalize_job_records([load_json("malformed_job.json")])  # type: ignore[call-overload]


def test_extract_job_posting_uses_injected_structured_extractor():
    class FakeExtractor:
        def invoke(self, raw_job_text: str):
            assert "Build backend services" in raw_job_text
            return load_json("jobs.json")[0]

    job = extract_job_posting(
        "Build backend services in Java and Python.",
        FakeExtractor(),
    )

    assert job.title == "Backend Software Engineer"


def test_scraper_exposes_linkedin_search_scraper():
    assert LinkedInSearchScraper is not None


def test_load_resume_text_supports_plain_text_fixture():
    text = load_resume_text(FIXTURES_DIR / "resume.txt")

    assert "Software engineer with 6 years of Java experience" in text


def test_load_resume_text_rejects_unsupported_file_types(tmp_path):
    path = tmp_path / "resume.md"
    path.write_text("resume", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported resume file type"):
        load_resume_text(path)


def test_load_resume_text_supports_pdf_inputs(monkeypatch, tmp_path):
    path = tmp_path / "resume.pdf"
    path.write_bytes(b"%PDF")

    fake_module = SimpleNamespace(
        PdfReader=lambda _: SimpleNamespace(
            pages=[
                SimpleNamespace(extract_text=lambda: "Page one"),
                SimpleNamespace(extract_text=lambda: "Page two"),
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "pypdf", fake_module)

    assert load_resume_text(path) == "Page one\nPage two"


def test_load_resume_text_supports_docx_inputs(monkeypatch, tmp_path):
    path = tmp_path / "resume.docx"
    path.write_bytes(b"docx")

    fake_module = SimpleNamespace(
        Document=lambda _: SimpleNamespace(
            paragraphs=[
                SimpleNamespace(text="Paragraph one"),
                SimpleNamespace(text="Paragraph two"),
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "docx", fake_module)

    assert load_resume_text(path) == "Paragraph one\nParagraph two"


def test_extract_candidate_profile_uses_injected_structured_extractor():
    class FakeExtractor:
        def invoke(self, resume_text: str):
            assert "Software engineer" in resume_text
            return load_json("candidate_profile.json")

    profile = extract_candidate_profile(
        load_resume_text(FIXTURES_DIR / "resume.txt"),
        FakeExtractor(),
    )

    assert profile.skills == ["Java", "Python", "Go"]
