from unittest.mock import MagicMock, patch

import pytest

from src.ingest.scraping import (
    LinkedInSearchScraper,
    _RESULTS_PER_PAGE,
    _build_page_url,
    _normalize_job_url,
)


# ── _build_page_url ────────────────────────────────────────────────────────────

def test_build_page_url_page1_sets_start_zero():
    url = _build_page_url("https://www.linkedin.com/jobs/search/?keywords=engineer", 1)
    assert "start=0" in url


def test_build_page_url_page2_sets_start_25():
    url = _build_page_url("https://www.linkedin.com/jobs/search/?keywords=engineer", 2)
    assert "start=25" in url


def test_build_page_url_page3_sets_start_50():
    url = _build_page_url("https://www.linkedin.com/jobs/search/?keywords=engineer", 3)
    assert "start=50" in url


def test_build_page_url_strips_current_job_id():
    url = _build_page_url(
        "https://www.linkedin.com/jobs/search/?keywords=engineer&currentJobId=123456",
        1,
    )
    assert "currentJobId" not in url


def test_build_page_url_overrides_existing_start_param():
    url = _build_page_url(
        "https://www.linkedin.com/jobs/search/?start=99&keywords=engineer",
        2,
    )
    assert "start=25" in url
    assert "start=99" not in url


def test_build_page_url_preserves_other_params():
    url = _build_page_url(
        "https://www.linkedin.com/jobs/search/?keywords=engineer&location=Seattle",
        1,
    )
    assert "keywords=engineer" in url
    assert "location=Seattle" in url


# ── _normalize_job_url ─────────────────────────────────────────────────────────

def test_normalize_job_url_absolute_url():
    url = _normalize_job_url("https://www.linkedin.com/jobs/view/1234567890/")
    assert url == "https://www.linkedin.com/jobs/view/1234567890/"


def test_normalize_job_url_relative_path():
    url = _normalize_job_url("/jobs/view/9876543210/?trackingId=abc")
    assert url == "https://www.linkedin.com/jobs/view/9876543210/"


def test_normalize_job_url_strips_query_params():
    url = _normalize_job_url("https://www.linkedin.com/jobs/view/1234567890/?foo=bar")
    assert url == "https://www.linkedin.com/jobs/view/1234567890/"


def test_normalize_job_url_adds_trailing_slash():
    url = _normalize_job_url("https://www.linkedin.com/jobs/view/1234567890")
    assert url == "https://www.linkedin.com/jobs/view/1234567890/"


def test_normalize_job_url_profile_link_returns_none():
    assert _normalize_job_url("https://www.linkedin.com/in/someprofile") is None


def test_normalize_job_url_empty_string_returns_none():
    assert _normalize_job_url("") is None


def test_normalize_job_url_non_job_path_returns_none():
    assert _normalize_job_url("/company/some-company/jobs/") is None


# ── LinkedInSearchScraper._extract_job_urls ────────────────────────────────────

def _make_card(href: str) -> MagicMock:
    link_el = MagicMock()
    link_el.get_attribute.return_value = href
    card = MagicMock()
    card.query_selector.return_value = link_el
    return card


def _make_card_no_link() -> MagicMock:
    card = MagicMock()
    card.query_selector.return_value = None
    return card


def test_extract_job_urls_returns_normalized_urls():
    scraper = LinkedInSearchScraper(cookie_file="dummy.json")
    page = MagicMock()
    page.query_selector_all.return_value = [
        _make_card("/jobs/view/1111111111/"),
        _make_card("/jobs/view/2222222222/"),
    ]
    urls = scraper._extract_job_urls(page, "li[data-occludable-job-id]")
    assert urls == [
        "https://www.linkedin.com/jobs/view/1111111111/",
        "https://www.linkedin.com/jobs/view/2222222222/",
    ]


def test_extract_job_urls_skips_cards_without_link():
    scraper = LinkedInSearchScraper(cookie_file="dummy.json")
    page = MagicMock()
    page.query_selector_all.return_value = [
        _make_card("/jobs/view/1111111111/"),
        _make_card_no_link(),
    ]
    urls = scraper._extract_job_urls(page, "li[data-occludable-job-id]")
    assert len(urls) == 1


def test_extract_job_urls_skips_non_job_hrefs():
    scraper = LinkedInSearchScraper(cookie_file="dummy.json")
    page = MagicMock()
    page.query_selector_all.return_value = [_make_card("/in/some-profile")]
    urls = scraper._extract_job_urls(page, "li[data-occludable-job-id]")
    assert urls == []


# ── LinkedInSearchScraper.collect_job_links (mocked BrowserSession) ────────────

def _setup_mock_session(mock_bs_cls: MagicMock) -> tuple[MagicMock, MagicMock]:
    """Return (mock_session, mock_page) wired into the BrowserSession context manager."""
    mock_session = MagicMock()
    mock_bs_cls.return_value.__enter__.return_value = mock_session
    mock_bs_cls.return_value.__exit__.return_value = False
    mock_page = MagicMock()
    mock_session.page = mock_page
    mock_page.goto.return_value = None
    mock_page.query_selector.return_value = None   # no login redirect
    mock_page.evaluate.return_value = None          # no scrollable container
    mock_page.wait_for_timeout.return_value = None
    return mock_session, mock_page


@patch("src.ingest.scraping.BrowserSession")
def test_collect_job_links_raises_on_missing_cookie_file(mock_bs_cls):
    mock_session, _ = _setup_mock_session(mock_bs_cls)
    mock_session.load_cookies.side_effect = FileNotFoundError(
        "LinkedIn cookie file not found"
    )
    scraper = LinkedInSearchScraper(cookie_file="nonexistent.json", max_pages=1)
    with pytest.raises(FileNotFoundError, match="cookie file"):
        scraper.collect_job_links("https://www.linkedin.com/jobs/search/")


@patch("src.ingest.scraping.BrowserSession")
def test_collect_job_links_raises_on_login_redirect(mock_bs_cls):
    mock_session, mock_page = _setup_mock_session(mock_bs_cls)
    mock_page.query_selector.return_value = MagicMock()  # login input found
    scraper = LinkedInSearchScraper(cookie_file="cookies.json", max_pages=1)
    with pytest.raises(RuntimeError, match="login"):
        scraper.collect_job_links("https://www.linkedin.com/jobs/search/")


@patch("src.ingest.scraping.BrowserSession")
def test_collect_job_links_raises_on_no_cards_page_1(mock_bs_cls):
    mock_session, mock_page = _setup_mock_session(mock_bs_cls)
    mock_page.wait_for_selector.side_effect = Exception("Timeout waiting for selector")
    scraper = LinkedInSearchScraper(cookie_file="cookies.json", max_pages=1)
    with pytest.raises(RuntimeError, match="page 1"):
        scraper.collect_job_links("https://www.linkedin.com/jobs/search/")


@patch("src.ingest.scraping.BrowserSession")
def test_collect_job_links_stops_on_partial_last_page(mock_bs_cls):
    mock_session, mock_page = _setup_mock_session(mock_bs_cls)
    mock_page.wait_for_selector.return_value = MagicMock()
    cards = [_make_card(f"/jobs/view/{i:010d}/") for i in range(10)]
    mock_page.query_selector_all.return_value = cards

    scraper = LinkedInSearchScraper(cookie_file="cookies.json", max_pages=5)
    links = scraper.collect_job_links("https://www.linkedin.com/jobs/search/")

    assert len(links) == 10
    assert mock_page.goto.call_count == 1  # stopped after page 1


@patch("src.ingest.scraping.BrowserSession")
def test_collect_job_links_paginates_full_pages(mock_bs_cls):
    mock_session, mock_page = _setup_mock_session(mock_bs_cls)
    mock_page.wait_for_selector.return_value = MagicMock()

    page_results = [
        [_make_card(f"/jobs/view/{i:010d}/") for i in range(_RESULTS_PER_PAGE)],  # full
        [_make_card(f"/jobs/view/{i:010d}/") for i in range(_RESULTS_PER_PAGE, _RESULTS_PER_PAGE + 5)],  # partial
    ]
    call_count = [0]

    def query_selector_all_side_effect(selector):
        result = page_results[call_count[0]]
        call_count[0] += 1
        return result

    mock_page.query_selector_all.side_effect = query_selector_all_side_effect

    scraper = LinkedInSearchScraper(cookie_file="cookies.json", max_pages=5)
    links = scraper.collect_job_links("https://www.linkedin.com/jobs/search/")

    assert len(links) == _RESULTS_PER_PAGE + 5
    assert mock_page.goto.call_count == 2


@patch("src.ingest.scraping.BrowserSession")
def test_collect_job_links_deduplicates_across_pages(mock_bs_cls):
    mock_session, mock_page = _setup_mock_session(mock_bs_cls)
    mock_page.wait_for_selector.return_value = MagicMock()

    # Page 1: full page; page 2: one duplicate + one new
    page1 = [_make_card(f"/jobs/view/{i:010d}/") for i in range(_RESULTS_PER_PAGE)]
    page2 = [
        _make_card("/jobs/view/0000000000/"),  # duplicate of page1[0]
        _make_card(f"/jobs/view/{_RESULTS_PER_PAGE:010d}/"),  # new
    ]
    call_count = [0]

    def side_effect(selector):
        result = page1 if call_count[0] == 0 else page2
        call_count[0] += 1
        return result

    mock_page.query_selector_all.side_effect = side_effect

    scraper = LinkedInSearchScraper(cookie_file="cookies.json", max_pages=5)
    links = scraper.collect_job_links("https://www.linkedin.com/jobs/search/")

    assert len(links) == _RESULTS_PER_PAGE + 1  # 25 unique from page1 + 1 new from page2
