import logging
import re
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from src.ingest.scraping_browser import BrowserSession

logger = logging.getLogger(__name__)

_RESULTS_PER_PAGE = 25

_JOB_CARD_SELECTORS = [
    "li[data-occludable-job-id]",
    "li.jobs-search-results__list-item",
]
_TITLE_LINK_SELECTORS = [
    "a.job-card-list__title--link",
    "a.job-card-list__title",
    "a.job-card-container__link",
]
_LOGIN_SELECTOR = 'input[name="session_key"]'


class JobScraper(Protocol):
    def collect_job_links(self, search_url: str) -> list[str]:
        ...


def _build_page_url(search_url: str, page_num: int) -> str:
    """Construct the paginated URL for page_num (1-based)."""
    parsed = urlparse(search_url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    params.pop("currentJobId", None)
    params["start"] = [str((page_num - 1) * _RESULTS_PER_PAGE)]
    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        urlencode(params, doseq=True),
        parsed.fragment,
    ))


def _normalize_job_url(href: str) -> str | None:
    """Return a canonical LinkedIn job URL, or None if href is not a job link."""
    if not href:
        return None
    url = href if href.startswith("http") else f"https://www.linkedin.com{href}"
    parsed = urlparse(url)
    if not re.search(r"/jobs/view/\d+", parsed.path):
        return None
    canonical_path = parsed.path.rstrip("/") + "/"
    return urlunparse((parsed.scheme, parsed.netloc, canonical_path, "", "", ""))


class LinkedInSearchScraper:
    """Scrapes job URLs from LinkedIn search result pages using Playwright."""

    def __init__(
        self,
        cookie_file: str | Path,
        headless: bool = True,
        max_pages: int = 20,
    ) -> None:
        self._cookie_file = Path(cookie_file)
        self._headless = headless
        self._max_pages = max_pages

    def collect_job_links(self, search_url: str) -> list[str]:
        """Return deduplicated job URLs collected from all search result pages."""
        collected: list[str] = []
        seen: set[str] = set()

        with BrowserSession(headless=self._headless) as session:
            session.load_cookies(self._cookie_file)
            page = session.page

            for page_num in range(1, self._max_pages + 1):
                url = _build_page_url(search_url, page_num)
                logger.info("Scraping page %d: %s", page_num, url)

                try:
                    page.goto(url, timeout=60_000, wait_until="domcontentloaded")
                except Exception as exc:
                    # Timeout here is not fatal — domcontentloaded may fire late on slow
                    # connections; _wait_for_cards below is the real readiness gate.
                    logger.warning("goto timeout on page %d (%s); continuing.", page_num, exc)

                if page.query_selector(_LOGIN_SELECTOR):
                    raise RuntimeError(
                        "LinkedIn redirected to login; cookies may have expired.\n"
                        "Run 'python tools/capture_cookies.py' to refresh your session, "
                        "then retry. Use --manual-links as a fallback."
                    )

                card_selector = self._wait_for_cards(page)
                if card_selector is None:
                    if page_num == 1:
                        raise RuntimeError(
                            "No job cards found on page 1. The search URL may be invalid "
                            "or LinkedIn is blocking access.\n"
                            "Use --manual-links as a fallback."
                        )
                    logger.info("No job cards on page %d; reached end of results.", page_num)
                    break

                self._scroll_job_list(page, card_selector)
                page_urls = self._extract_job_urls(page, card_selector)
                logger.info("  Page %d: %d URL(s) extracted", page_num, len(page_urls))

                for u in page_urls:
                    if u not in seen:
                        seen.add(u)
                        collected.append(u)
                        logger.info("    + %s", u)
                        print(f"  + {u}")

                if len(page_urls) < _RESULTS_PER_PAGE:
                    logger.info(
                        "  Page %d returned %d jobs (<%d); treating as last page.",
                        page_num,
                        len(page_urls),
                        _RESULTS_PER_PAGE,
                    )
                    break

                if page_num < self._max_pages:
                    session.human_delay()

        if not collected:
            raise RuntimeError(
                "Scraping finished but no job URLs were collected. "
                "Check the search URL and retry, or use --manual-links."
            )

        logger.info("Collected %d unique job URL(s).", len(collected))
        return collected

    def _wait_for_cards(self, page) -> str | None:
        """Wait for job cards to load; return the working selector or None."""
        for selector in _JOB_CARD_SELECTORS:
            try:
                page.wait_for_selector(selector, timeout=45_000)
                return selector
            except Exception:
                continue
        return None

    def _scroll_job_list(self, page, card_selector: str) -> None:
        """Scroll the job list to trigger lazy loading of all cards.

        Walks up the DOM from the first job card to find its scrollable ancestor
        so this works regardless of LinkedIn's container class names.
        """
        # Let the initial render settle before scrolling.
        page.wait_for_timeout(3000)

        # JS that scrolls 300 px inside the scrollable ancestor of the first card.
        # Returns {done: true} when we can no longer scroll (bottom reached).
        scroll_step_js = """
            (cardSelector) => {
                const card = document.querySelector(cardSelector);
                if (!card) return {done: true};
                let el = card.parentElement;
                while (el && el !== document.body) {
                    if (el.scrollHeight > el.clientHeight + 10) {
                        const before = el.scrollTop;
                        el.scrollTop += 300;
                        return {done: el.scrollTop === before && before > 0};
                    }
                    el = el.parentElement;
                }
                window.scrollBy(0, 300);
                return {done: false};
            }
        """

        for _ in range(30):
            try:
                result = page.evaluate(scroll_step_js, card_selector)
                page.wait_for_timeout(2500)
                if result and result.get("done"):
                    break
            except Exception:
                break

        # Final wait for LinkedIn to finish populating the last batch of cards.
        page.wait_for_timeout(5000)

    def _extract_job_urls(self, page, card_selector: str) -> list[str]:
        """Extract job URLs from loaded job cards."""
        urls: list[str] = []
        for card in page.query_selector_all(card_selector):
            for link_sel in _TITLE_LINK_SELECTORS:
                link_el = card.query_selector(link_sel)
                if link_el:
                    href = link_el.get_attribute("href") or ""
                    url = _normalize_job_url(href)
                    if url:
                        urls.append(url)
                        break
        return urls
