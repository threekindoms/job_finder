import json
import random
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class BrowserSession:
    """Context manager wrapping a single Playwright browser session."""

    def __init__(
        self,
        headless: bool = True,
        user_agent: str = _DEFAULT_USER_AGENT,
        viewport: tuple[int, int] = (1920, 1080),
    ) -> None:
        self._headless = headless
        self._user_agent = user_agent
        self._viewport = viewport
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    def __enter__(self) -> "BrowserSession":
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self._headless)
        w, h = self._viewport
        self._context = self._browser.new_context(
            user_agent=self._user_agent,
            viewport={"width": w, "height": h},
        )
        self._page = self._context.new_page()
        return self

    def __exit__(self, *_: Any) -> None:
        for obj in (self._page, self._context, self._browser):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("BrowserSession not started; use as a context manager")
        return self._page

    def load_cookies(self, cookie_file: Path) -> None:
        """Add saved LinkedIn session cookies to the browser context."""
        if not cookie_file.exists():
            raise FileNotFoundError(
                f"LinkedIn cookie file not found: {cookie_file}\n"
                "Run 'python tools/capture_cookies.py' to capture your LinkedIn session."
            )
        cookies = json.loads(cookie_file.read_text(encoding="utf-8"))
        assert self._context is not None
        self._context.add_cookies(cookies)

    @staticmethod
    def human_delay(min_sec: float = 5.0, max_sec: float = 10.0) -> None:
        time.sleep(random.uniform(min_sec, max_sec))
