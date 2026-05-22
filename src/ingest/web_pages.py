import json
import re
from html import unescape
from html.parser import HTMLParser
from urllib.request import Request, urlopen


class _TextExtractor(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        stripped = data.strip()
        if stripped:
            self.parts.append(stripped)


_MIN_DESCRIPTION_CHARS = 300


def _strip_html(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(unescape(value))
    return "\n".join(parser.parts)


def fetch_public_job_page_text(url: str) -> str:
    """Fetch a public job page and extract usable text, preferring JSON-LD."""
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=20) as response:
        html = response.read().decode("utf-8", errors="replace")

    for payload in re.findall(
        r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        parsed = json.loads(unescape(payload).strip())
        candidates = parsed if isinstance(parsed, list) else [parsed]
        for candidate in candidates:
            if candidate.get("@type") != "JobPosting":
                continue
            title = candidate.get("title", "").strip()
            description = _strip_html(candidate.get("description", ""))
            if len(description) >= _MIN_DESCRIPTION_CHARS:
                return f"Source URL: {url}\nTitle: {title}\n\n{description}".strip()

    return _strip_html(html)
