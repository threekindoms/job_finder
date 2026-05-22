from pathlib import Path

from pydantic import HttpUrl, TypeAdapter


_http_url_adapter = TypeAdapter(HttpUrl)


def load_job_links(path: str | Path) -> list[str]:
    """Load unique job links from a newline-delimited UTF-8 text file."""
    file_path = Path(path)
    raw_lines = file_path.read_text(encoding="utf-8").splitlines()

    links: list[str] = []
    seen: set[str] = set()
    for raw_line in raw_lines:
        link = raw_line.strip()
        if not link:
            continue

        validated_link = str(_http_url_adapter.validate_python(link))
        if validated_link in seen:
            continue

        seen.add(validated_link)
        links.append(validated_link)

    if not links:
        raise ValueError("manual job-link file contains no valid job URLs")

    return links
