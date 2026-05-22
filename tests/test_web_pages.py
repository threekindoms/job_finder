from src.ingest.web_pages import fetch_public_job_page_text


def test_fetch_public_job_page_text_prefers_jobposting_json_ld(monkeypatch):
    html = b"""
    <html>
      <script type="application/ld+json">
        {
          "@type": "JobPosting",
          "title": "Backend Engineer",
          "description": "<p>Build <strong>Python</strong> services. We are looking for a backend engineer to design and implement scalable Python microservices, RESTful APIs, and data pipelines. You will collaborate with cross-functional teams to deliver high-quality software, participate in code reviews, and help shape our technical roadmap. Strong proficiency in Python, SQL, and distributed systems is required. Experience with FastAPI, PostgreSQL, and Kubernetes is a plus.</p>"
        }
      </script>
    </html>
    """

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return html

    monkeypatch.setattr("src.ingest.web_pages.urlopen", lambda *_args, **_kwargs: FakeResponse())

    text = fetch_public_job_page_text("https://example.com/jobs/backend")

    assert "Source URL: https://example.com/jobs/backend" in text
    assert "Title: Backend Engineer" in text
    assert "Build" in text
    assert "Python" in text
