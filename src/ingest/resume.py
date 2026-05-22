from pathlib import Path
from typing import Protocol

from src.models import CandidateProfile


class CandidateProfileExtractor(Protocol):
    def invoke(self, resume_text: str) -> CandidateProfile | dict:
        ...


def load_resume_text(path: str | Path) -> str:
    """Load resume text from supported local file formats."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return _load_pdf_text(file_path)
    if suffix == ".docx":
        return _load_docx_text(file_path)

    raise ValueError(f"unsupported resume file type: {suffix or '<none>'}")


def extract_candidate_profile(
    resume_text: str,
    extractor: CandidateProfileExtractor,
) -> CandidateProfile:
    """Run an injectable structured extractor and validate its output."""
    extracted = extractor.invoke(resume_text)
    return CandidateProfile.model_validate(extracted)


def _load_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF loading requires the 'pypdf' package") from exc

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _load_docx_text(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("DOCX loading requires the 'python-docx' package") from exc

    document = Document(path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
