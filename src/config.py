import os
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl


class Settings(BaseModel):
    ollama_base_url: HttpUrl = "http://localhost:11434"
    local_prefilter_models: list[str] = Field(default_factory=list)
    ollama_cloud_base_url: HttpUrl = "https://api.ollama.com"
    ollama_cloud_api_key: str | None = None
    cloud_prefilter_models: list[str] = Field(default_factory=list)
    shortlist_scoring_provider: str = "ollama"
    shortlist_scoring_model: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    job_text_max_chars: int = Field(default=10_000, ge=1)
    max_paid_llm_jobs: int = Field(default=20, ge=1)
    skip_paid_llm_if_local_score_below: int = Field(default=80, ge=0, le=100)
    runs_dir: Path = Path("runs")
    linkedin_cookie_file: Path = Path("config/linkedin_cookies.json")
    linkedin_headless: bool = True
    linkedin_max_pages: int = Field(default=20, ge=1)
    storage_db_path: Path = Path("storage/store.db")
    backup_db_path: Path = Path.home() / "Documents/Storages/resume_matching"

    @classmethod
    def from_env(cls) -> "Settings":
        def _parse_model_list(env_var: str) -> list[str]:
            return [m.strip() for m in os.getenv(env_var, "").split(",") if m.strip()]

        # Resolve project-relative paths against the first PYTHONPATH entry so
        # the tool works correctly regardless of the working directory it is
        # invoked from (e.g. `PYTHONPATH=/path/to/project python -m src.cli`).
        pythonpath_root: Path | None = None
        raw_pythonpath = os.getenv("PYTHONPATH", "")
        if raw_pythonpath:
            first_entry = raw_pythonpath.split(":")[0].strip()
            if first_entry:
                pythonpath_root = Path(first_entry).resolve()

        def _abs(path: Path) -> Path:
            """Return path as-is if absolute; otherwise anchor it to PYTHONPATH root."""
            if path.is_absolute() or pythonpath_root is None:
                return path
            return pythonpath_root / path

        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            local_prefilter_models=_parse_model_list("LOCAL_PREFILTER_MODELS"),
            ollama_cloud_base_url=os.getenv("OLLAMA_CLOUD_BASE_URL", "https://api.ollama.com"),
            ollama_cloud_api_key=os.getenv("OLLAMA_CLOUD_API_KEY"),
            cloud_prefilter_models=_parse_model_list("CLOUD_PREFILTER_MODELS"),
            shortlist_scoring_provider=os.getenv("SHORTLIST_SCORING_PROVIDER", "ollama"),
            shortlist_scoring_model=os.getenv("SHORTLIST_SCORING_MODEL"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            job_text_max_chars=int(os.getenv("JOB_TEXT_MAX_CHARS", "10000")),
            max_paid_llm_jobs=int(os.getenv("MAX_PAID_LLM_JOBS", "20")),
            skip_paid_llm_if_local_score_below=int(
                os.getenv("SKIP_PAID_LLM_IF_LOCAL_SCORE_BELOW", "80")
            ),
            runs_dir=_abs(Path(os.getenv("RUNS_DIR", "runs"))),
            linkedin_cookie_file=_abs(
                Path(os.getenv("LINKEDIN_COOKIE_FILE", "config/linkedin_cookies.json"))
            ),
            linkedin_headless=os.getenv("LINKEDIN_HEADLESS", "true").lower()
            not in ("false", "0", "no"),
            linkedin_max_pages=int(os.getenv("LINKEDIN_MAX_PAGES", "20")),
            storage_db_path=_abs(Path(os.getenv("STORAGE_DB_PATH", "storage/store.db"))),
            backup_db_path=Path(
                os.getenv("BACKUP_DB_PATH", str(Path.home() / "Documents/Storages/resume_matching"))
            ),
        )
