import sqlite3
from datetime import datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidate_profiles (
    resume_hash  TEXT PRIMARY KEY,
    extracted_at TEXT NOT NULL,
    profile_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job_postings (
    url          TEXT PRIMARY KEY,
    fetched_at   TEXT NOT NULL,
    posting_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prefilter_results (
    candidate_hash TEXT NOT NULL,
    job_url        TEXT NOT NULL,
    model_name     TEXT NOT NULL,
    filtered_at    TEXT NOT NULL,
    result_json    TEXT NOT NULL,
    PRIMARY KEY (candidate_hash, job_url, model_name)
);
CREATE TABLE IF NOT EXISTS match_results (
    candidate_hash TEXT NOT NULL,
    job_url        TEXT NOT NULL,
    scoring_model  TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    scored_at      TEXT NOT NULL,
    result_json    TEXT NOT NULL,
    PRIMARY KEY (candidate_hash, job_url, scoring_model, prompt_version)
);
"""


def open_db(path: str | Path) -> sqlite3.Connection:
    """Open the SQLite store at path (or ':memory:'), creating schema if needed."""
    p = str(path)
    if p != ":memory:":
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def backup_db(conn: sqlite3.Connection, backup_dir: Path) -> Path:
    """Copy the live database to a timestamped file in backup_dir."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"store_{datetime.now().strftime('%Y%m%dT%H%M%S')}.db"
    with sqlite3.connect(str(dest)) as dst:
        conn.backup(dst)
    return dest
