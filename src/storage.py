import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models import RunReport
from src.reporting import build_artifacts


def create_run_dir(runs_dir: str | Path, timestamp: str | None = None) -> Path:
    """Create and return a timestamped run directory."""
    timestamp_value = timestamp or datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = Path(runs_dir) / timestamp_value
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_run_artifacts(run_dir: str | Path, report: RunReport) -> dict[str, Path]:
    """Write deterministic run artifacts and return their paths."""
    directory = Path(run_dir)
    artifacts = build_artifacts(report)

    paths = {
        "candidate_profile": directory / "candidate_profile.json",
        "jobs": directory / "jobs.json",
        "top_matches": directory / "top_matches.json",
        "remaining_ranked_jobs": directory / "remaining_ranked_jobs.json",
        "summary": directory / "summary.json",
        "usage": directory / "usage.json",
    }

    if artifacts["candidate_profile"] is not None:
        write_json(paths["candidate_profile"], artifacts["candidate_profile"])
    else:
        paths.pop("candidate_profile")
    write_json(paths["jobs"], artifacts["jobs"])
    write_json(paths["top_matches"], artifacts["top_matches"])
    write_json(paths["remaining_ranked_jobs"], artifacts["remaining_ranked_jobs"])
    write_json(paths["summary"], artifacts["summary"])
    if artifacts["usage"] is not None:
        write_json(paths["usage"], artifacts["usage"])
    else:
        paths.pop("usage")

    return paths
