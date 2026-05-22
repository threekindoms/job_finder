import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate

from src.cli import PipelineDependencies
from src.config import Settings
from src.ingest.jobs import extract_job_posting
from src.ingest.web_pages import fetch_public_job_page_text
from src.models import CandidateProfile, JobPosting, LocalPrefilterResult, MatchResult
from src.store import (
    get_candidate,
    get_job,
    get_match,
    get_prefilter,
    put_candidate,
    put_job,
    put_match,
    put_prefilter,
    resume_hash,
    scoring_prompt_version,
)
from src.prompts import (
    CANDIDATE_PROFILE_HUMAN_PROMPT,
    CANDIDATE_PROFILE_SYSTEM_PROMPT,
    JOB_POSTING_HUMAN_PROMPT,
    JOB_POSTING_SYSTEM_PROMPT,
    LOCAL_PREFILTER_HUMAN_PROMPT,
    LOCAL_PREFILTER_SYSTEM_PROMPT,
    SHORTLIST_SCORING_HUMAN_PROMPT,
    SHORTLIST_SCORING_SYSTEM_PROMPT_CLOUD,
    SHORTLIST_SCORING_SYSTEM_PROMPT_LOCAL,
)


def _build_cloud_ollama(
    settings: Settings,
    model: str,
    format: str | None = None,
) -> ChatOllama:
    """Return a ChatOllama pointed at the configured cloud endpoint."""
    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": str(settings.ollama_cloud_base_url),
        "temperature": 0,
    }
    if format:
        kwargs["format"] = format
    if settings.ollama_cloud_api_key:
        # client_kwargs is forwarded to the underlying httpx client; headers go in here.
        kwargs["client_kwargs"] = {
            "headers": {"Authorization": f"Bearer {settings.ollama_cloud_api_key}"}
        }
    return ChatOllama(**kwargs)


def _build_scoring_model(settings: Settings) -> Any:
    """Return a LangChain chat model for shortlist scoring based on configured provider."""
    provider = settings.shortlist_scoring_provider
    model = settings.shortlist_scoring_model
    if not model:
        raise ValueError("SHORTLIST_SCORING_MODEL is required")

    if provider == "ollama":
        return ChatOllama(
            model=model,
            base_url=str(settings.ollama_base_url),
            temperature=0,
        )
    if provider == "ollama_cloud":
        return _build_cloud_ollama(settings, model)
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("pip install langchain-anthropic to use the anthropic provider") from exc
        return ChatAnthropic(model=model, api_key=settings.anthropic_api_key, temperature=0)
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("pip install langchain-openai to use the openai provider") from exc
        return ChatOpenAI(model=model, api_key=settings.openai_api_key, temperature=0)
    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("pip install langchain-google-genai to use the google provider") from exc
        return ChatGoogleGenerativeAI(model=model, google_api_key=settings.google_api_key, temperature=0)

    raise ValueError(
        f"unknown shortlist scoring provider '{provider}'; "
        "expected one of: ollama, ollama_cloud, anthropic, openai, google"
    )


def load_saved_job_records(path: str | Path) -> list[dict[str, Any]]:
    """Load previously saved job records for the manual-link workflow."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 2.0


def _invoke_with_retry(fn: Any, label: str = "") -> Any:
    """Call fn(), retrying up to _RETRY_ATTEMPTS times on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < _RETRY_ATTEMPTS - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                prefix = f" ({label})" if label else ""
                print(f"    [transient error{prefix}, retry {attempt + 1}/{_RETRY_ATTEMPTS - 1} in {delay:.0f}s] {exc}")
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _parse_json_response(content: str) -> dict[str, Any]:
    """Extract a JSON object from a model response, tolerating common formatting issues.

    Handles: empty content, <think> blocks, markdown code fences, leading/trailing prose.
    """
    # Strip reasoning/thinking blocks produced by some models (e.g. deepseek-r1, ministral).
    text = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
    if not text:
        raise ValueError(
            f"model returned an empty response after stripping thinking tokens "
            f"(raw length: {len(content)})"
        )
    # Unwrap markdown code fences (```json ... ``` or ``` ... ```).
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    # Find the outermost JSON object in case the model added prose before/after.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


_PREFILTER_DESCRIPTION_MAX_CHARS = 3_000
_SCORING_DESCRIPTION_MAX_CHARS = 6_000


def _job_posting_for_prompt(job: JobPosting, max_description_chars: int) -> str:
    """Serialize a JobPosting for a model prompt with the description truncated."""
    data = job.model_dump(mode="json")
    if isinstance(data.get("description"), str):
        data["description"] = data["description"][:max_description_chars]
    return json.dumps(data)


def _extract_job_with_fallback(
    link: str,
    raw_text: str,
    chain: Any,
    max_chars: int,
) -> dict[str, Any]:
    """Extract a job posting, injecting link and description when the model omits them."""
    truncated = raw_text[:max_chars]
    raw = _invoke_with_retry(lambda: chain.invoke({"raw_job_text": truncated}), label=link)
    parsed: dict[str, Any] = _parse_json_response(raw.content)
    if not parsed.get("link"):
        parsed["link"] = link
    if not parsed.get("description"):
        parsed["description"] = truncated
    return parsed


def _invoke_prefilter(
    chain: Any,
    candidate_json: str,
    job: JobPosting,
) -> LocalPrefilterResult:
    """Invoke a prefilter chain and inject job_link when the model omits it."""
    raw = _invoke_with_retry(
        lambda: chain.invoke({
            "candidate_profile": candidate_json,
            "job_posting": _job_posting_for_prompt(job, _PREFILTER_DESCRIPTION_MAX_CHARS),
        }),
        label=str(job.link),
    )
    parsed: dict[str, Any] = _parse_json_response(raw.content)
    parsed["job_link"] = str(job.link)
    return LocalPrefilterResult.model_validate(parsed)


def build_ollama_dependencies(
    settings: Settings,
    jobs_file: str | Path | None,
    candidate_profile_file: str | Path | None = None,
    store_conn: sqlite3.Connection | None = None,
) -> PipelineDependencies:
    """Build concrete local-Ollama workflow dependencies."""
    _prefilter_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", LOCAL_PREFILTER_SYSTEM_PROMPT),
            ("human", LOCAL_PREFILTER_HUMAN_PROMPT),
        ]
    )
    prefilter_chains: dict[str, Any] = {
        model_name: (
            _prefilter_prompt
            | ChatOllama(
                model=model_name,
                base_url=str(settings.ollama_base_url),
                temperature=0,
                format="json",
            )
        )
        for model_name in settings.local_prefilter_models
    }
    for model_name in settings.cloud_prefilter_models:
        prefilter_chains[model_name] = (
            _prefilter_prompt | _build_cloud_ollama(settings, model_name, format="json")
        )

    shortlist_chat_model = _build_scoring_model(settings)
    scoring_system_prompt = (
        SHORTLIST_SCORING_SYSTEM_PROMPT_LOCAL
        if settings.shortlist_scoring_provider in ("ollama", "ollama_cloud")
        else SHORTLIST_SCORING_SYSTEM_PROMPT_CLOUD
    )
    shortlist_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", scoring_system_prompt),
                ("human", SHORTLIST_SCORING_HUMAN_PROMPT),
            ]
        )
        | shortlist_chat_model.with_structured_output(MatchResult, method="json_mode")
    )

    _scoring_model_id = settings.shortlist_scoring_model or ""
    _prompt_version = scoring_prompt_version(scoring_system_prompt)
    _state: dict[str, str] = {}  # shared between closures; holds "resume_hash"

    def candidate_extractor(resume_text: str) -> CandidateProfile:
        r_hash = resume_hash(resume_text)
        _state["resume_hash"] = r_hash

        if candidate_profile_file is not None:
            print(f"  Candidate profile: {candidate_profile_file}")
            profile = CandidateProfile.model_validate(
                json.loads(Path(candidate_profile_file).read_text(encoding="utf-8"))
            )
            if store_conn is not None:
                try:
                    put_candidate(store_conn, r_hash, profile)
                except Exception:
                    pass
            return profile

        if store_conn is not None:
            cached = get_candidate(store_conn, r_hash)
            if cached is not None:
                print("  Candidate profile: [cache hit]")
                return cached
        if settings.local_prefilter_models:
            model_name = settings.local_prefilter_models[0]
            chat_model = ChatOllama(
                model=model_name,
                base_url=str(settings.ollama_base_url),
                temperature=0,
            )
        elif settings.cloud_prefilter_models:
            model_name = settings.cloud_prefilter_models[0]
            chat_model = _build_cloud_ollama(settings, model_name)
        else:
            raise ValueError(
                "No prefilter models configured. "
                "Set LOCAL_PREFILTER_MODELS or CLOUD_PREFILTER_MODELS."
            )
        print(f"  Extracting candidate profile [{model_name}]...")
        candidate_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", CANDIDATE_PROFILE_SYSTEM_PROMPT),
                    ("human", CANDIDATE_PROFILE_HUMAN_PROMPT),
                ]
            )
            | chat_model.with_structured_output(CandidateProfile, method="json_mode")
        )
        profile = CandidateProfile.model_validate(candidate_chain.invoke({"resume_text": resume_text}))
        if store_conn is not None:
            try:
                put_candidate(store_conn, r_hash, profile)
            except Exception:
                pass
        return profile

    def job_loader(links: list[str]) -> list[dict[str, Any]]:
        if jobs_file is not None:
            print(f"  Saved jobs: {jobs_file}")
            records = load_saved_job_records(jobs_file)
            if store_conn is not None:
                for rec in records:
                    try:
                        put_job(store_conn, JobPosting.model_validate(rec))
                    except Exception:
                        pass
            return records

        if settings.local_prefilter_models:
            _job_chat = ChatOllama(
                model=settings.local_prefilter_models[0],
                base_url=str(settings.ollama_base_url),
                temperature=0,
                format="json",
            )
        elif settings.cloud_prefilter_models:
            _job_chat = _build_cloud_ollama(
                settings, settings.cloud_prefilter_models[0], format="json"
            )
        else:
            raise ValueError(
                "No prefilter models configured. "
                "Set LOCAL_PREFILTER_MODELS or CLOUD_PREFILTER_MODELS."
            )
        job_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", JOB_POSTING_SYSTEM_PROMPT),
                    ("human", JOB_POSTING_HUMAN_PROMPT),
                ]
            )
            | _job_chat
        )
        results = []
        n = len(links)
        for i, link in enumerate(links):
            print(f"  [{i + 1}/{n}] {link}")
            try:
                if store_conn is not None:
                    cached_job = get_job(store_conn, link)
                    if cached_job is not None:
                        print(f"        → {cached_job.title} [cached]")
                        results.append(cached_job.model_dump())
                        continue
                raw_text = fetch_public_job_page_text(link)
                parsed = _extract_job_with_fallback(link, raw_text, job_chain, settings.job_text_max_chars)
                print(f"        → {parsed.get('title') or '(no title)'}")
                job = JobPosting.model_validate(parsed)
                if store_conn is not None:
                    try:
                        put_job(store_conn, job)
                    except Exception:
                        pass
                results.append(job.model_dump())
            except Exception as exc:
                print(f"        DOWNLOAD_FAILED: {exc}")
        return results

    def prefilter_runner(
        candidate: CandidateProfile,
        jobs: list[JobPosting],
        _models: list[str],
    ) -> dict[str, list[LocalPrefilterResult]]:
        results: dict[str, list[LocalPrefilterResult]] = {}
        ordered_models = list(prefilter_chains.items())
        primary_models = ordered_models[:2]
        candidate_json = candidate.model_dump_json()
        r_hash = _state.get("resume_hash", "")
        for model_name, chain in primary_models:
            print(f"  [{model_name}]")
            model_results: list[LocalPrefilterResult] = []
            for job in jobs:
                try:
                    if store_conn is not None and r_hash:
                        cached = get_prefilter(store_conn, r_hash, str(job.link), model_name)
                        if cached is not None:
                            company_part = f" @ {job.company}" if job.company else ""
                            print(f"    [{cached.local_score:3d}] {job.title}{company_part} — {job.link} [cached]")
                            model_results.append(cached)
                            continue
                    result = _invoke_prefilter(chain, candidate_json, job)
                    company_part = f" @ {job.company}" if job.company else ""
                    print(f"    [{result.local_score:3d}] {job.title}{company_part} — {job.link}")
                    if store_conn is not None and r_hash:
                        try:
                            put_prefilter(store_conn, r_hash, str(job.link), model_name, result)
                        except Exception:
                            pass
                    model_results.append(result)
                except Exception as exc:
                    company_part = f" @ {job.company}" if job.company else ""
                    print(f"    PREFILTER_FAILED [{job.title}{company_part}]: {exc}")
            results[model_name] = model_results

        if len(ordered_models) < 3 or len(primary_models) < 2:
            return results

        first_model, second_model = primary_models
        first_by_link = {str(r.job_link): r for r in results[first_model[0]]}
        second_by_link = {str(r.job_link): r for r in results[second_model[0]]}

        # Categorize each job and decide which need tiebreaker resolution.
        # For jobs where exactly one primary failed (#3), a synthetic opposing
        # vote is added so the tiebreaker becomes the decisive third vote.
        needs_tiebreaker: list[JobPosting] = []
        synthetic_votes: list[LocalPrefilterResult] = []

        for job in jobs:
            link = str(job.link)
            in_first = link in first_by_link
            in_second = link in second_by_link

            if in_first and in_second:
                if first_by_link[link].should_advance != second_by_link[link].should_advance:
                    # Both primaries ran and disagree → tiebreaker required.
                    needs_tiebreaker.append(job)
            elif in_first:
                # Second model failed — add synthetic opposing vote so tiebreaker is decisive (#3).
                synthetic_votes.append(LocalPrefilterResult(
                    job_link=link,
                    local_score=0,
                    should_advance=not first_by_link[link].should_advance,
                    short_reason=f"{second_model[0]} failed; treated as opposing vote",
                ))
                needs_tiebreaker.append(job)
            elif in_second:
                # First model failed — add synthetic opposing vote so tiebreaker is decisive (#3).
                synthetic_votes.append(LocalPrefilterResult(
                    job_link=link,
                    local_score=0,
                    should_advance=not second_by_link[link].should_advance,
                    short_reason=f"{first_model[0]} failed; treated as opposing vote",
                ))
                needs_tiebreaker.append(job)
            else:
                # Both primaries failed — tiebreaker has the final decision (#5).
                needs_tiebreaker.append(job)

        if synthetic_votes:
            results["_failed_primary"] = synthetic_votes

        if not needs_tiebreaker:
            return results

        tiebreak_model_name, tiebreak_chain = ordered_models[2]
        print(f"  [tiebreaker: {tiebreak_model_name}] {len(needs_tiebreaker)} job(s) need resolution")
        tiebreak_results: list[LocalPrefilterResult] = []
        fallback_results: list[LocalPrefilterResult] = []

        for job in needs_tiebreaker:
            link = str(job.link)
            company_part = f" @ {job.company}" if job.company else ""
            try:
                if store_conn is not None and r_hash:
                    cached = get_prefilter(store_conn, r_hash, link, tiebreak_model_name)
                    if cached is not None:
                        print(f"    [{cached.local_score:3d}] {job.title}{company_part} — {job.link} [cached]")
                        tiebreak_results.append(cached)
                        continue
                result = _invoke_prefilter(tiebreak_chain, candidate_json, job)
                print(f"    [{result.local_score:3d}] {job.title}{company_part} — {job.link}")
                if store_conn is not None and r_hash:
                    try:
                        put_prefilter(store_conn, r_hash, link, tiebreak_model_name, result)
                    except Exception:
                        pass
                tiebreak_results.append(result)
            except Exception as exc:
                print(f"    PREFILTER_FAILED [{job.title}{company_part}]: {exc}")
                # Tiebreaker failed → default to advancing the job (recall-first, #4/#5).
                fallback_results.append(LocalPrefilterResult(
                    job_link=link,
                    local_score=50,
                    should_advance=True,
                    short_reason="tiebreaker failed; defaulting to advance (recall-first)",
                ))

        results[tiebreak_model_name] = tiebreak_results
        if fallback_results:
            results["_tiebreaker_fallback"] = fallback_results
        return results

    def scorer(candidate: CandidateProfile, job: JobPosting) -> MatchResult:
        r_hash = _state.get("resume_hash") or resume_hash(candidate.model_dump_json())
        if store_conn is not None:
            cached = get_match(store_conn, r_hash, str(job.link), _scoring_model_id, _prompt_version)
            if cached is not None:
                print(f"  {job.title} [cached]")
                return cached
        print(f"  {job.title}")
        result = shortlist_chain.invoke(
            {
                "candidate_profile": candidate.model_dump_json(),
                "job_posting": _job_posting_for_prompt(job, _SCORING_DESCRIPTION_MAX_CHARS),
            }
        )
        if store_conn is not None:
            try:
                validated = result if isinstance(result, MatchResult) else MatchResult.model_validate(result)
                put_match(store_conn, r_hash, str(job.link), _scoring_model_id, _prompt_version, validated)
            except Exception:
                pass
        return result

    return PipelineDependencies(candidate_extractor, job_loader, prefilter_runner, scorer)
