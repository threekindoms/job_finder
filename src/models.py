from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WorkExperienceEntry(BaseModel):
    summary: str
    keywords: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"summary": data, "keywords": []}
        return data


class CandidateProfile(BaseModel):
    professional_summary: str
    skills: list[str] = Field(default_factory=list)
    work_experience: list[WorkExperienceEntry] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    location: str | None = None


def _flatten_kv_dict(d: dict[str, Any]) -> list[str]:
    result = []
    for k, v in d.items():
        key, val = str(k).strip(), str(v).strip() if v else ""
        result.append(f"{key}: {val}" if val else key)
    return result


def _flatten_name_desc_dicts(items: list[dict[str, Any]]) -> list[str]:
    result = []
    for item in items:
        name = str(item.get("name", "")).strip()
        desc = str(item.get("description", "")).strip()
        text = f"{name}: {desc}" if name and desc else name or desc
        if text:
            result.append(text)
    return result


class JobPosting(BaseModel):
    title: str
    company: str = ""
    link: HttpUrl
    description: str
    requirements: list[str] = Field(default_factory=list)
    optional_requirements: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_common_job_keys(cls, data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict):
            return data
        aliases = {
            "Title": "title",
            "Company": "company",
            "Link": "link",
            "Description": "description",
            "Basic Qualifications": "requirements",
            "Preferred Qualifications": "optional_requirements",
        }
        normalized = dict(data)
        for source, target in aliases.items():
            if source in normalized and target not in normalized:
                normalized[target] = normalized[source]
        requirements = normalized.get("requirements")
        if isinstance(requirements, dict):
            normalized["requirements"] = _flatten_kv_dict(requirements)
        elif (
            isinstance(requirements, list)
            and requirements
            and all(isinstance(item, dict) for item in requirements)
        ):
            if any("items" in item for item in requirements):
                # Grouped format: [{"name": "Preferred Qualifications", "items": [...]}, ...]
                flattened_required: list[str] = []
                flattened_optional: list[str] = []
                for group in requirements:
                    items = group.get("items", [])
                    if not isinstance(items, list):
                        continue
                    if group.get("name") == "Preferred Qualifications":
                        flattened_optional.extend(str(item) for item in items)
                    else:
                        flattened_required.extend(str(item) for item in items)
                normalized["requirements"] = flattened_required
                if not normalized.get("optional_requirements"):
                    normalized["optional_requirements"] = flattened_optional
            else:
                # Individual dict format: [{"name": "...", "description": "..."}, ...]
                normalized["requirements"] = _flatten_name_desc_dicts(requirements)
        for field in ("requirements", "optional_requirements"):
            value = normalized.get(field)
            if value is None:
                normalized[field] = []
            elif isinstance(value, dict):
                normalized[field] = _flatten_kv_dict(value)
            elif (
                isinstance(value, list)
                and value
                and all(isinstance(item, dict) for item in value)
            ):
                normalized[field] = _flatten_name_desc_dicts(value)
        return normalized


class LocalPrefilterResult(BaseModel):
    job_link: HttpUrl
    local_score: int = Field(ge=0, le=100)
    should_advance: bool
    short_reason: str

    @field_validator("local_score", mode="before")
    @classmethod
    def normalize_fractional_score(cls, value: int | float) -> int | float:
        if isinstance(value, float) and 0 <= value <= 1:
            return round(value * 100)
        return value


class RequirementAssessment(BaseModel):
    requirement: str
    credit: float = Field(ge=0.0, le=1.0)
    reason: str

    @field_validator("credit", mode="before")
    @classmethod
    def normalize_credit(cls, value: Any) -> float:
        v = float(value)
        # Accept percentages (e.g. 100 → 1.0) from models that output 0/50/100
        if v > 1.0:
            v = v / 100.0
        return max(0.0, min(1.0, v))


SCORE_MAXIMA: dict[str, int] = {
    "required_qualifications_score": 60,
    "preferred_qualifications_score": 20,
    "experience_level_score": 9,
    "domain_fit_score": 9,
    "location_or_availability_score": 2,
}


class MatchResult(BaseModel):
    job_link: HttpUrl
    overall_score: int = Field(ge=0, le=100)
    required_qualifications_score: int = Field(default=0, ge=0, le=60)
    preferred_qualifications_score: int = Field(default=0, ge=0, le=20)
    experience_level_score: int = Field(ge=0, le=9)
    domain_fit_score: int = Field(ge=0, le=9)
    location_or_availability_score: int = Field(ge=0, le=2)
    requirement_assessments: list[RequirementAssessment] = Field(default_factory=list)
    preferred_requirement_assessments: list[RequirementAssessment] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    other_considerations: list[str] = Field(default_factory=list)
    resume_improvement_suggestions: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel

    @model_validator(mode="before")
    @classmethod
    def normalize_dimension_scores(cls, data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)

        # Compute required_qualifications_score from per-requirement credits when available.
        assessments = normalized.get("requirement_assessments")
        if assessments and isinstance(assessments, list):
            valid = [a for a in assessments if isinstance(a, dict) and "credit" in a]
            if valid:
                total_credit = sum(float(a.get("credit", 0)) for a in valid)
                normalized["required_qualifications_score"] = round(
                    total_credit / len(valid) * 60
                )

        # Compute preferred_qualifications_score from per-requirement credits when available.
        preferred_assessments = normalized.get("preferred_requirement_assessments")
        if preferred_assessments and isinstance(preferred_assessments, list):
            valid = [a for a in preferred_assessments if isinstance(a, dict) and "credit" in a]
            if valid:
                total_credit = sum(float(a.get("credit", 0)) for a in valid)
                normalized["preferred_qualifications_score"] = round(
                    total_credit / len(valid) * 20
                )

        for field_name, maximum in SCORE_MAXIMA.items():
            value = normalized.get(field_name)
            if isinstance(value, int):
                normalized[field_name] = max(0, min(value, maximum))
        if all(isinstance(normalized.get(f), int) for f in SCORE_MAXIMA):
            normalized["overall_score"] = sum(normalized[f] for f in SCORE_MAXIMA)
        return normalized

    @field_validator(
        "strengths",
        "gaps",
        "other_considerations",
        "resume_improvement_suggestions",
        mode="before",
    )
    @classmethod
    def normalize_singleton_text_lists(cls, value: list[str] | str) -> list[str] | str:
        if isinstance(value, str):
            return [value]
        return value


class UsageSummary(BaseModel):
    local_prefilter_models: list[str] = Field(default_factory=list)
    shortlist_scoring_provider: str | None = None
    shortlist_scoring_model: str | None = None
    searched_job_count: int = 0
    prefiltered_job_count: int = 0
    shortlist_job_count: int = 0
    scored_job_count: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: float | None = None


class RunReport(BaseModel):
    candidate_profile: CandidateProfile | None = None
    searched_jobs: list[JobPosting] = Field(default_factory=list)
    company_excluded_jobs: list[JobPosting] = Field(default_factory=list)
    top_matches: list[MatchResult] = Field(default_factory=list)
    remaining_ranked_jobs: list[dict[str, Any]] = Field(default_factory=list)
    usage: UsageSummary | None = None
