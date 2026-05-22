from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.models import CandidateProfile, JobPosting, LocalPrefilterResult, MatchResult
from src.prompts import (
    CANDIDATE_PROFILE_HUMAN_PROMPT,
    CANDIDATE_PROFILE_SYSTEM_PROMPT,
    JOB_POSTING_HUMAN_PROMPT,
    JOB_POSTING_SYSTEM_PROMPT,
    LOCAL_PREFILTER_HUMAN_PROMPT,
    LOCAL_PREFILTER_SYSTEM_PROMPT,
    SHORTLIST_SCORING_HUMAN_PROMPT,
    SHORTLIST_SCORING_SYSTEM_PROMPT_LOCAL,
)


def build_candidate_profile_chain(chat_model: Any) -> Any:
    """Build the LangChain structured-output chain for resume extraction."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CANDIDATE_PROFILE_SYSTEM_PROMPT),
            ("human", CANDIDATE_PROFILE_HUMAN_PROMPT),
        ]
    )
    structured_model = chat_model.with_structured_output(CandidateProfile)
    return prompt | structured_model


def build_job_posting_chain(chat_model: Any) -> Any:
    """Build the LangChain structured-output chain for job extraction."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JOB_POSTING_SYSTEM_PROMPT),
            ("human", JOB_POSTING_HUMAN_PROMPT),
        ]
    )
    structured_model = chat_model.with_structured_output(JobPosting)
    return prompt | structured_model


def build_local_prefilter_chain(chat_model: Any) -> Any:
    """Build the LangChain structured-output chain for local prefiltering."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", LOCAL_PREFILTER_SYSTEM_PROMPT),
            ("human", LOCAL_PREFILTER_HUMAN_PROMPT),
        ]
    )
    structured_model = chat_model.with_structured_output(LocalPrefilterResult)
    return prompt | structured_model


def build_shortlist_scoring_chain(chat_model: Any, system_prompt: str = SHORTLIST_SCORING_SYSTEM_PROMPT_LOCAL) -> Any:
    """Build the LangChain structured-output chain for detailed shortlist scoring."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", SHORTLIST_SCORING_HUMAN_PROMPT),
        ]
    )
    structured_model = chat_model.with_structured_output(MatchResult)
    return prompt | structured_model
