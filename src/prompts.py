CANDIDATE_PROFILE_SYSTEM_PROMPT = """Extract a candidate profile from the resume text.

Rules:
- Use only facts explicitly present in the resume text.
- Do not infer protected attributes.
- Return missing location as null.
- Preserve concise work-experience and education entries.
- Return exactly these fields:
  - professional_summary: string
  - skills: list of strings
  - work_experience: list of objects, each with:
      - summary: one-line description (company, role, date range)
      - keywords: technical skills, tools, and domain terms explicitly mentioned
        for that specific role in the resume — do not copy from other roles
  - education: list of strings
  - location: string or null
"""

CANDIDATE_PROFILE_HUMAN_PROMPT = """Resume text:
{resume_text}
"""

JOB_POSTING_SYSTEM_PROMPT = """Extract a structured job posting from raw web page text.

The input may contain page navigation, login UI, and company background — ignore all of that.

Rules:
- Use only facts present in the supplied text.
- title: the job title only.
- company: the hiring company name. Return an empty string if not found.
- link: the canonical job URL if one appears in the text.
- description: a concise summary of what the role does, drawn from the role overview or
  responsibilities sections. Do NOT use the generic company overview paragraph.
- requirements: extract each required/minimum qualification as a separate string.
  Look for sections labelled "Required Qualifications", "Requirements",
  "Minimum Qualifications", or "Qualifications". Each bullet or sentence becomes one item.
  If such a section exists in the text, this list MUST NOT be empty.
- optional_requirements: extract preferred/nice-to-have qualifications the same way.
  Look for "Preferred Qualifications", "Nice to have", or "Preferred/Additional Qualifications".
- Do not invent requirements when the text is silent on them.
- Return a flat JSON object without wrapper keys.
- Return exactly these fields: title, company, link, description, requirements, optional_requirements
"""

JOB_POSTING_HUMAN_PROMPT = """Job text:
{raw_job_text}
"""

LOCAL_PREFILTER_SYSTEM_PROMPT = """Evaluate whether the candidate should advance to paid scoring.

Rules:
- Favor recall over precision.
- Use only evidence explicitly present in the candidate profile and job posting.
- Return `should_advance = true` if the job may reasonably fit.
- Return `should_advance = false` only when the job is clearly weak or unsupported.
- Return one concise reason.
- Return `local_score` as an integer from 0 to 100.
- Return a flat JSON object without a wrapper key.
- Return exactly these fields:
  - job_link
  - local_score
  - should_advance
  - short_reason
"""

LOCAL_PREFILTER_HUMAN_PROMPT = """Candidate profile:
{candidate_profile}

Job posting:
{job_posting}
"""

SHORTLIST_SCORING_SYSTEM_PROMPT_LOCAL = """Score the job against the candidate profile.

Use this weighted rubric (scores are integers within each range):
- required qualifications: 0–60  (computed from requirement_assessments — see below)
- preferred qualifications: 0–20  (computed from preferred_requirement_assessments — see below)
- experience level: 0–9
- domain fit: 0–9
- location or availability: 0–2

For required_qualifications_score, fill `requirement_assessments` instead of computing
the number yourself. For every required qualification listed in the job posting, add one
entry with:
  - requirement: exact text of the qualification
  - credit: 1.0 if the candidate profile explicitly mentions that exact skill, technology,
    or experience; 0.5 if they have adjacent but not identical experience; 0.0 if absent.
    A "related" skill does NOT earn 1.0 — only direct, explicit evidence does.
  - reason: one sentence citing evidence from the profile or explaining the absence.
Set required_qualifications_score to 0; it will be computed automatically.

Do the same for preferred_qualifications_score: fill `preferred_requirement_assessments`
with one entry per preferred/optional qualification using the same credit rules.
Set preferred_qualifications_score to 0; it will be computed automatically.

Rules:
- Use only evidence explicitly present in the candidate profile and job posting.
- Do not infer protected attributes.
- Keep `overall_score` between 0 and 100.
- Return every score as an integer.
- Return strengths, gaps, other considerations, resume improvement suggestions, and confidence.
- Return `confidence` as exactly one of: `low`, `medium`, `high`.
- Return a flat JSON object without a wrapper key.
- Return exactly these fields:
  - job_link
  - overall_score
  - required_qualifications_score (set to 0)
  - preferred_qualifications_score (set to 0)
  - experience_level_score
  - domain_fit_score
  - location_or_availability_score
  - requirement_assessments
  - preferred_requirement_assessments
  - strengths
  - gaps
  - other_considerations
  - resume_improvement_suggestions
  - confidence
"""

SHORTLIST_SCORING_SYSTEM_PROMPT_CLOUD = """Score the job against the candidate profile.

Use this weighted rubric:
- required qualifications: 0–60
- preferred qualifications: 0–20
- experience level: 0–9
- domain fit: 0–9
- location or availability: 0–2

Scoring required_qualifications_score and preferred_qualifications_score — be rigorous:
- Assess each listed qualification independently:
    - Full credit (1.0): the profile explicitly names that exact skill, technology, or
      experience. Adjacent or related experience does NOT qualify for full credit.
    - Half credit (0.5): clearly overlapping but not identical — the exact skill is absent
      but a meaningful subset is directly present.
    - Zero credit (0.0): absent, only implied, or merely adjacent.
- required_qualifications_score = round(sum_of_credits / count_of_requirements * 60)
- preferred_qualifications_score = round(sum_of_credits / count_of_preferred * 20)
- If no preferred qualifications are listed, set preferred_qualifications_score to 0.
- location_or_availability_score: 2 if location matches or role is remote, 0 otherwise.

Rules:
- Use only evidence explicitly present in the candidate profile and job posting.
- Do not infer protected attributes.
- Return overall_score as the exact sum of all five dimension scores.
- Return every score as an integer.
- Return strengths, gaps, other considerations, resume improvement suggestions, and confidence.
- Return `confidence` as exactly one of: `low`, `medium`, `high`.
- Return a flat JSON object without a wrapper key.
- Return exactly these fields:
  - job_link
  - overall_score
  - required_qualifications_score
  - preferred_qualifications_score
  - experience_level_score
  - domain_fit_score
  - location_or_availability_score
  - strengths
  - gaps
  - other_considerations
  - resume_improvement_suggestions
  - confidence
"""

SHORTLIST_SCORING_HUMAN_PROMPT = """Candidate profile:
{candidate_profile}

Job posting:
{job_posting}
"""
