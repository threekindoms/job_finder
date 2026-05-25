# Job Matching Tool

Matches a resume against LinkedIn job postings using local LLMs for pre-filtering and a configurable model for detailed scoring. Results are cached in a local SQLite database so repeated runs skip expensive LLM calls for already-processed inputs.

Each run starts with scraping a searching URL for job listing with pagenations. Targeting LinkedIn but can be extended to other links. After all jobs are obtained, it matches an input resume with the job list and find all possible matches.

## How it works

1. Load job URLs — from a LinkedIn search URL (auto-scraped), a local file, or direct URLs.
2. Fetch each job page and extract a structured `JobPosting` (cached by URL, 7-day TTL).
3. Extract a `CandidateProfile` from the resume (cached by resume content hash).
4. Run all jobs through two local prefilter models; use a third as tiebreaker on disagreements (results cached per candidate + job + model).
5. Score shortlisted jobs with the configured scoring model — local or cloud (results cached per candidate + job + model + prompt version).
6. Write ranked results and run artifacts to `runs/<timestamp>/`.
7. Back up the SQLite store to `~/Documents/Storages/resume_matching/`.

Changing the scoring prompt or weights automatically invalidates cached scores. Changing the resume invalidates the candidate profile and all downstream scores.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with the models listed in `.env` (or an Ollama-compatible cloud endpoint)
  - Or [Cloud]() cloud models with API keys
- [Playwright](https://playwright.dev/python/) + a Chromium browser — only needed for `--search-url` scraping

## Installation

```bash
git clone <repo>
cd job_finder
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pull the default local models (only needed once):

```bash
ollama pull llama3.1:8b
ollama pull mistral:7b-instruct
ollama pull deepseek-r1:7b
```

If you plan to use `--search-url`, install the Playwright browser (only needed once):

```bash
playwright install chromium
```

## Configuration

Copy `.env.example` and edit as needed — all defaults use local Ollama models:

```bash
cp .env.example .env
```

### Local Ollama

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server URL |
| `LOCAL_PREFILTER_MODELS` | `llama3.1:8b,mistral:7b-instruct,deepseek-r1:7b` | Prefilter models (first two vote; third breaks ties) |
| `SHORTLIST_SCORING_PROVIDER` | `ollama` | Scoring provider: `ollama`, `ollama_cloud`, `anthropic`, `openai`, or `google` |
| `SHORTLIST_SCORING_MODEL` | `llama3.1:8b` | Model name for the scoring step |

### Ollama cloud

To use a remote Ollama endpoint (e.g. [api.ollama.com](https://api.ollama.com)) for prefiltering or scoring:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_CLOUD_BASE_URL` | `https://api.ollama.com` | Remote Ollama endpoint |
| `OLLAMA_CLOUD_API_KEY` | — | Bearer token for the cloud API |
| `CLOUD_PREFILTER_MODELS` | — | Comma-separated models to add to the prefilter pool via cloud |

Set `SHORTLIST_SCORING_PROVIDER=ollama_cloud` and `SHORTLIST_SCORING_MODEL` to use a cloud model for scoring.

### Cloud provider scoring

Uncomment the relevant block in `.env` and set the API key:

| Provider | `SHORTLIST_SCORING_PROVIDER` | Example model |
|---|---|---|
| Anthropic | `anthropic` | `claude-haiku-4-5-20251001` |
| OpenAI | `openai` | `gpt-4o-mini` |
| Google | `google` | `gemini-2.0-flash` |

### Other settings

| Variable | Default | Description |
|---|---|---|
| `MAX_PAID_LLM_JOBS` | `20` | Max jobs sent to the scoring step |
| `SKIP_PAID_LLM_IF_LOCAL_SCORE_BELOW` | `80` | Local score threshold for shortlist eligibility |
| `RUNS_DIR` | `runs` | Directory for run artifacts |
| `STORAGE_DB_PATH` | `storage/store.db` | SQLite cache database |
| `BACKUP_DB_PATH` | `~/Documents/Storages/resume_matching` | Backup destination after each run |
| `LINKEDIN_COOKIE_FILE` | `config/linkedin_cookies.json` | LinkedIn session cookies — required for `--search-url` |
| `LINKEDIN_HEADLESS` | `true` | Run browser headlessly during scraping |
| `LINKEDIN_MAX_PAGES` | `20` | Max search-result pages to scrape |

## Running

The CLI loads `.env` automatically on startup. No manual `source` step is needed.

### Using a config file (recommended)

```bash
cp config/run_config.example.toml config/run_config.toml
# edit config/run_config.toml
# at minimum, provide resume, a search URL for jobs (or a list of job links)
PYTHONPATH=. python -m src.cli --config config/run_config.toml
```

CLI flags override config file values when both are supplied.

### Using CLI flags directly

**From a LinkedIn search URL (auto-scrape):**

```bash
PYTHONPATH=. python -m src.cli \
  --resume path/to/resume.pdf \
  --search-url "https://www.linkedin.com/jobs/search/?keywords=Software+Engineer&location=Seattle"
```

Requires `config/linkedin_cookies.json` — run the one-time setup first:

```bash
python tools/capture_cookies.py
```

**Scrape only — save URLs to a file without running matching:**

```bash
PYTHONPATH=. python -m src.cli \
  --search-url "https://www.linkedin.com/jobs/search/?keywords=Software+Engineer" \
  --scrape-only
# prints: Saved to: /tmp/xyz_job_links.txt

PYTHONPATH=. python -m src.cli \
  --resume path/to/resume.pdf \
  --manual-links /tmp/xyz_job_links.txt
```

**From a job-link file** (one URL per line):

```bash
PYTHONPATH=. python -m src.cli \
  --resume path/to/resume.pdf \
  --manual-links path/to/job_links.txt
```

**From direct URLs:**

```bash
PYTHONPATH=. python -m src.cli \
  --resume path/to/resume.pdf \
  --job-url "https://www.linkedin.com/jobs/view/1234567890/" \
  --job-url "https://www.linkedin.com/jobs/view/9876543210/"
```

### Options

| Flag / config key | Default | Description |
|---|---|---|
| `--config` | — | Path to a TOML run config file |
| `--resume` / `resume` | *(required)* | Path to resume file (PDF, DOCX, or TXT) |
| `--search-url` / `search_url` | — | LinkedIn search URL; scrapes all pages automatically |
| `--scrape-only` / `scrape_only` | `false` | Scrape URLs to a temp file and exit; requires `--search-url` |
| `--manual-links` / `manual_links` | — | Path to newline-delimited job URL file |
| `--job-url` / `job_url` | — | Direct job URL; repeat / list for multiple |
| `--top-n` / `top_n` | `10` | Number of detailed match results to return |
| `--ignore-link` / `ignore_link` | — | Job URL to exclude; repeat / list for multiple |
| `--jobs-file` / `jobs_file` | — | Skip web fetching and load saved job records (JSON) |
| `--candidate-profile-file` / `candidate_profile_file` | — | Skip resume extraction and load a saved profile (JSON) |

### Run output

The run directory is printed on completion:

```
runs/<timestamp>/
  jobs.json                   # all fetched job postings
  top_matches.json            # detailed scores for top N jobs (includes title and company)
  remaining_ranked_jobs.json
  summary.json
  usage.json
```

## Scoring weights

| Dimension | Max points |
|---|---|
| Required qualifications | 60 |
| Preferred qualifications | 20 |
| Experience level | 9 |
| Domain fit | 9 |
| Location / availability | 2 |
| **Total** | **100** |

## Tests

```bash
PYTHONPATH=. pytest tests/
```


## Repo History
Most of the planning and implementation are completed by Codex and Claude Code.
