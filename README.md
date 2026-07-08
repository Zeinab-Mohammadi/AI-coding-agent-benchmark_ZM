# silent-bug-bench
## Benchmarking AI coding agents on silent numerical bugs in scientific Python

**Can a coding agent tell the difference between code that runs and code that's *right*?**

A reproducible evaluation framework for AI coding agents, built on real merged pull requests from [MNE-Python](https://github.com/mne-tools/mne-python) — a scientific Python library (7,500+ merged PRs) for EEG/MEG/fNIRS brain-signal analysis. It covers the full pipeline end to end: systematic task generation from repository history, Docker-based sandboxed execution, automatic `[0, 1]` scoring, and multi-model benchmarking with [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent).

The focus is on evaluating agentic coding ability beyond isolated function-writing: understanding an unfamiliar repository, making a targeted edit that requires *domain* reasoning (not just syntax), and getting scored against real tests where possible, and a graded partial-credit signal otherwise.

## Project Highlights

- Built a full benchmark on a large open-source repository (7,500+ merged PRs) — task generation, sandboxed execution, and scoring all automated, with no manually invented benchmark tasks
- Generated 50 tasks systematically by mining real `FIX:`/`BUG:` merged PRs and diffing base/head commits via the GitHub API
- Designed a reproducible Docker environment for isolated, side-effect-free agent execution
- Implemented a hybrid `[0, 1]` scorer: real test execution when possible, and a capped, graded similarity fallback — a real passing test always outweighs it — when the test suite can't run standalone
- Benchmarked 4 models/configurations head-to-head — Kimi K2.5, GPT-4o-mini, Llama 3.1 8B, Claude 3 Haiku — through an identical mini-swe-agent pipeline over OpenRouter
- Included a dedicated, isolated evaluation entry point for Kimi K2.5 (`scripts/evaluate_kimi.py`)
- Saved every agent's full trajectory (every tool call, every edit) for every task, so any reported score is traceable back to the transcript that produced it
- Structured the pipeline so the only repo-specific pieces are a PR-title convention and a topic-keyword set — designed with a path to scale across many repositories, not just one

## Motivation

Coding agents are usually evaluated on isolated, self-contained problems. Real software engineering isn't that — it requires navigating an unfamiliar codebase, figuring out which file actually needs to change, and making an edit that respects invariants the code never states out loud.

I went one step further than "does the agent navigate the repo correctly": the bugs in this benchmark specifically **don't crash**. They run, return a number, and the number is quietly wrong, because the fix requires understanding *why* the code exists, not just how it's shaped. Baseline-correcting over the wrong time window doesn't throw an exception — it silently biases every downstream analysis. Picking the wrong regularization in an inverse solution doesn't error out — it converges to a confidently wrong answer. That failure mode — passes review, runs clean, wrong under the hood — is a much closer analogue to what actually costs engineering time in any domain with a numerical correctness criterion, not just neuroscience.

To probe this directly, every task prompt includes a short plain-English explanation of what the surrounding code is *for*, not just what's broken. It's a small addition to the prompt, but it turns the task from "match a diff" into "use domain context to decide what correct even means" — and per the results below, that's still a hard problem for smaller models.

## Benchmark Design

Each task is a self-contained unit with:

- A fixed starting repository state (the file exactly as it was before the real fix landed)
- A natural-language prompt, including the domain-context hint
- A ground-truth expected fix (the file exactly as it was after the real fix merged)
- The task's original associated test file(s), carried through for scoring
- A final automatic score in `[0, 1]`

Tasks are split across 5 stages of the EEG/MEG analysis pipeline, chosen to require distinct kinds of reasoning so no single strategy generalizes across all of them:

| Topic | Tasks | Stage in Pipeline | Why Bugs Here Are Hard to Spot |
|---|---|---|---|
| Epoching | 10 | Raw data → time-locked segments | Baseline errors produce biased averages that still look numerically reasonable |
| Time-Frequency | 10 | Epoched data → spectral power | Wrong frequency axes/windowing produce smooth, believable spectrograms |
| Source Localization | 10 | Scalp signals → brain source estimates | The inverse problem is underdetermined; a wrong answer still converges |
| Artifact Rejection | 10 | Raw data → cleaned data | Removing the wrong ICA components deletes brain signal with no warning |
| Channel Handling | 10 | Sensor metadata → spatial analysis | Coordinate-frame errors are invisible until you plot and notice they're wrong |

## Evaluation Pipeline

1. **Generate** — mine merged PRs from GitHub history, filter by title convention + topic keywords, diff base/head SHAs, require an associated test file. (`scripts/generate_tasks.py`)
2. **Sandbox** — build an isolated Docker image with the target library installed from source. (`docker/Dockerfile`)
3. **Run** — a coding agent (one-shot or agentic via mini-swe-agent) receives the task prompt and produces a code change inside the sandbox.
4. **Score** — the scorer runs the task's real tests when they can execute standalone; otherwise it falls back to a graded line-level similarity metric against ground truth. (`scripts/score.py`)
5. **Log** — every score, and the full agent trajectory that produced it, is saved per task per model. (`results/`)
6. **Aggregate** — per-model and per-topic scores are rolled up for comparison. (`report/REPORT.md`)

This makes model comparison fully reproducible: same tasks, same sandbox, same scorer, only the model changes.

## Models Evaluated

All four run through the identical mini-swe-agent pipeline over OpenRouter — same prompts, same Docker image, same 300s budget, same scorer:

| Model | Mean Score | Notes |
|---|---:|---|
| Claude 3 Haiku (Anthropic) | **0.322** | 50/50 tasks completed |
| Kimi K2.5 (MoonshotAI) | 0.268 | Evaluated with `scripts/evaluate_kimi.py`; 50/50 completed |
| Llama 3.1 8B Instruct (Meta) | 0.214 | 50/50 completed; weakest at actually editing files vs. explaining |
| GPT-4o-mini (OpenAI) | 0.141 | 50/50 completed |

No model saturates the benchmark and none collapses to zero — the score spread is what makes this ranking meaningful rather than noise. These results reflect this exact benchmark, prompt format, time budget, and agent-runner configuration — not a general ranking of model capability. Source localization, which requires more domain-specific numerical reasoning, was the hardest category across models; artifact rejection and epoching scored highest across the board. Full per-topic breakdown and failure analysis: [`report/REPORT.md`](report/REPORT.md).

## Reproducibility and Auditability

- Every task is pulled from a PR that already existed and was merged by real maintainers — there's no hand-authored ground truth to overfit to.
- The scorer always prefers a real, passing test over any proxy metric; similarity scoring only activates when the actual test suite can't execute standalone, and it's capped well below what a passing test earns.
- Every model's complete agent trajectory is saved, so any reported score is independently auditable against the transcript that produced it.

## Repository Structure

```text
.
├── docker/
│   └── Dockerfile          # sandbox with the target library built from source
├── tasks/                  # 50 generated task JSON files + summary.json
├── scripts/
│   ├── generate_tasks.py   # automated task mining from GitHub PR history
│   ├── score.py            # hybrid scorer (tests + similarity) → [0,1]
│   ├── run_benchmark.py    # one-shot runner via OpenRouter
│   ├── run_miniswe.py      # agentic runner using mini-swe-agent
│   └── evaluate_kimi.py    # dedicated Kimi K2.5 evaluation entry point
├── results/                # per-model scores + full saved agent trajectories
└── report/
    └── REPORT.md           # environment, task design, scoring, results, limitations, scaling plan
```

## How to Run

```bash
export GITHUB_TOKEN="..."
export OPENROUTER_API_KEY="..."

# regenerate the 50 tasks from live GitHub PR history (optional — tasks/ is already populated)
python3 scripts/generate_tasks.py

# build the sandbox
docker build -t silent-bug-bench docker/

# evaluate a model with the agentic runner
python3 scripts/run_miniswe.py \
    --model openrouter/moonshotai/kimi-k2.5 \
    --tasks_dir tasks \
    --output results/miniswe_kimi.json

# swap in any other OpenRouter model with the same flag — no code changes
python3 scripts/run_miniswe.py \
    --model openrouter/anthropic/claude-3-haiku \
    --tasks_dir tasks \
    --output results/miniswe_haiku.json
```

## Scaling Beyond One Repository

The generator is written so the repo, its PR-title convention, and its topic keyword sets are the *only* repo-specific pieces — diffing, scoring, sandboxing, and the agent runner are all generic. Scaling to O(1000) repos means: a registry of repo metadata + PR conventions, parallelized task generation, and Docker images derived automatically from each repo's own CI config instead of a hand-written Dockerfile per repo. This plan, plus the benchmark's current shortcomings, is in [`report/REPORT.md`](report/REPORT.md).

## Tech Stack

Python · Docker · GitHub API · OpenRouter · mini-swe-agent · pytest · difflib-based similarity scoring · benchmark/task generation

## Technical Skills Demonstrated

This project demonstrates end-to-end AI evaluation engineering: task/dataset construction from real version-control history, sandboxed execution, multi-model benchmarking, scoring-function design, and empirical failure analysis.

AI agent evaluation and benchmark design · automated task generation from real-world version-control history · reproducible ML evaluation pipelines · Docker-based sandboxed execution · LLM tool-use via OpenRouter and mini-swe-agent · scoring-function design under partial ground truth · empirical, per-topic failure analysis across models · systems design for scaling an evaluation pipeline to hundreds of repositories
