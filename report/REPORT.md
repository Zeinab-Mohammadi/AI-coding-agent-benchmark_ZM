# Report: silent-bug-bench

*Evaluating AI coding agents on real neuroscience bug fixes*

**Repository benchmarked:** mne-tools/mne-python · **Tasks:** 50, auto-generated · **Models:** Kimi K2.5, GPT-4o-mini, Llama 3.1 8B, Claude 3 Haiku

## 1. Overview

I chose MNE-Python because it's a large, mature open-source library (7,500+ merged PRs) whose bugs are usually about scientific correctness rather than generic Python syntax. I wanted to test whether coding agents can make *useful* fixes to domain-specific numerical code — not just pattern-match to a nearby line. Tasks are generated from real merged PRs with a single script, and scoring runs end-to-end in Docker with no manual steps. The pipeline is also designed to generalize to other scientific Python libraries, not just MNE.

## 2. Evaluation Environment

Everything runs inside a Docker container (`python:3.11-slim`) with MNE-Python v1.7.0 installed from source at its official git tag. mini-swe-agent gets 300 seconds per task. The scorer's own test-execution step has a separate 120-second limit, and pytest itself is called with `--timeout=30` so a single hanging test can't stall a whole run. Docker isolation means one model's run can never affect another's.

## 3. Task Design

### 3.1 Generation Method

Tasks are generated automatically by `scripts/generate_tasks.py`, which:

1. Queries the GitHub API for merged PRs titled `FIX:` or `BUG:`.
2. Filters PRs by neuroscience-relevant keywords across 5 topic areas.
3. Extracts the broken and fixed versions of the changed file from the PR's base and head SHAs via the GitHub API.
4. Constructs a prompt that embeds a neuroscience-domain hint alongside the buggy code.

All 50 tasks come from real merged PRs — no bug was hand-written. Every one was found by a real user, fixed by a developer, and reviewed and merged by the MNE maintainers.

### 3.2 Neuroscience Topic Areas

The 50 tasks are split into five topic areas, 10 tasks each, chosen to cover distinct stages of the EEG/MEG analysis pipeline with distinct failure modes:

| Topic | Tasks | What It Tests |
|---|---|---|
| Epoching | 10 | Time-windowing around brain events, baseline correction |
| Time-Frequency | 10 | Power spectra, wavelets, frequency resolution |
| Source Localization | 10 | Mapping scalp signals to brain regions |
| Artifact Rejection | 10 | ICA, bad channel handling, noise removal |
| Channel Handling | 10 | Sensor types, montages, spatial positions |

### 3.3 Creativity and Novelty

**Why this domain.** I picked neuroscience analysis code specifically because its bugs are often silent — the code runs and produces numbers without crashing, but the science is wrong. Applying baseline correction over the wrong time window doesn't throw an exception; it produces results that look reasonable but corrupt every downstream analysis. A researcher could spend weeks interpreting data that was broken from the first step. I wanted the benchmark to test whether a model actually understands what the code is *supposed to do*, not just whether it can copy a nearby pattern.

That's what makes this harder than a typical coding benchmark: fixing MNE's epoching code requires knowing baseline correction has to come from a pre-stimulus window; fixing a time-frequency bug requires knowing frequency resolution is constrained by window length. Neither is recoverable by copying adjacent code.

**The domain-hint prompt.** Every task prompt includes a plain-English explanation of what the buggy code does scientifically and why the bug matters — e.g., a source-localization task explains that the inverse problem has no unique solution, so wrong regularization silently shifts the estimated activity to the wrong hemisphere. This hint is the main thing separating the task design here from a generic SWE-bench-style diff-matching task.

| Topic | Stage in Pipeline | Why Bugs Here Are Hard to Spot |
|---|---|---|
| Epoching | Raw continuous data → time-locked segments | Baseline correction errors produce biased averages that look numerically reasonable |
| Time-Frequency | Epoched data → spectral power | Wrong frequency axes or window parameters produce smooth, believable-looking spectrograms |
| Source Localization | Scalp signals → brain source estimates | The inverse problem has no unique solution; wrong regularization still converges to *an* answer |
| Artifact Rejection | Raw data → cleaned data | Removing the wrong ICA components permanently deletes brain signal with no warning |
| Channel Handling | Sensor metadata → spatial analysis | Coordinate-frame errors are invisible until you plot sensor positions and notice they're wrong |

**Versus SWE-bench.** SWE-bench samples issues from general repositories and tests whether a model can reproduce a patch — it doesn't probe whether the model understands *why* the code exists, just whether it can syntactically match a fix. This benchmark differs in three ways:

- **Domain-specific prompting** — each task explains not just what to fix, but why it matters scientifically, testing whether the model can use domain context to guide a fix rather than pattern-match a diff.
- **Hybrid scoring** — MNE's full test suite needs large data files that don't fit in the Docker image, so a pure pass/fail score would leave most tasks unscored. The scorer rewards partial correctness on a continuous scale instead, so every task gets a meaningful signal even when tests can't run.
- **Scientific correctness as the actual target** — ground truth here is "the code is scientifically correct," not just "the test passes." Those are related but not identical: a fix can pass tests while still being subtly wrong at the domain level. Domain-specific validators (e.g., asserting a baseline window is always pre-stimulus) are the natural next step to close this gap — see Section 7.

### 3.4 Task Format

Each task has a fixed structure: the starting state is the buggy file exactly as it existed before the PR's fix; the prompt describes the bug plus a neuroscience hint; the ground truth is the corrected file after merge. The task's associated MNE test file(s) are stored alongside it so the scorer can attempt to run them in Docker.

## 4. Scoring

Scoring is hybrid, combining test execution and code similarity to produce a score in `[0, 1]`.

### 4.1 Score Breakdown

Most MNE tests require external data files that don't fit in the Docker image, so a pure pass/fail score would leave most tasks unscored. Combining test execution with a similarity fallback means every task gets a meaningful score even when its tests can't run standalone.

| Score | Meaning |
|---|---|
| 1.0 | All relevant tests passed in Docker |
| 0.70 | Nearly identical to ground truth (similarity ≥ 0.999); tests couldn't run |
| 0.25 – 0.45 | Moderate similarity to ground truth — partial progress |
| 0.0 | Syntax error, agent failure, or no meaningful change |

### 4.2 Similarity Metric

For one-shot runs, large files are shortened to the changed region plus 40 lines of context on either side. For mini-swe-agent runs, the agent edits the full file, but the scorer compares only the changed region for large files, to avoid inflated similarity from large stretches of untouched code. In both cases, `difflib.SequenceMatcher` gives a line-level similarity ratio, so the scorer awards partial credit rather than a binary pass/fail.

### 4.3 Why Hybrid

MNE's test suite needs ~150MB of external test data not bundled into the Docker image. When tests can't run for that reason, similarity serves as a proxy — a known limitation discussed in Section 6.

### 4.4 Benchmark Approach

Kimi K2.5 was evaluated through the mini-swe-agent pipeline via `scripts/evaluate_kimi.py`. The other three models were evaluated with the same agentic runner, `scripts/run_miniswe.py`, so any difference in scores comes from the model itself and not from differences in setup. A one-shot (single prompt, single patch) evaluation mode is also implemented separately in `scripts/run_benchmark.py` for comparison.

**Agentic evaluation (mini-swe-agent).** In agentic mode, the model works iteratively — it can read files, run bash commands, make edits, and check its own work — all within a 300-second budget. This is closer to how a developer actually fixes a bug than a single-shot guess.

**Models evaluated:**
- GPT-4o-mini (OpenAI) via OpenRouter — `openai/gpt-4o-mini`
- Kimi K2.5 (MoonshotAI) via OpenRouter — `openrouter/moonshotai/kimi-k2.5`
- Llama 3.1 8B Instruct (Meta) via OpenRouter
- Claude 3 Haiku (Anthropic) via OpenRouter — `openrouter/anthropic/claude-3-haiku`

All four ran through the identical pipeline: same prompt construction, same Docker image, same 50 tasks, same scorer. Adding a new model is a one-line `--model` flag change, no code changes required.

## 5. Results

### 5.1 Agentic Results (mini-swe-agent, 300s timeout)

| Model | Average Score | Tasks Completed |
|---|---|---|
| Kimi K2.5 (MoonshotAI) | 0.2683 | 50 / 50 |
| GPT-4o-mini (OpenAI) | 0.1411 | 50 / 50 |
| Llama 3.1 8B | 0.2136 | 50 / 50 |
| Claude 3 Haiku (Anthropic) | 0.3217 | 50 / 50 |

All four models completed all 50 tasks.

### 5.2 Agentic Per-Topic Scores

| Topic | Kimi K2.5 | GPT-4o-mini | Llama 3.1 8B | Claude 3 Haiku |
|---|---|---|---|---|
| Epoching | 0.3379 | 0.2050 | 0.3000 | 0.3750 |
| Time-Frequency | 0.2289 | 0.0850 | 0.1601 | 0.3991 |
| Source Localization | 0.1483 | 0.0379 | 0.1483 | 0.2233 |
| Artifact Rejection | 0.3656 | 0.1851 | 0.2890 | 0.3406 |
| Channel Handling | 0.2606 | 0.1925 | 0.1706 | 0.2706 |

### 5.3 Analysis

Source localization was the hardest topic for every model by a clear margin — consistent with it requiring the most domain-specific reasoning (the inverse problem is underdetermined, so there's no obvious "shape" of a correct fix to pattern-match toward). Artifact rejection and epoching were the easiest across the board, likely because their bugs are closer to time-indexing and control-flow errors that don't require as much scientific context to reason about.

A few things went wrong along the way, worth recording honestly: an early run silently used a stale/unset API key and returned the original broken file for every task — caught only because two "different" models produced identical scores across the board. A first pass at a 120-second timeout was too short; the agent would start editing and run out of time mid-fix, so I raised it to 300 seconds (some larger files still time out at that budget). Not every candidate model ID worked through OpenRouter — of the ones I tried, Kimi K2.5, GPT-4o-mini, Llama 3.1 8B, and Claude 3 Haiku ran successfully end to end. Llama was the weakest in practice: it mostly printed explanations rather than actually editing files, while Kimi and GPT-4o-mini reliably produced real edits. Every model's full trajectory is saved under `results/trajectories/` so any of this is independently checkable, not just asserted.

## 6. Shortcomings

- **Test execution.** The biggest gap: many MNE tests need external datasets that didn't make it into the Docker image in the time available, so most tasks fall back to similarity scoring rather than a real pass/fail signal. It's not perfect, but it produces a usable score across all 50 tasks.
- **Wide-change tasks.** 15 of the 50 tasks came from PRs touching 100+ lines — harder for models to fix and harder to score cleanly. Filtering for smaller, more focused PRs would tighten this.
- **Similarity as a proxy.** Line-level similarity can give partial credit to code that looks close but is scientifically wrong. A structural or semantic comparison would be more faithful to what's actually being measured.
- **Single file per task.** Some MNE bugs span multiple files; each task here only captures the most-changed file, so some multi-file fixes are necessarily out of scope.
- **Model context limits.** Very large files had to be truncated to the changed region plus context. Smaller models (Llama, GPT-4o-mini) struggled more with these truncated views and sometimes returned incomplete code.

## 7. How I'd Improve and Scale This

**Scaling the generator.** The task generator already only depends on three repo-specific things: the repo name, its PR title convention, and its topic keyword sets. Scaling to O(1000) repos means adding a registry of repo metadata, parallelizing task generation across repos, and deriving each Docker image automatically from the target repo's own CI config instead of hand-writing one Dockerfile per repo.

**Improving task quality.** I'd filter for PRs with tighter diffs (≤50 lines, ≤2 files touched) so tasks stay focused, use AST-based extraction to isolate the exact buggy function instead of the whole file, and add genuinely multi-file tasks for bugs that require coordinated changes across files.

**Improving scoring.** I'd bundle the external test datasets into the Docker image so the real test suite can run on every task instead of falling back to similarity. I'd also explore semantic similarity via code embeddings instead of line-level diffing, and mutation testing to verify a fix actually addresses the *specific* bug rather than happening to look close to the reference patch.

**Adding models.** This is already the cheap part — any OpenRouter model is a `--model` flag away, no code changes.

## 8. Reproduction

```bash
export GITHUB_TOKEN='...'
export OPENROUTER_API_KEY='...'

python3 scripts/generate_tasks.py
docker build -t silent-bug-bench docker/
python3 scripts/evaluate_kimi.py
python3 scripts/run_miniswe.py --model openrouter/openai/gpt-4o-mini --tasks_dir tasks --output results/miniswe_gpt4omini.json
python3 scripts/run_miniswe.py --model openrouter/meta-llama/llama-3.1-8b-instruct --tasks_dir tasks --output results/miniswe_llama.json
python3 scripts/run_miniswe.py --model openrouter/anthropic/claude-3-haiku --tasks_dir tasks --output results/miniswe_haiku.json
```
