"""
Scorer for MNE-Python neuroscience benchmark.

Hybrid scoring strategy:
  1. Syntax check (fast, always runs)
  2. Try to run relevant pytest tests inside Docker
  3. If tests can't run (missing data/Qt), fall back to similarity scoring

Final score in [0, 1]:
  1.0   — tests passed (confirmed correct fix)
  0.70  — similarity >= 0.999 to ground truth (nearly identical)
  0.45  — similarity >= 0.995
  0.25  — similarity >= 0.98
  0.15  — similarity >= 0.90
  0.0   — syntax error, or similarity too low
"""

import os
import json
import subprocess
import tempfile
import ast
import sys
import difflib

USE_DOCKER = os.environ.get("USE_DOCKER", "1") == "1"
DOCKER_IMAGE = "silent-bug-bench"
TIMEOUT_SECONDS = 120


# ── 1. Syntax check ──────────────────────────────────────────────────────────

def check_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ── 2. Similarity score ───────────────────────────────────────────────────────

def similarity_score(solution: str, ground_truth: str) -> float:
    """
    Compute how similar the AI solution is to the ground truth fix.
    Uses line-level diff ratio — a standard string similarity measure.
    Returns a float in [0, 1].
    """
    sol_lines = solution.strip().splitlines()
    gt_lines = ground_truth.strip().splitlines()
    ratio = difflib.SequenceMatcher(None, sol_lines, gt_lines).ratio()
    return round(ratio, 4)


# ── 3. Test runner (Docker) ───────────────────────────────────────────────────

def run_tests_in_docker(task: dict, solution_code: str) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write solution
        with open(os.path.join(tmpdir, "solution.py"), "w") as f:
            f.write(solution_code)

        file_to_fix = task["file_to_fix"]
        test_files = task.get("test_files", [])

        if not test_files:
            return {"ran": False, "passed": False, "output": "No test files"}

        # Only use first test file to keep it fast
        test_file = test_files[0]

        runner = f"""#!/bin/bash
cd /mne-python
cp /workspace/solution.py {file_to_fix}
python -m pytest {test_file} -x -q --timeout=30 \
    --ignore-glob="**/test_*viz*" \
    -p no:pytest-qt \
    --no-header 2>&1 | tail -30
exit ${{PIPESTATUS[0]}}
"""
        runner_path = os.path.join(tmpdir, "run.sh")
        with open(runner_path, "w") as f:
            f.write(runner)
        os.chmod(runner_path, 0o755)

        try:
            result = subprocess.run(
                ["docker", "run", "--rm",
                 "-v", f"{tmpdir}:/workspace",
                 DOCKER_IMAGE,
                 "bash", "/workspace/run.sh"],
                capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
            )
            output = result.stdout + result.stderr

            # Check if tests actually ran (not just failed at collection/import)
            tests_ran = ("passed" in output or " failed" in output) and "ERROR collecting" not in output
            passed = result.returncode == 0 and "passed" in output and "ERROR" not in output

            return {"ran": tests_ran, "passed": passed, "output": output[:1500]}

        except subprocess.TimeoutExpired:
            return {"ran": False, "passed": False, "output": "TIMEOUT"}
        except Exception as e:
            return {"ran": False, "passed": False, "output": str(e)}


# ── 4. Main scorer ────────────────────────────────────────────────────────────

def find_change_region(broken_lines, fixed_lines, context_lines=40):
    """Find the region in broken_lines that needs to change, including insertions."""
    import difflib
    change_points = []
    matcher = difflib.SequenceMatcher(None, broken_lines, fixed_lines, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        if op == "insert":
            # Pure insertion at position i1 in broken
            change_points.append(i1)
        else:
            # replace or delete — lines i1:i2 changed
            change_points.extend(range(i1, max(i1 + 1, i2)))
    return change_points


def reconstruct_full_file(task: dict, solution_snippet: str) -> str:
    """
    For large files where the model only returned a snippet,
    splice the snippet back into the original file.
    """
    broken = task["broken_code"]

    import difflib
    broken_lines = broken.splitlines()
    fixed_lines = task["fixed_code"].splitlines()

    change_points = find_change_region(broken_lines, fixed_lines)

    if not change_points:
        return solution_snippet

    context_lines = 40
    start = max(0, min(change_points) - context_lines)
    end = min(len(broken_lines), max(change_points) + context_lines)

    snippet_lines = solution_snippet.splitlines()
    reconstructed = broken_lines[:start] + snippet_lines + broken_lines[end:]
    return "\n".join(reconstructed)


def extract_ground_truth_snippet(task: dict) -> str:
    """
    Extract the same region from the ground truth (fixed_code) that we sent to the model.
    This lets us compare the model's snippet directly to the correct snippet.
    """
    broken_lines = task["broken_code"].splitlines()
    fixed_lines = task["fixed_code"].splitlines()

    change_points = find_change_region(broken_lines, fixed_lines)
    if not change_points:
        return task["fixed_code"]

    context_lines = 40
    # In fixed_code, the insertion shifts lines — use same window on fixed_code
    start = max(0, min(change_points) - context_lines)
    end = min(len(fixed_lines), max(change_points) + context_lines + 5)  # +5 for insertions
    return "\n".join(fixed_lines[start:end])


def looks_like_full_file(solution_code: str, task: dict) -> bool:
    """Detect whether the model returned a full file vs. a snippet."""
    file_start = task["broken_code"].splitlines()[:20]
    sol_start = solution_code.splitlines()[:40]
    overlap = sum(1 for line in file_start if line.strip() and line in sol_start)
    return overlap >= 5 or len(solution_code) > 0.5 * len(task["broken_code"])


def extract_region_from_code(task: dict, code: str) -> str:
    """Extract only the changed region from a full-file solution, for focused comparison."""
    broken_lines = task["broken_code"].splitlines()
    fixed_lines = task["fixed_code"].splitlines()
    code_lines = code.splitlines()

    change_points = find_change_region(broken_lines, fixed_lines)
    if not change_points:
        return code

    context_lines = 40
    start = max(0, min(change_points) - context_lines)
    end = min(len(code_lines), max(change_points) + context_lines + 5)
    return "\n".join(code_lines[start:end])


def score_solution(task: dict, solution_code: str) -> dict:
    task_id = task["task_id"]
    broken = task["broken_code"]
    is_large_file = len(broken) > 10000

    # Detect whether we got a full file back (e.g. from mini-swe-agent) or a snippet
    returned_full_file = is_large_file and looks_like_full_file(solution_code, task)

    if is_large_file and not returned_full_file:
        # One-shot snippet mode: compare snippet to ground-truth snippet
        ground_truth_for_comparison = extract_ground_truth_snippet(task)
        compare_solution = solution_code
        test_code = reconstruct_full_file(task, solution_code)
    elif is_large_file and returned_full_file:
        # Full file from mini-swe-agent: compare only the changed region to avoid
        # inflated similarity from the huge unchanged portions of the file
        ground_truth_for_comparison = extract_ground_truth_snippet(task)
        compare_solution = extract_region_from_code(task, solution_code)
        test_code = solution_code
    else:
        # Small file: compare full solution to full ground truth
        ground_truth_for_comparison = task["fixed_code"]
        compare_solution = solution_code
        test_code = solution_code

    # Step 1: syntax check the actual code that would be tested
    # test_code is always a full file (reconstructed or returned directly)
    if not check_syntax(test_code):
        return {
            "task_id": task_id,
            "score": 0.0,
            "method": "syntax_fail",
            "passed_tests": False,
            "tests_ran": False,
            "similarity": 0.0,
            "output": "Syntax error in solution",
        }

    # Step 2: similarity to ground truth
    sim = similarity_score(compare_solution, ground_truth_for_comparison)

    # Step 3: try running tests in Docker
    if USE_DOCKER:
        test_result = run_tests_in_docker(task, test_code)
    else:
        test_result = {"ran": False, "passed": False, "output": "Docker disabled"}

    # Step 4: compute final score
    if test_result["ran"] and test_result["passed"]:
        # Tests confirmed correct — full score
        score = 1.0
        method = "tests_passed"
    elif test_result["ran"] and not test_result["passed"]:
        # Tests ran and failed — penalize but keep some similarity credit
        score = round(sim * 0.4, 4)
        method = "tests_failed"
    else:
        # Tests couldn't run — use similarity as proxy (conservative scale)
        # MNE PRs change very few lines, so high similarity doesn't guarantee correctness
        if sim >= 0.999:
            score = 0.70
        elif sim >= 0.995:
            score = 0.45
        elif sim >= 0.98:
            score = 0.25
        elif sim >= 0.90:
            score = 0.15
        else:
            score = round(sim * 0.1, 4)
        method = "similarity"

    return {
        "task_id": task_id,
        "score": score,
        "method": method,
        "passed_tests": test_result.get("passed", False),
        "tests_ran": test_result.get("ran", False),
        "similarity": sim,
        "output": test_result.get("output", "")[:500],
    }


def score_from_files(task_path: str, solution_path: str) -> dict:
    with open(task_path) as f:
        task = json.load(f)
    with open(solution_path) as f:
        solution_code = f.read()
    return score_solution(task, solution_code)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python score.py <task.json> <solution.py>")
        sys.exit(1)
    result = score_from_files(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2))
