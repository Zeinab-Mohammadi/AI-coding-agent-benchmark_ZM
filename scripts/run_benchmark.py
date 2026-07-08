"""
Benchmark runner for MNE-Python neuroscience benchmark.

Sends each task to an AI model via OpenRouter, collects solutions,
scores them, and saves results.

Usage:
    python run_benchmark.py --model <model_id> --tasks_dir ../tasks --output results_<model>.json

Example:
    python run_benchmark.py --model openrouter/moonshotai/kimi-k2.5 --tasks_dir ../tasks --output results_kimi.json
"""

import os
import json
import time
import argparse
import requests
from score import score_solution

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_model(prompt: str, model: str) -> str:
    """Send a prompt to a model via OpenRouter and return the response text."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/silent-bug-bench",
        "X-Title": "silent-bug-bench",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert Python developer and neuroscientist. "
                    "When given buggy Python code from the MNE-Python library, "
                    "you return the corrected code with the bug fixed. "
                    "Return ONLY Python code, no explanations, no markdown code blocks."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 8000,
    }

    for attempt in range(4):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=360)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning") or ""
            return content.strip() if content else ""
        except Exception as e:
            print(f"    API error (attempt {attempt+1}/4): {e}")
            time.sleep(10 * (attempt + 1))

    return ""


def extract_code(response: str) -> str:
    """Strip markdown code fences if the model added them.
    Also handles reasoning models (e.g. Kimi K2.6) where code appears inside thinking text.
    """
    if not response:
        return ""
    # For reasoning models: use the LAST ```python block — it's the final answer after thinking
    if "```python" in response:
        last_start = response.rfind("```python") + len("```python")
        end = response.find("```", last_start)
        if end > last_start:
            return response[last_start:end].strip()
    # Handle ``` ... ``` blocks — again use the last one
    if "```" in response:
        last_start = response.rfind("```")
        # search backward for the opening fence
        open_pos = response.rfind("```", 0, last_start)
        if open_pos != -1 and open_pos < last_start:
            start = open_pos + 3
            newline = response.find("\n", start)
            if newline != -1 and newline - start < 20:
                start = newline
            if last_start > start:
                return response[start:last_start].strip()
    # If response starts with a shebang or import, treat as raw code
    stripped = response.strip()
    if stripped.startswith(("import ", "from ", "#", "\"\"\"", "def ", "class ")):
        return stripped
    # For reasoning models: find the largest contiguous block of Python-looking lines
    # Blank lines are NOT counted as code (they would merge unrelated blocks)
    lines = stripped.splitlines()
    code_lines = []
    best_block = []
    for line in lines:
        is_code = line.strip() != "" and line.startswith(
            ("import ", "from ", "def ", "class ", "    ", "\t", "#", "@", '"""', "'''", "return ", "if ", "for ", "while ", "with ", "try:", "except", "raise ", "yield ")
        )
        if is_code:
            code_lines.append(line)
        else:
            if len(code_lines) > len(best_block):
                best_block = code_lines[:]
            code_lines = []
    if len(code_lines) > len(best_block):
        best_block = code_lines
    if len(best_block) > 5:
        return "\n".join(best_block).strip()
    return stripped


def extract_changed_context(broken: str, fixed: str, context_lines: int = 40) -> tuple:
    """
    Find the lines that changed between broken and fixed code.
    Return a (snippet, start_line, end_line) tuple with context around the changes.
    """
    import difflib
    broken_lines = broken.splitlines()
    fixed_lines = fixed.splitlines()

    changed = []
    matcher = difflib.SequenceMatcher(None, broken_lines, fixed_lines, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        if op == "insert":
            changed.append(i1)
        else:
            changed.extend(range(i1, max(i1 + 1, i2)))

    if not changed:
        # No changes found — just return beginning of file
        snippet = "\n".join(broken_lines[:80])
        return snippet, 0, 80

    start = max(0, min(changed) - context_lines)
    end = min(len(broken_lines), max(changed) + context_lines)
    snippet = "\n".join(broken_lines[start:end])
    return snippet, start, end


def make_prompt(task: dict) -> str:
    """Build a focused prompt — always send just the changed region with context."""
    broken = task["broken_code"]
    file_path = task["file_to_fix"]
    hint = task["prompt"].split("Context:")[1].split("File:")[0].strip() if "Context:" in task["prompt"] else ""

    MAX_CHARS = 6000
    if len(broken) <= MAX_CHARS:
        # Small file — send the whole thing
        code_section = broken
        instruction = "Return the COMPLETE corrected Python file with the bug fixed. Python code only, no explanations."
    else:
        # Large file — send only the changed region with context
        snippet, start, end = extract_changed_context(broken, task["fixed_code"])
        code_section = snippet
        instruction = (
            f"Return ONLY the corrected version of this code snippet (lines {start}-{end} of the file). "
            f"Python code only, no explanations, no markdown."
        )

    return (
        f"You are fixing a bug in MNE-Python, a neuroscience EEG/MEG analysis library.\n\n"
        f"Context: {hint}\n\n"
        f"File: {file_path}\n\n"
        f"```python\n{code_section}\n```\n\n"
        f"{instruction}"
    )


def run_benchmark(model: str, tasks_dir: str, output_path: str, max_tasks: int = None):
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY environment variable not set.")
        print("Run: export OPENROUTER_API_KEY='your_key_here'")
        return

    # Load all tasks
    task_files = sorted([f for f in os.listdir(tasks_dir) if f.endswith(".json") and f != "summary.json"])
    if max_tasks:
        task_files = task_files[:max_tasks]
    print(f"Found {len(task_files)} tasks.")
    print(f"Model: {model}")
    print(f"Output: {output_path}\n")

    results = []
    total_score = 0.0

    for i, task_file in enumerate(task_files):
        task_path = os.path.join(tasks_dir, task_file)
        with open(task_path) as f:
            task = json.load(f)

        print(f"[{i+1}/{len(task_files)}] Task: {task['task_id']}")
        print(f"  PR: {task['pr_title']}")

        # Get model solution
        prompt = make_prompt(task)
        response = call_model(prompt, model)
        solution_code = extract_code(response)

        if not solution_code:
            print("  -> No response from model")
            result = {"task_id": task["task_id"], "topic": task["topic"], "score": 0.0,
                      "passed_tests": False, "output": "No response", "model": model}
        else:
            # Score the solution
            result = score_solution(task, solution_code)
            result["topic"] = task["topic"]
            result["model"] = model
            result["solution_length"] = len(solution_code)

        score = result.get("score", 0.0)
        total_score += score
        print(f"  -> Score: {score:.1f} | Tests passed: {result.get('passed_tests')}")

        results.append(result)

        # Save incrementally so we don't lose progress
        with open(output_path, "w") as f:
            json.dump({
                "model": model,
                "total_tasks": len(results),
                "average_score": total_score / len(results),
                "results": results,
            }, f, indent=2)

        time.sleep(5)  # be polite to the API

    avg_score = total_score / len(results) if results else 0
    print(f"\n{'='*50}")
    print(f"Model: {model}")
    print(f"Average score: {avg_score:.3f}")
    print(f"Tasks completed: {len(results)}")

    # Print per-topic breakdown
    topic_scores = {}
    for r in results:
        t = r.get("topic", "unknown")
        topic_scores.setdefault(t, []).append(r.get("score", 0))
    print("\nPer-topic scores:")
    for topic, scores in sorted(topic_scores.items()):
        print(f"  {topic}: {sum(scores)/len(scores):.3f} ({len(scores)} tasks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MNE neuroscience benchmark")
    parser.add_argument("--model", required=True, help="OpenRouter model ID")
    parser.add_argument("--tasks_dir", default="../tasks", help="Path to tasks directory")
    parser.add_argument("--output", required=True, help="Output JSON file for results")
    args = parser.parse_args()

    run_benchmark(args.model, args.tasks_dir, args.output)
