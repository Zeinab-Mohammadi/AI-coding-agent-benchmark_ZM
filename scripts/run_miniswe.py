"""
Mini-SWE-Agent benchmark runner for MNE-Python neuroscience benchmark.

Uses mini-swe-agent as an iterative coding agent (not one-shot) to fix bugs.
The agent can explore files, run bash commands, and iteratively edit code.

Usage:
    python run_miniswe.py --model openrouter/anthropic/claude-3-haiku --output results/miniswe_haiku.json
    python run_miniswe.py --model openrouter/moonshotai/kimi-k2.5 --output results/miniswe_kimi.json
    python run_miniswe.py --model openrouter/meta-llama/llama-3.1-8b-instruct --output results/miniswe_llama.json
"""

import os
import sys
import json
import time
import tempfile
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(__file__))
from score import score_solution

TASKS_DIR = os.path.join(os.path.dirname(__file__), "..", "tasks")
TIMEOUT = 300  # seconds per task


def save_trajectory(trajectory_dir: str, task_id: str, stdout: str, stderr: str):
    """Save agent trajectory (stdout + stderr) to a file."""
    os.makedirs(trajectory_dir, exist_ok=True)
    path = os.path.join(trajectory_dir, f"{task_id}.txt")
    with open(path, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(stdout or "")
        f.write("\n=== STDERR ===\n")
        f.write(stderr or "")
    return path


def run_miniswe_on_task(task: dict, model: str, trajectory_dir: str) -> tuple:
    """
    Run mini-swe-agent on a single task.
    Sets up a temp directory with the buggy file, runs the agent,
    and returns (fixed_code, trajectory_path).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write the buggy file to a temp location
        file_to_fix = task["file_to_fix"]
        filename = os.path.basename(file_to_fix)
        local_path = os.path.join(tmpdir, filename)

        with open(local_path, "w") as f:
            f.write(task["broken_code"])

        # Build the task prompt for mini-swe-agent
        hint = ""
        if "Context:" in task["prompt"]:
            hint = task["prompt"].split("Context:")[1].split("File:")[0].strip()

        agent_task = (
            f"Fix the bug in the file {local_path}.\n\n"
            f"Context: {hint}\n\n"
            f"The file is from the MNE-Python neuroscience library (PR #{task['pr_number']}: {task['pr_title']}).\n"
            f"Read the file, identify the bug, and fix it in place. "
            f"Only modify {local_path}. Do not install packages or run tests."
        )

        # Run mini-swe-agent via subprocess
        env = os.environ.copy()
        env["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")
        env["MSWEA_MODEL_NAME"] = model
        env["MSWEA_COST_TRACKING"] = "ignore_errors"

        cmd = [
            "uvx", "mini-swe-agent",
            "--model", model,
            "--task", agent_task,
            "--yolo",
            "--cost-limit", "0.10",
        ]

        stdout_text = ""
        stderr_text = ""
        agent_failed = False
        failure_reason = ""

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                env=env,
                cwd=tmpdir,
            )
            stdout_text = result.stdout or ""
            stderr_text = result.stderr or ""
            combined = stdout_text + "\n" + stderr_text

            if "AuthenticationError" in combined or "No cookie auth credentials" in combined:
                agent_failed = True
                failure_reason = "OpenRouter authentication failed — check OPENROUTER_API_KEY"
                print(f"    -> Auth error: API key not reaching mini-swe-agent")
            elif result.returncode != 0:
                agent_failed = True
                failure_reason = f"mini-swe-agent exited with code {result.returncode}"

        except subprocess.TimeoutExpired as e:
            print(f"    -> Timeout after {TIMEOUT}s")
            stdout_text = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr_text = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")

        except Exception as e:
            print(f"    -> Error: {e}")
            agent_failed = True
            failure_reason = str(e)

        # Save trajectory
        traj_path = save_trajectory(trajectory_dir, task["task_id"], stdout_text, stderr_text)

        if agent_failed:
            return "", traj_path, failure_reason

        # Read the (hopefully fixed) file
        fixed_code = ""
        if os.path.exists(local_path):
            with open(local_path) as f:
                fixed_code = f.read()

        return fixed_code, traj_path, ""


def run_benchmark(model: str, tasks_dir: str, output_path: str, max_tasks: int = None):
    if model.startswith("openrouter/") and not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set. Run: export OPENROUTER_API_KEY='your_key'")

    task_files = sorted([
        f for f in os.listdir(tasks_dir)
        if f.endswith(".json") and f != "summary.json"
    ])
    if max_tasks:
        task_files = task_files[:max_tasks]

    print(f"Found {len(task_files)} tasks.")
    print(f"Model: {model}")
    print(f"Output: {output_path}\n")

    # Trajectory directory: results/trajectories/<model_shortname>/
    model_short = model.split("/")[-1]
    results_dir = os.path.dirname(output_path)
    trajectory_dir = os.path.join(results_dir, "trajectories", model_short)

    # Resume: load already-completed results from output file if it exists
    results = []
    total_score = 0.0
    completed_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        results = existing.get("results", [])
        total_score = sum(r.get("score", 0.0) for r in results)
        completed_ids = {r["task_id"] for r in results}
        if completed_ids:
            print(f"Resuming: {len(completed_ids)} tasks already done, skipping them.\n")

    for i, task_file in enumerate(task_files):
        with open(os.path.join(tasks_dir, task_file)) as f:
            task = json.load(f)

        # Skip already-completed tasks
        if task["task_id"] in completed_ids:
            continue

        print(f"[{i+1}/{len(task_files)}] Task: {task['task_id']}")
        print(f"  PR: {task['pr_title']}")

        fixed_code, traj_path, failure_reason = run_miniswe_on_task(task, model, trajectory_dir)

        if not fixed_code:
            result = {
                "task_id": task["task_id"],
                "topic": task["topic"],
                "score": 0.0,
                "method": "agent_failed",
                "passed_tests": False,
                "tests_ran": False,
                "similarity": 0.0,
                "output": failure_reason or "No output from agent",
                "trajectory": traj_path,
                "model": model,
            }
        else:
            result = score_solution(task, fixed_code)
            result["topic"] = task["topic"]
            result["model"] = model
            result["trajectory"] = traj_path

        score = result.get("score", 0.0)
        total_score += score
        print(
            f"  -> Score: {score:.4f} | "
            f"Method: {result.get('method')} | "
            f"Tests ran: {result.get('tests_ran')} | "
            f"Tests passed: {result.get('passed_tests')} | "
            f"Similarity: {result.get('similarity')}"
        )

        results.append(result)

        with open(output_path, "w") as f:
            json.dump({
                "model": model,
                "runner": "mini-swe-agent",
                "total_tasks": len(results),
                "average_score": total_score / len(results),
                "results": results,
            }, f, indent=2)

        time.sleep(3)

    avg = total_score / len(results) if results else 0
    print(f"\n{'='*50}")
    print(f"Model: {model}")
    print(f"Average score: {avg:.3f}")

    topic_scores = {}
    for r in results:
        t = r.get("topic", "unknown")
        topic_scores.setdefault(t, []).append(r.get("score", 0))
    print("\nPer-topic scores:")
    for topic, scores in sorted(topic_scores.items()):
        print(f"  {topic}: {sum(scores)/len(scores):.3f} ({len(scores)} tasks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model ID (e.g. openrouter/anthropic/claude-3-haiku)")
    parser.add_argument("--tasks_dir", default="../tasks")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_tasks", type=int, default=None)
    args = parser.parse_args()

    run_benchmark(args.model, args.tasks_dir, args.output, args.max_tasks)
