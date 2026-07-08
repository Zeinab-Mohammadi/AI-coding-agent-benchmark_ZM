"""
evaluate_kimi.py — Required evaluator for Kimi K2.5 by MoonshotAI.

Runs the full MNE-Python neuroscience benchmark against Kimi K2.5
using mini-swe-agent via OpenRouter and saves results to miniswe_kimi.json.

Usage:
    export OPENROUTER_API_KEY="your_key_here"
    python evaluate_kimi.py

Results are saved to: ../results/miniswe_kimi.json
"""

import os
import sys
import json

# Kimi K2.5 model ID for mini-swe-agent via OpenRouter
KIMI_MODEL = "openrouter/moonshotai/kimi-k2.5"  # Kimi K2.5 by MoonshotAI

TASKS_DIR = os.path.join(os.path.dirname(__file__), "..", "tasks")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "miniswe_kimi.json")


def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set.")
        print("Run: export OPENROUTER_API_KEY='your_key_here'")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("MNE-Python Neuroscience Benchmark")
    print(f"Model: Kimi K2.5 ({KIMI_MODEL})")
    print(f"Tasks: {TASKS_DIR}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)
    print()

    # Import and run the mini-swe-agent benchmark runner
    sys.path.insert(0, os.path.dirname(__file__))
    from run_miniswe import run_benchmark

    run_benchmark(
        model=KIMI_MODEL,
        tasks_dir=TASKS_DIR,
        output_path=OUTPUT_PATH,
    )

    # Print final summary
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            results = json.load(f)
        print("\n" + "=" * 60)
        print("KIMI K2.5 FINAL RESULTS")
        print("=" * 60)
        print(f"Average score: {results['average_score']:.3f} / 1.0")
        print(f"Total tasks: {results['total_tasks']}")
        print()
        print("Per-topic breakdown:")
        topic_scores = {}
        for r in results["results"]:
            t = r.get("topic", "unknown")
            topic_scores.setdefault(t, []).append(r.get("score", 0))
        for topic, scores in sorted(topic_scores.items()):
            avg = sum(scores) / len(scores)
            print(f"  {topic:25s}: {avg:.3f}  ({len(scores)} tasks)")


if __name__ == "__main__":
    main()
