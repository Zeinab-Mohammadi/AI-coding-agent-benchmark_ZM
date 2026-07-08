"""
Task generator for MNE-Python neuroscience benchmark.

Pulls real FIX: PRs from MNE-Python's history in 5 neuroscience topic areas,
extracts before/after code diffs, and saves 50 tasks as JSON files.
"""

import os
import json
import time
import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
REPO = "mne-tools/mne-python"
TASKS_DIR = os.path.join(os.path.dirname(__file__), "..", "tasks")
TASKS_PER_TOPIC = 10

# 5 neuroscience topic areas — keywords matched against PR titles and changed file paths
TOPICS = {
    "epoching": {
        "keywords": ["epoch", "baseline", "event", "tmin", "tmax", "crop"],
        "hint": (
            "This bug is in MNE-Python's epoching pipeline. "
            "Remember: in a standard EEG/MEG workflow, epochs are time-locked segments "
            "around experimental events. Baseline correction should use a pre-stimulus period. "
            "Incorrect time alignment or baseline logic corrupts all downstream analyses."
        ),
    },
    "time_frequency": {
        "keywords": ["spectrum", "psd", "tfr", "morlet", "welch", "frequency", "power", "spectral"],
        "hint": (
            "This bug is in MNE-Python's time-frequency or spectral analysis pipeline. "
            "Power spectral density (PSD) and time-frequency representations (TFR) must "
            "preserve correct frequency resolution and units. Dimension mismatches or "
            "incorrect normalization lead to wrong power estimates across all frequencies."
        ),
    },
    "source_localization": {
        "keywords": ["inverse", "forward", "dipole", "beamformer", "lcmv", "dics", "morph", "source"],
        "hint": (
            "This bug is in MNE-Python's source localization pipeline. "
            "Source localization maps scalp-level EEG/MEG signals back to brain regions. "
            "Errors in forward model computation, inverse operator application, or "
            "source morphing between brain templates produce incorrect brain activation maps."
        ),
    },
    "artifact_rejection": {
        "keywords": ["ica", "artifact", "bad", "interpolat", "ssp", "reject", "annotate"],
        "hint": (
            "This bug is in MNE-Python's artifact rejection pipeline. "
            "Artifact rejection removes noise from muscle movements, eye blinks, and bad electrodes. "
            "ICA (Independent Component Analysis) separates brain signals from artifacts. "
            "Bugs here mean artifact contamination survives into cleaned data."
        ),
    },
    "channel_handling": {
        "keywords": ["montage", "channel", "picks", "ch_type", "fnirs", "eeg", "meg", "sensor"],
        "hint": (
            "This bug is in MNE-Python's channel handling pipeline. "
            "Channels represent individual EEG/MEG/fNIRS sensors on the scalp or brain. "
            "Correct channel types, positions, and selection (picks) are critical — "
            "mixing up sensor types or locations invalidates spatial analyses."
        ),
    },
}


def get_merged_prs(page=1, per_page=100):
    url = f"https://api.github.com/repos/{REPO}/pulls"
    params = {"state": "closed", "per_page": per_page, "page": page, "sort": "updated", "direction": "desc"}
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return [pr for pr in resp.json() if pr.get("merged_at")]


def get_pr_files(pr_number):
    url = f"https://api.github.com/repos/{REPO}/pulls/{pr_number}/files"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def get_file_at_commit(path, commit_sha):
    url = f"https://api.github.com/repos/{REPO}/contents/{path}?ref={commit_sha}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return None
    import base64
    content = resp.json().get("content", "")
    try:
        return base64.b64decode(content).decode("utf-8", errors="replace")
    except Exception:
        return None


def pr_matches_topic(pr, topic_keywords):
    title = pr.get("title", "").lower()
    body = (pr.get("body") or "").lower()
    return any(kw in title or kw in body for kw in topic_keywords)


def is_fix_pr(pr):
    title = pr.get("title", "").upper()
    return title.startswith("FIX") or title.startswith("BUG") or "[FIX]" in title or "[BUG]" in title


def is_source_file(path):
    return (
        path.endswith(".py")
        and not path.startswith("doc/")
        and not path.startswith("examples/")
        and "changelog" not in path.lower()
        and "changes" not in path.lower()
    )


def is_test_file(path):
    return path.endswith(".py") and ("test" in path.lower())


def extract_task_from_pr(pr, topic_name, topic_info):
    pr_number = pr["number"]
    files = get_pr_files(pr_number)

    source_files = [f for f in files if is_source_file(f["filename"]) and not is_test_file(f["filename"])]
    test_files = [f for f in files if is_test_file(f["filename"])]

    if not source_files:
        return None

    # Pick the most changed non-test source file
    main_file = max(source_files, key=lambda f: f.get("changes", 0))

    base_sha = pr["base"]["sha"]
    head_sha = pr["head"]["sha"]

    broken_code = get_file_at_commit(main_file["filename"], base_sha)
    fixed_code = get_file_at_commit(main_file["filename"], head_sha)

    if not broken_code or not fixed_code or broken_code == fixed_code:
        return None

    # Require at least one test file — tasks without tests can only use similarity scoring
    test_paths = [f["filename"] for f in test_files]
    if not test_paths:
        return None

    task = {
        "task_id": f"{topic_name}_{pr_number}",
        "topic": topic_name,
        "pr_number": pr_number,
        "pr_title": pr["title"],
        "pr_url": pr["html_url"],
        "file_to_fix": main_file["filename"],
        "broken_code": broken_code,
        "fixed_code": fixed_code,
        "test_files": test_paths,
        "prompt": (
            f"The following Python file from the MNE-Python neuroscience library contains a bug "
            f"that was reported and fixed in PR #{pr_number} ('{pr['title']}').\n\n"
            f"Context: {topic_info['hint']}\n\n"
            f"File: {main_file['filename']}\n\n"
            f"Your task: Fix the bug in the code below. Return the complete corrected file.\n\n"
            f"```python\n{broken_code}\n```"
        ),
    }
    return task


def generate_tasks():
    os.makedirs(TASKS_DIR, exist_ok=True)
    all_tasks = []
    topic_counts = {t: 0 for t in TOPICS}

    print("Fetching merged PRs from MNE-Python...")
    page = 1
    seen_prs = set()

    while any(count < TASKS_PER_TOPIC for count in topic_counts.values()):
        prs = get_merged_prs(page=page, per_page=100)
        if not prs:
            print("No more PRs to fetch.")
            break

        for pr in prs:
            if pr["number"] in seen_prs:
                continue
            seen_prs.add(pr["number"])

            if not is_fix_pr(pr):
                continue

            for topic_name, topic_info in TOPICS.items():
                if topic_counts[topic_name] >= TASKS_PER_TOPIC:
                    continue
                if not pr_matches_topic(pr, topic_info["keywords"]):
                    continue

                print(f"  [{topic_name}] Trying PR #{pr['number']}: {pr['title']}")
                try:
                    task = extract_task_from_pr(pr, topic_name, topic_info)
                    if task:
                        all_tasks.append(task)
                        topic_counts[topic_name] += 1
                        task_path = os.path.join(TASKS_DIR, f"{task['task_id']}.json")
                        with open(task_path, "w") as f:
                            json.dump(task, f, indent=2)
                        print(f"    -> Saved task {task['task_id']} ({topic_counts[topic_name]}/{TASKS_PER_TOPIC})")
                        time.sleep(0.5)
                        break
                except Exception as e:
                    print(f"    -> Error: {e}")
                    time.sleep(1)

        page += 1
        time.sleep(1)

    print(f"\nDone. Generated {len(all_tasks)} tasks.")
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count} tasks")

    summary_path = os.path.join(TASKS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"total": len(all_tasks), "by_topic": topic_counts, "tasks": [t["task_id"] for t in all_tasks]}, f, indent=2)

    return all_tasks


if __name__ == "__main__":
    if not GITHUB_TOKEN:
        print("ERROR: GITHUB_TOKEN environment variable not set.")
        print("Run: export GITHUB_TOKEN='your_token_here'")
        exit(1)
    generate_tasks()
