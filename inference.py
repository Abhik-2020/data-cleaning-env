"""
Baseline inference script for Data Cleaning RL Environment.
Uses hackathon-provided LiteLLM proxy via API_BASE_URL and API_KEY env vars.
"""

import os
import requests
from openai import OpenAI

# ── Config — uses hackathon injected environment variables ─────────────────────
BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
API_KEY = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

TASKS = ["easy", "medium", "hard"]
ACTIONS = ["fill_missing", "fix_email", "remove_duplicates"]

# Initialize OpenAI client with hackathon LiteLLM proxy
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)


def call_llm(issues: list) -> str:
    """Ask LLM which action to take given current issues."""
    prompt = f"""You are a data cleaning agent. The current data issues are: {issues}
Available actions: fill_missing, fix_email, remove_duplicates
Reply with ONLY the action name, nothing else."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        action = response.choices[0].message.content.strip()
        if action in ACTIONS:
            return action
    except Exception as e:
        print(f"LLM error: {e}", flush=True)

    return pick_action_rule_based(issues)


def pick_action_rule_based(issues: list) -> str:
    """Rule-based fallback policy."""
    if "missing_age" in issues:
        return "fill_missing"
    elif "invalid_email" in issues:
        return "fix_email"
    elif "duplicates" in issues:
        return "remove_duplicates"
    return "fill_missing"


def run_task(task: str) -> float:
    """Run one full episode for a given task and return grade score."""
    r = requests.post(f"{BASE_URL}/reset", params={"task": task})
    obs = r.json()["observation"]

    done = False
    step = 0
    total_reward = 0.0

    print(f"[START] task={task}", flush=True)

    while not done:
        step += 1
        issues = obs["issues"]

        if not issues:
            break

        action = call_llm(issues)

        r = requests.post(f"{BASE_URL}/step", json={"action_type": action})
        result = r.json()
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        total_reward += reward

        print(f"[STEP] step={step} action={action} reward={reward}", flush=True)

    r = requests.get(f"{BASE_URL}/grade", params={"task": task})
    score = r.json()["score"]

    print(f"[END] task={task} score={score} steps={step}", flush=True)

    return score


def main():
    scores = {}
    for task in TASKS:
        scores[task] = run_task(task)

    avg = sum(scores.values()) / len(scores)
    print(f"[SUMMARY] easy={scores['easy']} medium={scores['medium']} hard={scores['hard']} avg={avg:.2f}", flush=True)


if __name__ == "__main__":
    main()