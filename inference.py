"""
Baseline inference script for Data Cleaning RL Environment.
Uses OpenAI-compatible client with Together AI.
Set API key via environment variable: OPENAI_API_KEY
"""

import os
import sys
import requests

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8b-chat-hf")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.together.xyz/v1")

TASKS = ["easy", "medium", "hard"]
ACTIONS = ["fill_missing", "fix_email", "remove_duplicates"]


def call_llm(prompt: str) -> str:
    """Call LLM via OpenAI-compatible API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM error: {e} — using rule-based fallback")
        return None


def pick_action_with_llm(issues: list) -> str:
    """Ask LLM which action to take given current issues."""
    if not OPENAI_API_KEY:
        return pick_action_rule_based(issues)

    prompt = f"""You are a data cleaning agent. The current data issues are: {issues}
Available actions: fill_missing, fix_email, remove_duplicates
Reply with ONLY the action name, nothing else."""

    response = call_llm(prompt)
    if response and response in ACTIONS:
        return response
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
    print(f"\n{'='*50}")
    print(f"TASK: {task.upper()}")
    print(f"{'='*50}")

    # Reset environment
    r = requests.post(f"{BASE_URL}/reset", params={"task": task})
    obs = r.json()["observation"]
    print(f"Initial issues: {obs['issues']}")

    done = False
    step = 0

    while not done:
        step += 1
        issues = obs["issues"]

        if not issues:
            break

        action = pick_action_with_llm(issues)
        print(f"[STEP {step}] Action: {action}")

        r = requests.post(f"{BASE_URL}/step", json={"action_type": action})
        result = r.json()
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        info = result.get("info", {})

        print(f"         Reward: {reward} | Issues left: {obs['issues']} | Done: {done}")

    # Get grade
    r = requests.get(f"{BASE_URL}/grade", params={"task": task})
    grade_result = r.json()
    score = grade_result["score"]
    print(f"\nFINAL SCORE for '{task}': {score} ({'PASSED' if score == 1.0 else 'PARTIAL'})")
    return score


def main():
    print("Data Cleaning RL Environment — Baseline Inference")
    print(f"Server: {BASE_URL}")
    print(f"Model: {MODEL_NAME if OPENAI_API_KEY else 'Rule-based (no API key)'}")

    scores = {}
    for task in TASKS:
        scores[task] = run_task(task)

    print(f"\n{'='*50}")
    print("BASELINE SCORES SUMMARY")
    print(f"{'='*50}")
    for task, score in scores.items():
        status = "✅ PASSED" if score == 1.0 else f"⚠️  PARTIAL ({score})"
        print(f"  {task.upper():<10} {status}")

    avg = sum(scores.values()) / len(scores)
    print(f"\n  AVERAGE SCORE: {avg:.2f}")
    print(f"{'='*50}")

    return scores


if __name__ == "__main__":
    main()