"""
Baseline inference script for Data Cleaning RL Environment.
Uses OpenAI-compatible client with Together AI.
Set API key via environment variable: OPENAI_API_KEY
"""

import os
import requests

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8b-chat-hf")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.together.xyz/v1")

TASKS = ["easy", "medium", "hard"]
ACTIONS = ["fill_missing", "fix_email", "remove_duplicates"]


def call_llm(prompt: str) -> str:
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
    except Exception:
        return None


def pick_action_with_llm(issues: list) -> str:
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
    if "missing_age" in issues:
        return "fill_missing"
    elif "invalid_email" in issues:
        return "fix_email"
    elif "duplicates" in issues:
        return "remove_duplicates"
    return "fill_missing"


def run_task(task: str) -> float:
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

        action = pick_action_with_llm(issues)

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