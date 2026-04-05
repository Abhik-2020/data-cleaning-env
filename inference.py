import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from server.environment import DataCleaningEnv
import pandas as pd

print("=" * 50)
print("       DATA CLEANING - INFERENCE DEMO")
print("=" * 50)

env = DataCleaningEnv()

if not os.path.exists("data.csv"):
    print("ERROR: data.csv file not found")
    exit()

env.load_data("data.csv")

# Show BEFORE
print("\n📋 BEFORE CLEANING (Raw Dirty Data):")
print("-" * 50)
print(env.df.to_string(index=False))
print(f"\n Issues detected:")
for issue in env._get_issues():
    print(f"   ❌ {issue}")

obs = env.reset()
done = False
step = 0

print("\n🤖 Cleaning started...\n")

while not done:
    step += 1
    issues = obs["issues"]

    if "missing_age" in issues:
        action = "fill_missing"
    elif "invalid_email" in issues:
        action = "fix_email"
    elif "duplicates" in issues:
        action = "remove_duplicates"
    else:
        break

    obs, reward, done = env.step(action)
    print(f"  [STEP {step}] ✅ {action} → reward={reward:.2f} | remaining issues: {obs['issues']}")

# Show AFTER
cleaned = env.get_clean_data()
file_path = os.path.abspath("cleaned_data.csv")
cleaned.to_csv(file_path, index=False)

print("\n📋 AFTER CLEANING (Clean Data):")
print("-" * 50)
print(cleaned.to_string(index=False))
print(f"\n✅ CLEANING COMPLETE in {step} steps!")
print(f"💾 Saved at: {file_path}")
print(f"\n Issues remaining: {env._get_issues() or '✅ None!'}")
