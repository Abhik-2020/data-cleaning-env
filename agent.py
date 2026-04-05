import requests
import random

BASE_URL = "https://abhik-2020-data-cleaning-env.hf.space"

ACTIONS = ["fill_missing", "fix_email", "remove_duplicates"]

Q = {}

alpha = 0.1
gamma = 0.9
epsilon = 0.2


def get_state_key(obs):
    issues = obs.get("issues", [])
    return tuple(sorted(issues))


def choose_action(state):
    if random.random() < epsilon:
        return random.choice(ACTIONS)

    if state not in Q:
        Q[state] = {a: 0 for a in ACTIONS}

    return max(Q[state], key=Q[state].get)


def update_q(state, action, reward, next_state):
    if state not in Q:
        Q[state] = {a: 0 for a in ACTIONS}

    if next_state not in Q:
        Q[next_state] = {a: 0 for a in ACTIONS}

    best_next = max(Q[next_state].values())

    Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])


def train(episodes=10):
    for ep in range(episodes):
        print(f"\n[EPISODE {ep+1}]")

        res = requests.post(f"{BASE_URL}/reset")
        data = res.json()

        obs = data["observation"]
        state = get_state_key(obs)

        done = False
        step = 0

        while not done:
            step += 1

            action = choose_action(state)

            res = requests.post(f"{BASE_URL}/step", json={
                "action_type": action
            })

            data = res.json()

            next_obs = data["observation"]
            reward = data["reward"]
            done = data["done"]

            next_state = get_state_key(next_obs)

            update_q(state, action, reward, next_state)

            print(f"[STEP {step}] {action} reward={reward}")

            state = next_state

    print("\n🔥 TRAINING COMPLETE")
    print(Q)


if __name__ == "__main__":
    train(episodes=20)