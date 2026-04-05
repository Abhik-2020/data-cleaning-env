# 🧹 Data Cleaning RL Environment

**Author:** Aditi Soni
**Version:** 1.0.0
**Built for:** OpenEnv Hackathon

---

## 🚀 What is this?

A **reinforcement learning environment** where an AI agent learns to clean messy tabular data — removing duplicates, filling missing values, and fixing broken email addresses — all through trial and reward.

Built on top of the **OpenEnv** spec: a standardized way to expose data environments to RL agents via a REST API.

---

## 🧠 How it Works

```
Agent --> POST /step (action) --> Environment
        <-- observation + reward + done <--
```

The environment wraps a dirty CSV file and exposes it as an RL gym:

| Component | Description |
|---|---|
| **State** | A snapshot of current data issues (`duplicates`, `missing_age`, `invalid_email`) |
| **Actions** | `fill_missing`, `fix_email`, `remove_duplicates` |
| **Reward** | +1 fix email/missing, +0.5 remove dups, -0.1/step, +5 bonus when clean |
| **Done** | When no issues remain OR step limit hit |

---

## 📁 Project Structure

```
data-cleaning-env/
├── server/
│   ├── __init__.py
│   ├── app.py            # FastAPI server (REST endpoints)
│   └── environment.py    # Core RL environment logic
├── agent.py              # Q-learning agent (trains via API)
├── inference.py          # Rule-based inference (greedy policy)
├── models.py             # Pydantic request/response models
├── data.csv              # Sample dirty dataset
├── openenv.yaml          # OpenEnv spec manifest
└── pyproject.toml        # Project config & dependencies
```

---

## ⚙️ Setup & Installation

**Requirements:** Python ≥ 3.11, `uv` (recommended) or `pip`

### Using uv (recommended)
```bash
uv venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install -e .
```

### Using pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn pydantic pandas openenv-core
pip install -e .
```

---

## 🏃 Running the Environment Server

```bash
uvicorn server.app:app --reload
```

Server starts at: **http://127.0.0.1:8000**

Interactive API docs: **http://127.0.0.1:8000/docs**

---

## 🤖 Running the RL Agent (Q-Learning)

In a second terminal (with the server running):

```bash
python agent.py
```

The agent runs **20 training episodes**, learning which action to apply given the current data state using a Q-table. It prints rewards per step and the final learned Q-values.

---

## 🔍 Running Inference (Greedy / Rule-Based)

For a deterministic, rule-based cleaning pass (no server needed):

```bash
python inference.py
```

This uses the environment directly and saves the cleaned output to `cleaned_data.csv`.

---

## 🌐 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Environment info |
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment to dirty state |
| `/observation` | GET | Get current data state |
| `/step` | POST | Apply an action |

### Example: `/step` Request
```json
POST /step
{
  "action_type": "fill_missing"
}
```

### Example: `/step` Response
```json
{
  "observation": {
    "data_preview": [...],
    "issues": ["invalid_email"]
  },
  "reward": 0.9,
  "done": false
}
```

---

## 🧪 Sample Data (`data.csv`)

| name | age | email |
|---|---|---|
| Rohit | 25 | rohit@gmail ❌ |
| Rohit | 25 | rohit@gmail ❌ (duplicate) |
| Anjali | *(missing)* | anjali@gmail.com ✅ |
| Amit | -5 ❌ | amit@@gmail.com ❌ |

The agent must learn to clean all of this in the fewest steps possible.

---

## 🏆 Reward Structure

| Event | Reward |
|---|---|
| Fixed missing/invalid age | +1.0 |
| Fixed invalid email(s) | +1.0 |
| Removed duplicate rows | +0.5 |
| Each step taken | -0.1 |
| **Dataset fully clean** | **+5.0 bonus** |

---

## 🔧 OpenEnv Compatibility

This environment is compliant with the **OpenEnv 1.0 spec** (`openenv.yaml`). It can be:
- Registered in the OpenEnv registry
- Loaded by any OpenEnv-compatible agent framework
- Extended with new actions or reward signals

---

## 📦 Dependencies

- `fastapi` — REST API server
- `uvicorn` — ASGI server
- `pydantic` — Data validation
- `pandas` — Data manipulation
- `openenv-core ≥ 0.2.0` — OpenEnv framework

---

## 💡 Ideas for Extension

- Add more data issues: type mismatches, outliers, wrong date formats
- Train a DQN agent instead of tabular Q-learning
- Support multi-column, multi-table datasets
- Add a web dashboard to visualize agent training live

---

*Built with ❤️ for the OpenEnv Hackathon by Aditi Soni*
