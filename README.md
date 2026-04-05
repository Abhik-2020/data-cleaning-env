# 🧹 Data Cleaning RL Environment

> An AI agent that learns to clean messy tabular data using Reinforcement Learning — built for the OpenEnv Hackathon 2026.

**🌐 Live API:** https://abhik-2020-data-cleaning-env.hf.space/docs  
**💻 GitHub:** https://github.com/Abhik-2020/data-cleaning-env  
**👤 Author:** Abhik  
**🏆 Built for:** OpenEnv Hackathon 2026  

---

## 🤔 What is This Project?

Most real-world datasets are messy — they have **missing values, broken emails, duplicate rows, and invalid data**. Normally, a developer writes fixed rules to clean this. But what if an AI could **learn to clean data by itself?**

That is exactly what this project does.

This is a **Reinforcement Learning Environment** where:
- The **environment** holds a dirty CSV file
- The **AI agent** tries different cleaning actions
- The **reward system** gives points for fixing problems
- The agent **learns over time** to clean data in the fewest steps possible

It is built on the **OpenEnv specification** — a standard way to expose RL environments as REST APIs, so any agent from anywhere can connect and interact.

---

## 🎯 The Problem We Are Solving

| Problem in Data | Example | Impact |
|----------------|---------|--------|
| Missing age | Rahul Verma — age is blank | Analysis breaks |
| Negative age | Amit Joshi: -3, Pooja Gupta: -8 | Impossible values |
| Duplicate rows | Same person appears 2-3 times | Inflated counts |
| Double @@ in email | rahul@@yahoo.com | Email undeliverable |
| No @ in email | amitjoshi | Completely invalid |
| Incomplete email | vikram@ | Cannot deliver |

**Before cleaning:** 15 rows, 3 types of issues  
**After cleaning:** 12 rows, zero issues ✅

---

## 🧠 How It Works

```
Dirty CSV Data
      ↓
  Environment (FastAPI Server)
      ↓
  Agent observes current issues
      ↓
  Agent picks an action
  (fill_missing / fix_email / remove_duplicates)
      ↓
  Environment applies action
      ↓
  Agent receives reward (+) or penalty (-)
      ↓
  Repeat until data is fully clean!
```

### Reward System

| Action | Reward | Why? |
|--------|--------|------|
| fill_missing (works) | +0.9 | Fixed missing/negative ages |
| fix_email (works) | +0.9 | Fixed broken emails |
| remove_duplicates (works) | +0.4 | Removed duplicate rows |
| Data fully clean! | +5.0 BONUS | Task complete! |
| Useless action | -0.1 | Wasted a step |

### Agent Learning Progress

```
Episode 1  → Takes 20 steps, never finishes  ❌
Episode 8  → Finishes in just 4 steps        ✅
Episode 15 → Finishes in 14 steps            ✅
Episode 18 → Finishes in 10 steps            ✅
```

The agent gets **smarter with every episode!**

---

## 🗂️ Project Structure

```
data-cleaning-env/
├── server/
│   ├── __init__.py       # Python package init
│   ├── app.py            # FastAPI server — all API endpoints
│   └── environment.py    # Core RL logic — cleaning, rewards, issue detection
├── agent.py              # Q-Learning agent — trains via API calls
├── inference.py          # Rule-based greedy cleaner — no server needed
├── models.py             # Pydantic request/response models
├── data.csv              # Dirty input dataset (15 messy rows)
├── openenv.yaml          # OpenEnv specification manifest
├── Dockerfile            # Docker container setup
├── docker-compose.yml    # Multi-container orchestration
└── pyproject.toml        # Project metadata and dependencies
```

---

## ⚙️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.11 | Core language |
| FastAPI | REST API server |
| Uvicorn | ASGI server |
| Pandas | Data manipulation |
| Pydantic | Data validation |
| Requests | HTTP client for agent |
| Q-Learning | RL algorithm |
| Regex | Email validation & fixing |
| Docker | Containerization |
| Docker Compose | Multi-container setup |
| Hugging Face Spaces | Cloud deployment |
| GitHub | Version control |
| OpenEnv | RL environment standard |

---

## 🚀 Quick Start

### Run Locally

```bash
# Step 1 - Install dependencies
pip install fastapi uvicorn pydantic pandas requests

# Step 2 - Start server (Terminal 1)
uvicorn server.app:app --reload

# Step 3 - Train AI agent (Terminal 2)
python agent.py

# Step 4 - Run rule-based cleaning
python inference.py

# Step 5 - See cleaned result
python -c "import pandas as pd; print(pd.read_csv('cleaned_data.csv'))"
```

### Run with Docker

```bash
# Build image
docker build -t data-cleaning-env .

# Start server
docker-compose up server

# Run agent (new terminal)
docker-compose run agent
```

---

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Environment info |
| `/health` | GET | Server health check |
| `/reset` | POST | Reset to dirty data state |
| `/observation` | GET | See current data without changes |
| `/step` | POST | Apply a cleaning action |

### Example Request & Response

```bash
POST /step
Content-Type: application/json

{"action_type": "fill_missing"}
```

```json
{
  "observation": {
    "issues": ["duplicates", "invalid_email"],
    "data_preview": [...]
  },
  "reward": 0.9,
  "done": false
}
```

---

## 🧪 How to Test

### Option 1 — Live Browser (No Setup Needed!)

Open: **https://abhik-2020-data-cleaning-env.hf.space/docs**

Interactive Swagger UI — test all endpoints by clicking buttons!

---

### Option 2 — Step by Step on /docs Page

#### Step 1 — Reset (Load dirty data)
- Click **POST /reset** → **Try it out** → **Execute**

✅ Response:
```json
{"observation": {"issues": ["duplicates", "missing_age", "invalid_email"]}}
```
> 3 problems detected!

---

#### Step 2 — Fix Ages
- Click **POST /step** → **Try it out**
- Paste: `{"action_type": "fill_missing"}`
- Click **Execute**

✅ Response:
```json
{"reward": 0.9, "done": false, "observation": {"issues": ["duplicates", "invalid_email"]}}
```
> missing_age fixed! Null → 27, Negative (-3, -8) → 27

---

#### Step 3 — Fix Emails
- Click **POST /step** → **Try it out**
- Paste: `{"action_type": "fix_email"}`
- Click **Execute**

✅ Response:
```json
{"reward": 0.9, "done": false, "observation": {"issues": ["duplicates"]}}
```
> Emails fixed! rahul@@yahoo.com → rahul@yahoo.com

---

#### Step 4 — Remove Duplicates
- Click **POST /step** → **Try it out**
- Paste: `{"action_type": "remove_duplicates"}`
- Click **Execute**

✅ Response:
```json
{"reward": 5.4, "done": true, "observation": {"issues": []}}
```
> 🎉 Data fully clean! 15 rows → 12 rows. done: true!

---

### Option 3 — Python Script

```python
import requests

BASE = "https://abhik-2020-data-cleaning-env.hf.space"

# Reset
r = requests.post(f"{BASE}/reset")
print("Issues found:", r.json()["observation"]["issues"])

# Fix ages
r = requests.post(f"{BASE}/step", json={"action_type": "fill_missing"})
print("Reward:", r.json()["reward"])

# Fix emails
r = requests.post(f"{BASE}/step", json={"action_type": "fix_email"})
print("Reward:", r.json()["reward"])

# Remove duplicates
r = requests.post(f"{BASE}/step", json={"action_type": "remove_duplicates"})
print("Done:", r.json()["done"])
print("Issues left:", r.json()["observation"]["issues"])
```

---

## 📊 OpenEnv Specification

This environment is fully compliant with the **OpenEnv 1.0 spec** (`openenv.yaml`):

```yaml
observation_space:
  - data_preview: array of row objects
  - issues: list of detected problems

action_space:
  - fill_missing
  - fix_email  
  - remove_duplicates

tasks:
  - easy:   Remove duplicates only
  - medium: Fill missing values
  - hard:   Fix all issues (full clean)
```

---

## 💡 Future Ideas

- Train a Deep Q-Network (DQN) for larger datasets
- Add more issue types: wrong dates, type mismatches, outliers
- Live dashboard to visualize agent training in real time
- Support multi-table and multi-column datasets
- Deploy multiple agents training simultaneously

---

*Made with ❤️ for OpenEnv Hackathon 2026 — Abhik*