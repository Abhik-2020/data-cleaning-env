from fastapi import FastAPI
from server.environment import DataCleaningEnv
from models import DataAction
import uvicorn
import os

app = FastAPI(
    title="Data Cleaning RL Environment",
    description="An OpenEnv-compatible reinforcement learning environment for cleaning tabular data.",
    version="1.0.0",
)

env = DataCleaningEnv()
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data.csv")


@app.on_event("startup")
def startup():
    env.load_data(DATA_PATH)
    env.reset()


@app.get("/")
def root():
    return {
        "name": "data-cleaning-env",
        "version": "1.0.0",
        "description": "RL environment for data cleaning tasks",
        "endpoints": ["/reset", "/step", "/state", "/observation", "/grade", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(task: str = "hard"):
    obs = env.reset(task=task)
    return {"observation": obs}


@app.get("/state")
def state():
    return env.get_state()


@app.get("/observation")
def get_observation():
    return {"observation": env._get_observation()}


@app.post("/step")
def step(action: DataAction):
    obs, reward, done, info = env.step(action.action_type)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/grade")
def grade(task: str = "hard"):
    score = env.grade(task)
    return {
        "task": task,
        "score": score,
        "passed": score == 1.0,
    }


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)


if __name__ == "__main__":
    main()