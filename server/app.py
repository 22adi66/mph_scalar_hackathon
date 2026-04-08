"""
FastAPI application for the Software-Defined Security Middle Platform (SDSmp) Environment.
"""

import asyncio
import json
import os
import traceback
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional

try:
    from .sdsmp_environment import SdsmpEnvironment, TASKS
except ImportError:
    from server.sdsmp_environment import SdsmpEnvironment, TASKS

app = FastAPI(
    title="SDSmp Cybersecurity Scheduler Environment",
    description="OpenEnv-compliant environment for IoT security job scheduling.",
    version="1.0.0",
)

_env = SdsmpEnvironment()

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/schema")
async def schema():
    try:
        from ..models import SdsmpAction, SdsmpObservation, SdsmpState
    except ImportError:
        from models import SdsmpAction, SdsmpObservation, SdsmpState
    return {
        "action": SdsmpAction.model_json_schema(),
        "observation": SdsmpObservation.model_json_schema(),
        "state": SdsmpState.model_json_schema(),
    }

@app.get("/tasks")
async def list_tasks():
    try:
        from ..models import SdsmpAction
    except ImportError:
        from models import SdsmpAction
    return {
        "tasks": list(TASKS.values()),
        "action_schema": SdsmpAction.model_json_schema(),
        "available_commands": {
            "schedule_job": {
                "description": "Schedule a pending job to a specific Smp VM",
                "parameters": {"job_id": "string (required)", "vm_id": "string (required)"},
            },
            "noop": {"description": "Do nothing (observe only)", "parameters": {}},
            "submit_evaluation": {"description": "End the episode and submit for grading", "parameters": {}},
        },
    }

class ResetRequest(BaseModel):
    seed: int = 42
    task_id: str = "easy"
    episode_id: str = ""

class StepRequest(BaseModel):
    command: str = "noop"
    parameters: Dict[str, Any] = {}

@app.post("/reset")
async def reset(req: Optional[ResetRequest] = Body(None)):
    if req is None:
        req = ResetRequest()
    obs = _env.reset(seed=req.seed, task_id=req.task_id, episode_id=req.episode_id or None)
    return obs.model_dump()

@app.post("/step")
async def step(req: StepRequest):
    obs = _env.step({"command": req.command, "parameters": req.parameters})
    return obs.model_dump()

@app.get("/state")
async def get_state():
    return _env.state.model_dump()

@app.post("/grader")
async def grader():
    return _env.get_grade()

class BaselineRequest(BaseModel):
    api_key: str = ""
    model: str = "gpt-4o-mini"

@app.post("/baseline")
async def run_baseline(req: BaselineRequest):
    try:
        from .baseline import run_baseline_all_tasks
    except ImportError:
        from server.baseline import run_baseline_all_tasks
    api_key = req.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return JSONResponse(status_code=400, content={"error": "No API key provided."})
    try:
        scores = run_baseline_all_tasks(api_key=api_key, model=req.model)
        return scores
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    env = SdsmpEnvironment()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "data": {"message": "Invalid JSON", "code": "INVALID_JSON"}})
                continue
            msg_type = msg.get("type", "")
            data = msg.get("data", {})
            if msg_type == "reset":
                obs = env.reset(seed=data.get("seed", 42), task_id=data.get("task_id", "easy"), episode_id=data.get("episode_id"))
                await ws.send_json({"type": "observation", "data": obs.model_dump()})
            elif msg_type == "step":
                action_data = data.get("action", data)
                obs = env.step(action_data)
                await ws.send_json({"type": "observation", "data": obs.model_dump()})
            elif msg_type == "state":
                await ws.send_json({"type": "state", "data": env.state.model_dump()})
            elif msg_type == "close":
                await ws.close()
                break
            else:
                await ws.send_json({"type": "error", "data": {"message": f"Unknown message type: {msg_type}", "code": "UNKNOWN_TYPE"}})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "data": {"message": str(e), "code": "EXECUTION_ERROR"}})
        except Exception:
            pass

def main():
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()