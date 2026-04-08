"""
server.py — RedlineEnv HTTP API Server

OpenEnv-compatible REST API. Runs on port 7860 (Hugging Face Spaces standard).

Endpoints:
  GET  /                         → environment info
  GET  /health                   → health check
  GET  /tasks                    → list all tasks
  GET  /tasks/{task_id}          → task detail
  POST /episode/start            → start new episode
  POST /episode/step             → take one action
  GET  /episode/{id}             → get episode state
  POST /episode/{id}/reset       → reset episode
  GET  /leaderboard              → global leaderboard
  POST /leaderboard/submit       → submit a score
  GET  /openenv.yaml             → environment spec (YAML)
  GET  /docs                     → Swagger UI
"""
from __future__ import annotations

import uuid
import time
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

from env import SpecGuardianEnv
from graders import grade_episode
from models import GradeResult, Observation
from tasks import TASKS, get_task, generate_task

app = FastAPI(
    title="RedlineEnv",
    description=(
        "OpenEnv evaluation environment for AI agent safety benchmarking. "
        "Tests whether agents understand systems before acting, apply only safe fixes, "
        "escalate unresolvable issues, and deliver accurate system health verdicts."
    ),
    version="2.0.0",
)

# In-memory stores
SESSIONS: Dict[str, Dict[str, Any]] = {}
LEADERBOARD: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    task_id: str = "easy_01"
    agent_name: Optional[str] = None  # for leaderboard attribution

class StepRequest(BaseModel):
    action: str
    args: Dict[str, Any] = {}

class LeaderboardEntry(BaseModel):
    agent_name: str
    session_id: str
    task_id: str
    score: float
    letter_grade: str
    success: bool
    total_reward: float
    steps_used: int
    system_integrity: float


# ---------------------------------------------------------------------------
# Root & Info
# ---------------------------------------------------------------------------

@app.get("/", response_class=JSONResponse)
def root():
    return {
        "name": "RedlineEnv",
        "version": "2.0",
        "tag": "openenv",
        "description": (
            "An OpenEnv evaluation environment that tests whether an AI agent "
            "understands a system before acting, applies only safe patches, "
            "escalates unresolvable issues, and delivers accurate system health verdicts."
        ),
        "tasks": {
            "handcrafted": list(TASKS.keys()),
            "procedural": "Use task_id format: proc_<easy|medium|hard>_<integer_seed> e.g. proc_hard_42",
        },
        "actions": ["inspect_file", "preview_patch", "apply_patch", "mark_issue", "escalate_issue", "submit_verdict"],
        "verdicts": ["system_healthy", "system_degraded", "system_critical"],
        "endpoints": {
            "GET  /":                          "This info",
            "GET  /health":                    "Health check",
            "GET  /tasks":                     "List all tasks",
            "GET  /tasks/{task_id}":           "Task detail",
            "POST /episode/start":             "Start new episode (body: {task_id, agent_name?})",
            "POST /episode/step?session_id=X": "Take one action (body: {action, args})",
            "GET  /episode/{id}":              "Get current episode state",
            "POST /episode/{id}/reset":        "Reset episode to start",
            "GET  /leaderboard":               "Global leaderboard",
            "POST /leaderboard/submit":        "Submit a score",
            "GET  /openenv.yaml":              "Full environment spec",
            "GET  /docs":                      "Interactive Swagger UI",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "sessions_active": len(SESSIONS), "leaderboard_entries": len(LEADERBOARD)}


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def openenv_spec():
    try:
        with open("openenv.yaml") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="openenv.yaml not found")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@app.get("/tasks")
def list_tasks():
    result = []
    for task_id, task in TASKS.items():
        state = task.build_state()
        result.append({
            "id": task_id,
            "difficulty": task.difficulty,
            "max_steps": state.max_steps,
            "correct_verdict": task.correct_verdict.value,
            "n_files": len(state.files),
            "n_issues": len(state.issues),
            "n_patches": len(state.patches),
            "n_fixable": len(task.fixable_issue_ids),
            "n_unfixable": len(task.non_fixable_issue_ids),
            "n_forbidden_patches": len(task.forbidden_patch_ids),
        })
    return {
        "count": len(result),
        "tasks": result,
        "procedural_info": "Additional tasks available via proc_<easy|medium|hard>_<seed>",
    }


@app.get("/tasks/{task_id}")
def task_detail(task_id: str):
    try:
        task = get_task(task_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    state = task.build_state()
    return {
        "id": task.id,
        "difficulty": task.difficulty,
        "correct_verdict": task.correct_verdict.value,
        "max_steps": state.max_steps,
        "files": [f.model_dump() for f in state.files],
        "issues": [i.model_dump() for i in state.issues],
        "patches": [
            {**p.model_dump(), "is_safe": p.id in task.safe_patch_ids}
            for p in state.patches
        ],
        "metadata": {
            "fixable_issue_ids": task.fixable_issue_ids,
            "non_fixable_issue_ids": task.non_fixable_issue_ids,
            "safe_patch_ids": task.safe_patch_ids,
            "forbidden_patch_ids": task.forbidden_patch_ids,
            "useful_file_ids": task.useful_file_ids,
        },
    }


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

@app.post("/episode/start")
def start_episode(req: StartRequest):
    try:
        task = get_task(req.task_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{req.task_id}' not found. Use /tasks to see available tasks, or proc_<diff>_<seed> for procedural."
        )

    session_id = str(uuid.uuid4())
    env = SpecGuardianEnv(task)
    obs = env.reset()
    state = env.state

    SESSIONS[session_id] = {
        "env": env,
        "task_id": req.task_id,
        "task": task,
        "agent_name": req.agent_name or "anonymous",
        "done": False,
        "started_at": time.time(),
    }

    return {
        "session_id": session_id,
        "task_id": req.task_id,
        "difficulty": task.difficulty,
        "max_steps": state.max_steps,
        "observation": obs.model_dump(),
        "tip": f"Take actions via POST /episode/step?session_id={session_id}",
    }


@app.post("/episode/step")
def step_episode(req: StepRequest, session_id: str = Query(...)):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /episode/start first.")

    session = SESSIONS[session_id]
    env: SpecGuardianEnv = session["env"]

    if session["done"]:
        raise HTTPException(status_code=400, detail="Episode is already done. Use /episode/{id}/reset to restart.")

    valid_actions = {"inspect_file", "preview_patch", "apply_patch", "mark_issue", "escalate_issue", "submit_verdict"}
    if req.action not in valid_actions:
        raise HTTPException(status_code=422,
            detail=f"Unknown action '{req.action}'. Valid: {sorted(valid_actions)}")

    obs, reward, done, event = env.step(req.action, req.args)
    session["done"] = done

    info: Dict[str, Any] = {
        "tags": event.tags,
        "detail": event.detail,
        "step": env.state.step_count,
        "steps_remaining": obs.steps_remaining,
        "system_integrity": obs.system_integrity,
    }

    if done:
        result = grade_episode(env.state, session["task"])
        session["result"] = result
        info["grade"] = _grade_dict(result)

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/episode/{session_id}")
def get_episode(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    session = SESSIONS[session_id]
    env: SpecGuardianEnv = session["env"]
    obs = env.observation()

    result = None
    if session["done"] and "result" in session:
        result = _grade_dict(session["result"])

    return {
        "session_id": session_id,
        "task_id": session["task_id"],
        "agent_name": session["agent_name"],
        "done": session["done"],
        "result": result,
        "observation": obs.model_dump(),
    }


@app.post("/episode/{session_id}/reset")
def reset_episode(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    session = SESSIONS[session_id]
    env: SpecGuardianEnv = session["env"]
    obs = env.reset()
    session["done"] = False
    session.pop("result", None)

    return {
        "session_id": session_id,
        "task_id": session["task_id"],
        "observation": obs.model_dump(),
        "message": "Episode reset successfully.",
    }


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

@app.get("/leaderboard")
def get_leaderboard(
    difficulty: Optional[str] = Query(None, description="Filter by: easy, medium, hard"),
    task_id: Optional[str] = Query(None, description="Filter by specific task"),
    limit: int = Query(50, le=200),
):
    entries = LEADERBOARD.copy()
    if difficulty:
        entries = [e for e in entries if e.get("difficulty") == difficulty]
    if task_id:
        entries = [e for e in entries if e.get("task_id") == task_id]
    entries.sort(key=lambda e: e["score"], reverse=True)
    entries = entries[:limit]

    return {
        "count": len(entries),
        "filters": {"difficulty": difficulty, "task_id": task_id},
        "entries": entries,
    }


@app.post("/leaderboard/submit")
def submit_leaderboard(entry: LeaderboardEntry):
    if entry.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found. Complete an episode first.")

    session = SESSIONS[entry.session_id]
    if not session.get("done"):
        raise HTTPException(status_code=400, detail="Episode not finished yet.")

    result = session.get("result")
    task = session["task"]

    record = {
        "agent_name": entry.agent_name,
        "session_id": entry.session_id,
        "task_id": session["task_id"],
        "difficulty": task.difficulty,
        "score": result.score if result else entry.score,
        "letter_grade": result.letter_grade if result else entry.letter_grade,
        "success": result.success if result else entry.success,
        "total_reward": result.total_reward if result else entry.total_reward,
        "steps_used": result.steps_used if result else entry.steps_used,
        "system_integrity": result.system_integrity if result else entry.system_integrity,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    LEADERBOARD.append(record)
    LEADERBOARD.sort(key=lambda e: e["score"], reverse=True)

    rank = next(i + 1 for i, e in enumerate(LEADERBOARD) if e["session_id"] == entry.session_id)
    return {"message": "Score submitted!", "rank": rank, "entry": record}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _grade_dict(result: GradeResult) -> Dict[str, Any]:
    return {
        "task_id": result.task_id,
        "difficulty": result.difficulty,
        "score": result.score,
        "success": result.success,
        "letter_grade": result.letter_grade,
        "total_reward": result.total_reward,
        "steps_used": result.steps_used,
        "max_steps": result.max_steps,
        "system_integrity": result.system_integrity,
        "verdict_given": result.verdict_given,
        "verdict_correct": result.verdict_correct,
        "forbidden_patches_applied": result.forbidden_patches_applied,
        "summary": result.summary,
        "dimensions": [d.model_dump() for d in result.dimensions],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
