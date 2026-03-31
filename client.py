"""
SDSmp Client for Cybersecurity Scheduler Environment.

Usage:
    from client import SdsmpClient

    client = SdsmpClient(base_url="http://localhost:8000")
    obs = client.reset(task_id="easy")
    obs = client.step(command="schedule_job", parameters={"job_id": "job-1", "vm_id": "vm-cpu-1"})
"""

from typing import Any, Dict, Optional
import requests


class SdsmpClient:
    """HTTP/REST client for the SDSmp Cybersecurity Scheduler environment."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def reset(self, seed: int = 42, task_id: str = "easy", episode_id: str = "") -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/reset", json={"seed": seed, "task_id": task_id, "episode_id": episode_id})
        resp.raise_for_status()
        return resp.json()

    def step(self, command: str = "noop", parameters: Optional[Dict] = None) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/step", json={"command": command, "parameters": parameters or {}})
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def get_tasks(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return resp.json()

    def get_grade(self) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/grader")
        resp.raise_for_status()
        return resp.json()

    def run_baseline(self, api_key: str = "", model: str = "gpt-4o-mini") -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/baseline", json={"api_key": api_key, "model": model})
        resp.raise_for_status()
        return resp.json()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def sync(self):
        return self