"""Test HTTP endpoints of the running server."""
import requests
import json
import time
import subprocess
import os

# Start the server in the background for testing
server_process = subprocess.Popen(["python", "-m", "server.app"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2) # Give it time to start

try:
    BASE = "http://localhost:8000"

    print("--- SDSmp Environment Test ---")

    # 1. Health check
    r = requests.get(f"{BASE}/health")
    print("1. Health:", r.json())

    # 2. Tasks
    r = requests.get(f"{BASE}/tasks")
    t = r.json()
    print(f"2. Tasks: {len(t['tasks'])} tasks available.")
    print(f"   Commands: {list(t['available_commands'].keys())}")

    # 3. Reset (medium task)
    r = requests.post(f"{BASE}/reset", json={"seed": 42, "task_id": "medium"})
    obs = r.json()
    print(f"3. Reset: {len(obs['pending_jobs'])} jobs, {len(obs['smp_vms'])} VMs.")

    # 4. Step (schedule a job)
    if obs['pending_jobs'] and obs['smp_vms']:
        job_id = obs['pending_jobs'][0]['job_id']
        vm_id = obs['smp_vms'][0]['vm_id']
        r = requests.post(f"{BASE}/step", json={
            "command": "schedule_job",
            "parameters": {"job_id": job_id, "vm_id": vm_id}
        })
        step_obs = r.json()
        print(f"4. Step (schedule_job {job_id} to {vm_id}):")
        print(f"   Reward: {step_obs['reward']:.4f}")
        print(f"   Log: {step_obs['execution_log']}")

    # 5. State
    r = requests.get(f"{BASE}/state")
    state = r.json()
    print(f"5. State: episode={state['episode_id'][:8]}..., steps={state['step_count']}, cost=${state['current_cost']:.4f}")

    # 6. Submit evaluation & grade
    r = requests.post(f"{BASE}/step", json={"command": "submit_evaluation", "parameters": {}})
    r = requests.post(f"{BASE}/grader")
    grade = r.json()
    print(f"6. Grader Score: {grade:.4f}")

    # 7. Schema
    r = requests.get(f"{BASE}/schema")
    schema = r.json()
    print(f"7. Schema properties accessible.")

    print("\n" + "="*60)
    print("ALL HTTP ENDPOINTS WORKING!")
    print("="*60)

finally:
    # Kill the server
    server_process.kill()
