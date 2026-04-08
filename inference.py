"""
Inference Script for OpenEnv Hackathon Submission
Ensures exact match with STDOUT formatting requirements.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from server.sdsmp_environment import SdsmpEnvironment, TASKS

# Environment Constants
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "sdsmp_cybersecurity"
MAX_STEPS = 20

def _parse_action(response_text: str) -> Dict[str, Any]:
    text = response_text.strip()
    try:
        # Simplistic JSON extraction block
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.rindex("```")
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.rindex("```")
            text = text[start:end].strip()
    except ValueError:
        pass

    try:
        action = json.loads(text)
        if isinstance(action, dict) and "command" in action:
            return action
    except BaseException:
        pass

    # Brute forcing the first JSON object
    for i in range(len(text)):
        if text[i] == "{":
            for j in range(len(text) - 1, i, -1):
                if text[j] == "}":
                    try:
                        action = json.loads(text[i : j + 1])
                        if isinstance(action, dict) and "command" in action:
                            return action
                    except BaseException:
                        continue
    return {"command": "noop", "parameters": {}}

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def run_task(client: OpenAI, task_id: str) -> None:
    env = SdsmpEnvironment()
    task = TASKS[task_id]
    
    obs = env.reset(seed=42, task_id=task_id)
    obs_data = obs.model_dump()

    system_msg = (
        "You are an autonomous Software-Defined Security Middle Platform (SDSmp) Job Scheduler Assistant.\n"
        "Your objective is to route incoming security jobs to the available pool of Virtual Machines in real-time.\n"
        "To maximize the system's Load Balancing Rate and efficiently pack resources, you must apply the Cannikin Law rule:\n"
        "- Route `compute-intensive` jobs EXCLUSIVELY to `high-cpu` VMs.\n"
        "- Route `io-intensive` jobs EXCLUSIVELY to `high-io` VMs.\n"
        "If you mismatch job processing capability, execution time incurs a massive 5x penalty, leading to system crash.\n\n"
        "Available Actions (Respond with JSON only):\n"
        "1. {\"command\": \"schedule_batch\", \"parameters\": {\"assignments\": [{\"job_id\": \"<ID1>\", \"vm_id\": \"<VM1>\"}, {\"job_id\": \"<ID2>\", \"vm_id\": \"<VM2>\"}]}}\n"
        "2. {\"command\": \"noop\", \"parameters\": {}}\n"
        "3. {\"command\": \"submit_evaluation\", \"parameters\": {}}\n\n"
        f"Task Description: {task['description']}\n"
        "Analyze ALL pending jobs, identify their types, and route ALL of them simultaneously in a single schedule_batch command to processing VMs."
    )

    messages = [{"role": "system", "content": system_msg}]
    rewards: List[float] = []
    
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            user_content = (
                f"Task: {task['name']}\n"
                f"Pending Jobs (Process ALL):\n{json.dumps(obs_data['pending_jobs'], indent=2)}\n\n"
                f"Smp VMs:\n{json.dumps(obs_data['smp_vms'], indent=2)}\n\n"
                f"Feedback from last step: {obs_data.get('execution_log', 'None')}\n"
                f"Metrics: Cost=${obs_data['current_cost']:.2f}, QoS={obs_data['qos_satisfaction_rate']:.2f}\n"
                "Pick your next action. Use JSON strictly."
            )

            messages.append({"role": "user", "content": user_content})

            # Call AI
            error_msg = None
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=300,
                )
                assistant_text = response.choices[0].message.content or ""
            except Exception as e:
                assistant_text = '{"command": "noop", "parameters": {}}'
                error_msg = str(e).replace("\n", " ")

            action_dict = _parse_action(assistant_text)
            action_str = json.dumps(action_dict).replace(" ", "")

            # Step Environment
            obs = env.step(action_dict)
            obs_data = obs.model_dump()
            steps_taken = step
            reward = obs.reward
            done = obs.done

            rewards.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)
            
            messages.append({"role": "assistant", "content": assistant_text})
            
            # Keep message context short 
            if len(messages) > 3:
                messages = messages[:1] + messages[-2:]

        # Grading logic
        score = env.get_grade()
        success = score > 0.6

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    
    # Do not print ANYTHING else to stdout as it will break the grader validation
    for task_id in ["easy", "medium", "hard"]:
        run_task(client, task_id)

