"""
Baseline inference script for the SDSmp Cybersecurity Environment.
Uses the OpenAI API client to run a model against all three tasks
and produce reproducible baseline scores.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

try:
    from .sdsmp_environment import SdsmpEnvironment, TASKS
except ImportError:
    from server.sdsmp_environment import SdsmpEnvironment, TASKS


SYSTEM_PROMPT = """You are an autonomous Software-Defined Security Middle Platform (SDSmp) Job Scheduler Assistant.

Your objective is to route incoming security jobs to the available pool of Virtual Machines in real-time.
To maximize the system's Load Balancing Rate and strictly respect QoS wait time limits, you must apply the Cannikin Law rule to capability matching:
- Route `compute-intensive` jobs EXCLUSIVELY to `high-cpu` VMs.
- Route `io-intensive` jobs EXCLUSIVELY to `high-io` VMs.
If you mismatch job processing capability, execution time incurs a massive 5x penalty, leading to system failure (especially in DDoS mode).

## Available Actions (Respond with JSON only)

1. **schedule_job**
   {{"command": "schedule_job", "parameters": {{"job_id": "<ID>", "vm_id": "<VM_ID>"}}}}

2. **noop**
   {{"command": "noop", "parameters": {{}}}}

3. **submit_evaluation**
   {{"command": "submit_evaluation", "parameters": {{}}}}

## Strategy
Always analyze the 'pending_jobs' array in your state. Pick the first job, identify its type, and schedule it to the matching VM with the LOWEST 'current_queue_length' among its type. If no pending jobs exist, submit the evaluation or take a noop to pass time.

## Current Task
{task_description}
"""

def _parse_action(response_text: str) -> Dict[str, Any]:
    text = response_text.strip()
    try:
        if "```json" in text:
            start = text.index("```json") + 7
            if "```" in text[start:]:
                end = text.index("```", start)
                text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            if "```" in text[start:]:
                end = text.index("```", start)
                text = text[start:end].strip()
    except ValueError:
        pass

    try:
        action = json.loads(text)
        if isinstance(action, dict) and "command" in action:
            return action
    except json.JSONDecodeError:
        pass

    for i in range(len(text)):
        if text[i] == "{":
            for j in range(len(text) - 1, i, -1):
                if text[j] == "}":
                    try:
                        action = json.loads(text[i : j + 1])
                        if isinstance(action, dict) and "command" in action:
                            return action
                    except json.JSONDecodeError:
                        continue
    return {"command": "noop", "parameters": {}}


def run_single_task(
    api_key: str,
    model: str,
    task_id: str,
    api_base_url: str = "https://api.openai.com/v1",
    max_agent_steps: int = 25,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError:
        return {"score": 0.0, "details": "openai package not installed.", "steps": 0, "trajectory": []}

    client = OpenAI(api_key=api_key, base_url=api_base_url)
    env = SdsmpEnvironment()
    task = TASKS[task_id]

    obs = env.reset(seed=42, task_id=task_id)
    obs_data = obs.model_dump()

    system_msg = SYSTEM_PROMPT.format(task_description=task["description"])

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": (
                f"Task: {task['name']}\n\n"
                f"Pending Jobs (First 5):\n{json.dumps(obs_data['pending_jobs'][:5], indent=2)}\n\n"
                f"Smp VMs:\n{json.dumps(obs_data['smp_vms'], indent=2)}\n\n"
                f"Current Metrics: Cost=${obs_data['current_cost']:.2f}, Load Balancing={obs_data['load_balancing_rate']:.2f}, QoS={obs_data['qos_satisfaction_rate']:.2f}\n\n"
                f"Pick your next action. Use JSON strictly."
            ),
        },
    ]

    trajectory = []
    steps = 0

    import time
    for step_num in range(max_agent_steps):
        time.sleep(4.0) # Pace API limits for free tier
        # Auto-retry block for free tier rate limits
        retries_limit = 3
        assistant_text = ""
        for attempt in range(retries_limit):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=300,
                )
                assistant_text = response.choices[0].message.content or ""
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RateLimit" in err_str:
                    print(f"  [Rate Limit Hit] Suspending script for 15 seconds to recover tokens (Attempt {attempt+1}/{retries_limit})...")
                    time.sleep(15.0)
                else:
                    print(f"\n[DEBUG] API EXCEPTION ENCOUNTERED: {e}")
                    trajectory.append({"step": step_num, "error": str(e)})
                    break
                    
        if not assistant_text:
            break

        action = _parse_action(assistant_text)
        print(f"  Step {step_num:02d} | Agent Action: {action.get('command')} -> {action.get('parameters', {}).get('vm_id', 'no-vm')}")
        trajectory.append({
            "step": step_num,
            "action": action,
            "raw_response": assistant_text[:200],
        })

        obs = env.step(action)
        obs_data = obs.model_dump()
        steps += 1

        messages.append({"role": "assistant", "content": assistant_text})
        
        user_content = f"Log: {obs_data['execution_log']}\nReward: {obs_data['reward']:.2f}\nDone: {obs_data['done']}\n"
        if not obs_data["done"]:
            user_content += (
                f"\nPending Jobs (First 5):\n{json.dumps(obs_data['pending_jobs'][:5], indent=2)}\n\n"
                f"Smp VMs:\n{json.dumps(obs_data['smp_vms'], indent=2)}\n\n"
                f"Metrics: Cost=${obs_data['current_cost']:.2f}, Load Balancing={obs_data['load_balancing_rate']:.2f}, QoS={obs_data['qos_satisfaction_rate']:.2f}\n\n"
                f"Pick next action. Use JSON strictly."
            )
        
        messages.append({"role": "user", "content": user_content})

        if obs_data["done"]:
            break

        if len(messages) > 4:
            messages = messages[:1] + messages[-3:]

    grade = env.get_grade()
    result = {"score": grade, "details": "Score generated", "steps": steps, "trajectory": trajectory}
    return result

def run_baseline_all_tasks(
    api_key: str,
    api_base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    results = {}
    for task_id in ["easy", "medium", "hard"]:
        print(f"Running baseline for task: {task_id}...")
        result = run_single_task(api_key, model, task_id, api_base_url)
        results[task_id] = {
            "score": result["score"],
            "details": result["details"],
            "steps": result["steps"],
        }
        print(f"  Score: {result['score']:.4f} ({result['details']})")

    aggregate = (
        results["easy"]["score"] * 0.2 +
        results["medium"]["score"] * 0.3 +
        results["hard"]["score"] * 0.5
    )
    results["aggregate_score"] = round(aggregate, 4)
    return results

if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
    scores = run_baseline_all_tasks(api_key, api_base_url, model)

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    for task_id in ["easy", "medium", "hard"]:
        s = scores[task_id]
        print(f"  {task_id:8s}: {s['score']:.4f}  ({s['details']})")
    print(f"\n  Aggregate: {scores['aggregate_score']:.4f}")
