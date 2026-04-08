"""Quick local verification before push — does not run full LLM loop."""
import sys, os
sys.path.insert(0, '.')

# 1. Environment boot test
from server.sdsmp_environment import SdsmpEnvironment, TASKS
env = SdsmpEnvironment()
obs = env.reset(seed=42, task_id='easy')
print(f"ENV BOOT OK — pending={len(obs.pending_jobs)} vms={len(obs.smp_vms)}")

job = obs.pending_jobs[0]
vm = 'vm-cpu-1' if job.job_type == 'compute-intensive' else 'vm-io-1'
obs2 = env.step({'command': 'schedule_batch', 'parameters': {'assignments': [{'job_id': job.job_id, 'vm_id': vm}]}})
print(f"STEP OK — reward={obs2.reward:.4f} done={obs2.done}")
env.close()
print("CLOSE OK")

# 2. All 3 tasks reset
for task_id in ['easy', 'medium', 'hard']:
    env2 = SdsmpEnvironment()
    obs3 = env2.reset(seed=42, task_id=task_id)
    env2.close()
    print(f"TASK {task_id} RESET OK — {len(obs3.pending_jobs)} jobs")

# 3. openai import (critical — must not break)
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("HF_TOKEN", "dummy"),
    base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
)
print("OPENAI CLIENT OK")

# 4. Log format verification (exact spec output)
from inference import log_start, log_step, log_end
print("--- STDOUT FORMAT CHECK ---")
log_start('easy', 'sdsmp_cybersecurity', 'Qwen/Qwen2.5-72B-Instruct')
log_step(1, '{"command":"noop"}', 0.05, False, None)
log_step(2, '{"command":"schedule_batch"}', 0.82, True, None)
log_end(True, 2, [0.05, 0.82])
print("--- END FORMAT CHECK ---")

print("\nALL CHECKS PASSED - ready to commit and push")
