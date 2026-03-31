import json
from server.sdsmp_environment import SdsmpEnvironment, TASKS

def run_heuristic(task_id: str):
    env = SdsmpEnvironment()
    obs = env.reset(seed=42, task_id=task_id)
    print(f"\n--- Testing '{task_id}' task with Heuristic Agent ---")
    
    for step in range(25):
        if obs.done:
            break
            
        pending = obs.pending_jobs
        if not pending:
            obs = env.step({"command": "submit_evaluation", "parameters": {}})
            continue
            
        job = pending[0]
        vms = obs.smp_vms
        
        # Heuristic: Find matching type VM with lowest queue
        best_vm = None
        lowest_queue = 999
        match_type = "high-cpu" if job.job_type == "compute-intensive" else "high-io"
        
        for vm in vms:
            if vm.vm_type == match_type and vm.current_queue_length < lowest_queue:
                lowest_queue = vm.current_queue_length
                best_vm = vm
        
        if best_vm:
            action = {
                "command": "schedule_job",
                "parameters": {
                    "job_id": job.job_id,
                    "vm_id": best_vm.vm_id
                }
            }
        else:
            action = {"command": "noop", "parameters": {}}
            
        obs = env.step(action)
        # Suppress long log outputs for readability
        log_snippet = obs.execution_log.split("(")[0].strip() if "(" in obs.execution_log else obs.execution_log
        print(f"[{step+1:02d}] Assigned {job.job_type[:11]} ({job.required_mips:.0f} MIPS) -> {action['parameters'].get('vm_id', 'None')} | {log_snippet[:35]} | QoS:{obs.qos_satisfaction_rate*100:3.0f}% Cost:${obs.current_cost:.3f}")
        
    grade = env.get_grade()
    print(f">>> Final Grade for {task_id}: {grade:.4f} <<<")

if __name__ == "__main__":
    for t in ["easy", "medium", "hard"]:
        run_heuristic(t)
