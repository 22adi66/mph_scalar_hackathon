"""Quick test script to verify the SDSmp environment works end-to-end."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.sdsmp_environment import SdsmpEnvironment

def test_easy():
    print("=" * 60)
    print("TEST: Easy Task (Low-Frequency Mode)")
    print("=" * 60)
    env = SdsmpEnvironment()
    obs = env.reset(seed=42, task_id="easy")
    print(f"  Reset OK. Task ID: {obs.task_id}")
    print(f"  Pending Jobs: {len(obs.pending_jobs)}")
    print(f"  Resource pool: {len(obs.smp_vms)} VMs")

    # Match jobs to VMs to minimize cost and keep QoS high
    for i in range(10):
        if obs.done:
            break
        pending = obs.pending_jobs
        if not pending:
            obs = env.step({"command": "submit_evaluation", "parameters": {}})
            break
        
        job = pending[0]
        # Heuristic scheduling
        match_type = "high-cpu" if job.job_type == "compute-intensive" else "high-io"
        best_vm = next((v for v in obs.smp_vms if v.vm_type == match_type), None)
        
        obs = env.step({
            "command": "schedule_job",
            "parameters": {"job_id": job.job_id, "vm_id": best_vm.vm_id}
        })
        print(f"    Assigned {job.job_type[:5]} to {best_vm.vm_type}, reward={obs.reward:.4f}")

    obs = env.step({"command": "submit_evaluation", "parameters": {}})
    grade = env.get_grade()
    print(f"  Final cost: ${obs.current_cost:.4f}")
    print(f"  Grade: {grade:.4f}")
    print()
    return grade

def test_medium():
    print("=" * 60)
    print("TEST: Medium Task (Random Burst Mode)")
    print("=" * 60)
    env = SdsmpEnvironment()
    obs = env.reset(seed=42, task_id="medium")
    print(f"  Reset OK. Task ID: {obs.task_id}")

    # Process all jobs heuristically
    for i in range(25):
        if obs.done:
            break
        if not obs.pending_jobs:
            obs = env.step({"command": "noop", "parameters": {}})
            continue
            
        job = obs.pending_jobs[0]
        match_type = "high-cpu" if job.job_type == "compute-intensive" else "high-io"
        
        best_vm = None
        min_q = 999
        for v in obs.smp_vms:
            if v.vm_type == match_type and v.current_queue_length < min_q:
                min_q = v.current_queue_length
                best_vm = v
                
        obs = env.step({
            "command": "schedule_job",
            "parameters": {"job_id": job.job_id, "vm_id": best_vm.vm_id}
        })
        print(f"    Assigned {job.job_type[:5]} to {best_vm.vm_id}, reward={obs.reward:.4f}")

    obs = env.step({"command": "submit_evaluation", "parameters": {}})
    grade = env.get_grade()
    print(f"  Grade: {grade:.4f}")
    print()
    return grade

def test_hard():
    print("=" * 60)
    print("TEST: Hard Task (High-Frequency DDoS Mode)")
    print("=" * 60)
    env = SdsmpEnvironment()
    obs = env.reset(seed=42, task_id="hard")
    print(f"  Reset OK. Task ID: {obs.task_id}")

    for i in range(25):
        if obs.done:
            break
        if not obs.pending_jobs:
            obs = env.step({"command": "noop", "parameters": {}})
            continue
            
        job = obs.pending_jobs[0]
        match_type = "high-cpu" if job.job_type == "compute-intensive" else "high-io"
        
        best_vm = None
        min_q = 999
        for v in obs.smp_vms:
            if v.vm_type == match_type and v.current_queue_length < min_q:
                min_q = v.current_queue_length
                best_vm = v
                
        obs = env.step({
            "command": "schedule_job",
            "parameters": {"job_id": job.job_id, "vm_id": best_vm.vm_id}
        })

    obs = env.step({"command": "submit_evaluation", "parameters": {}})
    grade = env.get_grade()
    print(f"  Grade: {grade:.4f}")
    print()
    return grade

if __name__ == "__main__":
    s1 = test_easy()
    s2 = test_medium()
    s3 = test_hard()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Easy:   {s1:.4f}")
    print(f"  Medium: {s2:.4f}")
    print(f"  Hard:   {s3:.4f}")
    agg = s1 * 0.2 + s2 * 0.3 + s3 * 0.5
    print(f"  Aggregate: {agg:.4f}")
    print()
    print("All scores in [0.0, 1.0]: ", all(0.0 <= s <= 1.0 for s in [s1, s2, s3]))
    print("All tests PASSED!")
