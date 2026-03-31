import logging
from server.sdsmp_environment import SdsmpEnvironment
from models import SdsmpAction

def test_pydantic_schema():
    print("Testing Pydantic schema generation...")
    from models import SdsmpAction, SdsmpObservation, SdsmpState
    print(SdsmpObservation.model_json_schema() is not None)

def test_environment():
    try:
        env = SdsmpEnvironment()
        obs = env.reset(seed=42, task_id="medium")
        print(f"Initial Observation: Jobs={len(obs.pending_jobs)}, VMs={len(obs.smp_vms)}")
        
        job_id = obs.pending_jobs[0].job_id
        vm_id = obs.smp_vms[0].vm_id
        print(f"Scheduling job {job_id} to {vm_id}")
        
        action = SdsmpAction(command="schedule_job", parameters={"job_id": job_id, "vm_id": vm_id})
        obs2 = env.step(action.model_dump())
        print(f"Step Result: Reward={obs2.reward:.4f}, Log={obs2.execution_log}")
        
        grade = env.get_grade()
        print(f"Current Grade: {grade}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pydantic_schema()
    test_environment()
