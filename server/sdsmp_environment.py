import math
from typing import Dict, Any, Tuple
from simulation import SdsmpSimulation
from models import SdsmpAction, SdsmpObservation, SdsmpState, JobInfo, VmInfo
import graders

TASKS = {
    "easy": {
        "id": "easy",
        "name": "Phase 1: Low-Frequency Smp Mode",
        "description": "Minimize cost during low-intensity traffic."
    },
    "medium": {
        "id": "medium",
        "name": "Phase 2: Random Burst Mode",
        "description": "Balance Cost and QoS Rate during unpredictable traffic."
    },
    "hard": {
        "id": "hard",
        "name": "Phase 3: High-Frequency DDoS Mode",
        "description": "Survive the traffic burst by perfectly balancing workloads across VMs."
    }
}

class SdsmpEnvironment:
    def __init__(self):
        self.task_id = "easy"
        self.workload_mode = "easy"
        self.task_description = TASKS[self.task_id]["description"]
        self.sim = SdsmpSimulation(seed=42, workload_mode=self.workload_mode)
        
        self.state = SdsmpState(
            episode_id="ep-1",
            step_count=0,
            task_id=self.task_id
        )
        self.max_steps = 20
        self.episode_log = []

    def reset(self, seed: int = 42, task_id: str = "easy", episode_id: str = None) -> SdsmpObservation:
        self.task_id = task_id
        if self.task_id not in TASKS:
            self.task_id = "easy"
            
        self.workload_mode = self.task_id
        self.task_description = TASKS[self.task_id]["description"]
            
        self.sim = SdsmpSimulation(seed=seed, workload_mode=self.workload_mode)
        self.state.step_count = 0
        self.state.current_cost = 0.0
        self.state.cumulative_reward = 0.0
        self.state.processed_jobs_count = 0
        self.state.qos_failed_count = 0
        self.state.task_id = self.task_id
        if episode_id:
            self.state.episode_id = episode_id
        self.episode_log = []
        return self._get_obs(msg="Environment reset.", done=False)

    def step(self, action_dict: Dict) -> SdsmpObservation:
        try:
            action = SdsmpAction(**action_dict)
        except Exception as e:
            return self._get_obs(msg=f"Invalid action schema: {str(e)}", done=False, reward=-1.0)
            
        self.state.step_count += 1
        
        done = False
        if self.state.step_count >= self.max_steps:
            done = True
            
        reward = 0.0
        msg = "No operation performed."
        
        if action.command == "submit_evaluation":
            done = True
            msg = "Evaluation submitted."
            return self._get_obs(msg, done)
            
        elif action.command == "schedule_job":
            job_id = action.parameters.get("job_id", "")
            vm_id = action.parameters.get("vm_id", "")
            
            success, log_msg, job_cost, qos_met = self.sim.schedule_job(job_id, vm_id)
            msg = log_msg
            
            if success:
                self.state.current_cost += job_cost
                self.state.processed_jobs_count += 1
                if not qos_met:
                    self.state.qos_failed_count += 1
                    
                # Equation 10 from paper: Nonlinear reward R = R_cost * R_QoS
                r_qos = 1.0 if qos_met else -1.0
                
                lambda_baseline_cost = 0.05
                j_cost = job_cost
                cost_penalty = -(2.0 / math.pi) * math.atan(j_cost - lambda_baseline_cost)
                
                if qos_met:
                    reward = r_qos + cost_penalty
                else:
                    reward = -2.0 # Severe penalty for missing QoS
                    
            else:
                reward = -0.5 # Penalty for invalid schedule
                
        # Advance simulation time
        self.sim.advance_time(50.0)
        
        self.state.cumulative_reward += reward
        self.episode_log.append(msg)
        
        # Determine if we should forcefully grade early
        metrics = self.sim.get_metrics()
        if metrics["avg_response_time_ms"] > 5000.0:
            done = True
            msg += " System crashed due to massive execution gridlock."
            
        return self._get_obs(msg, done, reward=reward)

    def _get_obs(self, msg: str, done: bool, reward: float = 0.0) -> SdsmpObservation:
        metrics = self.sim.get_metrics()
        
        pending_jobs = [JobInfo(**j) for j in self.sim.pending_jobs]
        smp_vms = [VmInfo(**v) for v in self.sim.vms.values()]
        
        return SdsmpObservation(
            done=done,
            reward=reward,
            current_cost=self.state.current_cost,
            load_balancing_rate=metrics["load_balancing_rate"],
            qos_satisfaction_rate=metrics["qos_satisfaction_rate"],
            avg_response_time_ms=metrics["avg_response_time_ms"],
            pending_jobs=pending_jobs,
            smp_vms=smp_vms,
            execution_log=msg,
            step_number=self.state.step_count,
            max_steps=self.max_steps,
            task_id=self.task_id,
            task_description=self.task_description
        )

    def get_grade(self) -> float:
        if self.task_id == "easy":
            return graders.grade_task_easy(self.state)
        elif self.task_id == "medium":
            return graders.grade_task_medium(self.state)
        elif self.task_id == "hard":
            return graders.grade_task_hard(self.state)
        return 0.0
