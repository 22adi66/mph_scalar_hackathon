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
        
        self._state = SdsmpState(
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
        self._state.step_count = 0
        self._state.current_cost = 0.0
        self._state.cumulative_reward = 0.0
        self._state.processed_jobs_count = 0
        self._state.qos_failed_count = 0
        self._state.critical_dropped_count = 0
        self._state.total_jobs_arrived = 0
        self._state.task_id = self.task_id
        if episode_id:
            self._state.episode_id = episode_id
        self.episode_log = []
        return self._get_obs(msg="Environment reset.", done=False)

    def step(self, action_dict: Dict) -> SdsmpObservation:
        try:
            action = SdsmpAction(**action_dict)
        except Exception as e:
            return self._get_obs(msg=f"Invalid action schema: {str(e)}", done=False, reward=0.01)
            
        self._state.step_count += 1
        
        done = False
        if self._state.step_count >= self.max_steps:
            done = True
            
        reward = 0.0
        msg = "No operation performed."
        
        if action.command == "submit_evaluation":
            done = True
            msg = "Evaluation submitted."
            return self._get_obs(msg, done)
            
        elif action.command == "schedule_batch":
            assignments = action.parameters.get("assignments", [])
            total_reward = 0.0
            msgs = []
            
            for assignment in assignments:
                job_id = assignment.get("job_id", "")
                vm_id = assignment.get("vm_id", "")

                # Peek at job priority BEFORE scheduling for reward shaping
                job_priority = next(
                    (j.get("priority", "NORMAL") for j in self.sim.pending_jobs if j["job_id"] == job_id),
                    "NORMAL"
                )

                success, log_msg, job_cost, qos_met = self.sim.schedule_job(job_id, vm_id)
                msgs.append(log_msg)

                if success:
                    self._state.current_cost += job_cost
                    self._state.processed_jobs_count += 1
                    if not qos_met:
                        self._state.qos_failed_count += 1

                    if qos_met:
                        cost_bonus = 0.5 * math.exp(-job_cost * 20.0)
                        # Priority bonus: reward quick response to zero-day alerts
                        priority_bonus = 0.3 if job_priority == "CRITICAL" else 0.1 if job_priority == "NORMAL" else 0.0
                        total_reward += 0.5 + cost_bonus + priority_bonus
                    else:
                        # QoS failure: contribute minimal reward (all rewards must stay >= 0.01)
                        total_reward += 0.02  # token positive signal even on failure

            if assignments:
                reward = total_reward / len(assignments)
            else:
                reward = 0.01  # noop or empty batch → minimum reward
                
            msg = " | ".join(msgs) if msgs else "No valid assignments submitted in batch."
            
        # Advance simulation time
        dropped_jobs, critical_dropped = self.sim.advance_time(50.0)

        # Sync total jobs ever generated - critical for anti-hack throughput check in graders
        self._state.total_jobs_arrived = self.sim.job_counter
        
        if dropped_jobs > 0:
            self._state.qos_failed_count += dropped_jobs
            self._state.processed_jobs_count += dropped_jobs
            if critical_dropped > 0:
                self._state.critical_dropped_count += critical_dropped
                # CRITICAL drop is bad but reward must stay >= 0.01
                reward = min(reward, 0.02)  # Signal: very bad step
                msg += f" [CRITICAL ALARM: {critical_dropped} CRITICAL priority job(s) timed out!]"
            else:
                reward = min(reward, 0.05)  # Standard drop: bad but not as bad
                msg += f" [ALARM: {dropped_jobs} job(s) timed out in pending queue and were dropped!]"

        # CRUCIAL: clamp all step rewards strictly to (0.01, 0.99) as required
        reward = round(float(max(0.01, min(0.99, reward))), 4)
            
        self._state.cumulative_reward += reward
        self.episode_log.append(msg)
        
        # Determine if we should forcefully grade early
        metrics = self.sim.get_metrics()
        if metrics["avg_response_time_ms"] > 5000.0:
            done = True
            msg += " System crashed due to massive execution gridlock."
            
        return self._get_obs(msg, done, reward=reward)

    def state(self) -> SdsmpState:
        """OpenEnv interface: returns the current environment state."""
        return self._state

    def _get_obs(self, msg: str, done: bool, reward: float = 0.01) -> SdsmpObservation:
        metrics = self.sim.get_metrics()

        # Enforce reward bounds on ALL code paths (including submit_evaluation, reset, errors)
        reward = round(float(max(0.01, min(0.99, reward))), 4)

        pending_jobs = [JobInfo(**j) for j in self.sim.pending_jobs]
        smp_vms = [VmInfo(**v) for v in self.sim.vms.values()]

        return SdsmpObservation(
            done=done,
            reward=reward,
            current_cost=self._state.current_cost,
            load_balancing_rate=metrics["load_balancing_rate"],
            qos_satisfaction_rate=metrics["qos_satisfaction_rate"],
            avg_response_time_ms=metrics["avg_response_time_ms"],
            pending_jobs=pending_jobs,
            smp_vms=smp_vms,
            execution_log=msg,
            step_number=self._state.step_count,
            max_steps=self.max_steps,
            task_id=self.task_id,
            task_description=self.task_description
        )

    def close(self) -> None:
        """No-op cleanup — required by OpenEnv interface."""
        pass

    def get_grade(self) -> float:
        if self.task_id == "easy":
            return graders.grade_task_easy(self._state)
        elif self.task_id == "medium":
            return graders.grade_task_medium(self._state)
        elif self.task_id == "hard":
            return graders.grade_task_hard(self._state)
        return 0.001
