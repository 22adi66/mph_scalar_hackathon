"""
Core simulation engine for the Software-Defined Security Middle Platform (SDSmp)
Models high-CPU and high-IO Smp VMs, handling real-time compute/IO intensive job scheduling.
"""

import random
import math
from typing import Dict, List, Tuple
from pydantic import BaseModel

class SdsmpSimulation:
    def __init__(self, seed: int = 42, workload_mode: str = "easy"):
        self.rng = random.Random(seed)
        self.workload_mode = workload_mode
        self.time_ms = 0.0
        
        self.vms: Dict[str, Dict] = {}
        self.pending_jobs: List[Dict] = []
        self.completed_jobs: List[Dict] = []
        
        self.total_cost = 0.0
        self.job_counter = 0
        
        self._initialize_vms()
        self._generate_jobs()

    def _initialize_vms(self):
        # 3 high-cpu VMs and 3 high-io VMs to act as the Smp Resource Pool
        for i in range(3):
            vid = f"vm-cpu-{i+1}"
            self.vms[vid] = {
                "vm_id": vid,
                "vm_type": "high-cpu",
                "speed_mips": 2000.0,
                "current_queue_length": 0,
                "wait_time_ms": 0.0,  # Accumulator for current queue
                "hourly_cost": 0.15
            }
        for i in range(3):
            vid = f"vm-io-{i+1}"
            self.vms[vid] = {
                "vm_id": vid,
                "vm_type": "high-io",
                "speed_mips": 2000.0,
                "current_queue_length": 0,
                "wait_time_ms": 0.0,
                "hourly_cost": 0.15
            }

    def _generate_jobs(self):
        # Emulates Table 4 from the research paper (Low, Random, High frequency modes)
        if self.workload_mode == "easy":
            num_jobs = self.rng.randint(2, 4)  # Low frequency (20-40)
        elif self.workload_mode == "medium":
            num_jobs = self.rng.randint(0, 10) # Random/Bursty (0-100)
        else:
            num_jobs = self.rng.randint(6, 10) # High frequency (60-80) DDoS scenario
            
        generated_in_batch = []
        for _ in range(num_jobs):
            self.job_counter += 1
            jtype = self.rng.choice(["compute-intensive", "io-intensive"])
            mips = max(50.0, self.rng.gauss(100.0, 20.0))
            
            p_roll = self.rng.random()
            if p_roll < 0.2:
                priority = "CRITICAL"
                qos = 150.0  # 3-step window: critical but achievable
            elif p_roll < 0.6:
                priority = "NORMAL"
                qos = 400.0
            else:
                priority = "LOW"
                qos = 2000.0

            job_id = f"job-{self.job_counter}"

            # CRITICAL jobs must NEVER have dependencies — they must be schedulable immediately
            depends_on = []
            if priority != "CRITICAL" and len(generated_in_batch) > 0 and self.rng.random() < 0.3:
                depends_on.append(generated_in_batch[-1])

            generated_in_batch.append(job_id)
            
            self.pending_jobs.append({
                "job_id": job_id,
                "job_type": jtype,
                "required_mips": mips,
                "qos_deadline_ms": qos,
                "arrival_time": self.time_ms,
                "priority": priority,
                "depends_on": depends_on
            })

    def advance_time(self, step_duration_ms: float = 50.0) -> Tuple[int, int]:
        """Moves environment time forward, clearing VM queues naturally."""
        self.time_ms += step_duration_ms
        for vm in self.vms.values():
            if vm["wait_time_ms"] > 0:
                vm["wait_time_ms"] = max(0.0, vm["wait_time_ms"] - step_duration_ms)
                # Approximation of queue clearing: if wait time drops, queue length drops
                if vm["wait_time_ms"] == 0.0:
                    vm["current_queue_length"] = 0
                else:
                    vm["current_queue_length"] = max(1, int(vm["wait_time_ms"] / 50.0))
        
        # Identify jobs that died while sitting in the pending queue
        dropped_count = 0
        critical_dropped = 0
        still_pending = []
        for job in self.pending_jobs:
            if self.time_ms > (job["arrival_time"] + job["qos_deadline_ms"]):
                dropped_count += 1
                if job.get("priority") == "CRITICAL":
                    critical_dropped += 1
                job["qos_met"] = False
                job["response_time_ms"] = self.time_ms - job["arrival_time"]
                job["cost"] = 0.0
                job["assigned_vm"] = "TIMEOUT_DROP"
                self.completed_jobs.append(job)
            else:
                still_pending.append(job)
                
        self.pending_jobs = still_pending
        
        # New jobs arrive at each time step
        self._generate_jobs()
        return dropped_count, critical_dropped

    def schedule_job(self, job_id: str, vm_id: str) -> Tuple[bool, str, float, bool]:
        """
        Calculates execution time utilizing the paper's Cannikin Law principle:
        Mismatched job types receive massive execution penalties.
        Returns: (success, message, cost_incurred, qos_met)
        """
        job = next((j for j in self.pending_jobs if j["job_id"] == job_id), None)
        if not job:
            return False, f"Job {job_id} not found in pending queue.", 0.0, False
            
        vm = self.vms.get(vm_id)
        if not vm:
            return False, f"VM {vm_id} not found in resource pool.", 0.0, False

        # Check DAG Dependencies
        for dep_id in job.get("depends_on", []):
            dep_completed = next((j for j in self.completed_jobs if j["job_id"] == dep_id), None)
            if not dep_completed or not dep_completed.get("qos_met", False):
                return False, f"[DAG BLOCK] Job {job_id} blocked. Dependency {dep_id} has not successfully completed.", 0.0, False

        # Calculate actual execution time base
        base_exec_time = (job["required_mips"] / vm["speed_mips"]) * 1000.0 # ms

        # Capability Mismatch Penalty (Cannikin Law)
        is_match = (job["job_type"] == "compute-intensive" and vm["vm_type"] == "high-cpu") or \
                   (job["job_type"] == "io-intensive" and vm["vm_type"] == "high-io")
                   
        if not is_match:
            base_exec_time *= 5.0 # Mismatch takes 5x longer

        # Actual start time is after current queue clears
        wait_time = vm["wait_time_ms"]
        response_time = wait_time + base_exec_time
        
        # Determine thermal/cooling state (Real-world complexity injection)
        # If the VM queue is severely backed up (>4 jobs), it requires intense
        # cooling power, significantly multiplying base infrastructural cost.
        cooling_multiplier = 1.0
        if vm["current_queue_length"] >= 4:
            cooling_multiplier = 1.5 + (0.1 * (vm["current_queue_length"] - 4))
            
        # Cost is calculated based strictly on runtime (Formula 8 from paper) modified by cooling
        # Cost in cents roughly
        job_cost = ((base_exec_time / 3600000.0) * vm["hourly_cost"] * 1000.0) * cooling_multiplier
        self.total_cost += job_cost

        # Update VM state
        vm["wait_time_ms"] += base_exec_time
        vm["current_queue_length"] += 1
        
        # Check QoS
        qos_met = response_time <= job["qos_deadline_ms"]
        
        # Log job completion
        job["response_time_ms"] = response_time
        job["qos_met"] = qos_met
        job["cost"] = job_cost
        job["assigned_vm"] = vm_id
        
        self.completed_jobs.append(job)
        self.pending_jobs.remove(job)
        
        msg = f"Job {job_id} assigned to {vm_id}. Resp_Time: {response_time:.1f}ms (QoS:{'PASS' if qos_met else 'FAIL'})"
        if cooling_multiplier > 1.0:
            msg += f" [THERMAL WARNING: VM queue overloaded. Cooling cost multiplier {cooling_multiplier:.1f}x applied.]"
            
        return True, msg, job_cost, qos_met

    def get_metrics(self) -> Dict:
        processed = len(self.completed_jobs)
        if processed == 0:
            return {
                "avg_response_time_ms": 0.0,
                "qos_satisfaction_rate": 1.0,
                "load_balancing_rate": 0.0,
                "total_cost": self.total_cost
            }
            
        avg_resp = sum(j["response_time_ms"] for j in self.completed_jobs) / processed
        qos_rate = sum(1 for j in self.completed_jobs if j["qos_met"]) / processed
        
        # Load balancing rate (coefficient of variation of queue lengths)
        queues = [v["current_queue_length"] for v in self.vms.values()]
        mean_q = sum(queues) / len(queues)
        if mean_q == 0:
            lb_rate = 0.0
        else:
            variance = sum((q - mean_q) ** 2 for q in queues) / len(queues)
            std_dev = math.sqrt(variance)
            lb_rate = std_dev / mean_q # Lower is better
            
        return {
            "avg_response_time_ms": avg_resp,
            "qos_satisfaction_rate": qos_rate,
            "load_balancing_rate": lb_rate,
            "total_cost": self.total_cost
        }
