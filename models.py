from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

class JobInfo(BaseModel):
    job_id: str = Field(description="Unique ID for the security operation job")
    job_type: str = Field(description="Type of job: 'compute-intensive' or 'io-intensive'")
    required_mips: float = Field(description="Processing capacity required in MIPS")
    qos_deadline_ms: float = Field(description="Maximum allowed response time before QoS fails")
    arrival_time: float = Field(description="Simulation timestamp when job arrived")
    priority: str = Field(description="Urgency: 'LOW', 'NORMAL', or 'CRITICAL'")
    depends_on: List[str] = Field(description="List of job_ids that MUST finish before this job can be scheduled", default_factory=list)

class VmInfo(BaseModel):
    vm_id: str = Field(description="Unique ID for the Smp virtual machine resource")
    vm_type: str = Field(description="VM capability type: 'high-cpu' or 'high-io'")
    speed_mips: float = Field(description="Processing speed in MIPS")
    current_queue_length: int = Field(description="Number of jobs currently waiting or executing")
    hourly_cost: float = Field(description="Cost per hour of running this VM")

class SdsmpAction(BaseModel):
    command: Literal["schedule_batch", "noop", "submit_evaluation"] = Field(
        ..., description="The action to perform in the SDSmp."
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="For schedule_batch, provide 'assignments': [{'job_id': 'x', 'vm_id': 'y'}, ...]"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SdsmpObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)

    current_cost: float = Field(description="Total operational cost incurred so far")
    load_balancing_rate: float = Field(description="Metric indicating how evenly jobs are distributed (0-1, lower is better)")
    qos_satisfaction_rate: float = Field(description="Percentage of jobs completed within their QoS deadline")
    avg_response_time_ms: float = Field(description="Average response time across all processed jobs")

    pending_jobs: List[JobInfo] = Field(description="Jobs waiting in the control plane to be scheduled")
    smp_vms: List[VmInfo] = Field(description="Available Virtual Machines in the Smp resource pool")

    execution_log: str = Field(description="Feedback from the last action")
    step_number: int = Field(description="Current step count")
    max_steps: int = Field(description="Maximum steps in the episode")
    task_id: str = Field(description="Current active task ID")
    task_description: str = Field(description="Description of the current active task")

    metadata: Dict[str, Any] = Field(default_factory=dict)

class SdsmpState(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    episode_id: Optional[str] = None
    step_count: int = 0
    task_id: str = "easy"
    current_cost: float = 0.0
    cumulative_reward: float = 0.0
    processed_jobs_count: int = 0
    qos_failed_count: int = 0
    critical_dropped_count: int = 0
    total_jobs_arrived: int = 0
