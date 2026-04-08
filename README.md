---
title: SDSmp Scheduler
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# SDSmp Cybersecurity Scheduler Environment

Welcome to the **Software-Defined Security Middle Platform (SDSmp) Job Scheduler**, a rigorous, OpenAI Gym / OpenEnv-compliant reinforcement learning environment. 

This environment is based on real-world Deep Reinforcement Learning (DRL) research for real-time cost and capability optimization in Cloud/IoT security infrastructure. As cyber threats (especially DDoS) increase in frequency, it is computationally prohibitive to process all incoming packets manually. This environment tasks an AI agent with orchestrating these workloads in real-time across constrained Virtual Machines (VMs).

## The Core Challenge
Your AI agent must dynamically route two types of incoming security operations:
1. `compute-intensive` jobs (e.g., deep packet inspection)
2. `io-intensive` jobs (e.g., firewall logging)

You are provided a Resource Pool of 6 Smp VMs (3 `high-cpu`, 3 `high-io`).

### The Cannikin Law Penalty
As defined in the reference literature, capability mismatch is disastrous. If an agent routes a `compute-intensive` job to a `high-io` VM, the processing execution time incurs a **5x overhead**. In high-frequency DDoS traffic, this instantly clogs the VM queue, crashes the Load Balancing Rate, and causes the entire system's QoS rate to plummet.

### The Reward Function
Our reward system uses a mathematically non-linear penalty to aggressively punish cost overruns while strictly enforcing Quality of Service (QoS) deadlines:
$$ R = R_{QoS} + \left( -\frac{2}{\pi} \arctan(J_{cost} - \lambda) \right) $$

## Tasks

| Task ID | Name | Difficulty | Objective |
|---|---|---|---|
| `easy` | Phase 1: Low-Frequency Smp Mode | 🟢 Easy | Minimize cost during low-intensity traffic. Weights: cost 60%, QoS 40%. |
| `medium` | Phase 2: Random Burst Mode | 🟡 Medium | Balance Cost and QoS equally under unpredictable traffic bursts. Weights: cost 45%, QoS 55%. |
| `hard` | Phase 3: High-Frequency DDoS Mode | 🔴 Hard | Survive the DDoS burst — pure QoS survival. Weights: QoS 80%, cost 20%. |

## Action Space

Actions are JSON objects with a `command` field (Pydantic `SdsmpAction`):

| Command | Parameters | Description |
|---|---|---|
| `schedule_batch` | `assignments: [{job_id, vm_id}, ...]` | Assign one or more pending jobs to specific VMs in a single step |
| `noop` | _(none)_ | Do nothing — observe and let time advance |
| `submit_evaluation` | _(none)_ | End the episode and trigger final grading |

## Observation Space

Observations are Pydantic `SdsmpObservation` objects with the following fields:

| Field | Type | Range / Values | Description |
|---|---|---|---|
| `done` | `bool` | `true` / `false` | Whether the episode has ended |
| `reward` | `float` | `[0.01, 0.99]` | Step-level reward signal |
| `current_cost` | `float` | `≥ 0.0` | Total operational cost incurred so far (in arbitrary cost units) |
| `load_balancing_rate` | `float` | `≥ 0.0` (lower is better) | Coefficient of variation of VM queue lengths — measures load evenness |
| `qos_satisfaction_rate` | `float` | `[0.0, 1.0]` | Fraction of completed jobs that met their QoS deadline |
| `avg_response_time_ms` | `float` | `≥ 0.0` | Average response time across all processed jobs in milliseconds |
| `pending_jobs` | `List[JobInfo]` | — | List of jobs waiting in the control plane queue |
| `smp_vms` | `List[VmInfo]` | — | Available Virtual Machines in the Smp resource pool |
| `execution_log` | `str` | — | Human-readable feedback from the last action |
| `step_number` | `int` | `[1, 20]` | Current step count within the episode |
| `max_steps` | `int` | `20` | Maximum steps per episode |
| `task_id` | `str` | `easy`, `medium`, `hard` | The active task identifier |
| `task_description` | `str` | — | Plain-text description of the current task |

### JobInfo fields
| Field | Type | Description |
|---|---|---|
| `job_id` | `str` | Unique job identifier (e.g. `job-1`) |
| `job_type` | `str` | `compute-intensive` or `io-intensive` |
| `required_mips` | `float` | MIPS required to process the job |
| `qos_deadline_ms` | `float` | Max allowed response time before QoS failure |
| `priority` | `str` | `CRITICAL`, `NORMAL`, or `LOW` |
| `depends_on` | `List[str]` | Job IDs that must complete before this job can be scheduled (DAG) |

### VmInfo fields
| Field | Type | Description |
|---|---|---|
| `vm_id` | `str` | Unique VM identifier (e.g. `vm-cpu-1`, `vm-io-2`) |
| `vm_type` | `str` | `high-cpu` or `high-io` |
| `speed_mips` | `float` | Processing capacity in MIPS |
| `current_queue_length` | `int` | Number of jobs currently queued or executing |
| `hourly_cost` | `float` | Cost per hour of running this VM |

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Validation Baseline**:
   ```bash
   export HF_TOKEN="hf_..."
   export API_BASE_URL="https://router.huggingface.co/v1"
   export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
   python inference.py
   ```

3. **Deploy to Hugging Face**:
   Currently structured perfectly for the OpenEnv Space specification.
   ```bash
   openenv validate
   git push origin main
   ```
## Baseline Performance Scores

The following baseline scores were obtained by running `inference.py` with `Qwen/Qwen2.5-72B-Instruct` via the HuggingFace Inference Router (seed=42, 20 steps per task):

| Task | Score | Difficulty | Agent Strategy |
|---|---|---|---|
| `easy` | ~0.55 | 🟢 Easy | LLM correctly matches job types to VMs; performs well under low load |
| `medium` | ~0.35 | 🟡 Medium | Struggles with bursty patterns; partial QoS maintenance |
| `hard` | ~0.15 | 🔴 Hard | DDoS overwhelms the context-window-limited LLM; drops critical jobs |
| **Aggregate** | **~0.27** | — | Weighted: easy×0.2 + medium×0.3 + hard×0.5 |

> Scores may vary slightly across API providers due to model temperature and token limits.

## Reward Function

Our reward system uses a mathematically non-linear component per step:

$$R_{step} = \frac{\text{(QoS bonus + cost bonus + priority bonus)}}{\text{assignments}} \in [0.01, 0.99]$$

Final episode grading combines cost efficiency and QoS satisfaction rate, weighted by task difficulty.

## OpenEnv Compliance

This environment fully implements the OpenEnv interface:
- `reset(seed, task_id)` → `SdsmpObservation`
- `step(action_dict)` → `SdsmpObservation` (reward + done embedded)
- `state()` → `SdsmpState`
- `close()` → no-op
- Metadata exposed via `openenv.yaml`

# Meta
