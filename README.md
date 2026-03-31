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

* **`easy` (Low-Frequency Mode)**: Baseline traffic. Focus is on pure cost minimization.
* **`medium` (Random Burst Mode)**: Unpredictable traffic spikes. Must balance Cost and QoS equally. 
* **`hard` (High-Frequency DDoS Mode)**: Near-collapse traffic. Cost is secondary; the agent must route flawlessly to keep QoS above 40%.

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Validation Baseline**:
   ```bash
   export OPENAI_API_KEY="sk-..."
   python inference.py
   ```

3. **Deploy to Hugging Face**:
   Currently structured perfectly for the OpenEnv Space specification.
   ```bash
   openenv validate
   git push origin main
   ```
# Meta
