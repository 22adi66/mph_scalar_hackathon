"""
Deep Q-Network (DQN) Training Agent for SDSmp Cybersecurity Scheduler
"""

import random
import math
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Any

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Install with: pip install torch")

from simulation import SdsmpSimulation
from models import SdsmpState


class SdsmpGymEnv:
    def __init__(self, task_id: str = "easy", seed: int = 42):
        self.task_id = task_id
        self.seed = seed
        self.sim = None
        self.state = None
        self.max_steps = 20
        self.step_count = 0
        self.n_actions = 7
        self.state_dim = 22
        
    def reset(self):
        self.sim = SdsmpSimulation(seed=self.seed, workload_mode=self.task_id)
        self.state = SdsmpState(episode_id="train", step_count=0, task_id=self.task_id)
        self.step_count = 0
        return self._get_state_vector()
    
    def _get_state_vector(self):
        metrics = self.sim.get_metrics()
        lb_rate = min(metrics.get("load_balancing_rate", 0.0), 10.0) / 10.0
        features = [lb_rate]
        if self.sim.pending_jobs:
            job = self.sim.pending_jobs[0]
            is_compute = 1.0 if job["job_type"] == "compute-intensive" else 0.0
            mips_norm = min(job["required_mips"] / 200.0, 1.0)
            qos_norm = min(job["qos_deadline_ms"] / 500.0, 1.0)
            features.extend([is_compute, mips_norm, qos_norm])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        vm_ids = sorted(self.sim.vms.keys())
        for vm_id in vm_ids:
            vm = self.sim.vms[vm_id]
            is_cpu = 1.0 if vm["vm_type"] == "high-cpu" else 0.0
            queue_norm = min(vm["current_queue_length"] / 10.0, 1.0)
            wait_norm = min(vm["wait_time_ms"] / 1000.0, 1.0)
            features.extend([is_cpu, queue_norm, wait_norm])
        
        return np.array(features, dtype=np.float32)
    
    def step(self, action: int):
        self.step_count += 1
        reward = 0.0
        info = {}
        
        if action == 6:
            self.sim.advance_time(50.0)
            reward = -0.01
        else:
            vm_ids = sorted(self.sim.vms.keys())
            vm_id = vm_ids[action]
            
            if self.sim.pending_jobs:
                job = self.sim.pending_jobs[0]
                job_id = job["job_id"]
                success, msg, cost, qos_met = self.sim.schedule_job(job_id, vm_id)
                
                if success:
                    self.state.processed_jobs_count += 1
                    self.state.current_cost += cost
                    
                    if qos_met:
                        job_type = job["job_type"]
                        vm_type = self.sim.vms[vm_id]["vm_type"]
                        is_match = (job_type == "compute-intensive" and vm_type == "high-cpu") or (job_type == "io-intensive" and vm_type == "high-io")
                        if is_match:
                            reward = 1.0 - cost * 10
                        else:
                            reward = 0.2 - cost * 10
                    else:
                        self.state.qos_failed_count += 1
                        reward = -2.0
                else:
                    reward = -0.5
                self.sim.advance_time(50.0)
            else:
                self.sim.advance_time(50.0)
                reward = -0.01
        
        done = self.step_count >= self.max_steps
        metrics = self.sim.get_metrics()
        if metrics["avg_response_time_ms"] > 5000.0:
            done = True
            reward = -10.0
            info["crashed"] = True
        
        return self._get_state_vector(), reward, done, info
    
    def get_grade(self):
        from graders import grade_task_easy, grade_task_medium, grade_task_hard
        self.state.step_count = self.step_count
        if self.task_id == "easy":
            return grade_task_easy(self.state)
        elif self.task_id == "medium":
            return grade_task_medium(self.state)
        else:
            return grade_task_hard(self.state)


if TORCH_AVAILABLE:
    class DQN(nn.Module):
        def __init__(self, state_dim, n_actions, hidden_dim=256):
            super(DQN, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions)
            )
        
        def forward(self, x):
            return self.network(x)

    class ReplayBuffer:
        def __init__(self, capacity=10000):
            self.buffer = deque(maxlen=capacity)
        
        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))
        
        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states), np.array(dones, dtype=np.float32))
        
        def __len__(self):
            return len(self.buffer)

    class DQNAgent:
        def __init__(self, state_dim, n_actions, lr=5e-4, gamma=0.95, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.99, batch_size=128, target_update=10):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            self.n_actions = n_actions
            self.gamma = gamma
            self.batch_size = batch_size
            self.target_update = target_update
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay
            self.policy_net = DQN(state_dim, n_actions).to(self.device)
            self.target_net = DQN(state_dim, n_actions).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.buffer = ReplayBuffer()
            self.update_count = 0
        
        def select_action(self, state, training=True):
            if training and random.random() < self.epsilon:
                return random.randrange(self.n_actions)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        
        def update(self):
            if len(self.buffer) < self.batch_size:
                return 0.0
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                target_q = rewards + self.gamma * next_q * (1 - dones)
            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_count += 1
            if self.update_count % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            return loss.item()
        
        def decay_epsilon(self):
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        def save(self, path):
            torch.save({'policy_net': self.policy_net.state_dict(), 'target_net': self.target_net.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epsilon': self.epsilon}, path)
            print(f"Model saved to {path}")
        
        def load(self, path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']


def train_agent(task_id="easy", n_episodes=500, save_path="dqn_sdsmp.pt"):
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch required. Install with: pip install torch")
        return None
    env = SdsmpGymEnv(task_id=task_id)
    agent = DQNAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    print(f"\n{'='*60}")
    print(f"Training DQN Agent on Task: {task_id}")
    print(f"{'='*60}\n")
    best_grade = 0.0
    episode_rewards = []
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        losses = []
        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            loss = agent.update()
            if loss > 0:
                losses.append(loss)
            total_reward += reward
            state = next_state
            if done:
                break
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        if (episode + 1) % 50 == 0:
            grade = env.get_grade()
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1:4d} | Reward: {avg_reward:7.2f} | Grade: {grade:.4f} | Epsilon: {agent.epsilon:.3f}")
            if grade > best_grade:
                best_grade = grade
                agent.save(save_path)
    print(f"\nTraining Complete! Best Grade: {best_grade:.4f}")
    return agent


def run_heuristic_baseline():
    print("\n" + "="*60)
    print("HEURISTIC BASELINE (Rule-Based)")
    print("="*60)
    for task_id in ["easy", "medium", "hard"]:
        env = SdsmpGymEnv(task_id=task_id)
        state = env.reset()
        while True:
            if env.sim.pending_jobs:
                job = env.sim.pending_jobs[0]
                job_type = job["job_type"]
                best_action = 6
                min_queue = float('inf')
                vm_ids = sorted(env.sim.vms.keys())
                for i, vm_id in enumerate(vm_ids):
                    vm = env.sim.vms[vm_id]
                    is_match = (job_type == "compute-intensive" and vm["vm_type"] == "high-cpu") or (job_type == "io-intensive" and vm["vm_type"] == "high-io")
                    if is_match and vm["current_queue_length"] < min_queue:
                        min_queue = vm["current_queue_length"]
                        best_action = i
                action = best_action
            else:
                action = 6
            state, _, done, _ = env.step(action)
            if done:
                break
        grade = env.get_grade()
        print(f"  {task_id:8s}: {grade:.4f}")


if __name__ == "__main__":
    run_heuristic_baseline()
    if TORCH_AVAILABLE:
        for task_id in ["easy", "medium", "hard"]:
            train_agent(task_id=task_id, n_episodes=300, save_path=f"dqn_{task_id}.pt")
    else:
        print("\nSkipping DQN training (PyTorch not installed)")