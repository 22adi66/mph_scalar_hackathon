import math
from models import SdsmpState

def grade_task_easy(state: SdsmpState) -> float:
    """
    Evaluates Phase 1: Low-Frequency Mode (20-40 intensity)
    Prioritizes cost optimization as resources are abundant.
    """
    if state.step_count < 10:
        return 0.001
        
    qos_rate = 1.0
    if state.processed_jobs_count > 0:
        qos_rate = max(0.0, 1.0 - (state.qos_failed_count / state.processed_jobs_count))
        
    # In low traffic, we expect near 100% QoS. If it falls, score plummets.
    if qos_rate < 0.9:
        return 0.001
        
    # Cost should be extremely low
    cost_score = max(0.0, 1.0 - (state.current_cost / max(1.0, float(state.processed_jobs_count) * 0.02)))
    
    score = (cost_score * 0.8) + (qos_rate * 0.2)
    return round(float(max(0.001, min(0.999, score))), 4)

def grade_task_medium(state: SdsmpState) -> float:
    """
    Evaluates Phase 2: Random Workload Mode (0-100 intensity)
    Prioritizes balancing cost and QoS equally.
    """
    if state.step_count < 10:
        return 0.001
        
    qos_rate = 1.0
    if state.processed_jobs_count > 0:
        qos_rate = max(0.0, 1.0 - (state.qos_failed_count / state.processed_jobs_count))
        
    if qos_rate < 0.7:
        return 0.001
        
    # Expect moderate cost
    cost_score = max(0.0, 1.0 - (state.current_cost / max(1.0, float(state.processed_jobs_count) * 0.05)))
    
    score = (cost_score * 0.5) + (qos_rate * 0.5)
    return round(float(max(0.001, min(0.999, score))), 4)

def grade_task_hard(state: SdsmpState) -> float:
    """
    Evaluates Phase 3: High-Frequency DDoS Mode (60-80 intensity)
    Prioritizes survival via pure QoS rate tracking under load.
    """
    if state.step_count < 10:
        return 0.001
        
    if state.processed_jobs_count == 0:
        return 0.001
        
    qos_rate = max(0.0, 1.0 - (state.qos_failed_count / state.processed_jobs_count))
    
    # Simple threshold survival score.
    # At high intensity, simply keeping QoS above 50% is a win due to Cannikin constraints.
    score = (qos_rate - 0.4) * 2.0
    return round(float(max(0.001, min(0.999, score))), 4)
