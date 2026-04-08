import math
from models import SdsmpState


def _throughput_penalty(state: SdsmpState) -> float:
    """
    Returns a multiplier [0.0, 1.0] penalizing very low throughput.
    Anti-hack: an agent that schedules NOTHING but has 0 failures would
    otherwise score 0.999. This forces meaningful participation.
    Throughput < 25% of arrived jobs → graduated penalty towards 0.
    Uses total_jobs_arrived to track all jobs ever generated.
    """
    total = getattr(state, "total_jobs_arrived", 0)
    if total == 0:
        return 1.0
    throughput = state.processed_jobs_count / total
    if throughput >= 0.25:
        return 1.0
    return throughput / 0.25  # Linear scale 0 → 1 as throughput goes 0% → 25%


def _qos_component(state: SdsmpState, min_acceptable: float = 0.5) -> float:
    """
    Proportional QoS score — NOT a binary cutoff.
    Returns 0.0 if qos_rate is at or below min_acceptable,
    scaling up to 1.0 as qos_rate approaches 1.0.
    This rewards partial improvement instead of penalizing all imperfect results.
    """
    if state.processed_jobs_count == 0:
        return 0.0
    qos_rate = max(0.0, 1.0 - (state.qos_failed_count / state.processed_jobs_count))
    if qos_rate <= min_acceptable:
        return 0.0
    return (qos_rate - min_acceptable) / (1.0 - min_acceptable)


def grade_task_easy(state: SdsmpState) -> float:
    """
    Phase 1: Low-Frequency Mode (20-40 intensity).
    Goal: Minimize cost while maintaining high QoS.
    Weights: cost 60%, qos 40%.

    Anti-hacks enforced:
      - No jobs processed at all → 0.001
      - Throughput < 25% overall → proportional multiplier
      - Any CRITICAL job dropped → score hard-capped at 0.4
    """
    if state.step_count < 10:
        return 0.001

    if state.processed_jobs_count == 0:
        return 0.001

    # Cost component: expects low cost relative to jobs processed
    cost_score = max(0.0, 1.0 - (
        state.current_cost / max(1.0, float(state.processed_jobs_count) * 0.02)
    ))

    # QoS component: proportional, min_acceptable = 0.6 for easy mode
    qos_component = _qos_component(state, min_acceptable=0.6)

    score = (cost_score * 0.6) + (qos_component * 0.4)

    # Zero-day hardcap: any CRITICAL drop severely limits max score
    if state.critical_dropped_count > 0:
        # Each critical drop reduces cap: first drop → 0.4, more → worse
        cap = max(0.1, 0.4 - (0.05 * (state.critical_dropped_count - 1)))
        score = min(score, cap)

    # Throughput anti-hack multiplier
    score = score * _throughput_penalty(state)

    return round(float(max(0.01, min(0.99, score))), 4)


def grade_task_medium(state: SdsmpState) -> float:
    """
    Phase 2: Random Workload Mode (0-100 intensity).
    Goal: Balance cost and QoS equally under unpredictable load.
    Weights: cost 45%, qos 55%.

    Anti-hacks enforced:
      - No jobs processed → 0.001
      - Throughput < 25% → proportional multiplier
      - CRITICAL jobs dropped → capped at 0.4
    """
    if state.step_count < 10:
        return 0.001

    if state.processed_jobs_count == 0:
        return 0.001

    # Cost component: moderate cost tolerance for bursty workloads
    cost_score = max(0.0, 1.0 - (
        state.current_cost / max(1.0, float(state.processed_jobs_count) * 0.05)
    ))

    # QoS component: min_acceptable = 0.5 for medium (burst mode is harder)
    qos_component = _qos_component(state, min_acceptable=0.5)

    score = (cost_score * 0.45) + (qos_component * 0.55)

    # Zero-day hardcap
    if state.critical_dropped_count > 0:
        cap = max(0.1, 0.4 - (0.05 * (state.critical_dropped_count - 1)))
        score = min(score, cap)

    # Throughput anti-hack multiplier
    score = score * _throughput_penalty(state)

    return round(float(max(0.01, min(0.99, score))), 4)


def grade_task_hard(state: SdsmpState) -> float:
    """
    Phase 3: High-Frequency DDoS Mode (60-80 intensity).
    Goal: Survive the burst — pure QoS survival matters most.
    Weights: qos 80%, cost 20%.

    Anti-hacks enforced:
      - No jobs processed → 0.001
      - Throughput < 25% → proportional multiplier
      - CRITICAL jobs dropped → capped at 0.4
    """
    if state.step_count < 10:
        return 0.001

    if state.processed_jobs_count == 0:
        return 0.001

    # QoS component: at DDoS intensity, surviving 40%+ is already impressive
    qos_component = _qos_component(state, min_acceptable=0.4)

    # Cost component: secondary in DDoS mode — just don't overspend catastrophically
    cost_score = max(0.0, 1.0 - (
        state.current_cost / max(1.0, float(state.processed_jobs_count) * 0.10)
    ))

    score = (qos_component * 0.80) + (cost_score * 0.20)

    # Zero-day hardcap
    if state.critical_dropped_count > 0:
        cap = max(0.1, 0.4 - (0.05 * (state.critical_dropped_count - 1)))
        score = min(score, cap)

    # Throughput anti-hack multiplier
    score = score * _throughput_penalty(state)

    return round(float(max(0.01, min(0.99, score))), 4)
