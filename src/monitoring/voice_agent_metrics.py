"""Voice-Agent Integration Metrics Collection"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class AgentTransitionMetrics:
    """Metrics for agent transitions"""
    call_id: str
    from_agent: str
    to_agent: str
    transition_duration_ms: float
    context_preserved: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class StateSyncMetrics:
    """Metrics for state synchronization"""
    call_id: str
    sync_duration_ms: float
    sync_status: str  # "success", "failed", "partial"
    error_count: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TierEscalationMetrics:
    """Metrics for tier escalation events"""
    call_id: str
    from_tier: str
    to_tier: str
    trigger: str
    qualification_score: float
    budget_available: bool
    escalation_approved: bool
    cost_impact: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class VoiceAgentPerformanceMetrics:
    """Performance metrics for voice-agent integration"""
    call_id: str
    agent_processing_latency_ms: float
    total_response_latency_ms: float
    fallback_triggered: bool = False
    circuit_breaker_tripped: bool = False
    cache_hit: bool = False
    timestamp: float = field(default_factory=time.time)


class VoiceAgentIntegrationMetrics:
    """Centralized metrics collection for voice-agent integration"""

    def __init__(self):
        self._agent_transitions: List[AgentTransitionMetrics] = []
        self._state_syncs: List[StateSyncMetrics] = []
        self._tier_escalations: List[TierEscalationMetrics] = []
        self._performance_metrics: List[VoiceAgentPerformanceMetrics] = []

        # Aggregated metrics
        self._agent_processing_times: Dict[str, List[float]] = defaultdict(list)
        self._transition_counts: Dict[str, int] = defaultdict(int)
        self._escalation_counts: Dict[str, int] = defaultdict(int)
        self._fallback_count: int = 0
        self._circuit_breaker_trips: int = 0
        self._sync_failures: int = 0

    def record_agent_transition(
        self,
        call_id: str,
        from_agent: str,
        to_agent: str,
        transition_duration_ms: float,
        context_preserved: bool = True
    ):
        """Record an agent transition event"""
        try:
            metric = AgentTransitionMetrics(
                call_id=call_id,
                from_agent=from_agent,
                to_agent=to_agent,
                transition_duration_ms=transition_duration_ms,
                context_preserved=context_preserved
            )
            self._agent_transitions.append(metric)

            transition_key = f"{from_agent}->{to_agent}"
            self._transition_counts[transition_key] += 1

            logger.debug(f"Recorded agent transition: {transition_key} for call {call_id}")

        except Exception as e:
            logger.error(f"Error recording agent transition: {e}")

    def record_state_sync(
        self,
        call_id: str,
        sync_duration_ms: float,
        sync_status: str,
        error_count: int = 0
    ):
        """Record a state synchronization event"""
        try:
            metric = StateSyncMetrics(
                call_id=call_id,
                sync_duration_ms=sync_duration_ms,
                sync_status=sync_status,
                error_count=error_count
            )
            self._state_syncs.append(metric)

            if sync_status == "failed":
                self._sync_failures += 1

            logger.debug(f"Recorded state sync: {sync_status} for call {call_id}")

        except Exception as e:
            logger.error(f"Error recording state sync: {e}")

    def record_tier_escalation(
        self,
        call_id: str,
        from_tier: str,
        to_tier: str,
        trigger: str,
        qualification_score: float,
        budget_available: bool,
        escalation_approved: bool,
        cost_impact: float
    ):
        """Record a tier escalation event"""
        try:
            metric = TierEscalationMetrics(
                call_id=call_id,
                from_tier=from_tier,
                to_tier=to_tier,
                trigger=trigger,
                qualification_score=qualification_score,
                budget_available=budget_available,
                escalation_approved=escalation_approved,
                cost_impact=cost_impact
            )
            self._tier_escalations.append(metric)

            escalation_key = f"{from_tier}->{to_tier}"
            self._escalation_counts[escalation_key] += 1

            logger.info(
                f"Recorded tier escalation: {escalation_key} for call {call_id} "
                f"(score: {qualification_score}, approved: {escalation_approved})"
            )

        except Exception as e:
            logger.error(f"Error recording tier escalation: {e}")

    def record_performance(
        self,
        call_id: str,
        agent_processing_latency_ms: float,
        total_response_latency_ms: float,
        fallback_triggered: bool = False,
        circuit_breaker_tripped: bool = False,
        cache_hit: bool = False
    ):
        """Record performance metrics"""
        try:
            metric = VoiceAgentPerformanceMetrics(
                call_id=call_id,
                agent_processing_latency_ms=agent_processing_latency_ms,
                total_response_latency_ms=total_response_latency_ms,
                fallback_triggered=fallback_triggered,
                circuit_breaker_tripped=circuit_breaker_tripped,
                cache_hit=cache_hit
            )
            self._performance_metrics.append(metric)

            if fallback_triggered:
                self._fallback_count += 1

            if circuit_breaker_tripped:
                self._circuit_breaker_trips += 1

            logger.debug(
                f"Recorded performance metrics for call {call_id}: "
                f"agent_latency={agent_processing_latency_ms}ms, "
                f"total_latency={total_response_latency_ms}ms"
            )

        except Exception as e:
            logger.error(f"Error recording performance metrics: {e}")

    def get_agent_processing_stats(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated agent processing statistics"""
        try:
            if agent_name:
                times = self._agent_processing_times.get(agent_name, [])
            else:
                times = [
                    t for times_list in self._agent_processing_times.values()
                    for t in times_list
                ]

            if not times:
                return {
                    "count": 0,
                    "avg_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "p95_ms": 0.0
                }

            sorted_times = sorted(times)
            count = len(sorted_times)
            p95_index = int(count * 0.95)

            return {
                "count": count,
                "avg_ms": sum(times) / count,
                "min_ms": sorted_times[0],
                "max_ms": sorted_times[-1],
                "p95_ms": sorted_times[p95_index] if p95_index < count else sorted_times[-1]
            }

        except Exception as e:
            logger.error(f"Error calculating agent processing stats: {e}")
            return {"error": str(e)}

    def get_transition_stats(self) -> Dict[str, int]:
        """Get agent transition statistics"""
        return dict(self._transition_counts)

    def get_escalation_rate(self) -> float:
        """Calculate tier escalation rate"""
        total_calls = len(set(m.call_id for m in self._performance_metrics))
        if total_calls == 0:
            return 0.0

        total_escalations = sum(self._escalation_counts.values())
        return (total_escalations / total_calls) * 100

    def get_fallback_rate(self) -> float:
        """Calculate fallback to direct LLM rate"""
        total_requests = len(self._performance_metrics)
        if total_requests == 0:
            return 0.0

        return (self._fallback_count / total_requests) * 100

    def get_sync_failure_rate(self) -> float:
        """Calculate state synchronization failure rate"""
        total_syncs = len(self._state_syncs)
        if total_syncs == 0:
            return 0.0

        return (self._sync_failures / total_syncs) * 100

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        try:
            return {
                "agent_transitions": {
                    "total": len(self._agent_transitions),
                    "by_type": self.get_transition_stats()
                },
                "state_synchronization": {
                    "total_syncs": len(self._state_syncs),
                    "failures": self._sync_failures,
                    "failure_rate_pct": self.get_sync_failure_rate()
                },
                "tier_escalations": {
                    "total": sum(self._escalation_counts.values()),
                    "by_type": dict(self._escalation_counts),
                    "escalation_rate_pct": self.get_escalation_rate()
                },
                "performance": {
                    "total_requests": len(self._performance_metrics),
                    "fallback_count": self._fallback_count,
                    "fallback_rate_pct": self.get_fallback_rate(),
                    "circuit_breaker_trips": self._circuit_breaker_trips,
                    "agent_processing": self.get_agent_processing_stats()
                }
            }

        except Exception as e:
            logger.error(f"Error generating metrics summary: {e}")
            return {"error": str(e)}

    def reset_metrics(self):
        """Reset all collected metrics"""
        self._agent_transitions.clear()
        self._state_syncs.clear()
        self._tier_escalations.clear()
        self._performance_metrics.clear()
        self._agent_processing_times.clear()
        self._transition_counts.clear()
        self._escalation_counts.clear()
        self._fallback_count = 0
        self._circuit_breaker_trips = 0
        self._sync_failures = 0
        logger.info("Metrics reset completed")


# Global metrics instance
voice_agent_metrics = VoiceAgentIntegrationMetrics()
