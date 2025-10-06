"""
Performance Optimizer for LLM Inference

This module provides auto-tuning capabilities for LLM inference performance,
including GPU utilization optimization, concurrency tuning, and batch size optimization.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque

from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PerformanceProfile:
    """Performance profile for a specific configuration."""
    concurrency: int
    batch_size: int
    avg_latency_ms: float
    throughput_rps: float
    gpu_utilization_percent: float
    success_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationMetrics:
    """Metrics for optimization decisions."""
    current_concurrency: int
    target_concurrency: int
    current_batch_size: int
    target_batch_size: int
    gpu_utilization: float
    target_gpu_utilization: float
    latency_p95_ms: float
    throughput_rps: float
    optimization_score: float


class PerformanceOptimizer:
    """
    Automatically tunes LLM performance parameters based on real-time metrics.

    Features:
    - GPU utilization-based concurrency tuning
    - Adaptive batch size optimization
    - Latency-aware scaling
    - Performance regression detection
    """

    def __init__(
        self,
        target_gpu_utilization: float = 0.75,
        target_latency_ms: float = 2000.0,
        min_concurrency: int = 3,
        max_concurrency: int = 10,
        adjustment_interval: float = 60.0
    ):
        """
        Initialize performance optimizer.

        Args:
            target_gpu_utilization: Target GPU utilization (0-1)
            target_latency_ms: Target latency threshold
            min_concurrency: Minimum concurrency level
            max_concurrency: Maximum concurrency level
            adjustment_interval: Seconds between adjustments
        """
        self.target_gpu_utilization = target_gpu_utilization
        self.target_latency_ms = target_latency_ms
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.adjustment_interval = adjustment_interval

        # Current configuration
        self.current_concurrency = settings.llm_max_concurrent
        self.current_batch_size = settings.ollama_batch_max_size

        # Performance history
        self.performance_history: deque = deque(maxlen=100)
        self.profiles: List[PerformanceProfile] = []

        # Optimization state
        self.last_adjustment_time = time.time()
        self.adjustment_count = 0
        self.improvement_streak = 0
        self.degradation_count = 0

        # Baseline metrics
        self.baseline_latency_ms: Optional[float] = None
        self.baseline_throughput_rps: Optional[float] = None

    def record_performance(
        self,
        latency_ms: float,
        throughput_rps: float,
        gpu_utilization: float,
        success_rate: float
    ):
        """
        Record performance metrics for a request.

        Args:
            latency_ms: Request latency in milliseconds
            throughput_rps: Throughput in requests per second
            gpu_utilization: GPU utilization percentage (0-1)
            success_rate: Success rate (0-1)
        """
        self.performance_history.append({
            "latency_ms": latency_ms,
            "throughput_rps": throughput_rps,
            "gpu_utilization": gpu_utilization,
            "success_rate": success_rate,
            "timestamp": time.time()
        })

        # Update baseline if not set
        if self.baseline_latency_ms is None:
            self.baseline_latency_ms = latency_ms
            self.baseline_throughput_rps = throughput_rps

    def should_optimize(self) -> bool:
        """
        Determine if optimization should be performed.

        Returns:
            True if optimization should run
        """
        # Need enough data
        if len(self.performance_history) < 10:
            return False

        # Respect adjustment interval
        time_since_adjustment = time.time() - self.last_adjustment_time
        if time_since_adjustment < self.adjustment_interval:
            return False

        return True

    def analyze_performance(self) -> OptimizationMetrics:
        """
        Analyze current performance and suggest optimizations.

        Returns:
            OptimizationMetrics with current and target values
        """
        if not self.performance_history:
            raise ValueError("No performance data available")

        # Calculate aggregate metrics
        recent_data = list(self.performance_history)[-20:]  # Last 20 samples

        avg_latency = sum(d["latency_ms"] for d in recent_data) / len(recent_data)
        avg_gpu_util = sum(d["gpu_utilization"] for d in recent_data) / len(recent_data)
        avg_throughput = sum(d["throughput_rps"] for d in recent_data) / len(recent_data)
        avg_success_rate = sum(d["success_rate"] for d in recent_data) / len(recent_data)

        # Calculate P95 latency
        latencies = sorted([d["latency_ms"] for d in recent_data])
        p95_index = int(len(latencies) * 0.95)
        latency_p95 = latencies[p95_index] if p95_index < len(latencies) else latencies[-1]

        # Determine target concurrency
        target_concurrency = self.current_concurrency

        if avg_gpu_util < self.target_gpu_utilization - 0.15:
            # GPU underutilized, increase concurrency
            if latency_p95 < self.target_latency_ms * 0.8:
                target_concurrency = min(self.current_concurrency + 1, self.max_concurrency)
                logger.info(
                    f"GPU underutilized ({avg_gpu_util*100:.1f}%), latency good "
                    f"({latency_p95:.0f}ms), suggesting concurrency increase"
                )

        elif avg_gpu_util > self.target_gpu_utilization + 0.15:
            # GPU overutilized, decrease concurrency
            target_concurrency = max(self.current_concurrency - 1, self.min_concurrency)
            logger.info(
                f"GPU overutilized ({avg_gpu_util*100:.1f}%), suggesting concurrency decrease"
            )

        elif latency_p95 > self.target_latency_ms:
            # Latency too high, decrease concurrency
            target_concurrency = max(self.current_concurrency - 1, self.min_concurrency)
            logger.info(
                f"Latency too high ({latency_p95:.0f}ms > {self.target_latency_ms:.0f}ms), "
                f"suggesting concurrency decrease"
            )

        # Determine target batch size (simple heuristic)
        target_batch_size = self.current_batch_size
        if avg_gpu_util < 0.5 and avg_success_rate > 0.95:
            # Can potentially increase batch size
            target_batch_size = min(self.current_batch_size + 1, 10)

        # Calculate optimization score
        gpu_score = 1.0 - abs(avg_gpu_util - self.target_gpu_utilization)
        latency_score = max(0, 1.0 - (latency_p95 / self.target_latency_ms))
        success_score = avg_success_rate

        optimization_score = (gpu_score + latency_score + success_score) / 3.0

        return OptimizationMetrics(
            current_concurrency=self.current_concurrency,
            target_concurrency=target_concurrency,
            current_batch_size=self.current_batch_size,
            target_batch_size=target_batch_size,
            gpu_utilization=avg_gpu_util,
            target_gpu_utilization=self.target_gpu_utilization,
            latency_p95_ms=latency_p95,
            throughput_rps=avg_throughput,
            optimization_score=optimization_score
        )

    def apply_optimization(self, metrics: OptimizationMetrics) -> Dict[str, Any]:
        """
        Apply optimization based on analysis.

        Args:
            metrics: OptimizationMetrics to apply

        Returns:
            Dictionary with applied changes
        """
        changes = {
            "concurrency_changed": False,
            "batch_size_changed": False,
            "previous_concurrency": self.current_concurrency,
            "new_concurrency": metrics.target_concurrency,
            "previous_batch_size": self.current_batch_size,
            "new_batch_size": metrics.target_batch_size
        }

        # Apply concurrency change
        if metrics.target_concurrency != self.current_concurrency:
            self.current_concurrency = metrics.target_concurrency
            changes["concurrency_changed"] = True
            logger.info(
                f"Concurrency adjusted: {changes['previous_concurrency']} -> "
                f"{changes['new_concurrency']}"
            )

        # Apply batch size change
        if metrics.target_batch_size != self.current_batch_size:
            self.current_batch_size = metrics.target_batch_size
            changes["batch_size_changed"] = True
            logger.info(
                f"Batch size adjusted: {changes['previous_batch_size']} -> "
                f"{changes['new_batch_size']}"
            )

        # Update optimization state
        self.last_adjustment_time = time.time()
        self.adjustment_count += 1

        # Track improvement/degradation
        if metrics.optimization_score > 0.75:
            self.improvement_streak += 1
            self.degradation_count = 0
        elif metrics.optimization_score < 0.50:
            self.degradation_count += 1
            self.improvement_streak = 0

        # Save profile
        profile = PerformanceProfile(
            concurrency=self.current_concurrency,
            batch_size=self.current_batch_size,
            avg_latency_ms=metrics.latency_p95_ms,
            throughput_rps=metrics.throughput_rps,
            gpu_utilization_percent=metrics.gpu_utilization * 100,
            success_rate=1.0  # Assume success if we got here
        )
        self.profiles.append(profile)

        # Keep only last 50 profiles
        if len(self.profiles) > 50:
            self.profiles.pop(0)

        return changes

    async def optimize(self) -> Optional[Dict[str, Any]]:
        """
        Perform optimization if conditions are met.

        Returns:
            Dictionary with optimization results, or None if no optimization performed
        """
        if not self.should_optimize():
            return None

        try:
            metrics = self.analyze_performance()
            changes = self.apply_optimization(metrics)

            result = {
                "optimized": changes["concurrency_changed"] or changes["batch_size_changed"],
                "metrics": {
                    "gpu_utilization": metrics.gpu_utilization,
                    "target_gpu_utilization": metrics.target_gpu_utilization,
                    "latency_p95_ms": metrics.latency_p95_ms,
                    "throughput_rps": metrics.throughput_rps,
                    "optimization_score": metrics.optimization_score
                },
                "changes": changes,
                "adjustment_count": self.adjustment_count,
                "improvement_streak": self.improvement_streak
            }

            if result["optimized"]:
                logger.info(
                    f"Performance optimized: score={metrics.optimization_score:.2f}, "
                    f"concurrency={self.current_concurrency}, batch_size={self.current_batch_size}"
                )

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return None

    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "concurrency": self.current_concurrency,
            "batch_size": self.current_batch_size,
            "target_gpu_utilization": self.target_gpu_utilization,
            "target_latency_ms": self.target_latency_ms
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.performance_history:
            return {"error": "No performance data"}

        recent = list(self.performance_history)[-50:]

        return {
            "total_samples": len(self.performance_history),
            "avg_latency_ms": sum(d["latency_ms"] for d in recent) / len(recent),
            "avg_throughput_rps": sum(d["throughput_rps"] for d in recent) / len(recent),
            "avg_gpu_utilization": sum(d["gpu_utilization"] for d in recent) / len(recent),
            "avg_success_rate": sum(d["success_rate"] for d in recent) / len(recent),
            "adjustment_count": self.adjustment_count,
            "improvement_streak": self.improvement_streak,
            "degradation_count": self.degradation_count,
            "profiles_count": len(self.profiles)
        }

    def reset(self):
        """Reset optimizer state."""
        self.performance_history.clear()
        self.profiles.clear()
        self.adjustment_count = 0
        self.improvement_streak = 0
        self.degradation_count = 0
        self.baseline_latency_ms = None
        self.baseline_throughput_rps = None
        logger.info("Performance optimizer reset")
