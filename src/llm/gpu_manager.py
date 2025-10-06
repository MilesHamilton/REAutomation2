"""
GPU Memory Manager for Ollama Models

This module monitors and manages GPU memory usage for LLM inference,
implementing automatic model unloading and preloading strategies.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import deque

from ..config import settings

logger = logging.getLogger(__name__)

# Try to import GPU monitoring libraries
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available, GPU monitoring disabled")


@dataclass
class ModelInfo:
    """Information about a loaded model"""
    name: str
    memory_mb: float = 0.0
    last_used: float = field(default_factory=time.time)
    access_count: int = 0
    load_time: float = 0.0


@dataclass
class GPUMetrics:
    """GPU usage metrics"""
    total_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    free_memory_mb: float = 0.0
    utilization_percent: float = 0.0
    temperature: float = 0.0
    models_loaded: int = 0
    models_unloaded: int = 0
    memory_threshold_breaches: int = 0


class GPUMemoryManager:
    """
    Manages GPU memory for Ollama models

    Features:
    - Monitors GPU memory usage
    - Automatically unloads models when memory is low
    - Preloads frequently used models
    - Tracks model usage patterns
    """

    def __init__(
        self,
        enabled: bool = True,
        memory_threshold_mb: int = 5120,
        monitoring_interval: int = 10,
        auto_unload: bool = True
    ):
        self.enabled = enabled and PYNVML_AVAILABLE
        self.memory_threshold_mb = memory_threshold_mb
        self.monitoring_interval = monitoring_interval
        self.auto_unload = auto_unload

        # GPU handle
        self._gpu_handle = None
        self._gpu_index = 0

        # Loaded models tracking
        self.loaded_models: Dict[str, ModelInfo] = {}

        # Usage history for prediction
        self.usage_history: deque = deque(maxlen=100)

        # Metrics
        self.metrics = GPUMetrics()

        # Monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

        if not self.enabled:
            logger.warning("GPU memory manager disabled (pynvml not available)")

    async def start(self):
        """Start the GPU memory manager"""
        if not self.enabled:
            logger.info("GPU memory manager not started (disabled)")
            return

        if self._running:
            logger.warning("GPU memory manager already running")
            return

        try:
            # Initialize NVML
            pynvml.nvmlInit()

            # Get GPU handle (use first GPU)
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)

            # Get total memory
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            self.metrics.total_memory_mb = memory_info.total / (1024 ** 2)

            logger.info(
                f"GPU memory manager started (GPU {self._gpu_index}, "
                f"Total memory: {self.metrics.total_memory_mb:.0f}MB, "
                f"Threshold: {self.memory_threshold_mb}MB)"
            )

            # Start monitoring task
            self._running = True
            self._monitor_task = asyncio.create_task(self._monitoring_loop())

        except Exception as e:
            logger.error(f"Failed to start GPU memory manager: {e}")
            self.enabled = False

    async def stop(self):
        """Stop the GPU memory manager"""
        if not self._running:
            return

        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Shutdown NVML
        if self.enabled:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

        logger.info("GPU memory manager stopped")

    async def _monitoring_loop(self):
        """Background loop for monitoring GPU memory"""
        logger.debug("GPU monitoring loop started")

        while self._running:
            try:
                # Update GPU metrics
                await self._update_metrics()

                # Check if memory threshold exceeded
                if self.metrics.used_memory_mb > self.memory_threshold_mb:
                    self.metrics.memory_threshold_breaches += 1
                    logger.warning(
                        f"GPU memory threshold exceeded: "
                        f"{self.metrics.used_memory_mb:.0f}MB > {self.memory_threshold_mb}MB"
                    )

                    # Auto-unload models if enabled
                    if self.auto_unload:
                        await self._unload_least_used_model()

                # Sleep until next check
                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                logger.debug("GPU monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)

        logger.debug("GPU monitoring loop stopped")

    async def _update_metrics(self):
        """Update GPU usage metrics"""
        if not self.enabled or not self._gpu_handle:
            return

        try:
            # Get memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            self.metrics.used_memory_mb = memory_info.used / (1024 ** 2)
            self.metrics.free_memory_mb = memory_info.free / (1024 ** 2)

            # Get utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            self.metrics.utilization_percent = utilization.gpu

            # Get temperature (if available)
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(
                    self._gpu_handle,
                    pynvml.NVML_TEMPERATURE_GPU
                )
                self.metrics.temperature = temperature
            except:
                pass

        except Exception as e:
            logger.error(f"Failed to update GPU metrics: {e}")

    async def track_model_usage(self, model_name: str):
        """Track that a model was used"""
        if model_name not in self.loaded_models:
            # New model loaded
            self.loaded_models[model_name] = ModelInfo(
                name=model_name,
                last_used=time.time(),
                access_count=1
            )
            self.metrics.models_loaded += 1
        else:
            # Existing model accessed
            model_info = self.loaded_models[model_name]
            model_info.last_used = time.time()
            model_info.access_count += 1

        # Add to usage history
        self.usage_history.append((time.time(), model_name))

    async def _unload_least_used_model(self):
        """Unload the least recently used model"""
        if not self.loaded_models:
            logger.warning("No models to unload")
            return

        # Find least recently used model
        lru_model = min(
            self.loaded_models.values(),
            key=lambda m: m.last_used
        )

        logger.info(
            f"Unloading least used model: {lru_model.name} "
            f"(last used {time.time() - lru_model.last_used:.0f}s ago)"
        )

        # Remove from tracking
        del self.loaded_models[lru_model.name]
        self.metrics.models_unloaded += 1

        # Note: Actual model unloading would need to call Ollama API
        # This is a placeholder for the tracking logic
        # In production, you would call: await self._ollama_unload_model(lru_model.name)

    async def get_recommended_models_to_preload(self, count: int = 3) -> List[str]:
        """
        Get list of models to preload based on usage patterns

        Returns the most frequently used models from recent history.
        """
        if not self.usage_history:
            return []

        # Count model usage in recent history
        model_counts: Dict[str, int] = {}
        for _, model_name in self.usage_history:
            model_counts[model_name] = model_counts.get(model_name, 0) + 1

        # Sort by usage count
        sorted_models = sorted(
            model_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top N models that aren't already loaded
        recommended = []
        for model_name, _ in sorted_models:
            if model_name not in self.loaded_models:
                recommended.append(model_name)
                if len(recommended) >= count:
                    break

        return recommended

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage breakdown"""
        if not self.enabled:
            return {
                "total_mb": 0.0,
                "used_mb": 0.0,
                "free_mb": 0.0,
                "used_percent": 0.0
            }

        used_percent = 0.0
        if self.metrics.total_memory_mb > 0:
            used_percent = (
                self.metrics.used_memory_mb / self.metrics.total_memory_mb * 100
            )

        return {
            "total_mb": self.metrics.total_memory_mb,
            "used_mb": self.metrics.used_memory_mb,
            "free_mb": self.metrics.free_memory_mb,
            "used_percent": used_percent,
            "threshold_mb": self.memory_threshold_mb,
            "above_threshold": self.metrics.used_memory_mb > self.memory_threshold_mb
        }

    async def get_metrics(self) -> Dict[str, any]:
        """Get comprehensive GPU metrics"""
        await self._update_metrics()

        return {
            "enabled": self.enabled,
            "total_memory_mb": self.metrics.total_memory_mb,
            "used_memory_mb": self.metrics.used_memory_mb,
            "free_memory_mb": self.metrics.free_memory_mb,
            "utilization_percent": self.metrics.utilization_percent,
            "temperature": self.metrics.temperature,
            "models_loaded": len(self.loaded_models),
            "models_loaded_total": self.metrics.models_loaded,
            "models_unloaded_total": self.metrics.models_unloaded,
            "memory_threshold_breaches": self.metrics.memory_threshold_breaches,
            "loaded_models": [
                {
                    "name": model.name,
                    "memory_mb": model.memory_mb,
                    "last_used_seconds_ago": time.time() - model.last_used,
                    "access_count": model.access_count
                }
                for model in self.loaded_models.values()
            ]
        }

    async def health_check(self) -> Dict[str, any]:
        """Check GPU manager health"""
        if not self.enabled:
            return {
                "status": "disabled",
                "issues": ["GPU monitoring not available (pynvml not installed)"]
            }

        await self._update_metrics()

        status = "healthy"
        issues = []

        # Check memory usage
        memory_usage = self.get_memory_usage()
        if memory_usage["above_threshold"]:
            status = "degraded"
            issues.append(
                f"GPU memory above threshold: {memory_usage['used_mb']:.0f}MB > "
                f"{memory_usage['threshold_mb']}MB"
            )

        # Check utilization
        if self.metrics.utilization_percent > 95:
            status = "degraded"
            issues.append(f"High GPU utilization: {self.metrics.utilization_percent:.1f}%")

        # Check temperature
        if self.metrics.temperature > 85:
            status = "degraded"
            issues.append(f"High GPU temperature: {self.metrics.temperature}°C")

        return {
            "status": status,
            "issues": issues,
            "running": self._running,
            **memory_usage
        }


# Global GPU manager instance
gpu_manager = GPUMemoryManager(
    enabled=settings.ollama_auto_unload_models,
    memory_threshold_mb=settings.ollama_gpu_memory_threshold_mb,
    monitoring_interval=settings.ollama_gpu_monitoring_interval,
    auto_unload=settings.ollama_auto_unload_models
)
