"""
Performance monitoring and metrics collection for REAutomation2
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
import psutil
import tracemalloc
from dataclasses import dataclass, field

from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from ..database.connection import get_database_session as get_db
from ..database.monitoring_models import PerformanceMetrics
from .models import MetricCategory, PerformanceMetric
from .langsmith_client import get_langsmith_client
from ..config import settings

logger = logging.getLogger(__name__)

# Create a simple MetricType enum for compatibility
class MetricType:
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    
    @property
    def value(self):
        return self


@dataclass
class MetricBuffer:
    """Buffer for collecting metrics before batch insertion"""
    metrics: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_flush: float = field(default_factory=time.time)
    flush_interval: float = 60.0  # Flush every minute


class PerformanceMonitor:
    """Centralized performance monitoring and metrics collection"""

    def __init__(self):
        # Check settings to determine if monitoring should be enabled
        self.enabled = settings.performance_monitoring_enabled and settings.metrics_enabled
        self.metric_buffers: Dict[str, MetricBuffer] = defaultdict(MetricBuffer)
        self.active_timers: Dict[str, float] = {}
        self.call_counters: Dict[str, int] = defaultdict(int)
        self.error_counters: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # System monitoring - respect settings
        self.system_metrics_enabled = settings.system_metrics_enabled and self.enabled
        self.memory_tracking_enabled = False

        # Background tasks
        self._monitoring_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize performance monitoring"""
        try:
            if not self.enabled:
                return

            logger.info("Initializing performance monitoring...")

            # Start memory tracking if enabled
            if self.memory_tracking_enabled:
                tracemalloc.start()

            # Start background monitoring tasks
            self._monitoring_tasks = [
                asyncio.create_task(self._system_metrics_collector()),
                asyncio.create_task(self._metric_flusher()),
                asyncio.create_task(self._performance_analyzer())
            ]

            logger.info("Performance monitoring initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize performance monitoring: {e}")
            self.enabled = False

    async def shutdown(self):
        """Shutdown performance monitoring"""
        try:
            self._shutdown_event.set()

            # Cancel background tasks
            for task in self._monitoring_tasks:
                task.cancel()

            # Wait for tasks to finish
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)

            # Flush remaining metrics
            await self._flush_all_metrics()

            if self.memory_tracking_enabled:
                tracemalloc.stop()

            logger.info("Performance monitoring shutdown complete")

        except Exception as e:
            logger.error(f"Error during performance monitoring shutdown: {e}")

    # Metric Collection Methods
    def record_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: str = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        call_id: Optional[str] = None
    ):
        """Record a performance metric"""
        if not self.enabled:
            return

        try:
            # Handle both string and MetricType inputs
            if hasattr(metric_type, 'value'):
                metric_type_str = metric_type.value
            else:
                metric_type_str = str(metric_type)
                
            metric_data = {
                "metric_name": metric_name,
                "metric_value": value,
                "metric_type": metric_type_str,
                "tags": tags or {},
                "call_id": call_id,
                "recorded_at": datetime.utcnow(),
                "source": "performance_monitor"
            }

            # Add to buffer
            buffer_key = f"{metric_type_str}:{metric_name}"
            self.metric_buffers[buffer_key].metrics.append(metric_data)

        except Exception as e:
            logger.error(f"Error recording metric {metric_name}: {e}")

    def start_timer(self, operation_name: str, call_id: Optional[str] = None) -> str:
        """Start timing an operation"""
        timer_key = f"{operation_name}:{call_id}" if call_id else operation_name
        self.active_timers[timer_key] = time.time()
        return timer_key

    def end_timer(self, timer_key: str, tags: Optional[Dict[str, str]] = None):
        """End timing an operation and record the duration"""
        if timer_key not in self.active_timers:
            logger.warning(f"Timer {timer_key} not found")
            return

        duration_ms = (time.time() - self.active_timers.pop(timer_key)) * 1000

        # Extract operation name
        operation_name = timer_key.split(':')[0]
        call_id = timer_key.split(':')[1] if ':' in timer_key else None

        # Record timing metric
        self.record_metric(
            f"{operation_name}_duration_ms",
            duration_ms,
            MetricType.HISTOGRAM,
            tags,
            call_id
        )

        # Update response time tracking
        self.response_times[operation_name].append(duration_ms)

    def increment_counter(self, counter_name: str, tags: Optional[Dict[str, str]] = None, call_id: Optional[str] = None):
        """Increment a counter metric"""
        self.call_counters[counter_name] += 1
        self.record_metric(
            counter_name,
            self.call_counters[counter_name],
            MetricType.COUNTER,
            tags,
            call_id
        )

    def record_error(self, error_type: str, error_message: str, call_id: Optional[str] = None):
        """Record an error occurrence"""
        self.error_counters[error_type] += 1

        tags = {
            "error_type": error_type,
            "error_message": error_message[:200]  # Truncate long messages
        }

        self.record_metric(
            "errors_total",
            self.error_counters[error_type],
            MetricType.COUNTER,
            tags,
            call_id
        )

    def record_agent_execution(
        self,
        agent_type: str,
        duration_ms: float,
        success: bool,
        call_id: Optional[str] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Record agent execution metrics"""
        tags = {
            "agent_type": agent_type,
            "success": str(success)
        }

        # Record duration
        self.record_metric(
            "agent_execution_duration_ms",
            duration_ms,
            MetricType.HISTOGRAM,
            tags,
            call_id
        )

        # Record success/failure
        status_metric = "agent_executions_success" if success else "agent_executions_failure"
        self.increment_counter(status_metric, tags, call_id)

        # Record additional metrics if provided
        if additional_metrics:
            for metric_name, value in additional_metrics.items():
                metric_tags = {**tags, "metric_source": "agent"}
                self.record_metric(
                    f"agent_{metric_name}",
                    value,
                    MetricType.GAUGE,
                    metric_tags,
                    call_id
                )

    def record_llm_metrics(
        self,
        provider: str,
        model: str,
        tokens_used: int,
        cost: float,
        duration_ms: float,
        call_id: Optional[str] = None
    ):
        """Record LLM usage metrics"""
        tags = {
            "provider": provider,
            "model": model
        }

        # Token usage
        self.record_metric("llm_tokens_used", tokens_used, MetricType.COUNTER, tags, call_id)

        # Cost
        self.record_metric("llm_cost", cost, MetricType.COUNTER, tags, call_id)

        # Duration
        self.record_metric("llm_duration_ms", duration_ms, MetricType.HISTOGRAM, tags, call_id)

        # Tokens per second
        if duration_ms > 0:
            tokens_per_second = (tokens_used / duration_ms) * 1000
            self.record_metric("llm_tokens_per_second", tokens_per_second, MetricType.GAUGE, tags, call_id)

    def record_gpu_metrics(
        self,
        gpu_id: int,
        utilization_percent: float,
        memory_used_mb: float,
        memory_total_mb: float,
        temperature_celsius: float,
        power_draw_watts: Optional[float] = None
    ):
        """Record GPU performance metrics"""
        tags = {"gpu_id": str(gpu_id)}

        # Utilization
        self.record_metric("gpu_utilization_percent", utilization_percent, MetricType.GAUGE, tags)

        # Memory
        self.record_metric("gpu_memory_used_mb", memory_used_mb, MetricType.GAUGE, tags)
        self.record_metric("gpu_memory_total_mb", memory_total_mb, MetricType.GAUGE, tags)

        memory_utilization = (memory_used_mb / memory_total_mb * 100) if memory_total_mb > 0 else 0
        self.record_metric("gpu_memory_utilization_percent", memory_utilization, MetricType.GAUGE, tags)

        # Temperature
        self.record_metric("gpu_temperature_celsius", temperature_celsius, MetricType.GAUGE, tags)

        # Power draw (if available)
        if power_draw_watts is not None:
            self.record_metric("gpu_power_draw_watts", power_draw_watts, MetricType.GAUGE, tags)

    def record_streaming_metrics(
        self,
        time_to_first_chunk_ms: float,
        total_chunks: int,
        total_tokens: int,
        total_duration_ms: float,
        throughput_tokens_per_second: float,
        call_id: Optional[str] = None
    ):
        """Record streaming response metrics"""
        tags = {"source": "streaming"}

        # Time to first chunk (TTFC) - critical for perceived latency
        self.record_metric("streaming_ttfc_ms", time_to_first_chunk_ms, MetricType.HISTOGRAM, tags, call_id)

        # Total chunks
        self.record_metric("streaming_chunks_total", total_chunks, MetricType.HISTOGRAM, tags, call_id)

        # Total tokens
        self.record_metric("streaming_tokens_total", total_tokens, MetricType.HISTOGRAM, tags, call_id)

        # Duration
        self.record_metric("streaming_duration_ms", total_duration_ms, MetricType.HISTOGRAM, tags, call_id)

        # Throughput
        self.record_metric("streaming_throughput_tokens_per_second", throughput_tokens_per_second, MetricType.GAUGE, tags, call_id)

        # Chunks per second
        if total_duration_ms > 0:
            chunks_per_second = (total_chunks / total_duration_ms) * 1000
            self.record_metric("streaming_chunks_per_second", chunks_per_second, MetricType.GAUGE, tags, call_id)

    # System Metrics Collection
    async def _system_metrics_collector(self):
        """Background task to collect system metrics"""
        while not self._shutdown_event.is_set():
            try:
                if self.system_metrics_enabled:
                    await self._collect_system_metrics()

                await asyncio.sleep(120)  # Collect every 2 minutes (reduced frequency)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system metrics collector: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system_cpu_percent", cpu_percent, MetricType.GAUGE)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system_memory_percent", memory.percent, MetricType.GAUGE)
            self.record_metric("system_memory_used_mb", memory.used / 1024 / 1024, MetricType.GAUGE)
            self.record_metric("system_memory_available_mb", memory.available / 1024 / 1024, MetricType.GAUGE)

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric("system_disk_percent", (disk.used / disk.total) * 100, MetricType.GAUGE)
            self.record_metric("system_disk_free_gb", disk.free / 1024 / 1024 / 1024, MetricType.GAUGE)

            # Network metrics (if available)
            try:
                net_io = psutil.net_io_counters()
                self.record_metric("system_network_bytes_sent", net_io.bytes_sent, MetricType.COUNTER)
                self.record_metric("system_network_bytes_recv", net_io.bytes_recv, MetricType.COUNTER)
            except:
                pass  # Network metrics might not be available

            # Process-specific metrics
            process = psutil.Process()
            self.record_metric("process_memory_mb", process.memory_info().rss / 1024 / 1024, MetricType.GAUGE)
            self.record_metric("process_cpu_percent", process.cpu_percent(), MetricType.GAUGE)

            # GPU metrics (if available)
            try:
                from ..llm.gpu_manager import gpu_manager
                gpu_metrics = await gpu_manager.get_metrics()

                if gpu_metrics and gpu_metrics.get("available", False):
                    self.record_gpu_metrics(
                        gpu_id=0,  # Primary GPU
                        utilization_percent=gpu_metrics.get("utilization_percent", 0),
                        memory_used_mb=gpu_metrics.get("used_memory_mb", 0),
                        memory_total_mb=gpu_metrics.get("total_memory_mb", 0),
                        temperature_celsius=gpu_metrics.get("temperature", 0),
                        power_draw_watts=gpu_metrics.get("power_draw", None)
                    )
            except Exception as e:
                logger.debug(f"GPU metrics not available: {e}")

            # Python-specific memory tracking
            if self.memory_tracking_enabled:
                current, peak = tracemalloc.get_traced_memory()
                self.record_metric("python_memory_current_mb", current / 1024 / 1024, MetricType.GAUGE)
                self.record_metric("python_memory_peak_mb", peak / 1024 / 1024, MetricType.GAUGE)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    # Metric Persistence
    async def _metric_flusher(self):
        """Background task to flush metrics to database"""
        while not self._shutdown_event.is_set():
            try:
                await self._flush_all_metrics()
                await asyncio.sleep(180)  # Flush every 3 minutes (reduced frequency)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metric flusher: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _flush_all_metrics(self):
        """Flush all metric buffers to database"""
        if not self.enabled:
            return

        total_flushed = 0

        for buffer_key, buffer in self.metric_buffers.items():
            if len(buffer.metrics) > 0:
                flushed = await self._flush_buffer(buffer)
                total_flushed += flushed

        if total_flushed > 0:
            logger.debug(f"Flushed {total_flushed} metrics to database")

    async def _flush_buffer(self, buffer: MetricBuffer) -> int:
        """Flush a specific metric buffer with proper async database operations"""
        if len(buffer.metrics) == 0:
            return 0

        try:
            # Convert metrics to database records
            db_records = []
            metrics_to_flush = []

            # Batch size reduced to prevent overwhelming the database
            while buffer.metrics and len(metrics_to_flush) < 50:  # Smaller batch size
                metrics_to_flush.append(buffer.metrics.popleft())

            for metric_data in metrics_to_flush:
                # Generate required IDs for the database model
                from ..database.monitoring_models import create_metric_id
                
                db_record = PerformanceMetrics(
                    metric_id=create_metric_id(
                        metric_data.get("metric_type", "gauge"), 
                        metric_data["metric_name"], 
                        metric_data["recorded_at"]
                    ),
                    metric_category=metric_data.get("metric_type", "gauge"),
                    metric_name=metric_data["metric_name"],
                    metric_value=metric_data["metric_value"],
                    call_id=metric_data.get("call_id"),
                    aggregation_level="call" if metric_data.get("call_id") else "system",
                    time_window="real_time",
                    recorded_at=metric_data["recorded_at"],
                    time_bucket=metric_data["recorded_at"].replace(second=0, microsecond=0),  # Round to minute
                    additional_metadata=metric_data.get("tags", {})
                )
                db_records.append(db_record)

            # Bulk insert to database using proper async context manager
            if db_records:
                from ..database.connection import db_manager
                async with db_manager.get_session() as db:
                    db.add_all(db_records)
                    await db.commit()

            buffer.last_flush = time.time()
            return len(db_records)

        except Exception as e:
            logger.error(f"Error flushing metrics buffer: {e}")
            # Put metrics back in buffer on error
            for metric in reversed(metrics_to_flush):
                buffer.metrics.appendleft(metric)
            return 0

    # Performance Analysis
    async def _performance_analyzer(self):
        """Background task to analyze performance patterns"""
        while not self._shutdown_event.is_set():
            try:
                await self._analyze_performance()
                await asyncio.sleep(300)  # Analyze every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance analyzer: {e}")
                await asyncio.sleep(600)  # Wait longer on error

    async def _analyze_performance(self):
        """Analyze performance metrics and identify issues"""
        try:
            # Analyze response times
            for operation, times in self.response_times.items():
                if len(times) >= 10:  # Need sufficient data
                    avg_time = sum(times) / len(times)
                    max_time = max(times)

                    # Check for performance degradation
                    if avg_time > 5000:  # 5 seconds
                        logger.warning(f"High average response time for {operation}: {avg_time:.2f}ms")

                    if max_time > 30000:  # 30 seconds
                        logger.warning(f"Very high max response time for {operation}: {max_time:.2f}ms")

            # Analyze error rates
            total_calls = sum(self.call_counters.values())
            total_errors = sum(self.error_counters.values())

            if total_calls > 0:
                error_rate = (total_errors / total_calls) * 100
                if error_rate > 5:  # 5% error rate threshold
                    logger.warning(f"High error rate: {error_rate:.2f}% ({total_errors}/{total_calls})")

            # Send metrics to LangSmith if enabled
            langsmith_client = get_langsmith_client()
            if langsmith_client.enabled:
                await self._send_performance_summary_to_langsmith()

        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")

    async def _send_performance_summary_to_langsmith(self):
        """Send performance summary to LangSmith"""
        try:
            langsmith_client = get_langsmith_client()

            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "active_timers": len(self.active_timers),
                "total_calls": sum(self.call_counters.values()),
                "total_errors": sum(self.error_counters.values()),
                "avg_response_times": {
                    op: sum(times) / len(times) if times else 0
                    for op, times in self.response_times.items()
                }
            }

            # This would be sent to LangSmith as a custom metric
            # Implementation depends on LangSmith's custom metrics API

        except Exception as e:
            logger.error(f"Error sending performance summary to LangSmith: {e}")

    # Query Methods
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            "enabled": self.enabled,
            "active_timers": len(self.active_timers),
            "metric_buffers": {
                key: len(buffer.metrics)
                for key, buffer in self.metric_buffers.items()
            },
            "call_counters": dict(self.call_counters),
            "error_counters": dict(self.error_counters),
            "avg_response_times": {
                op: sum(times) / len(times) if times else 0
                for op, times in self.response_times.items()
            }
        }

    async def get_metrics_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get metrics summary from database"""
        try:
            if start_time is None:
                start_time = datetime.utcnow() - timedelta(hours=1)
            if end_time is None:
                end_time = datetime.utcnow()

            async for db in get_db():
                query = db.query(PerformanceMetrics).filter(
                    and_(
                        PerformanceMetrics.recorded_at >= start_time,
                        PerformanceMetrics.recorded_at <= end_time
                    )
                )

                if metric_names:
                    query = query.filter(PerformanceMetrics.metric_name.in_(metric_names))

                metrics = query.all()
                break

            # Aggregate metrics
            summary = {
                "total_metrics": len(metrics),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "by_name": {},
                "by_type": {}
            }

            for metric in metrics:
                # By name
                if metric.metric_name not in summary["by_name"]:
                    summary["by_name"][metric.metric_name] = {
                        "count": 0,
                        "avg": 0,
                        "min": float('inf'),
                        "max": float('-inf'),
                        "sum": 0
                    }

                name_stats = summary["by_name"][metric.metric_name]
                name_stats["count"] += 1
                name_stats["sum"] += metric.metric_value
                name_stats["min"] = min(name_stats["min"], metric.metric_value)
                name_stats["max"] = max(name_stats["max"], metric.metric_value)
                name_stats["avg"] = name_stats["sum"] / name_stats["count"]

                # By type
                if metric.metric_type not in summary["by_type"]:
                    summary["by_type"][metric.metric_type] = 0
                summary["by_type"][metric.metric_type] += 1

            return summary

        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {"error": str(e)}


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Decorator for automatic performance monitoring
def monitor_performance(operation_name: Optional[str] = None):
    """Decorator to automatically monitor function performance"""
    def decorator(func: Callable):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"

        async def async_wrapper(*args, **kwargs):
            timer_key = performance_monitor.start_timer(operation_name)
            try:
                result = await func(*args, **kwargs)
                performance_monitor.end_timer(timer_key, {"success": "true"})
                performance_monitor.increment_counter(f"{operation_name}_calls")
                return result
            except Exception as e:
                performance_monitor.end_timer(timer_key, {"success": "false"})
                performance_monitor.record_error(type(e).__name__, str(e))
                performance_monitor.increment_counter(f"{operation_name}_errors")
                raise

        def sync_wrapper(*args, **kwargs):
            timer_key = performance_monitor.start_timer(operation_name)
            try:
                result = func(*args, **kwargs)
                performance_monitor.end_timer(timer_key, {"success": "true"})
                performance_monitor.increment_counter(f"{operation_name}_calls")
                return result
            except Exception as e:
                performance_monitor.end_timer(timer_key, {"success": "false"})
                performance_monitor.record_error(type(e).__name__, str(e))
                performance_monitor.increment_counter(f"{operation_name}_errors")
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Context manager for timing operations
class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self, operation_name: str, tags: Optional[Dict[str, str]] = None, call_id: Optional[str] = None):
        self.operation_name = operation_name
        self.tags = tags
        self.call_id = call_id
        self.timer_key = None

    def __enter__(self):
        self.timer_key = performance_monitor.start_timer(self.operation_name, self.call_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer_key:
            tags = self.tags or {}
            if exc_type:
                tags["success"] = "false"
                tags["error_type"] = exc_type.__name__
                performance_monitor.record_error(exc_type.__name__, str(exc_val), self.call_id)
            else:
                tags["success"] = "true"

            performance_monitor.end_timer(self.timer_key, tags)
