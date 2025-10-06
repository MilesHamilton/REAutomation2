import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import traceback
import json

from langsmith import Client
from langsmith.schemas import Run
from uuid import uuid4

from ..config.settings import settings
from ..database.monitoring_models import (
    WorkflowTrace,
    AgentExecution,
    create_trace_id,
    create_execution_id
)

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Simple circuit breaker for LangSmith API calls"""

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        elif self.state == "HALF_OPEN":
            return True
        return False

    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class LangSmithClient:
    """
    LangSmith client with circuit breaker, batching, and fallback mechanisms
    """

    def __init__(self):
        self.client: Optional[Client] = None
        self.enabled = settings.langsmith_enabled and settings.langsmith_api_key
        self.fallback_enabled = settings.langsmith_fallback_enabled
        self.batch_size = settings.langsmith_batch_size
        self.flush_interval = settings.langsmith_flush_interval

        # Circuit breaker for API calls
        self.circuit_breaker = CircuitBreaker()

        # Batch processing
        self.pending_traces: List[Dict[str, Any]] = []
        self.pending_executions: List[Dict[str, Any]] = []
        self.last_flush_time = time.time()

        # Local logging fallback
        self.fallback_logger = logging.getLogger("langsmith_fallback")

        # Initialize client if enabled
        if self.enabled:
            try:
                self.client = Client(
                    api_key=settings.langsmith_api_key,
                    api_url=settings.langsmith_endpoint
                )
                logger.info("LangSmith client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LangSmith client: {e}")
                if not self.fallback_enabled:
                    self.enabled = False

    async def create_workflow_trace(
        self,
        call_id: str,
        workflow_name: str,
        initial_context: Optional[Dict[str, Any]] = None,
        workflow_version: Optional[str] = None,
        parent_trace_id: Optional[str] = None
    ) -> str:
        """Create a new workflow trace"""
        trace_id = create_trace_id()

        trace_data = {
            "trace_id": trace_id,
            "call_id": call_id,
            "workflow_name": workflow_name,
            "workflow_version": workflow_version,
            "parent_trace_id": parent_trace_id,
            "status": "running",
            "start_time": datetime.utcnow(),
            "initial_context": initial_context or {},
            "total_agents_executed": 0,
            "total_llm_calls": 0,
            "total_cost": 0.0
        }

        if self.enabled and self.circuit_breaker.can_execute():
            try:
                # Create LangSmith run using new API
                run_id = uuid4()
                self.client.create_run(
                    id=run_id,
                    name=f"workflow_{workflow_name}",
                    run_type="chain",
                    inputs=initial_context or {},
                    project_name=settings.langsmith_project,
                    tags=[workflow_name, "workflow"],
                    extra={
                        "call_id": call_id,
                        "trace_id": trace_id,
                        "workflow_version": workflow_version
                    },
                    start_time=datetime.utcnow()
                )
                
                trace_data["langsmith_run_id"] = str(run_id)
                trace_data["langsmith_project"] = settings.langsmith_project
                trace_data["langsmith_url"] = f"{settings.langsmith_endpoint}/runs/{run_id}"

                self.circuit_breaker.record_success()
                logger.debug(f"Created LangSmith run for trace {trace_id}")

            except Exception as e:
                logger.error(f"Failed to create LangSmith run: {e}")
                self.circuit_breaker.record_failure()
                if not self.fallback_enabled:
                    raise

        # Add to batch for database storage
        self.pending_traces.append(trace_data)
        await self._check_flush()

        return trace_id

    async def update_workflow_trace(
        self,
        trace_id: str,
        status: str = None,
        final_context: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        total_cost: Optional[float] = None
    ):
        """Update an existing workflow trace"""
        update_data = {
            "trace_id": trace_id,
            "update_time": datetime.utcnow(),
        }

        if status:
            update_data["status"] = status
            if status in ["completed", "failed", "cancelled"]:
                update_data["end_time"] = datetime.utcnow()

        if final_context:
            update_data["final_context"] = final_context

        if error_message:
            update_data["error_occurred"] = True
            update_data["error_message"] = error_message
            update_data["error_type"] = error_type

        if total_cost is not None:
            update_data["total_cost"] = total_cost

        if self.enabled and self.circuit_breaker.can_execute():
            try:
                # Find the trace with LangSmith run ID
                trace = None
                for pending_trace in self.pending_traces:
                    if pending_trace["trace_id"] == trace_id:
                        trace = pending_trace
                        break

                if trace and trace.get("langsmith_run_id"):
                    # Update LangSmith run
                    run_update = {
                        "end_time": update_data.get("end_time"),
                        "outputs": final_context or {},
                        "error": error_message if error_message else None,
                        "extra": {
                            "status": status,
                            "total_cost": total_cost,
                            "error_type": error_type
                        }
                    }

                    self.client.update_run(trace["langsmith_run_id"], **run_update)
                    self.circuit_breaker.record_success()
                    logger.debug(f"Updated LangSmith run for trace {trace_id}")

            except Exception as e:
                logger.error(f"Failed to update LangSmith run: {e}")
                self.circuit_breaker.record_failure()
                if not self.fallback_enabled:
                    raise

        # Update pending traces
        for trace in self.pending_traces:
            if trace["trace_id"] == trace_id:
                trace.update(update_data)
                break

        await self._check_flush()

    async def create_agent_execution(
        self,
        trace_id: str,
        agent_type: str,
        agent_name: str,
        execution_order: int,
        input_data: Optional[Dict[str, Any]] = None,
        agent_state_before: Optional[Dict[str, Any]] = None,
        agent_version: Optional[str] = None
    ) -> str:
        """Create a new agent execution record"""
        execution_id = create_execution_id()

        execution_data = {
            "execution_id": execution_id,
            "trace_id": trace_id,
            "agent_type": agent_type,
            "agent_name": agent_name,
            "agent_version": agent_version,
            "execution_order": execution_order,
            "start_time": datetime.utcnow(),
            "input_data": input_data or {},
            "agent_state_before": agent_state_before or {},
            "llm_calls_made": 0,
            "tokens_consumed": 0,
            "processing_cost": 0.0,
            "retry_count": 0
        }

        if self.enabled and self.circuit_breaker.can_execute():
            try:
                # Create child run in LangSmith
                parent_trace = None
                for trace in self.pending_traces:
                    if trace["trace_id"] == trace_id:
                        parent_trace = trace
                        break

                if parent_trace and parent_trace.get("langsmith_run_id"):
                    # Create child run using new API
                    run_id = uuid4()
                    self.client.create_run(
                        id=run_id,
                        name=f"agent_{agent_type}_{agent_name}",
                        run_type="tool",
                        inputs=input_data or {},
                        project_name=settings.langsmith_project,
                        parent_run_id=parent_trace["langsmith_run_id"],
                        tags=[agent_type, agent_name, "agent_execution"],
                        extra={
                            "execution_id": execution_id,
                            "agent_version": agent_version,
                            "execution_order": execution_order
                        },
                        start_time=datetime.utcnow()
                    )

                    execution_data["langsmith_run_id"] = str(run_id)

                    self.circuit_breaker.record_success()
                    logger.debug(f"Created LangSmith run for execution {execution_id}")

            except Exception as e:
                logger.error(f"Failed to create LangSmith execution run: {e}")
                self.circuit_breaker.record_failure()
                if not self.fallback_enabled:
                    raise

        # Add to batch
        self.pending_executions.append(execution_data)
        await self._check_flush()

        return execution_id

    async def update_agent_execution(
        self,
        execution_id: str,
        output_data: Optional[Dict[str, Any]] = None,
        agent_state_after: Optional[Dict[str, Any]] = None,
        decision_rationale: Optional[str] = None,
        confidence_score: Optional[float] = None,
        routing_decision: Optional[str] = None,
        llm_calls_made: Optional[int] = None,
        tokens_consumed: Optional[int] = None,
        processing_cost: Optional[float] = None,
        error_message: Optional[str] = None,
        retry_count: Optional[int] = None
    ):
        """Update an existing agent execution"""
        update_data = {
            "execution_id": execution_id,
            "end_time": datetime.utcnow(),
        }

        if output_data:
            update_data["output_data"] = output_data
        if agent_state_after:
            update_data["agent_state_after"] = agent_state_after
        if decision_rationale:
            update_data["decision_rationale"] = decision_rationale
        if confidence_score is not None:
            update_data["confidence_score"] = confidence_score
        if routing_decision:
            update_data["routing_decision"] = routing_decision
        if llm_calls_made is not None:
            update_data["llm_calls_made"] = llm_calls_made
        if tokens_consumed is not None:
            update_data["tokens_consumed"] = tokens_consumed
        if processing_cost is not None:
            update_data["processing_cost"] = processing_cost
        if error_message:
            update_data["error_occurred"] = True
            update_data["error_message"] = error_message
        if retry_count is not None:
            update_data["retry_count"] = retry_count

        # Calculate duration
        for execution in self.pending_executions:
            if execution["execution_id"] == execution_id:
                start_time = execution["start_time"]
                duration = (update_data["end_time"] - start_time).total_seconds() * 1000
                update_data["duration_ms"] = duration
                break

        if self.enabled and self.circuit_breaker.can_execute():
            try:
                # Find execution with LangSmith run ID
                execution = None
                for pending_execution in self.pending_executions:
                    if pending_execution["execution_id"] == execution_id:
                        execution = pending_execution
                        break

                if execution and execution.get("langsmith_run_id"):
                    # Update LangSmith run
                    run_update = {
                        "end_time": update_data["end_time"],
                        "outputs": output_data or {},
                        "error": error_message if error_message else None,
                        "extra": {
                            "confidence_score": confidence_score,
                            "routing_decision": routing_decision,
                            "llm_calls_made": llm_calls_made,
                            "tokens_consumed": tokens_consumed,
                            "processing_cost": processing_cost,
                            "retry_count": retry_count
                        }
                    }

                    self.client.update_run(execution["langsmith_run_id"], **run_update)
                    self.circuit_breaker.record_success()
                    logger.debug(f"Updated LangSmith run for execution {execution_id}")

            except Exception as e:
                logger.error(f"Failed to update LangSmith execution run: {e}")
                self.circuit_breaker.record_failure()
                if not self.fallback_enabled:
                    raise

        # Update pending executions
        for execution in self.pending_executions:
            if execution["execution_id"] == execution_id:
                execution.update(update_data)
                break

        await self._check_flush()

    @asynccontextmanager
    async def trace_workflow(
        self,
        call_id: str,
        workflow_name: str,
        initial_context: Optional[Dict[str, Any]] = None,
        workflow_version: Optional[str] = None,
        parent_trace_id: Optional[str] = None
    ):
        """Context manager for workflow tracing"""
        trace_id = await self.create_workflow_trace(
            call_id=call_id,
            workflow_name=workflow_name,
            initial_context=initial_context,
            workflow_version=workflow_version,
            parent_trace_id=parent_trace_id
        )

        try:
            yield trace_id
            await self.update_workflow_trace(trace_id, status="completed")
        except Exception as e:
            await self.update_workflow_trace(
                trace_id,
                status="failed",
                error_message=str(e),
                error_type=type(e).__name__
            )
            raise

    @asynccontextmanager
    async def trace_agent_execution(
        self,
        trace_id: str,
        agent_type: str,
        agent_name: str,
        execution_order: int,
        input_data: Optional[Dict[str, Any]] = None,
        agent_state_before: Optional[Dict[str, Any]] = None,
        agent_version: Optional[str] = None
    ):
        """Context manager for agent execution tracing"""
        execution_id = await self.create_agent_execution(
            trace_id=trace_id,
            agent_type=agent_type,
            agent_name=agent_name,
            execution_order=execution_order,
            input_data=input_data,
            agent_state_before=agent_state_before,
            agent_version=agent_version
        )

        try:
            yield execution_id
        except Exception as e:
            await self.update_agent_execution(
                execution_id,
                error_message=str(e),
                retry_count=0
            )
            raise

    async def _check_flush(self):
        """Check if batch should be flushed"""
        current_time = time.time()
        should_flush = (
            len(self.pending_traces) >= self.batch_size or
            len(self.pending_executions) >= self.batch_size or
            current_time - self.last_flush_time >= self.flush_interval
        )

        if should_flush:
            await self.flush_batch()

    async def flush_batch(self):
        """Flush pending traces and executions to database/logs"""
        if not self.pending_traces and not self.pending_executions:
            return

        try:
            # In a real implementation, this would save to database
            # For now, we'll log the data if fallback is enabled
            if self.fallback_enabled:
                if self.pending_traces:
                    for trace in self.pending_traces:
                        self.fallback_logger.info(f"WORKFLOW_TRACE: {json.dumps(trace, default=str)}")

                if self.pending_executions:
                    for execution in self.pending_executions:
                        self.fallback_logger.info(f"AGENT_EXECUTION: {json.dumps(execution, default=str)}")

            logger.debug(f"Flushed {len(self.pending_traces)} traces and {len(self.pending_executions)} executions")

            # Clear batches
            self.pending_traces.clear()
            self.pending_executions.clear()
            self.last_flush_time = time.time()

        except Exception as e:
            logger.error(f"Failed to flush monitoring batch: {e}")
            if not self.fallback_enabled:
                raise

    async def shutdown(self):
        """Shutdown client and flush remaining data"""
        try:
            await self.flush_batch()
            logger.info("LangSmith client shutdown completed")
        except Exception as e:
            logger.error(f"Error during LangSmith client shutdown: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the client"""
        return {
            "enabled": self.enabled,
            "fallback_enabled": self.fallback_enabled,
            "circuit_breaker_state": self.circuit_breaker.state,
            "circuit_breaker_failures": self.circuit_breaker.failure_count,
            "pending_traces": len(self.pending_traces),
            "pending_executions": len(self.pending_executions),
            "last_flush_age_seconds": time.time() - self.last_flush_time
        }


# Global client instance
_langsmith_client: Optional[LangSmithClient] = None


def get_langsmith_client() -> LangSmithClient:
    """Get the global LangSmith client instance"""
    global _langsmith_client
    if _langsmith_client is None:
        _langsmith_client = LangSmithClient()
    return _langsmith_client


async def shutdown_langsmith_client():
    """Shutdown the global LangSmith client"""
    global _langsmith_client
    if _langsmith_client:
        await _langsmith_client.shutdown()
        _langsmith_client = None
