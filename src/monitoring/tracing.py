import asyncio
import functools
import logging
import time
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime

from ..agents.models import AgentType, WorkflowContext, AgentResponse
from .langsmith_client import get_langsmith_client
from .models import WorkflowStatus

logger = logging.getLogger(__name__)


class TracingContext:
    """Context for tracking tracing information"""

    def __init__(self):
        self.trace_id: Optional[str] = None
        self.execution_counter = 0
        self.workflow_start_time: Optional[float] = None
        self.total_cost = 0.0
        self.total_llm_calls = 0
        self.agent_executions: Dict[str, Dict[str, Any]] = {}


def trace_workflow(
    workflow_name: str = "agent_orchestration",
    workflow_version: Optional[str] = None
):
    """
    Decorator to trace entire workflow execution with LangSmith
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(self, call_id: str, *args, **kwargs):
            langsmith_client = get_langsmith_client()

            if not langsmith_client.enabled:
                # If monitoring is disabled, just execute normally
                return await func(self, call_id, *args, **kwargs)

            # Extract initial context for tracing
            initial_context = {
                "call_id": call_id,
                "function": func.__name__,
                "args": str(args)[:200],  # Truncate for storage
                "kwargs": {k: str(v)[:200] if not k.startswith('_') else '...' for k, v in kwargs.items()}
            }

            try:
                # Start workflow trace
                async with langsmith_client.trace_workflow(
                    call_id=call_id,
                    workflow_name=workflow_name,
                    initial_context=initial_context,
                    workflow_version=workflow_version
                ) as trace_id:

                    # Store trace context for agent executions
                    if not hasattr(self, '_tracing_context'):
                        self._tracing_context = {}

                    self._tracing_context[call_id] = TracingContext()
                    self._tracing_context[call_id].trace_id = trace_id
                    self._tracing_context[call_id].workflow_start_time = time.time()

                    # Execute the wrapped function
                    result = await func(self, call_id, *args, **kwargs)

                    # Update final context with results
                    final_context = {
                        "result_type": type(result).__name__ if result else "None",
                        "execution_time_ms": (time.time() - self._tracing_context[call_id].workflow_start_time) * 1000,
                        "total_cost": self._tracing_context[call_id].total_cost,
                        "total_llm_calls": self._tracing_context[call_id].total_llm_calls,
                        "agent_executions_count": self._tracing_context[call_id].execution_counter
                    }

                    if result and hasattr(result, 'workflow_state'):
                        final_context["final_workflow_state"] = result.workflow_state

                    # Update trace with final context and cost
                    await langsmith_client.update_workflow_trace(
                        trace_id,
                        status=WorkflowStatus.COMPLETED,
                        final_context=final_context,
                        total_cost=self._tracing_context[call_id].total_cost
                    )

                    # Clean up tracing context
                    if call_id in self._tracing_context:
                        del self._tracing_context[call_id]

                    return result

            except Exception as e:
                # Update trace with error information
                if hasattr(self, '_tracing_context') and call_id in self._tracing_context:
                    trace_id = self._tracing_context[call_id].trace_id

                    await langsmith_client.update_workflow_trace(
                        trace_id,
                        status=WorkflowStatus.FAILED,
                        error_message=str(e),
                        error_type=type(e).__name__
                    )

                    del self._tracing_context[call_id]

                # Re-raise the exception
                raise

        return wrapper
    return decorator


def trace_agent_execution(
    agent_type: Optional[Union[AgentType, str]] = None,
    agent_version: Optional[str] = None
):
    """
    Decorator to trace individual agent executions
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(self, state: WorkflowContext, *args, **kwargs):
            langsmith_client = get_langsmith_client()

            # Get agent type from parameter or try to infer
            if agent_type:
                current_agent_type = agent_type if isinstance(agent_type, str) else agent_type.value
            elif hasattr(self, 'agent_type'):
                current_agent_type = self.agent_type.value if hasattr(self.agent_type, 'value') else str(self.agent_type)
            else:
                current_agent_type = func.__name__.replace('_node', '').replace('_', '')

            agent_name = f"{current_agent_type}_{func.__name__}"

            # Check if we have a workflow trace to attach to
            call_id = state.call_id
            orchestrator = None
            trace_id = None
            execution_order = 0

            # Try to get tracing context from orchestrator
            if hasattr(self, '_tracing_context') and call_id in self._tracing_context:
                # This is the orchestrator
                orchestrator = self
                trace_id = self._tracing_context[call_id].trace_id
                execution_order = self._tracing_context[call_id].execution_counter
                self._tracing_context[call_id].execution_counter += 1
            elif hasattr(self, 'orchestrator') and hasattr(self.orchestrator, '_tracing_context'):
                # This is an agent with reference to orchestrator
                orchestrator = self.orchestrator
                if call_id in orchestrator._tracing_context:
                    trace_id = orchestrator._tracing_context[call_id].trace_id
                    execution_order = orchestrator._tracing_context[call_id].execution_counter
                    orchestrator._tracing_context[call_id].execution_counter += 1

            if not langsmith_client.enabled or not trace_id:
                # If monitoring is disabled or no trace context, execute normally
                return await func(self, state, *args, **kwargs)

            # Prepare input data for tracing
            input_data = {
                "workflow_state": state.workflow_state.value if hasattr(state.workflow_state, 'value') else str(state.workflow_state),
                "qualification_score": state.qualification_score,
                "conversation_history_length": len(state.conversation_history),
                "current_input": args[0] if args else kwargs.get('user_input', ''),
                "function": func.__name__
            }

            # Capture agent state before execution
            agent_state_before = {
                "workflow_state": input_data["workflow_state"],
                "qualification_score": state.qualification_score,
                "current_agent": str(state.current_agent) if state.current_agent else None,
                "metadata_keys": list(state.metadata.keys()) if state.metadata else []
            }

            try:
                # Start agent execution trace
                async with langsmith_client.trace_agent_execution(
                    trace_id=trace_id,
                    agent_type=current_agent_type,
                    agent_name=agent_name,
                    execution_order=execution_order,
                    input_data=input_data,
                    agent_state_before=agent_state_before,
                    agent_version=agent_version
                ) as execution_id:

                    start_time = time.time()

                    # Execute the wrapped function
                    result = await func(self, state, *args, **kwargs)

                    execution_time = time.time() - start_time

                    # Capture output data and state changes
                    output_data = {}
                    agent_state_after = {}

                    if isinstance(result, AgentResponse):
                        output_data = {
                            "response_text": result.response_text[:500] if result.response_text else None,  # Truncate
                            "agent_type": result.agent_type.value if hasattr(result.agent_type, 'value') else str(result.agent_type),
                            "requires_response": result.requires_response,
                            "should_escalate_tier": result.should_escalate_tier,
                            "confidence_score": result.confidence_score,
                            "cost": result.cost,
                            "processing_time_ms": result.processing_time_ms
                        }

                        # Extract decision rationale
                        decision_rationale = None
                        if hasattr(result, 'decision_rationale'):
                            decision_rationale = result.decision_rationale
                        elif result.state_updates and 'decision_rationale' in result.state_updates:
                            decision_rationale = result.state_updates['decision_rationale']

                        # Update cost and call tracking
                        if orchestrator and call_id in orchestrator._tracing_context:
                            if result.cost:
                                orchestrator._tracing_context[call_id].total_cost += result.cost
                            orchestrator._tracing_context[call_id].total_llm_calls += 1

                    elif isinstance(result, WorkflowContext):
                        # If returning the state directly
                        output_data = {
                            "workflow_state": result.workflow_state.value if hasattr(result.workflow_state, 'value') else str(result.workflow_state),
                            "qualification_score": result.qualification_score,
                            "current_agent": str(result.current_agent) if result.current_agent else None
                        }

                    # Capture agent state after execution
                    if isinstance(result, (AgentResponse, WorkflowContext)):
                        agent_state_after = {
                            "workflow_state": output_data.get("workflow_state", "unknown"),
                            "qualification_score": output_data.get("qualification_score", 0.0),
                            "execution_time_ms": execution_time * 1000
                        }

                    # Update the execution trace
                    await langsmith_client.update_agent_execution(
                        execution_id,
                        output_data=output_data,
                        agent_state_after=agent_state_after,
                        decision_rationale=decision_rationale,
                        confidence_score=output_data.get("confidence_score"),
                        routing_decision=output_data.get("workflow_state"),
                        llm_calls_made=1 if isinstance(result, AgentResponse) else 0,
                        tokens_consumed=getattr(result, 'tokens_used', 0) if hasattr(result, 'tokens_used') else 0,
                        processing_cost=output_data.get("cost", 0.0)
                    )

                    return result

            except Exception as e:
                # Update execution with error
                if 'execution_id' in locals():
                    await langsmith_client.update_agent_execution(
                        execution_id,
                        error_message=str(e),
                        retry_count=0
                    )

                raise

        return wrapper
    return decorator


def enable_orchestrator_tracing(orchestrator_class):
    """
    Class decorator to enable tracing for the AgentOrchestrator
    """
    # Wrap the process_input method with workflow tracing
    original_process_input = orchestrator_class.process_input
    orchestrator_class.process_input = trace_workflow("agent_orchestration", "1.0")(original_process_input)

    # Wrap all agent node methods with execution tracing
    node_methods = [
        "_conversation_node",
        "_qualification_node",
        "_objection_handler_node",
        "_scheduler_node",
        "_analytics_node"
    ]

    for method_name in node_methods:
        if hasattr(orchestrator_class, method_name):
            original_method = getattr(orchestrator_class, method_name)
            agent_type = method_name.replace("_node", "").replace("_", "")
            traced_method = trace_agent_execution(agent_type)(original_method)
            setattr(orchestrator_class, method_name, traced_method)

    return orchestrator_class


# Utility functions for manual tracing
async def trace_manual_workflow(
    call_id: str,
    workflow_name: str,
    workflow_function: Callable,
    initial_context: Optional[Dict[str, Any]] = None,
    workflow_version: Optional[str] = None
):
    """
    Manually trace a workflow function
    """
    langsmith_client = get_langsmith_client()

    if not langsmith_client.enabled:
        return await workflow_function()

    async with langsmith_client.trace_workflow(
        call_id=call_id,
        workflow_name=workflow_name,
        initial_context=initial_context,
        workflow_version=workflow_version
    ) as trace_id:
        try:
            result = await workflow_function()

            final_context = {
                "result_type": type(result).__name__ if result else "None",
                "success": True
            }

            await langsmith_client.update_workflow_trace(
                trace_id,
                status=WorkflowStatus.COMPLETED,
                final_context=final_context
            )

            return result

        except Exception as e:
            await langsmith_client.update_workflow_trace(
                trace_id,
                status=WorkflowStatus.FAILED,
                error_message=str(e),
                error_type=type(e).__name__
            )
            raise


async def trace_manual_agent_execution(
    trace_id: str,
    agent_type: str,
    agent_name: str,
    execution_order: int,
    agent_function: Callable,
    input_data: Optional[Dict[str, Any]] = None,
    agent_version: Optional[str] = None
):
    """
    Manually trace an agent execution
    """
    langsmith_client = get_langsmith_client()

    if not langsmith_client.enabled:
        return await agent_function()

    async with langsmith_client.trace_agent_execution(
        trace_id=trace_id,
        agent_type=agent_type,
        agent_name=agent_name,
        execution_order=execution_order,
        input_data=input_data,
        agent_version=agent_version
    ) as execution_id:
        try:
            result = await agent_function()

            output_data = {
                "result_type": type(result).__name__ if result else "None",
                "success": True
            }

            await langsmith_client.update_agent_execution(
                execution_id,
                output_data=output_data
            )

            return result

        except Exception as e:
            await langsmith_client.update_agent_execution(
                execution_id,
                error_message=str(e)
            )
            raise


# Context managers for manual tracing
class WorkflowTraceContext:
    """Context manager for manual workflow tracing"""

    def __init__(
        self,
        call_id: str,
        workflow_name: str,
        initial_context: Optional[Dict[str, Any]] = None,
        workflow_version: Optional[str] = None
    ):
        self.call_id = call_id
        self.workflow_name = workflow_name
        self.initial_context = initial_context
        self.workflow_version = workflow_version
        self.langsmith_client = get_langsmith_client()
        self.trace_id: Optional[str] = None

    async def __aenter__(self):
        if self.langsmith_client.enabled:
            self.trace_id = await self.langsmith_client.create_workflow_trace(
                self.call_id,
                self.workflow_name,
                self.initial_context,
                self.workflow_version
            )
        return self.trace_id

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.trace_id:
            if exc_type:
                await self.langsmith_client.update_workflow_trace(
                    self.trace_id,
                    status=WorkflowStatus.FAILED,
                    error_message=str(exc_val),
                    error_type=exc_type.__name__
                )
            else:
                await self.langsmith_client.update_workflow_trace(
                    self.trace_id,
                    status=WorkflowStatus.COMPLETED
                )


class AgentExecutionTraceContext:
    """Context manager for manual agent execution tracing"""

    def __init__(
        self,
        trace_id: str,
        agent_type: str,
        agent_name: str,
        execution_order: int,
        input_data: Optional[Dict[str, Any]] = None,
        agent_version: Optional[str] = None
    ):
        self.trace_id = trace_id
        self.agent_type = agent_type
        self.agent_name = agent_name
        self.execution_order = execution_order
        self.input_data = input_data
        self.agent_version = agent_version
        self.langsmith_client = get_langsmith_client()
        self.execution_id: Optional[str] = None

    async def __aenter__(self):
        if self.langsmith_client.enabled and self.trace_id:
            self.execution_id = await self.langsmith_client.create_agent_execution(
                self.trace_id,
                self.agent_type,
                self.agent_name,
                self.execution_order,
                self.input_data,
                agent_version=self.agent_version
            )
        return self.execution_id

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.execution_id:
            if exc_type:
                await self.langsmith_client.update_agent_execution(
                    self.execution_id,
                    error_message=str(exc_val)
                )
            else:
                await self.langsmith_client.update_agent_execution(
                    self.execution_id,
                    output_data={"success": True}
                )