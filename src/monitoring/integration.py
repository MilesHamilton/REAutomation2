"""
Integration module to enable LangSmith monitoring on the AgentOrchestrator
"""
import logging
from typing import Dict, Any, Optional
from functools import wraps

from ..agents.orchestrator import AgentOrchestrator, agent_orchestrator
from .tracing import trace_workflow, trace_agent_execution, get_langsmith_client
from .langsmith_client import shutdown_langsmith_client

logger = logging.getLogger(__name__)


def enable_monitoring():
    """
    Enable LangSmith monitoring for the global AgentOrchestrator instance
    """
    try:
        # Apply tracing to the process_input method
        if not hasattr(agent_orchestrator.process_input, '_monitoring_enabled'):
            original_process_input = agent_orchestrator.process_input

            @trace_workflow("agent_orchestration", "1.0")
            @wraps(original_process_input)
            async def traced_process_input(call_id: str, user_input: str, lead_data: Optional[Dict[str, Any]] = None):
                return await original_process_input(call_id, user_input, lead_data)

            traced_process_input._monitoring_enabled = True
            agent_orchestrator.process_input = traced_process_input.__get__(agent_orchestrator, AgentOrchestrator)

        # Apply tracing to agent node methods
        node_methods = [
            ("_conversation_node", "conversation"),
            ("_qualification_node", "qualification"),
            ("_objection_handler_node", "objection_handler"),
            ("_scheduler_node", "scheduler"),
            ("_analytics_node", "analytics")
        ]

        for method_name, agent_type in node_methods:
            if hasattr(agent_orchestrator, method_name):
                method = getattr(agent_orchestrator, method_name)

                if not hasattr(method, '_monitoring_enabled'):
                    @trace_agent_execution(agent_type, "1.0")
                    @wraps(method)
                    async def traced_method(state, original_method=method):
                        return await original_method(state)

                    traced_method._monitoring_enabled = True
                    setattr(agent_orchestrator, method_name, traced_method.__get__(agent_orchestrator, AgentOrchestrator))

        # Add monitoring context to orchestrator
        if not hasattr(agent_orchestrator, '_tracing_context'):
            agent_orchestrator._tracing_context = {}

        # Add health check for monitoring
        original_health_check = agent_orchestrator.health_check

        @wraps(original_health_check)
        async def enhanced_health_check():
            base_health = await original_health_check()
            langsmith_client = get_langsmith_client()

            base_health["monitoring"] = {
                "langsmith_enabled": langsmith_client.enabled,
                "langsmith_status": langsmith_client.get_health_status(),
                "tracing_contexts": len(getattr(agent_orchestrator, '_tracing_context', {}))
            }

            return base_health

        agent_orchestrator.health_check = enhanced_health_check.__get__(agent_orchestrator, AgentOrchestrator)

        # Add cleanup enhancement
        original_cleanup = agent_orchestrator.cleanup

        @wraps(original_cleanup)
        async def enhanced_cleanup():
            try:
                # Shutdown LangSmith client
                await shutdown_langsmith_client()
            except Exception as e:
                logger.error(f"Error shutting down monitoring: {e}")

            # Call original cleanup
            await original_cleanup()

        agent_orchestrator.cleanup = enhanced_cleanup.__get__(agent_orchestrator, AgentOrchestrator)

        logger.info("LangSmith monitoring enabled successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to enable monitoring: {e}")
        return False


def disable_monitoring():
    """
    Disable LangSmith monitoring (for testing or troubleshooting)
    """
    try:
        # Note: This is a simplified disable - in production you'd want to restore original methods
        langsmith_client = get_langsmith_client()
        langsmith_client.enabled = False
        logger.info("LangSmith monitoring disabled")
        return True
    except Exception as e:
        logger.error(f"Failed to disable monitoring: {e}")
        return False


def get_monitoring_status() -> Dict[str, Any]:
    """
    Get the current monitoring status
    """
    try:
        langsmith_client = get_langsmith_client()
        return {
            "enabled": langsmith_client.enabled,
            "health": langsmith_client.get_health_status(),
            "orchestrator_instrumented": hasattr(agent_orchestrator.process_input, '_monitoring_enabled'),
            "active_traces": len(getattr(agent_orchestrator, '_tracing_context', {}))
        }
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        return {"error": str(e)}


# Auto-enable monitoring on import if configured
def _auto_enable_monitoring():
    """Auto-enable monitoring if settings allow"""
    try:
        from ..config.settings import settings
        if settings.langsmith_enabled:
            enable_monitoring()
    except Exception as e:
        logger.debug(f"Auto-enable monitoring failed: {e}")


# Call auto-enable
_auto_enable_monitoring()