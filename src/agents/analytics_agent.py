import logging
from typing import Dict, Any

from .base_agent import BaseAgent
from .models import AgentType, AgentResponse, WorkflowContext, AnalyticsMetrics

logger = logging.getLogger(__name__)


class AnalyticsAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.ANALYTICS)

    async def process(self, context: WorkflowContext, user_input: str) -> AgentResponse:
        """Analyze conversation performance and generate insights"""
        try:
            # Generate analytics
            metrics = await self._generate_analytics(context)

            # This agent doesn't respond directly to users
            return AgentResponse(
                agent_type=self.agent_type,
                response_text=None,  # Analytics agent doesn't speak
                state_updates={"metadata.analytics": metrics.dict()}
            )

        except Exception as e:
            logger.error(f"Analytics agent error: {e}")
            return AgentResponse(agent_type=self.agent_type)

    async def _generate_analytics(self, context: WorkflowContext) -> AnalyticsMetrics:
        """Generate comprehensive analytics metrics"""
        import time
        duration = time.time() - context.created_at

        return AnalyticsMetrics(
            call_id=context.call_id,
            workflow_duration_ms=duration * 1000,
            qualification_progression=[context.qualification_score],
            tier_escalation_triggered=context.tier_escalated,
            outcome=self._determine_outcome(context)
        )

    def _determine_outcome(self, context: WorkflowContext) -> str:
        """Determine call outcome"""
        if context.qualification_score >= 0.8:
            return "qualified"
        elif context.qualification_score >= 0.5:
            return "potential"
        else:
            return "disqualified"

    def get_system_prompt(self, context: WorkflowContext) -> str:
        return "Analytics agent - processes conversation data in background."

    def can_handle(self, context: WorkflowContext) -> bool:
        return True  # Analytics runs in background