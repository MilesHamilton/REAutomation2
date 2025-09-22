import logging
from typing import Dict, Any

from .base_agent import BaseAgent
from .models import AgentType, AgentResponse, WorkflowContext, WorkflowState, ObjectionType, Objection

logger = logging.getLogger(__name__)


class ObjectionHandlerAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.OBJECTION_HANDLER)
        self.objection_responses = {
            ObjectionType.PRICE: [
                "I understand cost is important. Let me ask - what's it costing you to not solve this problem?",
                "That's a fair concern. Many of our clients said the same thing until they saw the ROI.",
            ],
            ObjectionType.TIME: [
                "I completely understand you're busy. That's exactly why this solution could help - it saves time.",
                "I appreciate that. How much time are you currently spending on [current process]?",
            ],
            ObjectionType.NEED: [
                "I hear that. Help me understand - what would have to happen for this to become important?",
                "That's good feedback. What if I told you that [specific benefit relevant to their situation]?",
            ]
        }

    async def process(self, context: WorkflowContext, user_input: str) -> AgentResponse:
        """Handle objections and convert them to opportunities"""
        try:
            # Identify the objection type and severity
            objection = await self._identify_objection(context, user_input)

            # Get appropriate response
            response_text = await self._generate_objection_response(context, objection)

            # Determine if objection was successfully handled
            next_state = await self._determine_post_objection_state(context, objection)

            return AgentResponse(
                agent_type=self.agent_type,
                response_text=response_text,
                state_updates={
                    "workflow_state": next_state,
                    "objection_count": context.objection_count + 1
                }
            )

        except Exception as e:
            logger.error(f"Objection handler error: {e}")
            return AgentResponse(
                agent_type=self.agent_type,
                response_text="I understand your concern. Let me address that..."
            )

    async def _identify_objection(self, context: WorkflowContext, user_input: str) -> Objection:
        """Identify objection type and severity"""
        schema = {
            "type": "object",
            "properties": {
                "objection_type": {"type": "string", "enum": ["price", "time", "need", "trust", "competition", "authority", "other"]},
                "severity": {"type": "integer", "minimum": 1, "maximum": 5},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "key_phrases": {"type": "array", "items": {"type": "string"}}
            }
        }

        result = await self._generate_structured_response(context, user_input, schema) or {}

        return Objection(
            type=ObjectionType(result.get("objection_type", "other")),
            text=user_input,
            confidence=result.get("confidence", 0.7),
            severity=result.get("severity", 3)
        )

    async def _generate_objection_response(self, context: WorkflowContext, objection: Objection) -> str:
        """Generate appropriate objection response"""
        responses = self.objection_responses.get(objection.type, ["I understand your concern."])
        base_response = responses[context.objection_count % len(responses)]

        return await self._generate_llm_response(context, f"Handle this objection: {objection.text}")

    async def _determine_post_objection_state(self, context: WorkflowContext, objection: Objection) -> WorkflowState:
        """Determine workflow state after handling objection"""
        if objection.severity <= 2:  # Soft objection
            return WorkflowState.QUALIFYING
        elif objection.severity >= 4:  # Hard objection
            return WorkflowState.CLOSING
        else:
            return WorkflowState.QUALIFYING

    def get_system_prompt(self, context: WorkflowContext) -> str:
        return f"""
        You are an expert objection handler. Your role is to:
        1. Acknowledge the objection respectfully
        2. Reframe it as an opportunity
        3. Ask questions to understand the real concern
        4. Provide value-based responses

        Current context: {context.workflow_state}
        Objections handled: {context.objection_count}
        """

    def can_handle(self, context: WorkflowContext) -> bool:
        return context.workflow_state == WorkflowState.HANDLING_OBJECTION