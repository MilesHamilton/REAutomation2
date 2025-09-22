import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from ..llm import llm_service, ConversationContext, Message, MessageRole
from .models import AgentType, AgentState, AgentResponse, WorkflowContext, AgentDecision

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.state = AgentState.IDLE
        self._processing_lock = asyncio.Lock()

    @abstractmethod
    async def process(self, context: WorkflowContext, user_input: str) -> AgentResponse:
        """Process user input and return agent response"""
        pass

    @abstractmethod
    def get_system_prompt(self, context: WorkflowContext) -> str:
        """Get system prompt for this agent"""
        pass

    async def _generate_llm_response(
        self,
        context: WorkflowContext,
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """Generate response using LLM service"""
        try:
            if not system_prompt:
                system_prompt = self.get_system_prompt(context)

            # Create conversation context
            conversation_context = ConversationContext(
                call_id=context.call_id,
                messages=[
                    Message(role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT, content=msg.content)
                    for i, msg in enumerate(context.conversation_history[-10:])  # Last 10 messages
                ],
                lead_info=context.lead_data
            )

            response = await llm_service.generate_response(
                context=conversation_context,
                user_input=user_input,
                system_prompt=system_prompt
            )

            return response.content if response else None

        except Exception as e:
            logger.error(f"LLM generation error for {self.agent_type}: {e}")
            return None

    async def _generate_structured_response(
        self,
        context: WorkflowContext,
        user_input: str,
        response_schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate structured response using LLM service"""
        try:
            if not system_prompt:
                system_prompt = self.get_system_prompt(context)

            conversation_context = ConversationContext(
                call_id=context.call_id,
                messages=[
                    Message(role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT, content=msg.content)
                    for i, msg in enumerate(context.conversation_history[-10:])
                ],
                lead_info=context.lead_data
            )

            response = await llm_service.generate_structured_response(
                context=conversation_context,
                user_input=user_input,
                response_schema=response_schema,
                system_prompt=system_prompt
            )

            return response.structured_data if response else None

        except Exception as e:
            logger.error(f"Structured LLM generation error for {self.agent_type}: {e}")
            return None

    async def _make_decision(
        self,
        context: WorkflowContext,
        user_input: str,
        decision_options: Dict[str, Any]
    ) -> Optional[AgentDecision]:
        """Make a structured decision based on context"""
        try:
            schema = {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(decision_options.keys())},
                    "reasoning": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "parameters": {"type": "object"}
                },
                "required": ["action", "reasoning", "confidence"]
            }

            system_prompt = f"""
            You are the {self.agent_type.value} agent. Based on the conversation context and user input,
            make a decision about what action to take.

            Available actions: {list(decision_options.keys())}

            Consider:
            - Current workflow state: {context.workflow_state}
            - Qualification score: {context.qualification_score}
            - Lead data: {context.lead_data}
            - Conversation history length: {len(context.conversation_history)}

            Provide your decision in the specified JSON format.
            """

            decision_data = await self._generate_structured_response(
                context=context,
                user_input=user_input,
                response_schema=schema,
                system_prompt=system_prompt
            )

            if decision_data:
                return AgentDecision(
                    agent_type=self.agent_type,
                    action=decision_data.get("action", "continue"),
                    reasoning=decision_data.get("reasoning", ""),
                    confidence=decision_data.get("confidence", 0.5),
                    parameters=decision_data.get("parameters", {})
                )

            return None

        except Exception as e:
            logger.error(f"Decision making error for {self.agent_type}: {e}")
            return None

    async def execute_with_timeout(
        self,
        context: WorkflowContext,
        user_input: str,
        timeout_seconds: float = 10.0
    ) -> AgentResponse:
        """Execute agent processing with timeout"""
        async with self._processing_lock:
            start_time = time.time()
            self.state = AgentState.THINKING

            try:
                # Execute with timeout
                response = await asyncio.wait_for(
                    self.process(context, user_input),
                    timeout=timeout_seconds
                )

                processing_time = (time.time() - start_time) * 1000
                response.processing_time_ms = processing_time

                self.state = AgentState.COMPLETED
                logger.debug(f"{self.agent_type} processed in {processing_time:.1f}ms")

                return response

            except asyncio.TimeoutError:
                logger.warning(f"{self.agent_type} timed out after {timeout_seconds}s")
                self.state = AgentState.ERROR

                return AgentResponse(
                    agent_type=self.agent_type,
                    response_text="I need a moment to process that. Could you repeat?",
                    processing_time_ms=(time.time() - start_time) * 1000
                )

            except Exception as e:
                logger.error(f"{self.agent_type} processing error: {e}")
                self.state = AgentState.ERROR

                return AgentResponse(
                    agent_type=self.agent_type,
                    response_text="I apologize, I encountered an issue. Let me try again.",
                    processing_time_ms=(time.time() - start_time) * 1000
                )

    def can_handle(self, context: WorkflowContext) -> bool:
        """Check if this agent can handle the current context"""
        return True  # Override in specific agents

    def get_confidence_threshold(self) -> float:
        """Get minimum confidence threshold for this agent"""
        return 0.7  # Override in specific agents

    async def cleanup(self):
        """Clean up agent resources"""
        self.state = AgentState.IDLE
        logger.debug(f"{self.agent_type} agent cleaned up")