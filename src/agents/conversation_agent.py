import logging
from typing import Dict, Any

from .base_agent import BaseAgent
from .models import AgentType, AgentResponse, WorkflowContext, WorkflowState, AgentMessage

logger = logging.getLogger(__name__)


class ConversationAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.CONVERSATION)
        self.conversation_starters = [
            "Hi {name}, this is calling from {company}. How are you doing today?",
            "Hello {name}, I hope I'm not catching you at a bad time. This is regarding {topic}.",
            "Good {time_of_day} {name}, I'm reaching out because {reason}."
        ]

    async def process(self, context: WorkflowContext, user_input: str) -> AgentResponse:
        """Process conversation and maintain natural flow"""
        try:
            # Determine conversation phase and appropriate response
            if context.workflow_state == WorkflowState.INITIALIZING:
                return await self._handle_greeting(context, user_input)
            elif context.workflow_state == WorkflowState.GREETING:
                return await self._handle_introduction(context, user_input)
            else:
                return await self._handle_general_conversation(context, user_input)

        except Exception as e:
            logger.error(f"Conversation agent error: {e}")
            return AgentResponse(
                agent_type=self.agent_type,
                response_text="I apologize, could you repeat that?"
            )

    async def _handle_greeting(self, context: WorkflowContext, user_input: str) -> AgentResponse:
        """Handle initial greeting phase"""
        # Analyze user's initial response
        greeting_analysis = await self._analyze_greeting_response(context, user_input)

        if greeting_analysis.get("receptive", False):
            response_text = await self._generate_introduction(context)

            return AgentResponse(
                agent_type=self.agent_type,
                response_text=response_text,
                state_updates={"workflow_state": WorkflowState.QUALIFYING}
            )
        else:
            # Handle negative or hesitant responses
            objection_type = greeting_analysis.get("objection_type", "time")
            response_text = await self._handle_initial_objection(context, objection_type)

            return AgentResponse(
                agent_type=self.agent_type,
                response_text=response_text,
                state_updates={"workflow_state": WorkflowState.HANDLING_OBJECTION}
            )

    async def _handle_introduction(self, context: WorkflowContext, user_input: str) -> AgentResponse:
        """Handle introduction phase"""
        response_text = await self._generate_llm_response(
            context=context,
            user_input=user_input
        )

        # Check if we should transition to qualification
        should_qualify = await self._should_transition_to_qualification(context, user_input)

        next_state = WorkflowState.QUALIFYING if should_qualify else WorkflowState.GREETING

        return AgentResponse(
            agent_type=self.agent_type,
            response_text=response_text,
            state_updates={"workflow_state": next_state}
        )

    async def _handle_general_conversation(self, context: WorkflowContext, user_input: str) -> AgentResponse:
        """Handle general conversation flow"""
        response_text = await self._generate_llm_response(
            context=context,
            user_input=user_input
        )

        # Analyze if we should escalate to another agent
        decision = await self._make_conversation_decision(context, user_input)

        response = AgentResponse(
            agent_type=self.agent_type,
            response_text=response_text,
            decision=decision
        )

        # Check for tier escalation based on engagement
        if self._should_escalate_tier(context):
            response.should_escalate_tier = True

        return response

    async def _analyze_greeting_response(self, context: WorkflowContext, user_input: str) -> Dict[str, Any]:
        """Analyze user's response to initial greeting"""
        schema = {
            "type": "object",
            "properties": {
                "receptive": {"type": "boolean"},
                "objection_type": {"type": "string", "enum": ["time", "interest", "authority", "other"]},
                "engagement_level": {"type": "number", "minimum": 0, "maximum": 1},
                "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]}
            }
        }

        system_prompt = """
        Analyze the user's response to determine their receptiveness to the sales call.
        Consider tone, word choice, and explicit statements.

        Receptive indicators: "Good", "Fine", "What's this about?", questions
        Non-receptive indicators: "Busy", "Not interested", "Remove me", hanging up sounds
        """

        return await self._generate_structured_response(
            context=context,
            user_input=user_input,
            response_schema=schema,
            system_prompt=system_prompt
        ) or {"receptive": False, "objection_type": "time"}

    async def _generate_introduction(self, context: WorkflowContext) -> str:
        """Generate personalized introduction"""
        lead_name = context.lead_data.get("name", "there")
        company = context.lead_data.get("company", "your business")

        return f"""Thanks {lead_name}! I'm calling because we work with businesses like {company}
        to help them save time and money on their operations. I know you're busy, so this will just take a minute.
        Can I ask you a quick question about how you currently handle [relevant business process]?"""

    async def _handle_initial_objection(self, context: WorkflowContext, objection_type: str) -> str:
        """Handle initial objections"""
        responses = {
            "time": "I completely understand you're busy. This will just take 30 seconds. If it's not a fit, I'll let you go immediately. Fair enough?",
            "interest": "I hear that a lot, which is exactly why I'm calling. Most people don't realize they're missing out on simple ways to save money. Can I share one quick example?",
            "authority": "No problem at all. Are you involved in decisions around [business area] or should I speak with someone else?",
            "other": "I understand. Let me ask you this - if I could show you a way to save 20% on [relevant cost], would that be worth 60 seconds of your time?"
        }

        return responses.get(objection_type, responses["other"])

    async def _should_transition_to_qualification(self, context: WorkflowContext, user_input: str) -> bool:
        """Determine if conversation should transition to qualification phase"""
        # Simple heuristics - in production, this would be more sophisticated
        qualifying_indicators = ["tell me more", "how does", "what do you", "interested", "sounds good"]
        user_lower = user_input.lower()

        return any(indicator in user_lower for indicator in qualifying_indicators)

    async def _make_conversation_decision(self, context: WorkflowContext, user_input: str) -> Dict[str, Any]:
        """Make decision about conversation flow"""
        decision_options = {
            "continue_conversation": "Keep the conversation flowing naturally",
            "transition_to_qualification": "Move to qualification questions",
            "handle_objection": "Address an objection that was raised",
            "schedule_callback": "Offer to call back at a better time"
        }

        return await self._make_decision(context, user_input, decision_options)

    def _should_escalate_tier(self, context: WorkflowContext) -> bool:
        """Determine if conversation quality warrants tier escalation"""
        # Check engagement indicators
        conversation_length = len(context.conversation_history)
        qualification_score = context.qualification_score

        # Escalate if conversation is progressing well
        return (
            conversation_length >= 6 and  # At least 3 back-and-forth exchanges
            qualification_score >= 0.4 and  # Some qualification detected
            not context.tier_escalated  # Haven't escalated yet
        )

    def get_system_prompt(self, context: WorkflowContext) -> str:
        """Get system prompt for conversation agent"""
        lead_info = context.lead_data

        return f"""
        You are an expert conversationalist conducting an outbound sales call. Your role is to:

        1. Maintain natural, friendly conversation flow
        2. Build rapport and trust
        3. Listen actively and respond appropriately
        4. Guide the conversation toward qualification
        5. Handle interruptions and objections gracefully

        Context:
        - Lead name: {lead_info.get('name', 'Unknown')}
        - Company: {lead_info.get('company', 'Unknown')}
        - Industry: {lead_info.get('industry', 'Unknown')}
        - Current workflow state: {context.workflow_state}
        - Qualification score: {context.qualification_score}

        Guidelines:
        - Keep responses conversational and under 2 sentences
        - Ask one question at a time
        - Acknowledge what the person says before moving forward
        - Use their name occasionally but not excessively
        - Match their communication style and energy level
        - If they seem rushed, offer to call back

        Current conversation context: This is message #{len(context.conversation_history) + 1} in the conversation.
        """

    def can_handle(self, context: WorkflowContext) -> bool:
        """Check if conversation agent can handle this context"""
        # Conversation agent can handle most states except specialized ones
        return context.workflow_state in [
            WorkflowState.INITIALIZING,
            WorkflowState.GREETING,
            WorkflowState.QUALIFYING,
            WorkflowState.CLOSING
        ]

    def get_confidence_threshold(self) -> float:
        """Conversation agent confidence threshold"""
        return 0.6  # Lower threshold since it's the general conversationalist