import logging
from typing import Dict, Any, List

from ..config import settings
from .base_agent import BaseAgent
from .models import (
    AgentType, AgentResponse, WorkflowContext, WorkflowState,
    QualificationFactors, AgentDecision
)

logger = logging.getLogger(__name__)


class QualificationAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.QUALIFICATION)
        self.qualification_questions = [
            {
                "factor": "intent",
                "question": "Are you currently looking for solutions to improve {business_area}?",
                "follow_up": "What specific challenges are you facing?"
            },
            {
                "factor": "budget",
                "question": "What kind of budget do you typically allocate for {solution_type}?",
                "follow_up": "Is budget something you control or would need approval for?"
            },
            {
                "factor": "timeline",
                "question": "When would you ideally like to have a solution in place?",
                "follow_up": "What's driving that timeline?"
            },
            {
                "factor": "authority",
                "question": "Are you involved in making decisions about {business_area}?",
                "follow_up": "Who else would be part of this decision?"
            },
            {
                "factor": "needs",
                "question": "Tell me about your current process for {business_area}.",
                "follow_up": "What works well and what could be improved?"
            }
        ]

    async def process(self, context: WorkflowContext, user_input: str) -> AgentResponse:
        """Process qualification questions and scoring"""
        try:
            # Analyze the user's response for qualification factors
            qualification_analysis = await self._analyze_qualification_response(context, user_input)

            # Update qualification score
            updated_factors = self._update_qualification_factors(context, qualification_analysis)

            # Generate next qualification question or decide on next action
            next_action = await self._determine_next_qualification_action(context, updated_factors)

            if next_action["action"] == "ask_question":
                response_text = await self._generate_next_question(context, next_action["factor"])

                return AgentResponse(
                    agent_type=self.agent_type,
                    response_text=response_text,
                    state_updates={
                        "qualification_score": updated_factors.overall_score,
                        "metadata.qualification_factors": updated_factors.dict()
                    }
                )

            elif next_action["action"] == "complete_qualification":
                response_text = await self._generate_qualification_summary(context, updated_factors)

                # Determine next workflow state based on score
                next_state = self._determine_next_workflow_state(updated_factors.overall_score)

                return AgentResponse(
                    agent_type=self.agent_type,
                    response_text=response_text,
                    state_updates={
                        "workflow_state": next_state,
                        "qualification_score": updated_factors.overall_score,
                        "metadata.qualification_factors": updated_factors.dict()
                    },
                    should_escalate_tier=updated_factors.overall_score >= settings.tier_escalation_threshold
                )

            else:  # Continue conversation
                response_text = await self._generate_llm_response(context, user_input)

                return AgentResponse(
                    agent_type=self.agent_type,
                    response_text=response_text,
                    state_updates={
                        "qualification_score": updated_factors.overall_score,
                        "metadata.qualification_factors": updated_factors.dict()
                    }
                )

        except Exception as e:
            logger.error(f"Qualification agent error: {e}")
            return AgentResponse(
                agent_type=self.agent_type,
                response_text="That's helpful to know. Let me ask you about something else..."
            )

    async def _analyze_qualification_response(self, context: WorkflowContext, user_input: str) -> Dict[str, Any]:
        """Analyze user response for qualification factors"""
        schema = {
            "type": "object",
            "properties": {
                "intent_indicators": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence": {"type": "string"}
                    }
                },
                "budget_indicators": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence": {"type": "string"}
                    }
                },
                "timeline_indicators": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence": {"type": "string"}
                    }
                },
                "authority_indicators": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence": {"type": "string"}
                    }
                },
                "needs_indicators": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence": {"type": "string"}
                    }
                },
                "overall_sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                "engagement_level": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }

        system_prompt = """
        Analyze this response for lead qualification factors:

        INTENT (buying intent, urgency, problem recognition):
        - High: "We need this ASAP", "This is a priority", "We're actively looking"
        - Medium: "We're exploring options", "It's on our radar", "Worth considering"
        - Low: "Not really a priority", "Happy with current solution", "Just curious"

        BUDGET (financial capacity, budget authority):
        - High: Specific budget mentioned, "We have allocated funds"
        - Medium: General budget ranges, "We'd need to see ROI"
        - Low: "No budget", "Very tight budget", "Would need to find money"

        TIMELINE (urgency, decision timeline):
        - High: "Within next month", "ASAP", "This quarter"
        - Medium: "Next few months", "By end of year", "Soon"
        - Low: "Someday", "No rush", "Maybe next year"

        AUTHORITY (decision-making power):
        - High: "I make these decisions", "It's up to me", "I'm the owner"
        - Medium: "I'm part of the team", "I have input", "I can recommend"
        - Low: "I'd need to check", "My boss decides", "Not my area"

        NEEDS (problem/solution fit):
        - High: Specific problems mentioned that match your solution
        - Medium: General challenges, some alignment
        - Low: No clear problems or misaligned needs

        Score each factor 0-1 based on the evidence in their response.
        """

        return await self._generate_structured_response(
            context=context,
            user_input=user_input,
            response_schema=schema,
            system_prompt=system_prompt
        ) or self._get_default_qualification_analysis()

    def _update_qualification_factors(self, context: WorkflowContext, analysis: Dict[str, Any]) -> QualificationFactors:
        """Update qualification factors based on analysis"""
        # Get existing factors from context
        existing_factors = context.metadata.get("qualification_factors", {})

        # Extract scores from analysis
        intent = analysis.get("intent_indicators", {}).get("score", existing_factors.get("intent", 0.0))
        budget = analysis.get("budget_indicators", {}).get("score", existing_factors.get("budget", 0.0))
        timeline = analysis.get("timeline_indicators", {}).get("score", existing_factors.get("timeline", 0.0))
        authority = analysis.get("authority_indicators", {}).get("score", existing_factors.get("authority", 0.0))
        needs = analysis.get("needs_indicators", {}).get("score", existing_factors.get("needs", 0.0))

        # Use weighted average to incorporate new information with existing
        weight_new = 0.6  # Give more weight to new information
        weight_old = 0.4

        return QualificationFactors(
            intent=max(intent, existing_factors.get("intent", 0.0) * weight_old + intent * weight_new),
            budget=max(budget, existing_factors.get("budget", 0.0) * weight_old + budget * weight_new),
            timeline=max(timeline, existing_factors.get("timeline", 0.0) * weight_old + timeline * weight_new),
            authority=max(authority, existing_factors.get("authority", 0.0) * weight_old + authority * weight_new),
            needs=max(needs, existing_factors.get("needs", 0.0) * weight_old + needs * weight_new)
        )

    async def _determine_next_qualification_action(self, context: WorkflowContext, factors: QualificationFactors) -> Dict[str, Any]:
        """Determine the next action in qualification process"""
        # Check which factors need more information
        factor_scores = {
            "intent": factors.intent,
            "budget": factors.budget,
            "timeline": factors.timeline,
            "authority": factors.authority,
            "needs": factors.needs
        }

        # Find the lowest scoring factor that we should focus on
        lowest_factor = min(factor_scores.items(), key=lambda x: x[1])
        conversation_length = len(context.conversation_history)

        # Decision logic
        if conversation_length >= 8:  # Asked enough questions
            return {"action": "complete_qualification"}
        elif lowest_factor[1] < 0.5 and conversation_length < 6:  # Need more info on weak factor
            return {"action": "ask_question", "factor": lowest_factor[0]}
        elif factors.overall_score >= 0.7:  # High qualification score
            return {"action": "complete_qualification"}
        else:
            return {"action": "continue_conversation"}

    async def _generate_next_question(self, context: WorkflowContext, factor: str) -> str:
        """Generate the next qualification question"""
        # Find appropriate question for the factor
        question_template = next(
            (q for q in self.qualification_questions if q["factor"] == factor),
            self.qualification_questions[0]
        )

        # Personalize the question
        business_area = context.lead_data.get("industry", "your business operations")
        solution_type = "business solutions"  # Could be more specific based on context

        question = question_template["question"].format(
            business_area=business_area,
            solution_type=solution_type
        )

        # Make it conversational
        conversational_lead_ins = [
            "That makes sense. ",
            "I appreciate you sharing that. ",
            "Thanks for the insight. ",
            "Good to know. "
        ]

        lead_in = conversational_lead_ins[len(context.conversation_history) % len(conversational_lead_ins)]
        return f"{lead_in}{question}"

    async def _generate_qualification_summary(self, context: WorkflowContext, factors: QualificationFactors) -> str:
        """Generate a summary response after qualification"""
        if factors.overall_score >= settings.qualification_threshold:
            return """Based on what you've shared, it sounds like there's a real opportunity for us to help.
            I'd love to show you exactly how we can address your specific needs.
            Would you be open to a brief demo to see if this makes sense for your situation?"""
        elif factors.overall_score >= 0.5:
            return """I can see there are some interesting possibilities here.
            Let me ask you this - if I could show you a solution that addresses your main concern,
            would it be worth 15 minutes of your time to take a look?"""
        else:
            return """I appreciate you taking the time to speak with me today.
            It sounds like the timing might not be quite right.
            Would it be helpful if I checked back with you in a few months?"""

    def _determine_next_workflow_state(self, qualification_score: float) -> WorkflowState:
        """Determine next workflow state based on qualification score"""
        if qualification_score >= settings.qualification_threshold:
            return WorkflowState.SCHEDULING
        elif qualification_score >= 0.5:
            return WorkflowState.CLOSING  # Soft close attempt
        else:
            return WorkflowState.CLOSING  # Polite close

    def _get_default_qualification_analysis(self) -> Dict[str, Any]:
        """Get default qualification analysis when structured response fails"""
        return {
            "intent_indicators": {"score": 0.3, "evidence": "No clear buying intent expressed"},
            "budget_indicators": {"score": 0.3, "evidence": "Budget not discussed"},
            "timeline_indicators": {"score": 0.3, "evidence": "Timeline not specified"},
            "authority_indicators": {"score": 0.5, "evidence": "Authority level unclear"},
            "needs_indicators": {"score": 0.4, "evidence": "Some needs mentioned"},
            "overall_sentiment": "neutral",
            "engagement_level": 0.5
        }

    def get_system_prompt(self, context: WorkflowContext) -> str:
        """Get system prompt for qualification agent"""
        current_factors = context.metadata.get("qualification_factors", {})

        return f"""
        You are an expert qualification agent conducting B2B lead qualification. Your role is to:

        1. Ask strategic questions to uncover qualification factors
        2. Listen actively and dig deeper when you hear buying signals
        3. Maintain natural conversation flow while gathering information
        4. Build value perception while qualifying

        Current Qualification Status:
        - Intent: {current_factors.get('intent', 'Unknown')}
        - Budget: {current_factors.get('budget', 'Unknown')}
        - Timeline: {current_factors.get('timeline', 'Unknown')}
        - Authority: {current_factors.get('authority', 'Unknown')}
        - Needs: {current_factors.get('needs', 'Unknown')}
        - Overall Score: {context.qualification_score}

        Lead Context:
        - Company: {context.lead_data.get('company', 'Unknown')}
        - Industry: {context.lead_data.get('industry', 'Unknown')}
        - Size: {context.lead_data.get('company_size', 'Unknown')}

        Guidelines:
        - Ask one qualification question at a time
        - Use information they've already shared to ask better follow-up questions
        - If they give a vague answer, ask a clarifying follow-up
        - Always acknowledge their answer before asking the next question
        - Balance qualification with value building

        Remember: You're not interrogating them - you're having a consultative conversation
        to see if there's a mutual fit.
        """

    def can_handle(self, context: WorkflowContext) -> bool:
        """Check if qualification agent can handle this context"""
        return context.workflow_state == WorkflowState.QUALIFYING

    def get_confidence_threshold(self) -> float:
        """Qualification agent confidence threshold"""
        return 0.8  # Higher threshold for qualification accuracy