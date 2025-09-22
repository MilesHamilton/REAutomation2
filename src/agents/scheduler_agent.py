import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

from .base_agent import BaseAgent
from .models import AgentType, AgentResponse, WorkflowContext, WorkflowState, SchedulingSlot

logger = logging.getLogger(__name__)


class SchedulerAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.SCHEDULER)

    async def process(self, context: WorkflowContext, user_input: str) -> AgentResponse:
        """Handle scheduling requests and calendar management"""
        try:
            # Parse scheduling preferences
            preferences = await self._parse_scheduling_preferences(context, user_input)

            # Generate available slots
            available_slots = self._generate_available_slots(preferences)

            # Create scheduling response
            response_text = await self._generate_scheduling_response(context, available_slots)

            return AgentResponse(
                agent_type=self.agent_type,
                response_text=response_text,
                state_updates={
                    "scheduling_attempts": context.scheduling_attempts + 1,
                    "metadata.available_slots": [slot.dict() for slot in available_slots]
                }
            )

        except Exception as e:
            logger.error(f"Scheduler agent error: {e}")
            return AgentResponse(
                agent_type=self.agent_type,
                response_text="Let me check my calendar and get back to you with some options."
            )

    async def _parse_scheduling_preferences(self, context: WorkflowContext, user_input: str) -> Dict[str, Any]:
        """Parse user's scheduling preferences"""
        schema = {
            "type": "object",
            "properties": {
                "preferred_days": {"type": "array", "items": {"type": "string"}},
                "preferred_times": {"type": "array", "items": {"type": "string"}},
                "timezone": {"type": "string"},
                "duration_preference": {"type": "string"},
                "urgency": {"type": "string", "enum": ["urgent", "normal", "flexible"]}
            }
        }

        return await self._generate_structured_response(context, user_input, schema) or {}

    def _generate_available_slots(self, preferences: Dict[str, Any]) -> List[SchedulingSlot]:
        """Generate available scheduling slots"""
        slots = []
        now = datetime.now()

        # Generate next 5 business days
        for i in range(1, 8):
            date = now + timedelta(days=i)
            if date.weekday() < 5:  # Weekdays only
                # Morning slot
                morning = date.replace(hour=10, minute=0, second=0, microsecond=0)
                slots.append(SchedulingSlot(
                    datetime=morning.isoformat(),
                    duration_minutes=30,
                    timezone="UTC"
                ))

                # Afternoon slot
                afternoon = date.replace(hour=14, minute=0, second=0, microsecond=0)
                slots.append(SchedulingSlot(
                    datetime=afternoon.isoformat(),
                    duration_minutes=30,
                    timezone="UTC"
                ))

        return slots[:6]  # Return first 6 slots

    async def _generate_scheduling_response(self, context: WorkflowContext, slots: List[SchedulingSlot]) -> str:
        """Generate response with scheduling options"""
        if not slots:
            return "I'm not seeing any availability right now. Can I have someone from my team reach out to coordinate a time?"

        slot_descriptions = []
        for i, slot in enumerate(slots[:3]):  # Show top 3 options
            dt = datetime.fromisoformat(slot.datetime)
            day = dt.strftime("%A, %B %d")
            time = dt.strftime("%I:%M %p")
            slot_descriptions.append(f"{day} at {time}")

        options_text = " or ".join(slot_descriptions)

        return f"Great! I have some time available {options_text}. What works better for your schedule?"

    def get_system_prompt(self, context: WorkflowContext) -> str:
        return f"""
        You are a scheduling agent focused on converting qualified leads to booked meetings.
        Your role is to:
        1. Offer specific, convenient time slots
        2. Handle scheduling objections and conflicts
        3. Confirm meeting details and expectations
        4. Create urgency when appropriate

        Current context: Scheduling attempt #{context.scheduling_attempts + 1}
        Qualification score: {context.qualification_score}
        """

    def can_handle(self, context: WorkflowContext) -> bool:
        return context.workflow_state == WorkflowState.SCHEDULING