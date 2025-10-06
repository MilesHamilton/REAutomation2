import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .models import (
    AgentType, WorkflowState, WorkflowContext, AgentMessage,
    AgentResponse, AgentDecision
)
from .conversation_agent import ConversationAgent
from .qualification_agent import QualificationAgent
from .objection_handler import ObjectionHandlerAgent
from .scheduler_agent import SchedulerAgent
from .analytics_agent import AnalyticsAgent
from ..monitoring.tracing import trace_workflow, trace_agent_execution

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    def __init__(self):
        self.agents = {
            AgentType.CONVERSATION: ConversationAgent(),
            AgentType.QUALIFICATION: QualificationAgent(),
            AgentType.OBJECTION_HANDLER: ObjectionHandlerAgent(),
            AgentType.SCHEDULER: SchedulerAgent(),
            AgentType.ANALYTICS: AnalyticsAgent()
        }

        self.workflow_graph: Optional[StateGraph] = None
        self.active_contexts: Dict[str, WorkflowContext] = {}
        self.is_initialized = False

        # Callbacks
        self._on_agent_transition: Optional[Callable] = None
        self._on_workflow_complete: Optional[Callable] = None
        self._on_tier_escalation: Optional[Callable] = None

    async def initialize(self):
        """Initialize the orchestrator and build the workflow graph"""
        try:
            logger.info("Initializing Agent Orchestrator...")

            # Build LangGraph workflow
            self._build_workflow_graph()

            self.is_initialized = True
            logger.info("Agent Orchestrator initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Agent Orchestrator initialization failed: {e}")
            return False

    def _build_workflow_graph(self):
        """Build the LangGraph workflow"""
        # Define the state structure
        workflow = StateGraph(WorkflowContext)

        # Add nodes for each agent
        workflow.add_node("conversation", self._conversation_node)
        workflow.add_node("qualification", self._qualification_node)
        workflow.add_node("objection_handler", self._objection_handler_node)
        workflow.add_node("scheduler", self._scheduler_node)
        workflow.add_node("analytics", self._analytics_node)

        # Set entry point
        workflow.set_entry_point("conversation")

        # Define conditional edges based on workflow state
        workflow.add_conditional_edges(
            "conversation",
            self._route_from_conversation,
            {
                "qualification": "qualification",
                "objection_handler": "objection_handler",
                "scheduler": "scheduler",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "qualification",
            self._route_from_qualification,
            {
                "conversation": "conversation",
                "objection_handler": "objection_handler",
                "scheduler": "scheduler",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "objection_handler",
            self._route_from_objection_handler,
            {
                "conversation": "conversation",
                "qualification": "qualification",
                "scheduler": "scheduler",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "scheduler",
            self._route_from_scheduler,
            {
                "conversation": "conversation",
                "qualification": "qualification",
                "end": END
            }
        )

        # Analytics node always goes to end
        workflow.add_edge("analytics", END)

        # Set up checkpointer for state persistence
        memory = MemorySaver()
        self.workflow_graph = workflow.compile(checkpointer=memory)

    async def process_input(
        self,
        call_id: str,
        user_input: str,
        lead_data: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentResponse]:
        """Process user input through the agent workflow"""
        try:
            if not self.is_initialized:
                logger.error("Agent Orchestrator not initialized")
                return None

            # Get or create workflow context
            context = self._get_or_create_context(call_id, lead_data)

            # Add user message to conversation history
            user_message = AgentMessage(
                agent_type=AgentType.CONVERSATION,  # User input attributed to conversation
                content=user_input,
                timestamp=time.time()
            )
            context.conversation_history.append(user_message)
            context.updated_at = time.time()

            # Process through workflow
            config = {"configurable": {"thread_id": call_id}}

            # Update context with user input for processing
            context.metadata["current_input"] = user_input

            # Run the workflow
            result = await self.workflow_graph.ainvoke(context, config)

            # Get the final response from the workflow
            if hasattr(result, 'metadata') and 'agent_response' in result.metadata:
                agent_response = result.metadata['agent_response']

                # Add agent response to conversation history
                if agent_response.response_text:
                    agent_message = AgentMessage(
                        agent_type=agent_response.agent_type,
                        content=agent_response.response_text,
                        timestamp=time.time()
                    )
                    result.conversation_history.append(agent_message)

                # Update stored context
                self.active_contexts[call_id] = result

                # Check for tier escalation
                if agent_response.should_escalate_tier and self._on_tier_escalation:
                    self._on_tier_escalation(call_id, result.qualification_score)

                return agent_response

            return None

        except Exception as e:
            logger.error(f"Error processing input for call {call_id}: {e}")
            return AgentResponse(
                agent_type=AgentType.CONVERSATION,
                response_text="I apologize, I experienced a technical issue. Could you repeat that?"
            )

    async def process_voice_input(
        self,
        call_id: str,
        user_input: str,
        lead_data: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentResponse]:
        """Process voice input with voice-specific optimizations"""
        try:
            # Get or create voice-specific context
            context = self._get_or_create_voice_context(call_id, lead_data)

            # Process through standard workflow
            agent_response = await self.process_input(call_id, user_input, lead_data)

            if agent_response:
                # Optimize response for voice delivery
                agent_response = self._optimize_for_voice(agent_response)

                # Check if tier escalation is needed based on qualification
                if self._check_tier_escalation_needed(context):
                    agent_response.should_escalate_tier = True

            return agent_response

        except Exception as e:
            logger.error(f"Error processing voice input for call {call_id}: {e}")
            return self._create_fallback_response()

    def _get_or_create_voice_context(
        self,
        call_id: str,
        lead_data: Optional[Dict[str, Any]] = None
    ) -> WorkflowContext:
        """Get or create context with voice-specific metadata"""
        context = self._get_or_create_context(call_id, lead_data)

        # Add voice-specific metadata
        if "voice_mode" not in context.metadata:
            context.metadata["voice_mode"] = True
            context.metadata["response_style"] = "concise"
            context.metadata["max_response_length"] = 150  # words

        return context

    def _optimize_for_voice(self, response: AgentResponse) -> AgentResponse:
        """Optimize agent response for voice delivery"""
        try:
            if response.response_text:
                # Remove markdown formatting
                text = response.response_text.replace("*", "").replace("#", "")

                # Simplify complex punctuation
                text = text.replace("  ", " ").strip()

                # Ensure conversational flow
                if len(text.split()) > 150:
                    # Truncate to ~150 words at sentence boundary
                    sentences = text.split(". ")
                    truncated = []
                    word_count = 0

                    for sentence in sentences:
                        words = sentence.split()
                        if word_count + len(words) <= 150:
                            truncated.append(sentence)
                            word_count += len(words)
                        else:
                            break

                    text = ". ".join(truncated)
                    if not text.endswith("."):
                        text += "."

                response.response_text = text

            return response

        except Exception as e:
            logger.error(f"Error optimizing response for voice: {e}")
            return response

    def _check_tier_escalation_needed(self, context: WorkflowContext) -> bool:
        """Check if tier escalation should be triggered"""
        try:
            # Check qualification score
            if context.qualification_score >= 0.7:  # Threshold from settings
                return True

            # Check if lead has high-value indicators
            lead_data = context.lead_data
            if lead_data.get("budget") and lead_data.get("budget") > 10000:
                return True

            if lead_data.get("urgency") == "high":
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking tier escalation: {e}")
            return False

    def _create_fallback_response(self) -> AgentResponse:
        """Create a fallback response for error conditions"""
        return AgentResponse(
            agent_type=AgentType.CONVERSATION,
            response_text="I apologize for the brief interruption. Could you please repeat that?"
        )

    def _get_or_create_context(
        self,
        call_id: str,
        lead_data: Optional[Dict[str, Any]] = None
    ) -> WorkflowContext:
        """Get existing context or create new one"""
        if call_id in self.active_contexts:
            return self.active_contexts[call_id]

        context = WorkflowContext(
            call_id=call_id,
            lead_data=lead_data or {},
            workflow_state=WorkflowState.INITIALIZING
        )

        self.active_contexts[call_id] = context
        return context

    # Node functions for LangGraph
    async def _conversation_node(self, state: WorkflowContext) -> WorkflowContext:
        """Process conversation agent"""
        agent = self.agents[AgentType.CONVERSATION]
        user_input = state.metadata.get("current_input", "")

        response = await agent.execute_with_timeout(state, user_input)

        # Update state based on agent response
        if response.state_updates:
            for key, value in response.state_updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)
                else:
                    state.metadata[key] = value

        # Store response for return
        state.metadata["agent_response"] = response
        state.current_agent = AgentType.CONVERSATION

        return state

    async def _qualification_node(self, state: WorkflowContext) -> WorkflowContext:
        """Process qualification agent"""
        agent = self.agents[AgentType.QUALIFICATION]
        user_input = state.metadata.get("current_input", "")

        response = await agent.execute_with_timeout(state, user_input)

        # Update state
        if response.state_updates:
            for key, value in response.state_updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)
                else:
                    state.metadata[key] = value

        state.metadata["agent_response"] = response
        state.current_agent = AgentType.QUALIFICATION

        return state

    async def _objection_handler_node(self, state: WorkflowContext) -> WorkflowContext:
        """Process objection handler agent"""
        agent = self.agents[AgentType.OBJECTION_HANDLER]
        user_input = state.metadata.get("current_input", "")

        response = await agent.execute_with_timeout(state, user_input)

        # Update state
        if response.state_updates:
            for key, value in response.state_updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)
                else:
                    state.metadata[key] = value

        state.metadata["agent_response"] = response
        state.current_agent = AgentType.OBJECTION_HANDLER

        return state

    async def _scheduler_node(self, state: WorkflowContext) -> WorkflowContext:
        """Process scheduler agent"""
        agent = self.agents[AgentType.SCHEDULER]
        user_input = state.metadata.get("current_input", "")

        response = await agent.execute_with_timeout(state, user_input)

        # Update state
        if response.state_updates:
            for key, value in response.state_updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)
                else:
                    state.metadata[key] = value

        state.metadata["agent_response"] = response
        state.current_agent = AgentType.SCHEDULER

        return state

    async def _analytics_node(self, state: WorkflowContext) -> WorkflowContext:
        """Process analytics agent (background)"""
        agent = self.agents[AgentType.ANALYTICS]
        user_input = state.metadata.get("current_input", "")

        response = await agent.execute_with_timeout(state, user_input)

        # Analytics doesn't change workflow flow, just collects data
        if response.state_updates:
            state.metadata.update(response.state_updates)

        return state

    # Routing functions
    def _route_from_conversation(self, state: WorkflowContext) -> str:
        """Route from conversation agent based on workflow state"""
        if state.workflow_state == WorkflowState.QUALIFYING:
            return "qualification"
        elif state.workflow_state == WorkflowState.HANDLING_OBJECTION:
            return "objection_handler"
        elif state.workflow_state == WorkflowState.SCHEDULING:
            return "scheduler"
        elif state.workflow_state in [WorkflowState.COMPLETED, WorkflowState.FAILED]:
            return "end"
        else:
            return "end"  # Stay in conversation by default

    def _route_from_qualification(self, state: WorkflowContext) -> str:
        """Route from qualification agent"""
        if state.workflow_state == WorkflowState.HANDLING_OBJECTION:
            return "objection_handler"
        elif state.workflow_state == WorkflowState.SCHEDULING:
            return "scheduler"
        elif state.workflow_state == WorkflowState.GREETING:
            return "conversation"
        else:
            return "end"

    def _route_from_objection_handler(self, state: WorkflowContext) -> str:
        """Route from objection handler"""
        if state.workflow_state == WorkflowState.QUALIFYING:
            return "qualification"
        elif state.workflow_state == WorkflowState.SCHEDULING:
            return "scheduler"
        elif state.workflow_state == WorkflowState.GREETING:
            return "conversation"
        else:
            return "end"

    def _route_from_scheduler(self, state: WorkflowContext) -> str:
        """Route from scheduler"""
        if state.workflow_state == WorkflowState.QUALIFYING:
            return "qualification"
        elif state.workflow_state == WorkflowState.GREETING:
            return "conversation"
        else:
            return "end"

    # Context management
    def get_context(self, call_id: str) -> Optional[WorkflowContext]:
        """Get workflow context for a call"""
        return self.active_contexts.get(call_id)

    def end_workflow(self, call_id: str) -> bool:
        """End workflow for a call"""
        try:
            if call_id in self.active_contexts:
                context = self.active_contexts[call_id]
                context.workflow_state = WorkflowState.COMPLETED

                # Trigger completion callback
                if self._on_workflow_complete:
                    self._on_workflow_complete(context)

                # Run final analytics
                asyncio.create_task(self.agents[AgentType.ANALYTICS].process(context, ""))

                del self.active_contexts[call_id]
                logger.info(f"Workflow ended for call {call_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error ending workflow for call {call_id}: {e}")
            return False

    def get_active_workflows(self) -> Dict[str, WorkflowContext]:
        """Get all active workflow contexts"""
        return self.active_contexts.copy()

    # Callbacks
    def on_agent_transition(self, callback: Callable[[str, AgentType, AgentType], None]):
        """Set callback for agent transitions"""
        self._on_agent_transition = callback

    def on_workflow_complete(self, callback: Callable[[WorkflowContext], None]):
        """Set callback for workflow completion"""
        self._on_workflow_complete = callback

    def on_tier_escalation(self, callback: Callable[[str, float], None]):
        """Set callback for tier escalation"""
        self._on_tier_escalation = callback

    async def health_check(self) -> dict:
        """Check orchestrator health"""
        try:
            if not self.is_initialized:
                return {
                    "status": "unhealthy",
                    "error": "Orchestrator not initialized"
                }

            return {
                "status": "healthy",
                "active_workflows": len(self.active_contexts),
                "agents": list(self.agents.keys()),
                "workflow_graph_ready": self.workflow_graph is not None,
                "initialized": self.is_initialized
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self):
        """Clean up orchestrator resources"""
        try:
            # End all active workflows
            for call_id in list(self.active_contexts.keys()):
                self.end_workflow(call_id)

            # Clean up agents
            for agent in self.agents.values():
                await agent.cleanup()

            self.is_initialized = False
            logger.info("Agent Orchestrator cleanup complete")

        except Exception as e:
            logger.error(f"Error during orchestrator cleanup: {e}")


# Global orchestrator instance
agent_orchestrator = AgentOrchestrator()
