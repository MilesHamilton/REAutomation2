import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ServiceCosts:
    """Cost breakdown for different services"""
    # LLM costs (per token)
    ollama_local_cost_per_token: float = 0.0  # Local inference is essentially free
    openai_gpt4_cost_per_token: float = 0.03 / 1000  # $0.03 per 1K tokens
    openai_gpt35_cost_per_token: float = 0.002 / 1000  # $0.002 per 1K tokens

    # TTS costs
    local_piper_cost_per_char: float = 0.0  # Local TTS is essentially free
    local_coqui_cost_per_char: float = 0.0  # Local TTS is essentially free
    elevenlabs_cost_per_char: float = 0.0002  # $0.0002 per character

    # STT costs
    whisper_local_cost_per_minute: float = 0.0  # Local inference is essentially free
    whisper_api_cost_per_minute: float = 0.006  # $0.006 per minute

    # Twilio costs
    twilio_voice_cost_per_minute: float = 0.0085  # $0.0085 per minute
    twilio_sms_cost_per_message: float = 0.0075  # $0.0075 per SMS

    # Infrastructure costs (amortized per call)
    infrastructure_cost_per_call: float = 0.001  # $0.001 per call for compute/storage

    # Rate limiting factors
    max_tokens_per_request: int = 4000
    max_characters_per_tts: int = 5000


class CostCalculator:
    """Real-time cost calculation for voice calls"""

    def __init__(self, service_costs: Optional[ServiceCosts] = None):
        self.costs = service_costs or ServiceCosts()
        self.call_costs: Dict[str, Dict[str, float]] = {}

    def initialize_call_costs(self, call_id: str):
        """Initialize cost tracking for a new call"""
        self.call_costs[call_id] = {
            "llm": 0.0,
            "tts": 0.0,
            "stt": 0.0,
            "twilio": 0.0,
            "infrastructure": self.costs.infrastructure_cost_per_call,
            "total": self.costs.infrastructure_cost_per_call,
            "started_at": datetime.now().timestamp()
        }
        logger.debug(f"Initialized cost tracking for call {call_id}")

    def calculate_llm_cost(
        self,
        call_id: str,
        provider: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate LLM inference cost"""
        try:
            total_tokens = input_tokens + output_tokens

            # Get cost per token based on provider
            if provider == "ollama":
                cost_per_token = self.costs.ollama_local_cost_per_token
            elif provider == "openai-gpt-4":
                cost_per_token = self.costs.openai_gpt4_cost_per_token
            elif provider == "openai-gpt-3.5":
                cost_per_token = self.costs.openai_gpt35_cost_per_token
            else:
                logger.warning(f"Unknown LLM provider: {provider}, using default cost")
                cost_per_token = self.costs.ollama_local_cost_per_token

            cost = total_tokens * cost_per_token

            # Update call costs
            if call_id in self.call_costs:
                self.call_costs[call_id]["llm"] += cost
                self.call_costs[call_id]["total"] += cost

            logger.debug(f"Call {call_id}: LLM cost ${cost:.6f} ({total_tokens} tokens)")
            return cost

        except Exception as e:
            logger.error(f"Error calculating LLM cost for call {call_id}: {e}")
            return 0.0

    def calculate_tts_cost(
        self,
        call_id: str,
        provider: str,
        character_count: int
    ) -> float:
        """Calculate TTS synthesis cost"""
        try:
            # Get cost per character based on provider
            if provider in ["piper", "coqui", "local"]:
                cost_per_char = self.costs.local_piper_cost_per_char
            elif provider == "elevenlabs":
                cost_per_char = self.costs.elevenlabs_cost_per_char
            else:
                logger.warning(f"Unknown TTS provider: {provider}, using local cost")
                cost_per_char = self.costs.local_piper_cost_per_char

            cost = character_count * cost_per_char

            # Update call costs
            if call_id in self.call_costs:
                self.call_costs[call_id]["tts"] += cost
                self.call_costs[call_id]["total"] += cost

            logger.debug(f"Call {call_id}: TTS cost ${cost:.6f} ({character_count} chars)")
            return cost

        except Exception as e:
            logger.error(f"Error calculating TTS cost for call {call_id}: {e}")
            return 0.0

    def calculate_stt_cost(
        self,
        call_id: str,
        provider: str,
        duration_minutes: float
    ) -> float:
        """Calculate STT transcription cost"""
        try:
            # Get cost per minute based on provider
            if provider == "whisper-local":
                cost_per_minute = self.costs.whisper_local_cost_per_minute
            elif provider == "whisper-api":
                cost_per_minute = self.costs.whisper_api_cost_per_minute
            else:
                logger.warning(f"Unknown STT provider: {provider}, using local cost")
                cost_per_minute = self.costs.whisper_local_cost_per_minute

            cost = duration_minutes * cost_per_minute

            # Update call costs
            if call_id in self.call_costs:
                self.call_costs[call_id]["stt"] += cost
                self.call_costs[call_id]["total"] += cost

            logger.debug(f"Call {call_id}: STT cost ${cost:.6f} ({duration_minutes:.2f} min)")
            return cost

        except Exception as e:
            logger.error(f"Error calculating STT cost for call {call_id}: {e}")
            return 0.0

    def calculate_twilio_cost(
        self,
        call_id: str,
        duration_minutes: float,
        sms_count: int = 0
    ) -> float:
        """Calculate Twilio communication cost"""
        try:
            voice_cost = duration_minutes * self.costs.twilio_voice_cost_per_minute
            sms_cost = sms_count * self.costs.twilio_sms_cost_per_message
            total_cost = voice_cost + sms_cost

            # Update call costs
            if call_id in self.call_costs:
                self.call_costs[call_id]["twilio"] += total_cost
                self.call_costs[call_id]["total"] += total_cost

            logger.debug(f"Call {call_id}: Twilio cost ${total_cost:.6f} ({duration_minutes:.2f} min, {sms_count} SMS)")
            return total_cost

        except Exception as e:
            logger.error(f"Error calculating Twilio cost for call {call_id}: {e}")
            return 0.0

    def get_call_cost_breakdown(self, call_id: str) -> Dict[str, float]:
        """Get detailed cost breakdown for a call"""
        if call_id not in self.call_costs:
            logger.warning(f"No cost data found for call {call_id}")
            return {}

        return self.call_costs[call_id].copy()

    def get_total_call_cost(self, call_id: str) -> float:
        """Get total cost for a call"""
        if call_id not in self.call_costs:
            return 0.0
        return self.call_costs[call_id].get("total", 0.0)

    def estimate_remaining_cost(
        self,
        call_id: str,
        estimated_duration_minutes: float,
        estimated_messages: int,
        estimated_tts_characters: int,
        tier: str = "local"
    ) -> float:
        """Estimate remaining cost for a call"""
        try:
            # Estimate remaining Twilio cost
            remaining_twilio = estimated_duration_minutes * self.costs.twilio_voice_cost_per_minute

            # Estimate remaining TTS cost
            if tier == "premium":
                remaining_tts = estimated_tts_characters * self.costs.elevenlabs_cost_per_char
            else:
                remaining_tts = estimated_tts_characters * self.costs.local_piper_cost_per_char

            # Estimate remaining LLM cost (using Ollama local)
            estimated_tokens = estimated_messages * 100  # Rough estimate
            remaining_llm = estimated_tokens * self.costs.ollama_local_cost_per_token

            # Estimate remaining STT cost (using local Whisper)
            remaining_stt = estimated_duration_minutes * self.costs.whisper_local_cost_per_minute

            total_estimated = remaining_twilio + remaining_tts + remaining_llm + remaining_stt

            logger.debug(f"Call {call_id}: Estimated remaining cost ${total_estimated:.6f}")
            return total_estimated

        except Exception as e:
            logger.error(f"Error estimating remaining cost for call {call_id}: {e}")
            return 0.0

    def finalize_call_costs(self, call_id: str) -> Dict[str, float]:
        """Finalize and return costs for a completed call"""
        if call_id not in self.call_costs:
            logger.warning(f"No cost data found for finalizing call {call_id}")
            return {}

        costs = self.call_costs[call_id].copy()
        costs["ended_at"] = datetime.now().timestamp()

        # Calculate call duration
        if "started_at" in costs:
            duration = costs["ended_at"] - costs["started_at"]
            costs["duration_seconds"] = duration

        logger.info(f"Call {call_id} finalized: Total cost ${costs['total']:.6f}")

        # Keep cost data for reporting but mark as finalized
        self.call_costs[call_id]["finalized"] = True

        return costs

    def cleanup_old_calls(self, max_age_hours: int = 24):
        """Clean up cost data for old calls"""
        current_time = datetime.now().timestamp()
        cutoff_time = current_time - (max_age_hours * 3600)

        calls_to_remove = []
        for call_id, costs in self.call_costs.items():
            if costs.get("finalized") and costs.get("ended_at", 0) < cutoff_time:
                calls_to_remove.append(call_id)

        for call_id in calls_to_remove:
            del self.call_costs[call_id]

        if calls_to_remove:
            logger.info(f"Cleaned up cost data for {len(calls_to_remove)} old calls")

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get overall cost summary across all active calls"""
        try:
            active_calls = [call_id for call_id, costs in self.call_costs.items()
                           if not costs.get("finalized", False)]

            total_active_cost = sum(
                costs["total"] for costs in self.call_costs.values()
                if not costs.get("finalized", False)
            )

            finalized_calls = [call_id for call_id, costs in self.call_costs.items()
                             if costs.get("finalized", False)]

            total_finalized_cost = sum(
                costs["total"] for costs in self.call_costs.values()
                if costs.get("finalized", False)
            )

            return {
                "active_calls": len(active_calls),
                "finalized_calls": len(finalized_calls),
                "total_active_cost": total_active_cost,
                "total_finalized_cost": total_finalized_cost,
                "total_cost": total_active_cost + total_finalized_cost,
                "average_cost_per_call": (total_finalized_cost / max(len(finalized_calls), 1))
                                        if finalized_calls else 0.0
            }

        except Exception as e:
            logger.error(f"Error generating cost summary: {e}")
            return {"error": str(e)}