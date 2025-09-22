from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import uuid
import time
import json

from ...config import settings
from ...voice.pipeline import VoicePipeline
from ...voice.models import TTSProvider, VoiceCallState
from ...agents.orchestrator import agent_orchestrator

router = APIRouter()


class CallStartRequest(BaseModel):
    phone_number: str
    lead_data: Optional[Dict[str, Any]] = None
    priority: int = 1
    tier: str = "local"  # local or premium


class CallResponse(BaseModel):
    call_id: str
    status: str
    phone_number: str
    created_at: float
    tier: str
    estimated_cost: float


class CallStatusResponse(BaseModel):
    call_id: str
    status: str
    duration_seconds: Optional[float] = None
    tier: str
    actual_cost: Optional[float] = None
    qualification_score: Optional[float] = None
    conversation_summary: Optional[str] = None


# Global voice pipeline instance
voice_pipeline = VoicePipeline()

# Simple in-memory call tracking (replace with proper database)
active_calls: Dict[str, Dict[str, Any]] = {}
call_history: List[Dict[str, Any]] = []


@router.post("/start", response_model=CallResponse)
async def start_call(request: CallStartRequest):
    """
    Start a new outbound call with voice pipeline integration
    """
    try:
        # Check concurrent call limits
        active_count = len([call for call in active_calls.values() if call["status"] in ["ringing", "in_progress"]])

        if active_count >= settings.max_concurrent_calls:
            raise HTTPException(
                status_code=429,
                detail=f"Maximum concurrent calls ({settings.max_concurrent_calls}) reached"
            )

        call_id = str(uuid.uuid4())
        created_at = time.time()

        # Determine initial TTS tier
        initial_tier = TTSProvider.LOCAL_PIPER if request.tier == "local" else TTSProvider.ELEVENLABS

        # Estimate cost based on tier
        estimated_cost = 0.03 if initial_tier == TTSProvider.LOCAL_PIPER else 0.08

        call_data = {
            "call_id": call_id,
            "phone_number": request.phone_number,
            "lead_data": request.lead_data or {},
            "priority": request.priority,
            "tier": initial_tier.value,
            "status": "initializing",
            "created_at": created_at,
            "estimated_cost": estimated_cost,
            "actual_cost": None,
            "qualification_score": None,
            "conversation_id": None,
            "conversation_summary": None
        }

        active_calls[call_id] = call_data

        # Initialize voice pipeline if not already done
        if not voice_pipeline.is_initialized:
            pipeline_ready = await voice_pipeline.initialize()
            if not pipeline_ready:
                raise HTTPException(status_code=503, detail="Voice pipeline initialization failed")

        # Start the actual voice call through pipeline
        call_started = await voice_pipeline.start_call(
            call_id=call_id,
            phone_number=request.phone_number,
            lead_data=request.lead_data,
            initial_tier=initial_tier
        )

        if call_started:
            call_data["status"] = "ringing"

            # Set up agent orchestrator for this call
            # The agent orchestrator will handle the conversation flow

            return CallResponse(
                call_id=call_id,
                status=call_data["status"],
                phone_number=request.phone_number,
                created_at=created_at,
                tier=initial_tier.value,
                estimated_cost=estimated_cost
            )
        else:
            # Remove failed call
            del active_calls[call_id]
            raise HTTPException(status_code=500, detail="Failed to initiate voice call")

    except Exception as e:
        if call_id in active_calls:
            del active_calls[call_id]
        raise HTTPException(status_code=500, detail=f"Failed to start call: {e}")


@router.get("/{call_id}/status", response_model=CallStatusResponse)
async def get_call_status(call_id: str):
    """
    Get the current status of a call
    """
    if call_id not in active_calls:
        # Check call history
        historical_call = next(
            (call for call in call_history if call["call_id"] == call_id),
            None
        )
        if not historical_call:
            raise HTTPException(status_code=404, detail="Call not found")
        call_data = historical_call
    else:
        call_data = active_calls[call_id]

    duration = None
    if call_data.get("ended_at") and call_data.get("started_at"):
        duration = call_data["ended_at"] - call_data["started_at"]

    return CallStatusResponse(
        call_id=call_id,
        status=call_data["status"],
        duration_seconds=duration,
        tier=call_data["tier"],
        actual_cost=call_data.get("actual_cost"),
        qualification_score=call_data.get("qualification_score"),
        conversation_summary=call_data.get("conversation_summary")
    )


@router.post("/{call_id}/end")
async def end_call(call_id: str):
    """
    End an active call
    """
    if call_id not in active_calls:
        raise HTTPException(status_code=404, detail="Active call not found")

    call_data = active_calls[call_id]

    if call_data["status"] not in ["ringing", "in_progress"]:
        raise HTTPException(status_code=400, detail="Call is not active")

    call_data["status"] = "completed"
    call_data["ended_at"] = time.time()

    # Calculate actual cost (placeholder logic)
    duration = call_data.get("ended_at", time.time()) - call_data.get("started_at", time.time())
    base_cost = 0.03 if call_data["tier"] == "local" else 0.08
    call_data["actual_cost"] = min(base_cost * (duration / 60), settings.cost_per_call_limit)

    # Move to history
    call_history.append(call_data.copy())
    del active_calls[call_id]

    return {"status": "ended", "call_id": call_id, "actual_cost": call_data["actual_cost"]}


@router.get("/")
async def list_calls():
    """
    List all calls (active and recent history)
    """
    recent_history = call_history[-50:]  # Last 50 calls

    return {
        "active_calls": list(active_calls.values()),
        "recent_calls": recent_history,
        "summary": {
            "active_count": len(active_calls),
            "total_calls_today": len([
                call for call in call_history
                if call.get("created_at", 0) > time.time() - 86400
            ])
        }
    }


@router.get("/metrics")
async def call_metrics():
    """
    Get call performance metrics
    """
    now = time.time()
    today_start = now - 86400  # Last 24 hours

    today_calls = [
        call for call in call_history
        if call.get("created_at", 0) > today_start
    ]

    completed_calls = [
        call for call in today_calls
        if call["status"] == "completed"
    ]

    total_cost = sum(call.get("actual_cost", 0) for call in completed_calls)
    avg_cost = total_cost / max(len(completed_calls), 1)

    qualified_leads = [
        call for call in completed_calls
        if call.get("qualification_score", 0) >= settings.qualification_threshold
    ]

    return {
        "today": {
            "total_calls": len(today_calls),
            "completed_calls": len(completed_calls),
            "qualified_leads": len(qualified_leads),
            "total_cost": round(total_cost, 2),
            "average_cost_per_call": round(avg_cost, 3),
            "qualification_rate": len(qualified_leads) / max(len(completed_calls), 1),
        },
        "active": {
            "concurrent_calls": len(active_calls),
            "max_concurrent": settings.max_concurrent_calls,
            "utilization": len(active_calls) / settings.max_concurrent_calls
        },
        "budget": {
            "daily_budget": settings.daily_budget,
            "spent_today": round(total_cost, 2),
            "remaining": round(settings.daily_budget - total_cost, 2)
        }
    }


@router.websocket("/ws/{call_id}")
async def websocket_call_handler(websocket: WebSocket, call_id: str):
    """
    WebSocket endpoint for real-time call audio processing
    """
    await websocket.accept()

    try:
        if call_id not in active_calls:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": "Call not found"
            }))
            await websocket.close()
            return

        call_session = voice_pipeline.get_call_session(call_id)
        if not call_session:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": "Voice session not found"
            }))
            await websocket.close()
            return

        await websocket.send_text(json.dumps({
            "event": "connected",
            "call_id": call_id,
            "status": call_session.state.value
        }))

        # Handle WebSocket messages (audio data from Twilio)
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                event = data.get("event")

                if event == "media":
                    # Forward audio data to voice pipeline
                    # This would typically be handled by the Twilio integration
                    pass
                elif event == "start":
                    await websocket.send_text(json.dumps({
                        "event": "media_started",
                        "call_id": call_id
                    }))
                elif event == "stop":
                    await websocket.send_text(json.dumps({
                        "event": "media_stopped",
                        "call_id": call_id
                    }))
                    break

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "event": "error",
                    "message": "Invalid JSON format"
                }))

    except WebSocketDisconnect:
        # Handle client disconnect
        if call_id in active_calls:
            await voice_pipeline.end_call(call_id, "websocket_disconnect")
    except Exception as e:
        await websocket.send_text(json.dumps({
            "event": "error",
            "message": f"WebSocket error: {e}"
        }))
    finally:
        # Cleanup
        await websocket.close()


@router.post("/twilio/twiml/{call_id}")
async def generate_twiml(call_id: str):
    """
    Generate TwiML response for Twilio call setup
    """
    if call_id not in active_calls:
        raise HTTPException(status_code=404, detail="Call not found")

    # Generate TwiML that connects to our WebSocket
    twiml_response = voice_pipeline.twilio_integration.generate_twiml_response(call_id)

    return {
        "content": twiml_response,
        "media_type": "application/xml"
    }


@router.post("/twilio/status/{call_id}")
async def handle_twilio_status(call_id: str, status_data: dict):
    """
    Handle Twilio status callbacks
    """
    if call_id in active_calls:
        call_data = active_calls[call_id]
        twilio_status = status_data.get('CallStatus', 'unknown')

        # Update call status based on Twilio callback
        status_mapping = {
            'initiated': 'ringing',
            'ringing': 'ringing',
            'answered': 'in_progress',
            'completed': 'completed',
            'failed': 'failed',
            'busy': 'busy',
            'no-answer': 'no_answer'
        }

        new_status = status_mapping.get(twilio_status, 'unknown')
        call_data["status"] = new_status

        # Update voice pipeline if needed
        if new_status == "completed" or new_status == "failed":
            await voice_pipeline.end_call(call_id, new_status)
            # Move to history
            call_history.append(call_data.copy())
            del active_calls[call_id]

        # Handle status callback in Twilio integration
        voice_pipeline.twilio_integration.handle_status_callback(call_id, status_data)

    return {"status": "received"}


@router.post("/{call_id}/tier-switch")
async def switch_call_tier(call_id: str, target_tier: str):
    """
    Switch TTS tier for an active call
    """
    if call_id not in active_calls:
        raise HTTPException(status_code=404, detail="Call not found")

    try:
        tier_map = {
            "local": TTSProvider.LOCAL_PIPER,
            "premium": TTSProvider.ELEVENLABS,
            "elevenlabs": TTSProvider.ELEVENLABS
        }

        if target_tier not in tier_map:
            raise HTTPException(status_code=400, detail="Invalid tier specified")

        new_tier = tier_map[target_tier]
        success = await voice_pipeline.switch_tier(call_id, new_tier, "manual")

        if success:
            active_calls[call_id]["tier"] = new_tier.value
            return {
                "status": "success",
                "call_id": call_id,
                "new_tier": new_tier.value
            }
        else:
            raise HTTPException(status_code=500, detail="Tier switch failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error switching tier: {e}")


@router.get("/{call_id}/voice-metrics")
async def get_voice_metrics(call_id: str):
    """
    Get voice processing metrics for a call
    """
    if call_id not in active_calls:
        raise HTTPException(status_code=404, detail="Call not found")

    call_session = voice_pipeline.get_call_session(call_id)
    if not call_session:
        raise HTTPException(status_code=404, detail="Voice session not found")

    return {
        "call_id": call_id,
        "metrics": call_session.metrics.dict(),
        "state": call_session.state.value,
        "current_tier": call_session.current_tier.value,
        "tier_switches": call_session.metrics.tier_switches
    }