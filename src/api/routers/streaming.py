"""
Streaming API endpoints for real-time LLM responses

Provides SSE (Server-Sent Events) and WebSocket endpoints for streaming
LLM responses with low latency.
"""

import asyncio
import json
import logging
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...llm.service import llm_service
from ...llm.models import ConversationContext, Message, MessageRole
from ...monitoring.performance import performance_monitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/streaming", tags=["streaming"])


class StreamRequest(BaseModel):
    """Request model for streaming endpoint."""
    call_id: str = Field(..., description="Unique call identifier")
    user_input: str = Field(..., description="User message to stream response for")
    conversation_history: list = Field(default_factory=list, description="Conversation history")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    for_voice: bool = Field(default=False, description="Optimize chunks for voice synthesis")
    lead_info: dict = Field(default_factory=dict, description="Lead information")
    conversation_state: str = Field(default="active", description="Conversation state")


class StreamChunk(BaseModel):
    """Stream chunk model."""
    type: str = Field(..., description="Chunk type: 'chunk', 'done', 'error'")
    content: Optional[str] = Field(None, description="Chunk content")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


@router.post("/sse")
async def stream_sse(request: StreamRequest):
    """
    Stream LLM response using Server-Sent Events (SSE).

    SSE is ideal for unidirectional streaming from server to client.
    Compatible with standard HTTP clients.

    Returns:
        StreamingResponse: SSE stream of response chunks
    """
    if not llm_service.is_ready():
        raise HTTPException(status_code=503, detail="LLM service not ready")

    async def event_generator():
        """Generate SSE events from LLM stream."""
        try:
            # Create conversation context
            messages = []
            for msg in request.conversation_history:
                messages.append(Message(
                    role=MessageRole(msg.get("role", "user")),
                    content=msg.get("content", ""),
                    timestamp=msg.get("timestamp")
                ))

            context = ConversationContext(
                call_id=request.call_id,
                messages=messages,
                lead_info=request.lead_info,
                conversation_state=request.conversation_state
            )

            # Start timer for metrics
            timer_key = performance_monitor.start_timer("streaming_sse", request.call_id)

            # Stream response
            chunk_count = 0
            async for chunk in llm_service.generate_response_streaming(
                context=context,
                user_input=request.user_input,
                system_prompt=request.system_prompt,
                for_voice=request.for_voice
            ):
                chunk_count += 1

                # Format as SSE event
                chunk_data = StreamChunk(
                    type="chunk",
                    content=chunk,
                    metadata={"chunk_index": chunk_count}
                )

                # SSE format: "data: {json}\n\n"
                yield f"data: {chunk_data.json()}\n\n"

                # Small delay to prevent overwhelming client
                await asyncio.sleep(0.01)

            # Send completion event
            done_data = StreamChunk(
                type="done",
                content=None,
                metadata={"total_chunks": chunk_count}
            )
            yield f"data: {done_data.json()}\n\n"

            # End timer
            performance_monitor.end_timer(timer_key, tags={"method": "sse"})

            logger.info(f"SSE stream completed for call {request.call_id}: {chunk_count} chunks")

        except Exception as e:
            logger.error(f"SSE streaming error for call {request.call_id}: {e}")

            # Send error event
            error_data = StreamChunk(
                type="error",
                content=None,
                metadata={"error": str(e)}
            )
            yield f"data: {error_data.json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.websocket("/ws/{call_id}")
async def stream_websocket(websocket: WebSocket, call_id: str):
    """
    Stream LLM response using WebSocket.

    WebSocket provides bidirectional communication with lower overhead
    than SSE. Ideal for interactive applications.

    Args:
        websocket: WebSocket connection
        call_id: Unique call identifier
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established for call {call_id}")

    try:
        # Wait for initial request
        request_data = await websocket.receive_json()

        user_input = request_data.get("user_input")
        if not user_input:
            await websocket.send_json({
                "type": "error",
                "error": "user_input is required"
            })
            await websocket.close()
            return

        # Parse conversation history
        messages = []
        for msg in request_data.get("conversation_history", []):
            messages.append(Message(
                role=MessageRole(msg.get("role", "user")),
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp")
            ))

        # Create context
        context = ConversationContext(
            call_id=call_id,
            messages=messages,
            lead_info=request_data.get("lead_info", {}),
            conversation_state=request_data.get("conversation_state", "active")
        )

        # Start timer
        timer_key = performance_monitor.start_timer("streaming_websocket", call_id)

        # Stream response
        chunk_count = 0
        for_voice = request_data.get("for_voice", False)

        async for chunk in llm_service.generate_response_streaming(
            context=context,
            user_input=user_input,
            system_prompt=request_data.get("system_prompt"),
            for_voice=for_voice
        ):
            chunk_count += 1

            # Send chunk via WebSocket
            await websocket.send_json({
                "type": "chunk",
                "content": chunk,
                "metadata": {
                    "chunk_index": chunk_count,
                    "call_id": call_id
                }
            })

        # Send completion message
        await websocket.send_json({
            "type": "done",
            "metadata": {
                "total_chunks": chunk_count,
                "call_id": call_id
            }
        })

        # End timer
        performance_monitor.end_timer(timer_key, tags={"method": "websocket"})

        logger.info(f"WebSocket stream completed for call {call_id}: {chunk_count} chunks")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for call {call_id}")

    except Exception as e:
        logger.error(f"WebSocket streaming error for call {call_id}: {e}")

        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e),
                "metadata": {"call_id": call_id}
            })
        except:
            pass  # Connection might be closed

    finally:
        try:
            await websocket.close()
        except:
            pass


@router.get("/metrics")
async def get_streaming_metrics():
    """
    Get streaming performance metrics.

    Returns:
        dict: Streaming metrics including TTFC, throughput, etc.
    """
    if not llm_service.is_ready():
        raise HTTPException(status_code=503, detail="LLM service not ready")

    metrics = llm_service.get_streaming_metrics()

    return {
        "streaming_metrics": metrics,
        "status": "active" if metrics.get("total_streams", 0) > 0 else "inactive"
    }


@router.get("/health")
async def streaming_health():
    """
    Check streaming endpoint health.

    Returns:
        dict: Health status
    """
    return {
        "status": "healthy" if llm_service.is_ready() else "unhealthy",
        "service": "streaming",
        "endpoints": {
            "sse": "/streaming/sse",
            "websocket": "/streaming/ws/{call_id}",
            "metrics": "/streaming/metrics"
        }
    }
