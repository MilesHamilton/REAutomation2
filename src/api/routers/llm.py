from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from ...llm import llm_service, Message, MessageRole, ConversationContext, RequestPriority

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    system_prompt: Optional[str] = None
    max_tokens: int = 150
    temperature: float = 0.7


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    usage_tokens: int
    response_time_ms: float
    confidence_score: Optional[float] = None


class StructuredRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    system_prompt: Optional[str] = None
    response_schema: Dict[str, Any]


class StructuredResponse(BaseModel):
    structured_data: Dict[str, Any]
    raw_response: str
    conversation_id: str
    usage_tokens: int
    response_time_ms: float


class QualificationRequest(BaseModel):
    conversation_id: str


class QualificationResponse(BaseModel):
    qualification_score: float
    confidence: float
    factors: Dict[str, float]
    reasoning: str
    recommended_action: str
    conversation_id: str


# Simple in-memory conversation storage (replace with proper database)
conversations: Dict[str, ConversationContext] = {}


def get_or_create_conversation(conversation_id: Optional[str] = None) -> ConversationContext:
    if conversation_id and conversation_id in conversations:
        return conversations[conversation_id]

    import uuid
    new_id = conversation_id or str(uuid.uuid4())

    context = ConversationContext(
        call_id=new_id,
        messages=[],
        conversation_state="active"
    )
    conversations[new_id] = context
    return context


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Generate a conversational response using the LLM
    """
    try:
        if not llm_service.is_ready():
            raise HTTPException(status_code=503, detail="LLM service not ready")

        context = get_or_create_conversation(request.conversation_id)

        response = await llm_service.generate_response(
            context=context,
            user_input=request.message,
            system_prompt=request.system_prompt
        )

        # Update conversation context
        context.messages.append(Message(role=MessageRole.USER, content=request.message))
        context.messages.append(Message(role=MessageRole.ASSISTANT, content=response.content))

        return ChatResponse(
            response=response.content,
            conversation_id=context.call_id,
            usage_tokens=response.usage_tokens,
            response_time_ms=response.response_time_ms,
            confidence_score=response.confidence_score
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {e}")


@router.post("/structured", response_model=StructuredResponse)
async def structured_generation(request: StructuredRequest):
    """
    Generate structured output using the LLM
    """
    try:
        if not llm_service.is_ready():
            raise HTTPException(status_code=503, detail="LLM service not ready")

        context = get_or_create_conversation(request.conversation_id)

        response = await llm_service.generate_structured_response(
            context=context,
            user_input=request.message,
            response_schema=request.response_schema,
            system_prompt=request.system_prompt
        )

        return StructuredResponse(
            structured_data=response.structured_data or {},
            raw_response=response.content,
            conversation_id=context.call_id,
            usage_tokens=response.usage_tokens,
            response_time_ms=response.response_time_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Structured generation failed: {e}")


@router.post("/qualify", response_model=QualificationResponse)
async def qualify_lead(request: QualificationRequest):
    """
    Analyze conversation for lead qualification
    """
    try:
        if not llm_service.is_ready():
            raise HTTPException(status_code=503, detail="LLM service not ready")

        if request.conversation_id not in conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")

        context = conversations[request.conversation_id]
        qualification_data = await llm_service.analyze_qualification(context)

        # Update context with qualification score
        context.qualification_score = qualification_data.get("qualification_score", 0.0)

        return QualificationResponse(
            qualification_score=qualification_data.get("qualification_score", 0.0),
            confidence=qualification_data.get("confidence", 0.0),
            factors=qualification_data.get("factors", {}),
            reasoning=qualification_data.get("reasoning", ""),
            recommended_action=qualification_data.get("recommended_action", "continue"),
            conversation_id=context.call_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lead qualification failed: {e}")


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Get conversation history and context
    """
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    context = conversations[conversation_id]
    return {
        "conversation_id": context.call_id,
        "messages": [msg.dict() for msg in context.messages],
        "qualification_score": context.qualification_score,
        "conversation_state": context.conversation_state,
        "lead_info": context.lead_info,
        "metadata": context.metadata
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation from memory
    """
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"status": "deleted", "conversation_id": conversation_id}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/conversations")
async def list_conversations():
    """
    List all active conversations
    """
    return {
        "conversations": [
            {
                "conversation_id": context.call_id,
                "message_count": len(context.messages),
                "qualification_score": context.qualification_score,
                "state": context.conversation_state
            }
            for context in conversations.values()
        ],
        "total_count": len(conversations)
    }


@router.get("/metrics")
async def get_llm_metrics():
    """
    Get detailed LLM service metrics including cache and queue performance
    """
    try:
        if not llm_service.is_ready():
            raise HTTPException(status_code=503, detail="LLM service not ready")

        metrics = await llm_service.get_detailed_metrics()
        return metrics

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {e}")


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear the LLM response cache
    """
    try:
        from ...llm import llm_cache
        success = await llm_cache.clear()

        if success:
            return {"status": "success", "message": "Cache cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear cache")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {e}")


@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache performance statistics
    """
    try:
        from ...llm import llm_cache
        stats = llm_cache.get_stats()
        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {e}")


@router.get("/queue/status")
async def get_queue_status():
    """
    Get current queue status and metrics
    """
    try:
        from ...llm import request_queue
        status = await request_queue.get_queue_status()
        return status

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {e}")


@router.post("/queue/clear")
async def clear_queue():
    """
    Clear all pending requests from the queue
    """
    try:
        from ...llm import request_queue
        cleared_count = await request_queue.clear_queues()

        return {
            "status": "success",
            "message": f"Cleared {cleared_count} pending requests"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear queue: {e}")


class PriorityRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    priority: str = "normal"  # low, normal, high, critical
    system_prompt: Optional[str] = None
    max_tokens: int = 150
    temperature: float = 0.7


@router.post("/chat/priority", response_model=ChatResponse)
async def chat_with_priority(request: PriorityRequest):
    """
    Generate a conversational response with specified priority
    """
    try:
        if not llm_service.is_ready():
            raise HTTPException(status_code=503, detail="LLM service not ready")

        context = get_or_create_conversation(request.conversation_id)

        # Map string priority to enum
        priority_map = {
            "low": RequestPriority.LOW,
            "normal": RequestPriority.NORMAL,
            "high": RequestPriority.HIGH,
            "critical": RequestPriority.CRITICAL
        }

        priority = priority_map.get(request.priority, RequestPriority.NORMAL)

        response = await llm_service.generate_response(
            context=context,
            user_input=request.message,
            system_prompt=request.system_prompt,
            priority=priority
        )

        # Update conversation context
        context.messages.append(Message(role=MessageRole.USER, content=request.message))
        context.messages.append(Message(role=MessageRole.ASSISTANT, content=response.content))

        return ChatResponse(
            response=response.content,
            conversation_id=context.call_id,
            usage_tokens=response.usage_tokens,
            response_time_ms=response.response_time_ms,
            confidence_score=response.confidence_score
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Priority chat generation failed: {e}")