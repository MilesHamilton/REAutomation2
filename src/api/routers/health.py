from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ...llm import llm_service
from ...config import settings

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check for all services
    """
    health_status = {
        "service": "re-automation-tts",
        "version": "0.1.0",
        "status": "healthy",
        "timestamp": 0,
        "components": {}
    }

    try:
        # Check LLM service
        llm_health = await llm_service.health_check()
        health_status["components"]["llm"] = {
            "status": llm_health.status,
            "response_time_ms": llm_health.response_time_ms,
            "concurrent_requests": llm_health.concurrent_requests,
            "details": llm_health.details
        }

        # Overall status based on component health
        component_statuses = [
            health_status["components"]["llm"]["status"]
        ]

        if any(status == "unhealthy" for status in component_statuses):
            health_status["status"] = "unhealthy"
        elif any(status == "degraded" for status in component_statuses):
            health_status["status"] = "degraded"

        import time
        health_status["timestamp"] = time.time()

        return health_status

    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        return health_status


@router.get("/llm")
async def llm_health():
    """
    Detailed health check for LLM service
    """
    try:
        health = await llm_service.health_check()
        metrics = await llm_service.get_metrics()

        return {
            "health": health.dict(),
            "metrics": metrics.dict(),
            "ready": llm_service.is_ready()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM service unavailable: {e}")


@router.get("/config")
async def config_info():
    """
    Return non-sensitive configuration information
    """
    return {
        "llm_model": settings.ollama_model,
        "max_concurrent_calls": settings.max_concurrent_calls,
        "max_concurrent_llm": settings.llm_max_concurrent,
        "tts_engine": settings.tts_engine,
        "debug": settings.debug,
        "metrics_enabled": settings.metrics_enabled
    }