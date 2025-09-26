import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import settings
from ..llm import llm_service
from .routers import llm, health, calls, integrations, monitoring
from .middleware import RequestLoggingMiddleware, MetricsMiddleware
from ..monitoring.performance import performance_monitor
from ..monitoring.alerts import alert_manager
from ..monitoring.dashboard import dashboard_manager

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting REAutomation TTS API server...")

    # Initialize LLM service
    startup_success = await llm_service.startup()
    if not startup_success:
        logger.error("Failed to initialize LLM service")
        raise RuntimeError("LLM service initialization failed")

    # Initialize monitoring systems
    try:
        logger.info("Initializing monitoring systems...")
        await performance_monitor.initialize()
        await alert_manager.initialize()
        await dashboard_manager.start_real_time_updates()
        logger.info("Monitoring systems initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize monitoring systems: {e}")
        # Don't fail the entire startup if monitoring fails
        pass

    logger.info("All services started successfully")

    yield

    # Shutdown
    logger.info("Shutting down services...")
    await llm_service.shutdown()

    # Shutdown monitoring systems
    try:
        logger.info("Shutting down monitoring systems...")
        await performance_monitor.shutdown()
        await alert_manager.shutdown()
        logger.info("Monitoring systems shutdown complete")
    except Exception as e:
        logger.error(f"Error shutting down monitoring systems: {e}")

    logger.info("Shutdown complete")


# Create FastAPI app with lifespan management
app = FastAPI(
    title="REAutomation Two-Tier TTS API",
    description="Intelligent Two-Tier Voice Lead Generation System",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(RequestLoggingMiddleware)
if settings.metrics_enabled:
    app.add_middleware(MetricsMiddleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(llm.router, prefix="/llm", tags=["llm"])
app.include_router(calls.router, prefix="/calls", tags=["calls"])
app.include_router(integrations.router, tags=["integrations"])
app.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])


@app.get("/")
async def root():
    return {
        "service": "REAutomation Two-Tier TTS API",
        "version": "0.1.0",
        "status": "operational",
        "docs": "/docs" if settings.debug else "disabled"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
