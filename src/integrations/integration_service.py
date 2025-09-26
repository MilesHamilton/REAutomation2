"""
Integration Service Manager

This module provides a unified interface for managing all external integrations
including ElevenLabs, Redis session management, CRM connectors, and scheduling.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import os

from .elevenlabs import ElevenLabsService, TTSRequest, TTSResponse
from .redis_session import RedisSessionManager, ConversationState, SessionConfig
from .crm_connectors import CRMConnectorManager, CRMConfig, WebhookPayload, WebhookEvent
from .scheduling import SchedulingProvider, AppointmentRequest, Appointment, CalendarProvider

logger = logging.getLogger(__name__)


@dataclass
class IntegrationHealth:
    """Health status for all integrations"""
    elevenlabs_healthy: bool = False
    redis_healthy: bool = False
    crm_healthy: bool = False
    scheduling_healthy: bool = False
    overall_healthy: bool = False
    last_check: Optional[datetime] = None
    details: Optional[Dict[str, Any]] = None


class IntegrationServiceManager:
    """
    Unified manager for all external integration services

    Provides a single interface for voice services, session management,
    CRM operations, and scheduling functionality.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Service instances
        self.elevenlabs: Optional[ElevenLabsService] = None
        self.redis_session: Optional[RedisSessionManager] = None
        self.crm_manager: Optional[CRMConnectorManager] = None
        self.scheduling: Optional[SchedulingProvider] = None

        # Health status
        self._health_status = IntegrationHealth()
        self._health_check_interval = 300  # 5 minutes
        self._health_check_task: Optional[asyncio.Task] = None

        # Service statistics
        self.stats = {
            "tts_requests": 0,
            "sessions_created": 0,
            "webhooks_sent": 0,
            "appointments_booked": 0,
            "errors": 0,
            "uptime_start": datetime.utcnow()
        }

        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize all integration services"""
        try:
            logger.info("Initializing integration services...")

            # Initialize ElevenLabs service
            await self._initialize_elevenlabs()

            # Initialize Redis session management
            await self._initialize_redis_session()

            # Initialize CRM connectors
            await self._initialize_crm_connectors()

            # Initialize scheduling service
            await self._initialize_scheduling()

            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_check_loop())

            # Perform initial health check
            await self.health_check()

            logger.info("Integration services initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize integration services: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up all services and resources"""
        logger.info("Cleaning up integration services...")

        # Stop health check task
        self._shutdown_event.set()
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Clean up services
        cleanup_tasks = []

        if self.elevenlabs:
            cleanup_tasks.append(self.elevenlabs.cleanup())

        if self.redis_session:
            cleanup_tasks.append(self.redis_session.cleanup())

        if self.crm_manager:
            cleanup_tasks.append(self.crm_manager.cleanup())

        if self.scheduling:
            cleanup_tasks.append(self.scheduling.cleanup())

        # Wait for all cleanup tasks
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        logger.info("Integration services cleaned up")

    # ElevenLabs TTS Operations
    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        voice_settings: Optional[Dict[str, float]] = None
    ) -> Optional[TTSResponse]:
        """
        Synthesize speech using ElevenLabs

        Args:
            text: Text to synthesize
            voice_id: Voice ID (optional)
            voice_settings: Voice configuration (optional)

        Returns:
            TTSResponse with audio data, None if service unavailable
        """
        if not self.elevenlabs:
            logger.warning("ElevenLabs service not available")
            return None

        try:
            response = await self.elevenlabs.synthesize_speech(
                text=text,
                voice_id=voice_id,
                voice_settings=voice_settings
            )

            self.stats["tts_requests"] += 1
            return response

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            self.stats["errors"] += 1
            return None

    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available ElevenLabs voices"""
        if not self.elevenlabs:
            return []

        try:
            voices = await self.elevenlabs.get_available_voices()
            return [voice.dict() for voice in voices]
        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            return []

    async def estimate_tts_cost(self, text: str) -> float:
        """Estimate TTS cost for given text"""
        if not self.elevenlabs:
            return 0.0

        return self.elevenlabs.estimate_cost(text)

    # Session Management Operations
    async def create_session(
        self,
        call_id: str,
        phone_number: str,
        initial_data: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> Optional[ConversationState]:
        """
        Create a new conversation session

        Args:
            call_id: Unique call identifier
            phone_number: Contact phone number
            initial_data: Initial session data
            ttl: Session time-to-live

        Returns:
            ConversationState if successful, None otherwise
        """
        if not self.redis_session:
            logger.warning("Redis session service not available")
            return None

        try:
            session = await self.redis_session.create_session(
                call_id=call_id,
                phone_number=phone_number,
                initial_data=initial_data,
                ttl=ttl
            )

            self.stats["sessions_created"] += 1
            return session

        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            self.stats["errors"] += 1
            return None

    async def get_session(self, call_id: str) -> Optional[ConversationState]:
        """Get session by call ID"""
        if not self.redis_session:
            return None

        try:
            return await self.redis_session.get_session(call_id)
        except Exception as e:
            logger.error(f"Session retrieval failed: {e}")
            return None

    async def update_session(
        self,
        call_id: str,
        updates: Dict[str, Any],
        extend_ttl: bool = True
    ) -> bool:
        """Update session data"""
        if not self.redis_session:
            return False

        try:
            return await self.redis_session.update_session(
                call_id=call_id,
                updates=updates,
                extend_ttl=extend_ttl
            )
        except Exception as e:
            logger.error(f"Session update failed: {e}")
            return False

    async def delete_session(self, call_id: str) -> bool:
        """Delete session"""
        if not self.redis_session:
            return False

        try:
            return await self.redis_session.delete_session(call_id)
        except Exception as e:
            logger.error(f"Session deletion failed: {e}")
            return False

    async def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        if not self.redis_session:
            return []

        try:
            return await self.redis_session.get_active_sessions()
        except Exception as e:
            logger.error(f"Failed to get active sessions: {e}")
            return []

    # CRM Operations
    async def send_webhook(
        self,
        event_type: WebhookEvent,
        call_id: str,
        phone_number: str,
        data: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Send webhook to all configured CRM systems

        Args:
            event_type: Type of webhook event
            call_id: Call identifier
            phone_number: Contact phone number
            data: Event data

        Returns:
            Dictionary mapping CRM name to success status
        """
        if not self.crm_manager:
            logger.warning("CRM manager not available")
            return {}

        try:
            import uuid
            payload = WebhookPayload(
                event_type=event_type,
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                call_id=call_id,
                phone_number=phone_number,
                data=data
            )

            results = await self.crm_manager.broadcast_webhook(payload)
            self.stats["webhooks_sent"] += len([r for r in results.values() if r])

            return results

        except Exception as e:
            logger.error(f"Webhook broadcast failed: {e}")
            self.stats["errors"] += 1
            return {}

    async def export_lead_data(
        self,
        lead_data: Dict[str, Any],
        call_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Export lead data to all CRM systems

        Args:
            lead_data: Lead information
            call_data: Call details

        Returns:
            Dictionary mapping CRM name to export results
        """
        if not self.crm_manager:
            return {}

        try:
            return await self.crm_manager.export_to_all_crms(lead_data, call_data)
        except Exception as e:
            logger.error(f"Lead data export failed: {e}")
            self.stats["errors"] += 1
            return {}

    # Scheduling Operations
    async def get_availability(
        self,
        start_date: datetime,
        end_date: datetime,
        duration_minutes: int = 60,
        timezone_str: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get available appointment slots

        Args:
            start_date: Start of availability window
            end_date: End of availability window
            duration_minutes: Required appointment duration
            timezone_str: Timezone for slots

        Returns:
            List of available time slot dictionaries
        """
        if not self.scheduling:
            logger.warning("Scheduling service not available")
            return []

        try:
            slots = await self.scheduling.get_availability(
                start_date=start_date,
                end_date=end_date,
                duration_minutes=duration_minutes,
                timezone_str=timezone_str
            )

            return [slot.to_dict() for slot in slots]

        except Exception as e:
            logger.error(f"Failed to get availability: {e}")
            return []

    async def book_appointment(
        self,
        contact_name: str,
        contact_email: str,
        contact_phone: str,
        preferred_datetime: datetime,
        meeting_type: str = "consultation",
        duration_minutes: int = 60,
        timezone_str: str = "America/New_York",
        notes: Optional[str] = None,
        lead_data: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Book an appointment

        Args:
            contact_name: Contact name
            contact_email: Contact email
            contact_phone: Contact phone number
            preferred_datetime: Preferred appointment time
            meeting_type: Type of meeting
            duration_minutes: Meeting duration
            timezone_str: Timezone
            notes: Additional notes
            lead_data: Lead information
            call_id: Associated call ID

        Returns:
            Appointment dictionary if successful, None otherwise
        """
        if not self.scheduling:
            logger.warning("Scheduling service not available")
            return None

        try:
            from .scheduling import MeetingType

            request = AppointmentRequest(
                contact_name=contact_name,
                contact_email=contact_email,
                contact_phone=contact_phone,
                meeting_type=MeetingType(meeting_type),
                preferred_datetime=preferred_datetime,
                timezone=timezone_str,
                duration_minutes=duration_minutes,
                notes=notes,
                lead_data=lead_data,
                call_id=call_id
            )

            appointment = await self.scheduling.book_appointment(request)
            self.stats["appointments_booked"] += 1

            # Send webhook notification about appointment
            if appointment and self.crm_manager:
                await self.send_webhook(
                    WebhookEvent.APPOINTMENT_SCHEDULED,
                    call_id or "unknown",
                    contact_phone,
                    appointment.to_dict()
                )

            return appointment.to_dict() if appointment else None

        except Exception as e:
            logger.error(f"Appointment booking failed: {e}")
            self.stats["errors"] += 1
            return None

    async def cancel_appointment(
        self,
        appointment_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """Cancel an appointment"""
        if not self.scheduling:
            return False

        try:
            return await self.scheduling.cancel_appointment(appointment_id, reason)
        except Exception as e:
            logger.error(f"Appointment cancellation failed: {e}")
            return False

    # Health and Monitoring
    async def health_check(self) -> IntegrationHealth:
        """Perform comprehensive health check of all services"""
        try:
            health_checks = {}

            # Check ElevenLabs
            if self.elevenlabs:
                elevenlabs_health = await self.elevenlabs.health_check()
                health_checks["elevenlabs"] = elevenlabs_health
                self._health_status.elevenlabs_healthy = elevenlabs_health.get("status") == "healthy"
            else:
                self._health_status.elevenlabs_healthy = False

            # Check Redis session manager
            if self.redis_session:
                redis_health = await self.redis_session.health_check()
                health_checks["redis"] = redis_health
                self._health_status.redis_healthy = redis_health.get("status") == "healthy"
            else:
                self._health_status.redis_healthy = False

            # Check CRM connectors
            if self.crm_manager:
                crm_health = await self.crm_manager.health_check_all()
                health_checks["crm"] = crm_health
                self._health_status.crm_healthy = any(
                    h.get("status") == "healthy" for h in crm_health.values()
                )
            else:
                self._health_status.crm_healthy = False

            # Check scheduling service
            if self.scheduling:
                scheduling_health = await self.scheduling.health_check()
                health_checks["scheduling"] = scheduling_health
                self._health_status.scheduling_healthy = scheduling_health.get("status") == "healthy"
            else:
                self._health_status.scheduling_healthy = False

            # Overall health
            self._health_status.overall_healthy = (
                self._health_status.elevenlabs_healthy and
                self._health_status.redis_healthy and
                (self._health_status.crm_healthy or len(health_checks.get("crm", {})) == 0) and
                (self._health_status.scheduling_healthy or not self.scheduling)
            )

            self._health_status.last_check = datetime.utcnow()
            self._health_status.details = health_checks

            return self._health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._health_status.overall_healthy = False
            self._health_status.last_check = datetime.utcnow()
            return self._health_status

    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        uptime = datetime.utcnow() - self.stats["uptime_start"]

        stats = {
            **self.stats,
            "uptime_seconds": uptime.total_seconds(),
            "uptime_hours": uptime.total_seconds() / 3600,
            "health_status": {
                "elevenlabs": self._health_status.elevenlabs_healthy,
                "redis": self._health_status.redis_healthy,
                "crm": self._health_status.crm_healthy,
                "scheduling": self._health_status.scheduling_healthy,
                "overall": self._health_status.overall_healthy,
                "last_check": self._health_status.last_check.isoformat() if self._health_status.last_check else None
            }
        }

        # Add service-specific stats
        if self.elevenlabs:
            stats["elevenlabs_stats"] = self.elevenlabs.get_service_stats()

        if self.redis_session:
            redis_stats = asyncio.create_task(self.redis_session.get_session_stats())
            # Note: In a real implementation, you'd handle this async call properly

        if self.scheduling:
            stats["scheduling_stats"] = self.scheduling.stats

        return stats

    # Private initialization methods
    async def _initialize_elevenlabs(self) -> None:
        """Initialize ElevenLabs service"""
        elevenlabs_config = self.config.get("elevenlabs", {})

        if elevenlabs_config.get("enabled", True):
            try:
                self.elevenlabs = ElevenLabsService(
                    api_key=elevenlabs_config.get("api_key"),
                    max_requests_per_minute=elevenlabs_config.get("rate_limit", 120),
                    default_voice_id=elevenlabs_config.get("default_voice", "21m00Tcm4TlvDq8ikWAM")
                )

                await self.elevenlabs.initialize()
                logger.info("ElevenLabs service initialized")

            except Exception as e:
                logger.warning(f"ElevenLabs initialization failed: {e}")
                self.elevenlabs = None

    async def _initialize_redis_session(self) -> None:
        """Initialize Redis session manager"""
        redis_config = self.config.get("redis", {})

        if redis_config.get("enabled", True):
            try:
                session_config = SessionConfig(
                    default_ttl=redis_config.get("default_ttl", 3600),
                    max_ttl=redis_config.get("max_ttl", 86400),
                    cleanup_interval=redis_config.get("cleanup_interval", 300)
                )

                self.redis_session = RedisSessionManager(
                    redis_url=redis_config.get("url"),
                    redis_host=redis_config.get("host", "localhost"),
                    redis_port=redis_config.get("port", 6379),
                    redis_db=redis_config.get("db", 0),
                    redis_password=redis_config.get("password"),
                    config=session_config
                )

                await self.redis_session.initialize()
                logger.info("Redis session manager initialized")

            except Exception as e:
                logger.warning(f"Redis session manager initialization failed: {e}")
                self.redis_session = None

    async def _initialize_crm_connectors(self) -> None:
        """Initialize CRM connectors"""
        crm_configs = self.config.get("crm_connectors", [])

        if crm_configs:
            try:
                self.crm_manager = CRMConnectorManager()

                for crm_config_dict in crm_configs:
                    if crm_config_dict.get("enabled", True):
                        crm_config = CRMConfig(**crm_config_dict)
                        await self.crm_manager.add_connector(
                            crm_config.name,
                            crm_config
                        )

                logger.info(f"CRM connectors initialized: {len(self.crm_manager.connectors)}")

            except Exception as e:
                logger.warning(f"CRM connectors initialization failed: {e}")
                self.crm_manager = None

    async def _initialize_scheduling(self) -> None:
        """Initialize scheduling service"""
        scheduling_config = self.config.get("scheduling", {})

        if scheduling_config.get("enabled", False):
            try:
                provider = CalendarProvider(scheduling_config.get("provider", "internal"))
                credentials = scheduling_config.get("credentials", {})

                self.scheduling = SchedulingProvider(
                    provider=provider,
                    credentials=credentials,
                    config=scheduling_config
                )

                await self.scheduling.initialize()
                logger.info("Scheduling service initialized")

            except Exception as e:
                logger.warning(f"Scheduling service initialization failed: {e}")
                self.scheduling = None

    async def _health_check_loop(self) -> None:
        """Background health check loop"""
        while not self._shutdown_event.is_set():
            try:
                await self.health_check()
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._health_check_interval
                )
            except asyncio.TimeoutError:
                continue  # Normal timeout, continue health checks
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    # Context manager support
    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()