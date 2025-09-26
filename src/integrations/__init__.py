"""
Integration services for external data sources and APIs

This package provides comprehensive integration services for:
- ElevenLabs premium TTS service
- Redis session management
- External CRM connectors (Salesforce, HubSpot, Pipedrive, etc.)
- Calendar and scheduling systems
"""

from .elevenlabs import (
    ElevenLabsService,
    TTSRequest,
    TTSResponse,
    VoiceModel,
    ElevenLabsError,
    RateLimitError,
    QuotaExceededError
)

from .redis_session import (
    RedisSessionManager,
    ConversationState,
    SessionConfig,
    SessionError,
    SessionNotFoundError,
    SessionExpiredError
)

from .crm_connectors import (
    CRMConnector,
    CRMConnectorManager,
    CRMConfig,
    WebhookPayload,
    WebhookEvent,
    CRMType,
    AuthType,
    CRMError,
    AuthenticationError,
    WebhookDeliveryError,
    DataExportError
)

from .scheduling import (
    SchedulingProvider,
    AppointmentRequest,
    Appointment,
    TimeSlot,
    CalendarProvider,
    MeetingType,
    AppointmentStatus,
    SchedulingError,
    TimeSlotNotAvailableError,
    CalendarIntegrationError
)

from .integration_service import (
    IntegrationServiceManager,
    IntegrationHealth
)

__all__ = [
    # ElevenLabs
    "ElevenLabsService",
    "TTSRequest",
    "TTSResponse",
    "VoiceModel",
    "ElevenLabsError",
    "RateLimitError",
    "QuotaExceededError",

    # Redis Session Management
    "RedisSessionManager",
    "ConversationState",
    "SessionConfig",
    "SessionError",
    "SessionNotFoundError",
    "SessionExpiredError",

    # CRM Connectors
    "CRMConnector",
    "CRMConnectorManager",
    "CRMConfig",
    "WebhookPayload",
    "WebhookEvent",
    "CRMType",
    "AuthType",
    "CRMError",
    "AuthenticationError",
    "WebhookDeliveryError",
    "DataExportError",

    # Scheduling
    "SchedulingProvider",
    "AppointmentRequest",
    "Appointment",
    "TimeSlot",
    "CalendarProvider",
    "MeetingType",
    "AppointmentStatus",
    "SchedulingError",
    "TimeSlotNotAvailableError",
    "CalendarIntegrationError",

    # Integration Manager
    "IntegrationServiceManager",
    "IntegrationHealth"
]
