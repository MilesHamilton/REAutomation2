"""
Scheduling System Integration

This module provides integration with calendar systems for appointment booking,
timezone handling, availability management, and meeting coordination.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta, timezone
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
import os
from zoneinfo import ZoneInfo

import aiohttp
from pydantic import BaseModel, validator
from icalendar import Calendar, Event as iCalEvent, vDDDTypes
import pytz

logger = logging.getLogger(__name__)


class SchedulingError(Exception):
    """Base exception for scheduling errors"""
    pass


class TimeSlotNotAvailableError(SchedulingError):
    """Exception raised when requested time slot is not available"""
    pass


class CalendarIntegrationError(SchedulingError):
    """Exception raised for calendar integration failures"""
    pass


class CalendarProvider(str, Enum):
    """Supported calendar providers"""
    GOOGLE_CALENDAR = "google_calendar"
    OUTLOOK = "outlook"
    CALENDLY = "calendly"
    ACUITY = "acuity"
    GENERIC_CALDAV = "caldav"
    INTERNAL = "internal"


class MeetingType(str, Enum):
    """Meeting types"""
    CONSULTATION = "consultation"
    PROPERTY_SHOWING = "property_showing"
    CLOSING = "closing"
    FOLLOW_UP = "follow_up"
    DISCOVERY_CALL = "discovery_call"


class AppointmentStatus(str, Enum):
    """Appointment status"""
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    NO_SHOW = "no_show"
    RESCHEDULED = "rescheduled"


@dataclass
class TimeSlot:
    """Available time slot"""
    start_time: datetime
    end_time: datetime
    timezone: str
    available: bool = True
    calendar_id: Optional[str] = None
    buffer_before: int = 15  # minutes
    buffer_after: int = 15   # minutes
    metadata: Optional[Dict[str, Any]] = None

    def duration_minutes(self) -> int:
        """Get slot duration in minutes"""
        return int((self.end_time - self.start_time).total_seconds() / 60)

    def overlaps_with(self, other: 'TimeSlot') -> bool:
        """Check if this slot overlaps with another"""
        return (self.start_time < other.end_time and
                self.end_time > other.start_time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "timezone": self.timezone,
            "available": self.available,
            "duration_minutes": self.duration_minutes(),
            "calendar_id": self.calendar_id,
            "buffer_before": self.buffer_before,
            "buffer_after": self.buffer_after,
            "metadata": self.metadata
        }


class AppointmentRequest(BaseModel):
    """Appointment booking request"""
    contact_name: str
    contact_email: str
    contact_phone: str
    meeting_type: MeetingType
    preferred_datetime: datetime
    timezone: str
    duration_minutes: int = 60
    notes: Optional[str] = None
    lead_data: Optional[Dict[str, Any]] = None
    call_id: Optional[str] = None

    @validator('preferred_datetime')
    def validate_datetime_future(cls, v):
        if v <= datetime.utcnow():
            raise ValueError('Preferred datetime must be in the future')
        return v

    @validator('duration_minutes')
    def validate_duration(cls, v):
        if v < 15 or v > 480:  # 15 minutes to 8 hours
            raise ValueError('Duration must be between 15 and 480 minutes')
        return v


@dataclass
class Appointment:
    """Scheduled appointment"""
    appointment_id: str
    contact_name: str
    contact_email: str
    contact_phone: str
    meeting_type: MeetingType
    start_time: datetime
    end_time: datetime
    timezone: str
    status: AppointmentStatus
    calendar_event_id: Optional[str] = None
    meeting_link: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    confirmation_sent: bool = False
    reminder_sent: bool = False
    lead_data: Optional[Dict[str, Any]] = None
    call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        result = asdict(self)
        # Handle datetime serialization
        for field in ['start_time', 'end_time', 'created_at', 'updated_at']:
            if result.get(field):
                result[field] = result[field].isoformat()
        return result

    def duration_minutes(self) -> int:
        """Get appointment duration in minutes"""
        return int((self.end_time - self.start_time).total_seconds() / 60)


class SchedulingProvider:
    """
    Base scheduling provider for calendar integration

    Handles calendar operations, availability checking, and appointment management
    """

    def __init__(
        self,
        provider: CalendarProvider,
        credentials: Dict[str, str],
        config: Optional[Dict[str, Any]] = None
    ):
        self.provider = provider
        self.credentials = credentials
        self.config = config or {}

        # Session for API calls
        self._session: Optional[aiohttp.ClientSession] = None

        # Business hours configuration
        self.business_hours = self.config.get("business_hours", {
            "monday": {"start": "09:00", "end": "17:00"},
            "tuesday": {"start": "09:00", "end": "17:00"},
            "wednesday": {"start": "09:00", "end": "17:00"},
            "thursday": {"start": "09:00", "end": "17:00"},
            "friday": {"start": "09:00", "end": "17:00"},
            "saturday": {"start": "10:00", "end": "14:00"},
            "sunday": None  # Closed
        })

        self.default_timezone = self.config.get("timezone", "America/New_York")
        self.advance_booking_days = self.config.get("advance_booking_days", 30)
        self.minimum_advance_hours = self.config.get("minimum_advance_hours", 2)

        # Statistics
        self.stats = {
            "appointments_created": 0,
            "appointments_cancelled": 0,
            "availability_checks": 0,
            "calendar_syncs": 0,
            "reminders_sent": 0,
            "api_errors": 0
        }

    async def initialize(self) -> None:
        """Initialize the scheduling provider"""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=30.0)
            self._session = aiohttp.ClientSession(timeout=timeout)

        await self._authenticate()

        logger.info(f"Scheduling provider initialized: {self.provider}")

    async def cleanup(self) -> None:
        """Clean up resources"""
        if self._session:
            await self._session.close()
            self._session = None

    async def get_availability(
        self,
        start_date: datetime,
        end_date: datetime,
        duration_minutes: int = 60,
        timezone_str: str = None
    ) -> List[TimeSlot]:
        """
        Get available time slots within date range

        Args:
            start_date: Start of availability window
            end_date: End of availability window
            duration_minutes: Required appointment duration
            timezone_str: Timezone for the slots

        Returns:
            List of available TimeSlot objects
        """
        timezone_str = timezone_str or self.default_timezone
        self.stats["availability_checks"] += 1

        try:
            # Get busy times from calendar
            busy_times = await self._get_busy_times(start_date, end_date)

            # Generate potential slots based on business hours
            potential_slots = self._generate_business_hour_slots(
                start_date, end_date, duration_minutes, timezone_str
            )

            # Filter out busy times
            available_slots = []
            for slot in potential_slots:
                if not self._slot_conflicts_with_busy_times(slot, busy_times):
                    available_slots.append(slot)

            return available_slots

        except Exception as e:
            logger.error(f"Failed to get availability: {e}")
            self.stats["api_errors"] += 1
            return []

    async def book_appointment(self, request: AppointmentRequest) -> Appointment:
        """
        Book an appointment

        Args:
            request: Appointment booking request

        Returns:
            Appointment object with booking details

        Raises:
            TimeSlotNotAvailableError: If requested time is not available
            SchedulingError: For other booking failures
        """
        try:
            # Validate time slot availability
            slot_end = request.preferred_datetime + timedelta(minutes=request.duration_minutes)
            requested_slot = TimeSlot(
                start_time=request.preferred_datetime,
                end_time=slot_end,
                timezone=request.timezone
            )

            # Check if slot is available
            availability = await self.get_availability(
                request.preferred_datetime - timedelta(hours=1),
                request.preferred_datetime + timedelta(hours=2),
                request.duration_minutes,
                request.timezone
            )

            slot_available = any(
                slot.start_time <= request.preferred_datetime and
                slot.end_time >= slot_end
                for slot in availability
            )

            if not slot_available:
                raise TimeSlotNotAvailableError(
                    f"Time slot {request.preferred_datetime} is not available"
                )

            # Create calendar event
            calendar_event = await self._create_calendar_event(request)

            # Generate appointment ID
            appointment_id = str(uuid.uuid4())

            # Create appointment object
            appointment = Appointment(
                appointment_id=appointment_id,
                contact_name=request.contact_name,
                contact_email=request.contact_email,
                contact_phone=request.contact_phone,
                meeting_type=request.meeting_type,
                start_time=request.preferred_datetime,
                end_time=slot_end,
                timezone=request.timezone,
                status=AppointmentStatus.SCHEDULED,
                calendar_event_id=calendar_event.get("id"),
                meeting_link=calendar_event.get("meeting_link"),
                location=calendar_event.get("location"),
                notes=request.notes,
                created_at=datetime.utcnow(),
                lead_data=request.lead_data,
                call_id=request.call_id
            )

            # Send confirmation
            await self._send_appointment_confirmation(appointment)

            self.stats["appointments_created"] += 1
            logger.info(f"Appointment booked: {appointment_id}")

            return appointment

        except TimeSlotNotAvailableError:
            raise
        except Exception as e:
            logger.error(f"Failed to book appointment: {e}")
            self.stats["api_errors"] += 1
            raise SchedulingError(f"Appointment booking failed: {e}")

    async def cancel_appointment(
        self,
        appointment_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Cancel an appointment

        Args:
            appointment_id: Appointment to cancel
            reason: Cancellation reason

        Returns:
            True if cancellation was successful
        """
        try:
            # Get appointment details
            appointment = await self.get_appointment(appointment_id)
            if not appointment:
                return False

            # Cancel calendar event
            if appointment.calendar_event_id:
                await self._cancel_calendar_event(appointment.calendar_event_id)

            # Send cancellation notification
            await self._send_appointment_cancellation(appointment, reason)

            self.stats["appointments_cancelled"] += 1
            logger.info(f"Appointment cancelled: {appointment_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to cancel appointment {appointment_id}: {e}")
            return False

    async def reschedule_appointment(
        self,
        appointment_id: str,
        new_datetime: datetime,
        new_duration: Optional[int] = None
    ) -> Optional[Appointment]:
        """
        Reschedule an existing appointment

        Args:
            appointment_id: Appointment to reschedule
            new_datetime: New appointment time
            new_duration: New duration (optional)

        Returns:
            Updated Appointment object if successful
        """
        try:
            # Get existing appointment
            old_appointment = await self.get_appointment(appointment_id)
            if not old_appointment:
                return None

            # Create new booking request
            duration = new_duration or old_appointment.duration_minutes()
            new_request = AppointmentRequest(
                contact_name=old_appointment.contact_name,
                contact_email=old_appointment.contact_email,
                contact_phone=old_appointment.contact_phone,
                meeting_type=old_appointment.meeting_type,
                preferred_datetime=new_datetime,
                timezone=old_appointment.timezone,
                duration_minutes=duration,
                notes=old_appointment.notes,
                lead_data=old_appointment.lead_data,
                call_id=old_appointment.call_id
            )

            # Cancel old appointment
            await self.cancel_appointment(appointment_id, "Rescheduled")

            # Book new appointment
            new_appointment = await self.book_appointment(new_request)
            new_appointment.status = AppointmentStatus.RESCHEDULED

            return new_appointment

        except Exception as e:
            logger.error(f"Failed to reschedule appointment {appointment_id}: {e}")
            return None

    async def get_appointment(self, appointment_id: str) -> Optional[Appointment]:
        """Get appointment by ID"""
        # This would typically query a database
        # For now, return None as this is provider-specific
        return None

    async def get_upcoming_appointments(
        self,
        days_ahead: int = 7
    ) -> List[Appointment]:
        """Get upcoming appointments"""
        try:
            end_date = datetime.utcnow() + timedelta(days=days_ahead)
            events = await self._get_calendar_events(datetime.utcnow(), end_date)

            appointments = []
            for event in events:
                appointment = self._event_to_appointment(event)
                if appointment:
                    appointments.append(appointment)

            return appointments

        except Exception as e:
            logger.error(f"Failed to get upcoming appointments: {e}")
            return []

    async def send_reminder(self, appointment: Appointment) -> bool:
        """Send appointment reminder"""
        try:
            # Send email reminder
            await self._send_email_reminder(appointment)

            # Send SMS reminder if phone number available
            if appointment.contact_phone:
                await self._send_sms_reminder(appointment)

            appointment.reminder_sent = True
            self.stats["reminders_sent"] += 1

            return True

        except Exception as e:
            logger.error(f"Failed to send reminder for {appointment.appointment_id}: {e}")
            return False

    # Private implementation methods
    async def _authenticate(self) -> None:
        """Authenticate with calendar provider"""
        if self.provider == CalendarProvider.GOOGLE_CALENDAR:
            await self._google_authenticate()
        elif self.provider == CalendarProvider.OUTLOOK:
            await self._outlook_authenticate()
        # Add other providers as needed

    async def _google_authenticate(self) -> None:
        """Google Calendar OAuth authentication"""
        # Implementation would handle OAuth flow
        pass

    async def _outlook_authenticate(self) -> None:
        """Outlook/Microsoft Graph authentication"""
        # Implementation would handle Microsoft OAuth
        pass

    async def _get_busy_times(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Get busy times from calendar"""
        try:
            if self.provider == CalendarProvider.GOOGLE_CALENDAR:
                return await self._get_google_busy_times(start_date, end_date)
            elif self.provider == CalendarProvider.OUTLOOK:
                return await self._get_outlook_busy_times(start_date, end_date)
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to get busy times: {e}")
            return []

    def _generate_business_hour_slots(
        self,
        start_date: datetime,
        end_date: datetime,
        duration_minutes: int,
        timezone_str: str
    ) -> List[TimeSlot]:
        """Generate time slots based on business hours"""
        slots = []
        tz = ZoneInfo(timezone_str)

        # Minimum advance time
        min_start = datetime.utcnow() + timedelta(hours=self.minimum_advance_hours)

        current_date = max(start_date.date(), min_start.date())
        end_date = min(end_date.date(),
                      (datetime.utcnow() + timedelta(days=self.advance_booking_days)).date())

        while current_date <= end_date.date():
            day_name = current_date.strftime("%A").lower()
            business_hours = self.business_hours.get(day_name)

            if business_hours:
                # Parse business hours
                start_time_str = business_hours["start"]  # "09:00"
                end_time_str = business_hours["end"]      # "17:00"

                start_hour, start_min = map(int, start_time_str.split(":"))
                end_hour, end_min = map(int, end_time_str.split(":"))

                # Create datetime objects
                day_start = datetime.combine(current_date, datetime.min.time().replace(
                    hour=start_hour, minute=start_min
                )).replace(tzinfo=tz)

                day_end = datetime.combine(current_date, datetime.min.time().replace(
                    hour=end_hour, minute=end_min
                )).replace(tzinfo=tz)

                # Generate slots for this day
                slot_start = day_start
                while slot_start + timedelta(minutes=duration_minutes) <= day_end:
                    slot_end = slot_start + timedelta(minutes=duration_minutes)

                    # Ensure slot is in the future
                    if slot_start >= min_start:
                        slots.append(TimeSlot(
                            start_time=slot_start,
                            end_time=slot_end,
                            timezone=timezone_str,
                            available=True
                        ))

                    # Move to next slot (typically 15 or 30 minute intervals)
                    slot_interval = self.config.get("slot_interval_minutes", 30)
                    slot_start += timedelta(minutes=slot_interval)

            current_date += timedelta(days=1)

        return slots

    def _slot_conflicts_with_busy_times(
        self,
        slot: TimeSlot,
        busy_times: List[Tuple[datetime, datetime]]
    ) -> bool:
        """Check if slot conflicts with busy times"""
        # Add buffer time
        buffered_start = slot.start_time - timedelta(minutes=slot.buffer_before)
        buffered_end = slot.end_time + timedelta(minutes=slot.buffer_after)

        for busy_start, busy_end in busy_times:
            if buffered_start < busy_end and buffered_end > busy_start:
                return True

        return False

    async def _create_calendar_event(self, request: AppointmentRequest) -> Dict[str, Any]:
        """Create calendar event for appointment"""
        event_data = {
            "summary": f"{request.meeting_type.value.title()} - {request.contact_name}",
            "description": f"Contact: {request.contact_name}\nPhone: {request.contact_phone}\nEmail: {request.contact_email}\n\nNotes: {request.notes or 'N/A'}",
            "start": {
                "dateTime": request.preferred_datetime.isoformat(),
                "timeZone": request.timezone
            },
            "end": {
                "dateTime": (request.preferred_datetime + timedelta(minutes=request.duration_minutes)).isoformat(),
                "timeZone": request.timezone
            },
            "attendees": [
                {"email": request.contact_email, "displayName": request.contact_name}
            ],
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "email", "minutes": 24 * 60},  # 24 hours
                    {"method": "popup", "minutes": 30}        # 30 minutes
                ]
            }
        }

        # Add meeting link if configured
        if self.config.get("create_meeting_links"):
            event_data["conferenceData"] = {
                "createRequest": {
                    "requestId": str(uuid.uuid4()),
                    "conferenceSolutionKey": {"type": "hangoutsMeet"}
                }
            }

        if self.provider == CalendarProvider.GOOGLE_CALENDAR:
            return await self._create_google_event(event_data)
        elif self.provider == CalendarProvider.OUTLOOK:
            return await self._create_outlook_event(event_data)
        else:
            return {"id": str(uuid.uuid4()), "status": "created"}

    async def _send_appointment_confirmation(self, appointment: Appointment) -> None:
        """Send appointment confirmation email/SMS"""
        # Implementation would send actual notifications
        logger.info(f"Sending confirmation for appointment {appointment.appointment_id}")

    async def _send_appointment_cancellation(
        self,
        appointment: Appointment,
        reason: Optional[str]
    ) -> None:
        """Send appointment cancellation notification"""
        logger.info(f"Sending cancellation for appointment {appointment.appointment_id}")

    async def _send_email_reminder(self, appointment: Appointment) -> None:
        """Send email reminder"""
        logger.info(f"Sending email reminder for appointment {appointment.appointment_id}")

    async def _send_sms_reminder(self, appointment: Appointment) -> None:
        """Send SMS reminder"""
        logger.info(f"Sending SMS reminder for appointment {appointment.appointment_id}")

    def _event_to_appointment(self, event: Dict[str, Any]) -> Optional[Appointment]:
        """Convert calendar event to Appointment object"""
        try:
            # This would parse calendar event and create Appointment
            # Implementation depends on calendar provider format
            return None
        except Exception as e:
            logger.error(f"Failed to convert event to appointment: {e}")
            return None

    # Provider-specific implementations
    async def _get_google_busy_times(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Get busy times from Google Calendar"""
        # Implementation for Google Calendar FreeBusy API
        return []

    async def _create_google_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create Google Calendar event"""
        # Implementation for Google Calendar Events API
        return {"id": str(uuid.uuid4()), "status": "created"}

    async def _get_outlook_busy_times(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Get busy times from Outlook Calendar"""
        # Implementation for Microsoft Graph Calendar API
        return []

    async def _create_outlook_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create Outlook calendar event"""
        # Implementation for Microsoft Graph Events API
        return {"id": str(uuid.uuid4()), "status": "created"}

    async def _get_calendar_events(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get calendar events in date range"""
        # Provider-specific implementation
        return []

    async def _cancel_calendar_event(self, event_id: str) -> bool:
        """Cancel calendar event"""
        # Provider-specific implementation
        logger.info(f"Cancelling calendar event: {event_id}")
        return True

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test authentication
            auth_valid = await self._test_authentication()

            return {
                "status": "healthy" if auth_valid else "unhealthy",
                "provider": self.provider,
                "authentication": "valid" if auth_valid else "invalid",
                "business_hours_configured": bool(self.business_hours),
                "stats": self.stats
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": self.provider
            }

    async def _test_authentication(self) -> bool:
        """Test authentication with calendar provider"""
        try:
            # Provider-specific authentication test
            return True
        except:
            return False