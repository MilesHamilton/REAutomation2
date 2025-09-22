import asyncio
import aiohttp
import logging
import base64
import json
from typing import Optional, Dict, Any, AsyncGenerator
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import websockets

from ..config import settings
from .models import AudioChunk, AudioConfig

logger = logging.getLogger(__name__)


class TwilioIntegration:
    def __init__(self):
        self.client: Optional[Client] = None
        self.websocket_connections: Dict[str, Any] = {}
        self.audio_streams: Dict[str, asyncio.Queue] = {}
        self.is_initialized = False

    async def initialize(self):
        """Initialize Twilio client and services"""
        try:
            if not settings.twilio_account_sid or not settings.twilio_auth_token:
                logger.error("Twilio credentials not configured")
                return False

            # Initialize Twilio client
            self.client = Client(
                settings.twilio_account_sid,
                settings.twilio_auth_token
            )

            # Test connection
            try:
                account = self.client.api.account.fetch()
                logger.info(f"Twilio client initialized for account: {account.friendly_name}")
            except TwilioRestException as e:
                logger.error(f"Twilio authentication failed: {e}")
                return False

            self.is_initialized = True
            logger.info("Twilio integration initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Twilio initialization failed: {e}")
            return False

    async def start_call(self, call_id: str, phone_number: str) -> bool:
        """Start an outbound call"""
        try:
            if not self.is_initialized:
                logger.error("Twilio integration not initialized")
                return False

            # Clean phone number format
            clean_number = self._clean_phone_number(phone_number)
            if not clean_number:
                logger.error(f"Invalid phone number format: {phone_number}")
                return False

            # Create TwiML for the call with WebSocket connection
            twiml_url = self._generate_twiml_url(call_id)

            # Make the call
            call = self.client.calls.create(
                to=clean_number,
                from_=settings.twilio_phone_number,
                url=twiml_url,
                method='POST',
                status_callback=f"{self._get_base_url()}/twilio/status/{call_id}",
                status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
                status_callback_method='POST'
            )

            logger.info(f"Twilio call initiated: {call.sid} for call_id: {call_id}")

            # Set up audio stream queue for this call
            self.audio_streams[call_id] = asyncio.Queue()

            return True

        except TwilioRestException as e:
            logger.error(f"Twilio call creation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Error starting call {call_id}: {e}")
            return False

    def _clean_phone_number(self, phone_number: str) -> Optional[str]:
        """Clean and validate phone number format"""
        # Remove all non-digit characters
        digits = ''.join(filter(str.isdigit, phone_number))

        # Handle US numbers
        if len(digits) == 10:
            return f"+1{digits}"
        elif len(digits) == 11 and digits.startswith('1'):
            return f"+{digits}"
        elif phone_number.startswith('+'):
            return phone_number
        else:
            return None

    def _generate_twiml_url(self, call_id: str) -> str:
        """Generate TwiML URL for the call"""
        base_url = self._get_base_url()
        return f"{base_url}/twilio/twiml/{call_id}"

    def _get_base_url(self) -> str:
        """Get base URL for webhooks"""
        # In production, this would be your public domain
        # For development, you might use ngrok
        return f"http://localhost:{settings.api_port}"

    async def handle_websocket_connection(self, call_id: str, websocket_url: str):
        """Handle WebSocket connection for real-time audio"""
        try:
            async with websockets.connect(websocket_url) as websocket:
                self.websocket_connections[call_id] = websocket

                logger.info(f"WebSocket connected for call {call_id}")

                # Handle incoming audio messages
                async for message in websocket:
                    await self._process_websocket_message(call_id, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed for call {call_id}")
        except Exception as e:
            logger.error(f"WebSocket error for call {call_id}: {e}")
        finally:
            if call_id in self.websocket_connections:
                del self.websocket_connections[call_id]

    async def _process_websocket_message(self, call_id: str, message: str):
        """Process incoming WebSocket messages"""
        try:
            data = json.loads(message)
            event = data.get("event")

            if event == "media":
                # Process audio data
                media = data.get("media", {})
                payload = media.get("payload")
                timestamp = media.get("timestamp", "0")

                if payload:
                    # Decode base64 audio data
                    audio_data = base64.b64decode(payload)

                    # Create audio chunk
                    chunk = AudioChunk(
                        data=audio_data,
                        timestamp=float(timestamp) / 1000.0,  # Convert to seconds
                        chunk_id=int(timestamp),
                        sample_rate=8000,  # Twilio uses 8kHz µ-law
                        channels=1,
                        is_speech=True  # Assume speech for now
                    )

                    # Queue audio chunk for processing
                    if call_id in self.audio_streams:
                        await self.audio_streams[call_id].put(chunk)

            elif event == "start":
                logger.info(f"Call {call_id} audio stream started")

            elif event == "stop":
                logger.info(f"Call {call_id} audio stream stopped")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in WebSocket message for call {call_id}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message for call {call_id}: {e}")

    async def get_audio_stream(self, call_id: str) -> AsyncGenerator[AudioChunk, None]:
        """Get audio stream for a call"""
        if call_id not in self.audio_streams:
            logger.error(f"No audio stream found for call {call_id}")
            return

        queue = self.audio_streams[call_id]

        try:
            while True:
                # Wait for audio chunk
                chunk = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield chunk

        except asyncio.TimeoutError:
            # No audio received in timeout period, continue
            pass
        except Exception as e:
            logger.error(f"Error in audio stream for call {call_id}: {e}")

    async def play_audio(self, call_id: str, audio_data: bytes) -> bool:
        """Play audio through the call"""
        try:
            if call_id not in self.websocket_connections:
                logger.error(f"No WebSocket connection for call {call_id}")
                return False

            websocket = self.websocket_connections[call_id]

            # Convert audio to µ-law format (Twilio requirement)
            mulaw_data = self._convert_to_mulaw(audio_data)

            # Encode as base64
            payload = base64.b64encode(mulaw_data).decode('utf-8')

            # Create media message
            media_message = {
                "event": "media",
                "streamSid": call_id,
                "media": {
                    "payload": payload
                }
            }

            # Send through WebSocket
            await websocket.send(json.dumps(media_message))

            logger.debug(f"Audio sent to call {call_id}")
            return True

        except Exception as e:
            logger.error(f"Error playing audio for call {call_id}: {e}")
            return False

    def _convert_to_mulaw(self, pcm_data: bytes) -> bytes:
        """Convert PCM audio to µ-law format"""
        try:
            import audioop
            # Convert 16-bit PCM to µ-law
            return audioop.lin2ulaw(pcm_data, 2)
        except ImportError:
            logger.warning("audioop not available, returning PCM data as-is")
            return pcm_data
        except Exception as e:
            logger.error(f"Error converting audio to µ-law: {e}")
            return pcm_data

    async def end_call(self, call_id: str) -> bool:
        """End a call"""
        try:
            # Close WebSocket connection
            if call_id in self.websocket_connections:
                websocket = self.websocket_connections[call_id]
                await websocket.close()
                del self.websocket_connections[call_id]

            # Clean up audio stream
            if call_id in self.audio_streams:
                del self.audio_streams[call_id]

            logger.info(f"Call {call_id} ended and cleaned up")
            return True

        except Exception as e:
            logger.error(f"Error ending call {call_id}: {e}")
            return False

    def generate_twiml_response(self, call_id: str) -> str:
        """Generate TwiML response for call connection"""
        websocket_url = f"wss://localhost:{settings.api_port}/twilio/ws/{call_id}"

        twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">Hello, please hold while we connect you.</Say>
    <Connect>
        <Stream url="{websocket_url}">
            <Parameter name="callId" value="{call_id}"/>
        </Stream>
    </Connect>
</Response>'''
        return twiml

    def handle_status_callback(self, call_id: str, status_data: Dict[str, Any]):
        """Handle Twilio status callbacks"""
        try:
            call_status = status_data.get('CallStatus')
            call_sid = status_data.get('CallSid')

            logger.info(f"Call {call_id} (SID: {call_sid}) status: {call_status}")

            # You can add logic here to update call status in your system
            # For example, update database, trigger events, etc.

        except Exception as e:
            logger.error(f"Error handling status callback for call {call_id}: {e}")

    async def health_check(self) -> dict:
        """Check Twilio integration health"""
        try:
            if not self.is_initialized:
                return {
                    "status": "unhealthy",
                    "error": "Twilio integration not initialized"
                }

            # Test API connection
            account = self.client.api.account.fetch()

            return {
                "status": "healthy",
                "account_sid": account.sid,
                "account_name": account.friendly_name,
                "active_connections": len(self.websocket_connections),
                "active_streams": len(self.audio_streams),
                "phone_number": settings.twilio_phone_number
            }

        except TwilioRestException as e:
            return {
                "status": "unhealthy",
                "error": f"Twilio API error: {e}"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self):
        """Clean up Twilio integration resources"""
        try:
            # Close all WebSocket connections
            for call_id, websocket in self.websocket_connections.items():
                try:
                    await websocket.close()
                except Exception as e:
                    logger.error(f"Error closing WebSocket for call {call_id}: {e}")

            # Clear all data
            self.websocket_connections.clear()
            self.audio_streams.clear()
            self.is_initialized = False

            logger.info("Twilio integration cleanup complete")

        except Exception as e:
            logger.error(f"Error during Twilio cleanup: {e}")