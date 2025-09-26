"""
External CRM Connectors Service

This module provides connectors for external CRM systems,
supporting webhook notifications, data export, and synchronization.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid
import hmac
import hashlib
from dataclasses import dataclass, asdict
import os

import aiohttp
from pydantic import BaseModel
import jwt

logger = logging.getLogger(__name__)


class CRMError(Exception):
    """Base exception for CRM integration errors"""
    pass


class AuthenticationError(CRMError):
    """Exception raised for authentication failures"""
    pass


class WebhookDeliveryError(CRMError):
    """Exception raised for webhook delivery failures"""
    pass


class DataExportError(CRMError):
    """Exception raised for data export failures"""
    pass


class CRMType(str, Enum):
    """Supported CRM types"""
    SALESFORCE = "salesforce"
    HUBSPOT = "hubspot"
    PIPEDRIVE = "pipedrive"
    ZOHO = "zoho"
    CUSTOM = "custom"
    WEBHOOK_ONLY = "webhook_only"


class WebhookEvent(str, Enum):
    """Webhook event types"""
    CALL_STARTED = "call.started"
    CALL_ENDED = "call.ended"
    LEAD_QUALIFIED = "lead.qualified"
    APPOINTMENT_SCHEDULED = "appointment.scheduled"
    CALL_FAILED = "call.failed"
    AGENT_ESCALATION = "agent.escalation"
    COST_THRESHOLD = "cost.threshold"


class AuthType(str, Enum):
    """Authentication types"""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    CUSTOM = "custom"


@dataclass
class WebhookPayload:
    """Webhook payload structure"""
    event_type: WebhookEvent
    event_id: str
    timestamp: datetime
    call_id: str
    phone_number: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


class CRMConfig(BaseModel):
    """CRM configuration"""
    crm_type: CRMType
    name: str
    base_url: str
    auth_type: AuthType
    credentials: Dict[str, str]
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    enabled: bool = True
    rate_limit: int = 100  # requests per minute
    retry_attempts: int = 3
    timeout: float = 30.0
    custom_headers: Dict[str, str] = {}
    data_mapping: Dict[str, str] = {}  # Field mapping for data export


class WebhookDelivery:
    """Webhook delivery tracker"""

    def __init__(self):
        self.delivery_id: str = str(uuid.uuid4())
        self.event_id: str = ""
        self.webhook_url: str = ""
        self.payload: Dict[str, Any] = {}
        self.attempts: int = 0
        self.max_attempts: int = 3
        self.status: str = "pending"  # pending, delivered, failed
        self.last_attempt: Optional[datetime] = None
        self.next_retry: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.created_at: datetime = datetime.utcnow()


class CRMConnector:
    """
    Base CRM connector with common functionality for external integrations

    Handles authentication, data transformation, and error handling
    for various CRM systems.
    """

    def __init__(self, config: CRMConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._auth_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None

        # Rate limiting
        self._request_times: List[float] = []

        # Statistics
        self.stats = {
            "webhooks_sent": 0,
            "webhooks_failed": 0,
            "api_calls": 0,
            "api_errors": 0,
            "auth_renewals": 0,
            "data_exports": 0,
            "rate_limits_hit": 0
        }

    async def initialize(self) -> None:
        """Initialize the connector"""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            connector = aiohttp.TCPConnector(limit=100)

            headers = {
                "User-Agent": "REAutomation2-CRM-Connector/1.0",
                "Content-Type": "application/json",
                **self.config.custom_headers
            }

            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers
            )

        # Authenticate if needed
        await self._authenticate()

        logger.info(f"CRM connector initialized for {self.config.name}")

    async def cleanup(self) -> None:
        """Clean up resources"""
        if self._session:
            await self._session.close()
            self._session = None

    async def send_webhook(self, payload: WebhookPayload) -> bool:
        """
        Send webhook notification to CRM

        Args:
            payload: Webhook payload to send

        Returns:
            True if webhook was delivered successfully

        Raises:
            WebhookDeliveryError: If webhook delivery fails
        """
        if not self.config.webhook_url or not self.config.enabled:
            return False

        delivery = WebhookDelivery()
        delivery.event_id = payload.event_id
        delivery.webhook_url = self.config.webhook_url
        delivery.payload = payload.to_dict()
        delivery.max_attempts = self.config.retry_attempts

        return await self._deliver_webhook(delivery)

    async def export_lead_data(
        self,
        lead_data: Dict[str, Any],
        call_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Export lead and call data to CRM

        Args:
            lead_data: Lead information
            call_data: Call details and results

        Returns:
            Export result with CRM record ID and status

        Raises:
            DataExportError: If data export fails
        """
        if not self.config.enabled:
            return {"status": "disabled", "exported": False}

        try:
            # Apply data mapping
            mapped_data = self._map_data({**lead_data, **call_data})

            # Send to CRM based on type
            result = await self._export_to_crm(mapped_data)

            self.stats["data_exports"] += 1

            return {
                "status": "success",
                "exported": True,
                "crm_id": result.get("id"),
                "crm_url": result.get("url"),
                "mapped_fields": len(mapped_data)
            }

        except Exception as e:
            logger.error(f"Failed to export data to {self.config.name}: {e}")
            self.stats["api_errors"] += 1
            raise DataExportError(f"Data export failed: {e}")

    async def sync_contact_data(
        self,
        phone_number: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Synchronize contact data with CRM

        Args:
            phone_number: Contact identifier
            updates: Data to update

        Returns:
            True if sync was successful
        """
        if not self.config.enabled:
            return False

        try:
            # Find contact in CRM
            contact = await self._find_contact_by_phone(phone_number)

            if contact:
                # Update existing contact
                result = await self._update_contact(contact["id"], updates)
            else:
                # Create new contact
                result = await self._create_contact(phone_number, updates)

            return result.get("success", False)

        except Exception as e:
            logger.error(f"Failed to sync contact data: {e}")
            return False

    async def get_contact_data(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve contact data from CRM

        Args:
            phone_number: Phone number to search for

        Returns:
            Contact data if found, None otherwise
        """
        if not self.config.enabled:
            return None

        try:
            contact = await self._find_contact_by_phone(phone_number)
            return contact
        except Exception as e:
            logger.error(f"Failed to get contact data: {e}")
            return None

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of CRM connection"""
        try:
            if not self._session:
                await self.initialize()

            # Test authentication
            auth_valid = await self._test_authentication()

            # Test webhook endpoint if configured
            webhook_reachable = False
            if self.config.webhook_url:
                webhook_reachable = await self._test_webhook_endpoint()

            return {
                "status": "healthy" if auth_valid else "unhealthy",
                "crm_type": self.config.crm_type,
                "name": self.config.name,
                "authentication": "valid" if auth_valid else "invalid",
                "webhook_configured": bool(self.config.webhook_url),
                "webhook_reachable": webhook_reachable,
                "enabled": self.config.enabled,
                "stats": self.stats
            }

        except Exception as e:
            logger.error(f"CRM health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "crm_type": self.config.crm_type,
                "name": self.config.name
            }

    # Private methods
    async def _authenticate(self) -> None:
        """Handle authentication based on auth type"""
        if self.config.auth_type == AuthType.API_KEY:
            # API key is usually passed in headers
            pass
        elif self.config.auth_type == AuthType.OAUTH2:
            await self._oauth2_authenticate()
        elif self.config.auth_type == AuthType.BEARER_TOKEN:
            self._auth_token = self.config.credentials.get("token")
        elif self.config.auth_type == AuthType.BASIC_AUTH:
            # Basic auth is handled in request headers
            pass

    async def _oauth2_authenticate(self) -> None:
        """Handle OAuth2 authentication"""
        try:
            if self._token_expires and datetime.utcnow() < self._token_expires:
                return  # Token still valid

            auth_data = {
                "grant_type": "client_credentials",
                "client_id": self.config.credentials.get("client_id"),
                "client_secret": self.config.credentials.get("client_secret")
            }

            auth_url = f"{self.config.base_url}/oauth/token"

            async with self._session.post(auth_url, data=auth_data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self._auth_token = token_data.get("access_token")
                    expires_in = token_data.get("expires_in", 3600)
                    self._token_expires = datetime.utcnow() + timedelta(seconds=expires_in - 60)

                    self.stats["auth_renewals"] += 1
                    logger.debug(f"OAuth2 authentication successful for {self.config.name}")
                else:
                    raise AuthenticationError(f"OAuth2 authentication failed: {response.status}")

        except Exception as e:
            logger.error(f"OAuth2 authentication error: {e}")
            raise AuthenticationError(f"Authentication failed: {e}")

    async def _deliver_webhook(self, delivery: WebhookDelivery) -> bool:
        """Deliver webhook with retry logic"""
        while delivery.attempts < delivery.max_attempts:
            delivery.attempts += 1
            delivery.last_attempt = datetime.utcnow()

            try:
                await self._check_rate_limit()

                # Create webhook signature if secret is configured
                headers = {"Content-Type": "application/json"}
                payload_json = json.dumps(delivery.payload)

                if self.config.webhook_secret:
                    signature = hmac.new(
                        self.config.webhook_secret.encode(),
                        payload_json.encode(),
                        hashlib.sha256
                    ).hexdigest()
                    headers["X-Signature-SHA256"] = f"sha256={signature}"
                    headers["X-REAutomation-Event"] = delivery.payload.get("event_type", "")

                async with self._session.post(
                    delivery.webhook_url,
                    data=payload_json,
                    headers=headers
                ) as response:

                    if response.status in [200, 201, 202, 204]:
                        delivery.status = "delivered"
                        self.stats["webhooks_sent"] += 1
                        logger.debug(f"Webhook delivered: {delivery.delivery_id}")
                        return True
                    else:
                        error_text = await response.text()
                        delivery.error_message = f"HTTP {response.status}: {error_text}"

            except Exception as e:
                delivery.error_message = str(e)
                logger.error(f"Webhook delivery attempt {delivery.attempts} failed: {e}")

            # Calculate next retry time with exponential backoff
            if delivery.attempts < delivery.max_attempts:
                backoff_seconds = (2 ** delivery.attempts) * 60  # 2, 4, 8 minutes
                delivery.next_retry = datetime.utcnow() + timedelta(seconds=backoff_seconds)
                await asyncio.sleep(min(backoff_seconds, 300))  # Max 5 minutes

        delivery.status = "failed"
        self.stats["webhooks_failed"] += 1
        logger.error(f"Webhook delivery failed permanently: {delivery.delivery_id}")
        return False

    async def _export_to_crm(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Export data to specific CRM type"""
        await self._check_rate_limit()

        if self.config.crm_type == CRMType.SALESFORCE:
            return await self._export_to_salesforce(data)
        elif self.config.crm_type == CRMType.HUBSPOT:
            return await self._export_to_hubspot(data)
        elif self.config.crm_type == CRMType.PIPEDRIVE:
            return await self._export_to_pipedrive(data)
        elif self.config.crm_type == CRMType.CUSTOM:
            return await self._export_to_custom_crm(data)
        else:
            # Generic REST API export
            return await self._export_generic(data)

    async def _export_to_salesforce(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Export to Salesforce"""
        headers = self._get_auth_headers()
        endpoint = f"{self.config.base_url}/services/data/v52.0/sobjects/Lead/"

        # Map to Salesforce Lead object
        sf_data = {
            "FirstName": data.get("first_name", ""),
            "LastName": data.get("last_name", "Unknown"),
            "Company": data.get("company", "Unknown"),
            "Phone": data.get("phone_number"),
            "Email": data.get("email"),
            "LeadSource": "Voice AI Call",
            "Status": "Open - Not Contacted",
            "Description": f"Auto-generated from voice call. Call ID: {data.get('call_id')}"
        }

        async with self._session.post(endpoint, json=sf_data, headers=headers) as response:
            if response.status in [200, 201]:
                result = await response.json()
                return {"id": result.get("id"), "success": True}
            else:
                error = await response.text()
                raise DataExportError(f"Salesforce export failed: {error}")

    async def _export_to_hubspot(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Export to HubSpot"""
        headers = self._get_auth_headers()
        endpoint = f"{self.config.base_url}/crm/v3/objects/contacts"

        # Map to HubSpot Contact
        hs_data = {
            "properties": {
                "firstname": data.get("first_name", ""),
                "lastname": data.get("last_name", "Unknown"),
                "company": data.get("company", ""),
                "phone": data.get("phone_number"),
                "email": data.get("email"),
                "lifecyclestage": "lead",
                "lead_source": "Voice AI",
                "hs_lead_status": "NEW"
            }
        }

        async with self._session.post(endpoint, json=hs_data, headers=headers) as response:
            if response.status in [200, 201]:
                result = await response.json()
                return {"id": result.get("id"), "success": True}
            else:
                error = await response.text()
                raise DataExportError(f"HubSpot export failed: {error}")

    async def _export_to_pipedrive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Export to Pipedrive"""
        headers = self._get_auth_headers()

        # Create person first
        person_endpoint = f"{self.config.base_url}/v1/persons"
        person_data = {
            "name": f"{data.get('first_name', '')} {data.get('last_name', 'Unknown')}".strip(),
            "phone": [{"value": data.get("phone_number"), "primary": True}],
            "email": [{"value": data.get("email"), "primary": True}] if data.get("email") else [],
            "org_id": None  # Could be mapped if company info available
        }

        async with self._session.post(person_endpoint, json=person_data, headers=headers) as response:
            if response.status in [200, 201]:
                person_result = await response.json()
                person_id = person_result["data"]["id"]

                # Create deal
                deal_endpoint = f"{self.config.base_url}/v1/deals"
                deal_data = {
                    "title": f"Voice Call Lead - {data.get('phone_number')}",
                    "person_id": person_id,
                    "status": "open",
                    "stage_id": 1  # Usually first stage
                }

                async with self._session.post(deal_endpoint, json=deal_data, headers=headers) as deal_response:
                    if deal_response.status in [200, 201]:
                        deal_result = await deal_response.json()
                        return {"id": deal_result["data"]["id"], "person_id": person_id, "success": True}

            error = await response.text()
            raise DataExportError(f"Pipedrive export failed: {error}")

    async def _export_generic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generic REST API export"""
        headers = self._get_auth_headers()
        endpoint = f"{self.config.base_url}/leads"  # Assume generic endpoint

        async with self._session.post(endpoint, json=data, headers=headers) as response:
            if response.status in [200, 201]:
                result = await response.json()
                return {"id": result.get("id"), "success": True}
            else:
                error = await response.text()
                raise DataExportError(f"Generic CRM export failed: {error}")

    def _map_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data mapping configuration"""
        if not self.config.data_mapping:
            return data

        mapped_data = {}
        for source_field, target_field in self.config.data_mapping.items():
            if source_field in data:
                mapped_data[target_field] = data[source_field]

        # Include unmapped fields if no mapping exists for them
        for key, value in data.items():
            if key not in self.config.data_mapping:
                mapped_data[key] = value

        return mapped_data

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers based on auth type"""
        headers = {}

        if self.config.auth_type == AuthType.API_KEY:
            api_key = self.config.credentials.get("api_key")
            headers["Authorization"] = f"Bearer {api_key}"
        elif self.config.auth_type == AuthType.BEARER_TOKEN:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        elif self.config.auth_type == AuthType.BASIC_AUTH:
            import base64
            username = self.config.credentials.get("username")
            password = self.config.credentials.get("password")
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"

        return headers

    async def _check_rate_limit(self) -> None:
        """Enforce rate limiting"""
        import time

        now = time.time()
        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self.config.rate_limit:
            sleep_time = 60 - (now - self._request_times[0])
            if sleep_time > 0:
                self.stats["rate_limits_hit"] += 1
                await asyncio.sleep(sleep_time)
                self._request_times.pop(0)

        self._request_times.append(now)
        self.stats["api_calls"] += 1

    async def _test_authentication(self) -> bool:
        """Test if authentication is working"""
        try:
            headers = self._get_auth_headers()
            test_endpoint = f"{self.config.base_url}/user" if self.config.crm_type == CRMType.CUSTOM else f"{self.config.base_url}"

            async with self._session.get(test_endpoint, headers=headers) as response:
                return response.status in [200, 201, 204]
        except:
            return False

    async def _test_webhook_endpoint(self) -> bool:
        """Test if webhook endpoint is reachable"""
        try:
            test_payload = {
                "event_type": "test.connection",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"test": True}
            }

            async with self._session.post(
                self.config.webhook_url,
                json=test_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                return response.status < 500  # Accept any response except server errors
        except:
            return False

    # Contact management methods
    async def _find_contact_by_phone(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """Find contact by phone number - CRM-specific implementation"""
        # This would be implemented differently for each CRM
        return None

    async def _create_contact(self, phone_number: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new contact - CRM-specific implementation"""
        return {"success": False, "error": "Not implemented"}

    async def _update_contact(self, contact_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing contact - CRM-specific implementation"""
        return {"success": False, "error": "Not implemented"}


class CRMConnectorManager:
    """
    Manager for multiple CRM connectors

    Handles routing of webhooks and data exports to multiple CRM systems
    with load balancing and failover support.
    """

    def __init__(self):
        self.connectors: Dict[str, CRMConnector] = {}
        self.webhook_queue: asyncio.Queue = asyncio.Queue()
        self._worker_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

    async def add_connector(self, name: str, config: CRMConfig) -> None:
        """Add a CRM connector"""
        connector = CRMConnector(config)
        await connector.initialize()
        self.connectors[name] = connector
        logger.info(f"Added CRM connector: {name}")

    async def remove_connector(self, name: str) -> None:
        """Remove a CRM connector"""
        if name in self.connectors:
            await self.connectors[name].cleanup()
            del self.connectors[name]
            logger.info(f"Removed CRM connector: {name}")

    async def broadcast_webhook(self, payload: WebhookPayload) -> Dict[str, bool]:
        """Send webhook to all enabled connectors"""
        results = {}

        tasks = []
        for name, connector in self.connectors.items():
            if connector.config.enabled and connector.config.webhook_url:
                task = connector.send_webhook(payload)
                tasks.append((name, task))

        if tasks:
            completed_tasks = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )

            for (name, _), result in zip(tasks, completed_tasks):
                if isinstance(result, Exception):
                    results[name] = False
                    logger.error(f"Webhook failed for {name}: {result}")
                else:
                    results[name] = result

        return results

    async def export_to_all_crms(
        self,
        lead_data: Dict[str, Any],
        call_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Export data to all enabled CRM systems"""
        results = {}

        for name, connector in self.connectors.items():
            if connector.config.enabled:
                try:
                    result = await connector.export_lead_data(lead_data, call_data)
                    results[name] = result
                except Exception as e:
                    results[name] = {
                        "status": "error",
                        "exported": False,
                        "error": str(e)
                    }

        return results

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all connectors"""
        results = {}

        for name, connector in self.connectors.items():
            results[name] = await connector.health_check()

        return results

    async def cleanup(self) -> None:
        """Clean up all connectors and workers"""
        self._shutdown_event.set()

        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()

        # Wait for workers to finish
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        # Clean up connectors
        for connector in self.connectors.values():
            await connector.cleanup()

        self.connectors.clear()
        logger.info("CRM connector manager cleaned up")