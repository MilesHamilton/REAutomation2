"""
Integration Service Layer

This module provides a service layer that integrates Google Sheets with the agent system,
handling contact management, call result tracking, and status updates.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from .google_sheets import GoogleSheetsClient
from .models import Contact, CallResult, ContactStatus, ContactFilter, SheetsOperationResult
from ..agents.models import WorkflowContext, AnalyticsMetrics
from ..config.settings import settings

logger = logging.getLogger(__name__)


class IntegrationService:
    """Service layer for managing integrations with external data sources."""
    
    def __init__(self):
        """Initialize the integration service."""
        self.sheets_client = GoogleSheetsClient()
        logger.info("Integration service initialized")
    
    async def get_next_contacts_to_call(self, limit: int = 10) -> List[Contact]:
        """
        Get the next batch of contacts to call from Google Sheets.
        
        Args:
            limit: Maximum number of contacts to return
            
        Returns:
            List of Contact objects ready to be called
        """
        try:
            contacts = await self.sheets_client.get_contacts_to_call(limit=limit)
            logger.info(f"Retrieved {len(contacts)} contacts ready to call")
            return contacts
        except Exception as e:
            logger.error(f"Error getting contacts to call: {e}")
            return []
    
    async def get_contact_by_phone(self, phone_number: str) -> Optional[Contact]:
        """
        Get contact information by phone number.
        
        Args:
            phone_number: The phone number to search for
            
        Returns:
            Contact object or None if not found
        """
        try:
            contact = await self.sheets_client.get_contact_by_phone(phone_number)
            if contact:
                logger.info(f"Found contact for phone {phone_number}: {contact.name}")
            else:
                logger.warning(f"No contact found for phone {phone_number}")
            return contact
        except Exception as e:
            logger.error(f"Error getting contact by phone {phone_number}: {e}")
            return None
    
    async def mark_contact_as_called(self, phone_number: str, call_id: str, 
                                   status: str = "Called - Vapi") -> SheetsOperationResult:
        """
        Mark a contact as called in the Google Sheets.
        
        Args:
            phone_number: The phone number of the contact
            call_id: The unique call ID
            status: The status to set (default: "Called - Vapi")
            
        Returns:
            SheetsOperationResult indicating success/failure
        """
        try:
            contact_status = ContactStatus(
                phone_number=phone_number,
                status=status,
                call_id=call_id,
                timestamp=datetime.now().isoformat()
            )
            
            result = await self.sheets_client.update_contact_status(contact_status)
            if result.success:
                logger.info(f"Successfully marked {phone_number} as called with ID {call_id}")
            else:
                logger.error(f"Failed to mark {phone_number} as called: {result.message}")
            
            return result
        except Exception as e:
            logger.error(f"Error marking contact as called: {e}")
            return SheetsOperationResult(
                success=False,
                message=f"Error marking contact as called: {str(e)}",
                error=str(e),
                records_affected=0
            )
    
    async def save_call_results(self, workflow_context: WorkflowContext, 
                              analytics: AnalyticsMetrics) -> SheetsOperationResult:
        """
        Save call results to Google Sheets based on workflow context and analytics.
        
        Args:
            workflow_context: The workflow context containing call information
            analytics: Analytics metrics from the call
            
        Returns:
            SheetsOperationResult indicating success/failure
        """
        try:
            # Extract contact information from lead_data
            lead_data = workflow_context.lead_data
            
            # Create call result object
            call_result = CallResult(
                name=lead_data.get('name', ''),
                phone_number=lead_data.get('phone_number', ''),
                property_address=lead_data.get('property_address', ''),
                call_date=datetime.now().isoformat(),
                call_summary=self._generate_call_summary(workflow_context, analytics),
                selling_timeline=lead_data.get('selling_timeline', ''),
                bedrooms=lead_data.get('bedrooms', ''),
                bathrooms=lead_data.get('bathrooms', ''),
                asking_price=lead_data.get('asking_price', ''),
                call_duration=f"{analytics.workflow_duration_ms / 1000:.1f}s",
                qualification_score=workflow_context.qualification_score,
                interested=analytics.outcome in ['qualified', 'scheduled'],
                follow_up_needed=analytics.outcome == 'callback',
                appointment_scheduled=analytics.outcome == 'scheduled',
                appointment_date=lead_data.get('appointment_date', ''),
                notes=lead_data.get('notes', '')
            )
            
            result = await self.sheets_client.write_call_results([call_result])
            if result.success:
                logger.info(f"Successfully saved call results for {call_result.phone_number}")
            else:
                logger.error(f"Failed to save call results: {result.message}")
            
            return result
        except Exception as e:
            logger.error(f"Error saving call results: {e}")
            return SheetsOperationResult(
                success=False,
                message=f"Error saving call results: {str(e)}",
                error=str(e),
                records_affected=0
            )
    
    def _generate_call_summary(self, workflow_context: WorkflowContext, 
                             analytics: AnalyticsMetrics) -> str:
        """
        Generate a call summary based on workflow context and analytics.
        
        Args:
            workflow_context: The workflow context
            analytics: Analytics metrics
            
        Returns:
            A formatted call summary string
        """
        try:
            summary_parts = []
            
            # Add outcome
            summary_parts.append(f"Outcome: {analytics.outcome.title()}")
            
            # Add qualification score
            if workflow_context.qualification_score > 0:
                summary_parts.append(f"Qualification Score: {workflow_context.qualification_score:.2f}")
            
            # Add objection count if any
            if workflow_context.objection_count > 0:
                summary_parts.append(f"Objections Handled: {workflow_context.objection_count}")
            
            # Add tier escalation info
            if workflow_context.tier_escalated:
                summary_parts.append("Tier Escalated: Yes")
            
            # Add agent transitions
            if analytics.agent_transitions:
                agents_used = [transition.get('to_agent', '') for transition in analytics.agent_transitions]
                unique_agents = list(set(filter(None, agents_used)))
                if unique_agents:
                    summary_parts.append(f"Agents Used: {', '.join(unique_agents)}")
            
            # Add conversation highlights from metadata
            if workflow_context.metadata:
                highlights = workflow_context.metadata.get('conversation_highlights', [])
                if highlights:
                    summary_parts.append(f"Key Points: {'; '.join(highlights[:3])}")  # Limit to 3 highlights
            
            return " | ".join(summary_parts) if summary_parts else "Call completed"
        except Exception as e:
            logger.error(f"Error generating call summary: {e}")
            return "Call completed - summary generation failed"
    
    async def update_contact_status(self, phone_number: str, status: str, 
                                  call_id: Optional[str] = None) -> SheetsOperationResult:
        """
        Update the status of a contact in Google Sheets.
        
        Args:
            phone_number: The phone number of the contact
            status: The new status to set
            call_id: Optional call ID
            
        Returns:
            SheetsOperationResult indicating success/failure
        """
        try:
            contact_status = ContactStatus(
                phone_number=phone_number,
                status=status,
                call_id=call_id,
                timestamp=datetime.now().isoformat()
            )
            
            result = await self.sheets_client.update_contact_status(contact_status)
            logger.info(f"Updated status for {phone_number} to {status}")
            return result
        except Exception as e:
            logger.error(f"Error updating contact status: {e}")
            return SheetsOperationResult(
                success=False,
                message=f"Error updating contact status: {str(e)}",
                error=str(e),
                records_affected=0
            )
    
    async def search_contacts(self, contact_filter: ContactFilter) -> List[Contact]:
        """
        Search for contacts based on filter criteria.
        
        Args:
            contact_filter: Filter criteria for the search
            
        Returns:
            List of Contact objects matching the criteria
        """
        try:
            contacts = await self.sheets_client.read_contacts(contact_filter)
            logger.info(f"Found {len(contacts)} contacts matching filter criteria")
            return contacts
        except Exception as e:
            logger.error(f"Error searching contacts: {e}")
            return []
    
    def is_sheets_enabled(self) -> bool:
        """
        Check if Google Sheets integration is enabled and configured.
        
        Returns:
            True if sheets integration is available, False otherwise
        """
        return self.sheets_client.is_enabled()
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """
        Get the status of all integrations.
        
        Returns:
            Dictionary containing integration status information
        """
        status = {
            "google_sheets": {
                "enabled": self.sheets_client.is_enabled(),
                "input_spreadsheet_configured": bool(settings.input_spreadsheet_id),
                "output_spreadsheet_configured": bool(settings.output_spreadsheet_id),
                "credentials_file_exists": bool(
                    settings.google_sheets_credentials_file and 
                    os.path.exists(settings.google_sheets_credentials_file)
                )
            }
        }
        
        # Test connection if enabled
        if self.sheets_client.is_enabled():
            try:
                # Try to read a small number of contacts to test connection
                test_contacts = await self.sheets_client.read_contacts(ContactFilter(limit=1))
                status["google_sheets"]["connection_test"] = "success"
                status["google_sheets"]["test_message"] = f"Successfully connected, found {len(test_contacts)} contacts"
            except Exception as e:
                status["google_sheets"]["connection_test"] = "failed"
                status["google_sheets"]["test_message"] = str(e)
        
        return status


# Global integration service instance
integration_service = IntegrationService()
