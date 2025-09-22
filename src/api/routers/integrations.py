"""
Integration API Routes

This module provides API endpoints for managing integrations with external data sources,
particularly Google Sheets for contact management and call result tracking.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any
import logging

from ...integrations.service import integration_service
from ...integrations.models import Contact, ContactFilter, ContactStatus, SheetsOperationResult
from ...agents.models import WorkflowContext, AnalyticsMetrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/integrations", tags=["integrations"])


@router.get("/status")
async def get_integration_status() -> Dict[str, Any]:
    """
    Get the status of all integrations.
    
    Returns:
        Dictionary containing integration status information
    """
    try:
        status = await integration_service.get_integration_status()
        return {
            "success": True,
            "data": status
        }
    except Exception as e:
        logger.error(f"Error getting integration status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/contacts", response_model=List[Contact])
async def get_contacts(
    limit: Optional[int] = Query(None, description="Maximum number of contacts to return"),
    exclude_called: bool = Query(True, description="Exclude contacts that have already been called"),
    name_contains: Optional[str] = Query(None, description="Filter by name containing this text"),
    address_contains: Optional[str] = Query(None, description="Filter by address containing this text"),
    phone_number: Optional[str] = Query(None, description="Filter by specific phone number")
) -> List[Contact]:
    """
    Get contacts from Google Sheets with optional filtering.
    
    Args:
        limit: Maximum number of contacts to return
        exclude_called: Whether to exclude contacts that have already been called
        name_contains: Filter by name containing this text
        address_contains: Filter by address containing this text
        phone_number: Filter by specific phone number
        
    Returns:
        List of Contact objects
    """
    try:
        if not integration_service.is_sheets_enabled():
            raise HTTPException(
                status_code=503, 
                detail="Google Sheets integration is not enabled or configured"
            )
        
        contact_filter = ContactFilter(
            limit=limit,
            exclude_called=exclude_called,
            name_contains=name_contains,
            address_contains=address_contains,
            phone_number=phone_number
        )
        
        contacts = await integration_service.search_contacts(contact_filter)
        return contacts
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting contacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/contacts/to-call", response_model=List[Contact])
async def get_contacts_to_call(
    limit: int = Query(10, description="Maximum number of contacts to return")
) -> List[Contact]:
    """
    Get the next batch of contacts that are ready to be called.
    
    Args:
        limit: Maximum number of contacts to return
        
    Returns:
        List of Contact objects ready to be called
    """
    try:
        if not integration_service.is_sheets_enabled():
            raise HTTPException(
                status_code=503, 
                detail="Google Sheets integration is not enabled or configured"
            )
        
        contacts = await integration_service.get_next_contacts_to_call(limit=limit)
        return contacts
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting contacts to call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/contacts/phone/{phone_number}", response_model=Contact)
async def get_contact_by_phone(phone_number: str) -> Contact:
    """
    Get contact information by phone number.
    
    Args:
        phone_number: The phone number to search for
        
    Returns:
        Contact object
    """
    try:
        if not integration_service.is_sheets_enabled():
            raise HTTPException(
                status_code=503, 
                detail="Google Sheets integration is not enabled or configured"
            )
        
        contact = await integration_service.get_contact_by_phone(phone_number)
        if not contact:
            raise HTTPException(
                status_code=404, 
                detail=f"Contact with phone number {phone_number} not found"
            )
        
        return contact
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting contact by phone: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/contacts/mark-called")
async def mark_contact_as_called(
    phone_number: str,
    call_id: str,
    status: str = "Called - Vapi"
) -> Dict[str, Any]:
    """
    Mark a contact as called in Google Sheets.
    
    Args:
        phone_number: The phone number of the contact
        call_id: The unique call ID
        status: The status to set (default: "Called - Vapi")
        
    Returns:
        Operation result
    """
    try:
        if not integration_service.is_sheets_enabled():
            raise HTTPException(
                status_code=503, 
                detail="Google Sheets integration is not enabled or configured"
            )
        
        result = await integration_service.mark_contact_as_called(
            phone_number=phone_number,
            call_id=call_id,
            status=status
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.message)
        
        return {
            "success": True,
            "message": result.message,
            "records_affected": result.records_affected
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking contact as called: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/contacts/update-status")
async def update_contact_status(
    phone_number: str,
    status: str,
    call_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update the status of a contact in Google Sheets.
    
    Args:
        phone_number: The phone number of the contact
        status: The new status to set
        call_id: Optional call ID
        
    Returns:
        Operation result
    """
    try:
        if not integration_service.is_sheets_enabled():
            raise HTTPException(
                status_code=503, 
                detail="Google Sheets integration is not enabled or configured"
            )
        
        result = await integration_service.update_contact_status(
            phone_number=phone_number,
            status=status,
            call_id=call_id
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.message)
        
        return {
            "success": True,
            "message": result.message,
            "records_affected": result.records_affected
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating contact status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/call-results/save")
async def save_call_results(
    workflow_context: WorkflowContext,
    analytics: AnalyticsMetrics
) -> Dict[str, Any]:
    """
    Save call results to Google Sheets.
    
    Args:
        workflow_context: The workflow context containing call information
        analytics: Analytics metrics from the call
        
    Returns:
        Operation result
    """
    try:
        if not integration_service.is_sheets_enabled():
            raise HTTPException(
                status_code=503, 
                detail="Google Sheets integration is not enabled or configured"
            )
        
        result = await integration_service.save_call_results(
            workflow_context=workflow_context,
            analytics=analytics
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.message)
        
        return {
            "success": True,
            "message": result.message,
            "records_affected": result.records_affected
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving call results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sheets/test-connection")
async def test_sheets_connection() -> Dict[str, Any]:
    """
    Test the Google Sheets connection and return basic information.
    
    Returns:
        Connection test results
    """
    try:
        if not integration_service.is_sheets_enabled():
            return {
                "success": False,
                "message": "Google Sheets integration is not enabled or configured",
                "enabled": False
            }
        
        # Try to get a small number of contacts to test the connection
        contacts = await integration_service.get_next_contacts_to_call(limit=1)
        
        return {
            "success": True,
            "message": f"Successfully connected to Google Sheets. Found {len(contacts)} contacts available.",
            "enabled": True,
            "sample_contacts_count": len(contacts)
        }
    except Exception as e:
        logger.error(f"Error testing sheets connection: {e}")
        return {
            "success": False,
            "message": f"Connection test failed: {str(e)}",
            "enabled": integration_service.is_sheets_enabled(),
            "error": str(e)
        }


# Health check endpoint specific to integrations
@router.get("/health")
async def integration_health_check() -> Dict[str, Any]:
    """
    Health check for integration services.
    
    Returns:
        Health status of integration services
    """
    health_status = {
        "status": "healthy",
        "integrations": {
            "google_sheets": {
                "enabled": integration_service.is_sheets_enabled(),
                "status": "healthy" if integration_service.is_sheets_enabled() else "disabled"
            }
        }
    }
    
    # Test Google Sheets connection if enabled
    if integration_service.is_sheets_enabled():
        try:
            await integration_service.get_next_contacts_to_call(limit=1)
            health_status["integrations"]["google_sheets"]["status"] = "healthy"
            health_status["integrations"]["google_sheets"]["last_test"] = "success"
        except Exception as e:
            health_status["integrations"]["google_sheets"]["status"] = "error"
            health_status["integrations"]["google_sheets"]["last_test"] = "failed"
            health_status["integrations"]["google_sheets"]["error"] = str(e)
            health_status["status"] = "degraded"
    
    return health_status
