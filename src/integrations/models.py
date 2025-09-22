from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import time


class Contact(BaseModel):
    """Contact information from Google Sheets."""
    name: str
    phone_number: str
    property_address: str
    bedrooms: Optional[str] = ""
    bathrooms: Optional[str] = ""
    phone_column: Optional[str] = None  # Which phone column was used
    
    # Additional fields that might be available
    owner_last_name: Optional[str] = ""
    city: Optional[str] = ""
    state: Optional[str] = ""
    zip_code: Optional[str] = ""
    property_type: Optional[str] = ""
    estimated_value: Optional[str] = ""
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CallResult(BaseModel):
    """Call result data to write back to Google Sheets."""
    name: str
    phone_number: str
    property_address: str
    call_date: str
    call_summary: str
    selling_timeline: Optional[str] = ""
    bedrooms: Optional[str] = ""
    bathrooms: Optional[str] = ""
    asking_price: Optional[str] = ""
    
    # Additional result fields
    call_duration: Optional[str] = ""
    qualification_score: Optional[float] = None
    interested: Optional[bool] = None
    follow_up_needed: Optional[bool] = None
    appointment_scheduled: Optional[bool] = None
    appointment_date: Optional[str] = ""
    notes: Optional[str] = ""
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ContactStatus(BaseModel):
    """Contact status tracking."""
    phone_number: str
    status: str  # e.g., 'Called - Vapi', 'No Answer', 'Completed', 'Scheduled'
    call_id: Optional[str] = None
    timestamp: Optional[str] = None
    last_updated: float = Field(default_factory=time.time)


class SheetsConfig(BaseModel):
    """Configuration for Google Sheets integration."""
    credentials_file: str
    input_spreadsheet_id: Optional[str] = None
    output_spreadsheet_id: Optional[str] = None
    enabled: bool = True
    
    # Column mapping configuration
    name_column: str = "Owner 1 First Name"
    address_column: str = "Address"
    phone_columns: List[str] = Field(default_factory=lambda: ["Phone 1", "Phone 2", "Phone 3"])
    bedrooms_columns: List[str] = Field(default_factory=lambda: ["Bedrooms", "Bed", "BR"])
    bathrooms_columns: List[str] = Field(default_factory=lambda: ["Bathrooms", "Bath", "BA"])
    
    # Output sheet configuration
    output_headers: List[str] = Field(default_factory=lambda: [
        "Name", "Phone Number", "Property Address", "Call Date",
        "Selling Timeline", "Bedrooms", "Bathrooms", "Asking Price",
        "Call Summary", "Qualification Score", "Interested", 
        "Follow Up Needed", "Appointment Scheduled", "Notes"
    ])


class SheetsOperationResult(BaseModel):
    """Result of a Google Sheets operation."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    operation_time: float = Field(default_factory=time.time)
    records_affected: int = 0


class ContactFilter(BaseModel):
    """Filter criteria for retrieving contacts."""
    limit: Optional[int] = None
    exclude_called: bool = True
    status_filter: Optional[List[str]] = None
    phone_number: Optional[str] = None
    name_contains: Optional[str] = None
    address_contains: Optional[str] = None
