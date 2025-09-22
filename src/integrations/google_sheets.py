#!/usr/bin/env python3
"""
Google Sheets Client for Real Estate Automation

This module provides a client for interacting with Google Sheets API using gspread.
It handles authentication and provides methods for reading and writing data.
Integrated with the REAutomation2 project architecture.
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional

import gspread
from google.oauth2.service_account import Credentials

from ..config.settings import settings
from .models import (
    Contact, CallResult, ContactStatus, SheetsConfig, 
    SheetsOperationResult, ContactFilter
)

# Configure logging
logger = logging.getLogger(__name__)

# Google Sheets API configuration
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]


class GoogleSheetsClient:
    """Client for interacting with Google Sheets API using gspread."""
    
    def __init__(self, config: Optional[SheetsConfig] = None):
        """
        Initialize the Google Sheets client.
        
        Args:
            config: Optional SheetsConfig object. If not provided, uses settings from environment.
        """
        self.client = None
        self.config = config or SheetsConfig(
            credentials_file=settings.google_sheets_credentials_file,
            input_spreadsheet_id=settings.input_spreadsheet_id,
            output_spreadsheet_id=settings.output_spreadsheet_id,
            enabled=settings.sheets_enabled
        )
        
        if self.config.enabled:
            self._authenticate()
    
    def _safe_str_strip(self, value) -> str:
        """
        Safely convert a value to string and strip whitespace.
        Handles both string and numeric types from Google Sheets.
        
        Args:
            value: Value from Google Sheets (can be string, int, float, or None)
            
        Returns:
            Stripped string or empty string if None/empty
        """
        if value is None or value == '':
            return ''
        
        # Convert to string first, then strip
        try:
            return str(value).strip()
        except (AttributeError, TypeError):
            # Fallback for any unexpected types
            return str(value) if value is not None else ''
    
    def _parse_phone_number(self, phone_raw) -> Optional[str]:
        """
        Parse and format a phone number from raw input.
        
        Handles formats like:
        - (330) 329-0287 - Wireless
        - 330-329-0287
        - 3303290287
        - +1 330 329 0287
        - etc.
        
        Args:
            phone_raw: Raw phone number from spreadsheet (can be string, int, or float)
            
        Returns:
            Formatted phone number with +1 prefix, or None if invalid
        """
        if phone_raw is None or phone_raw == '':
            return None
        
        # Convert to string and strip whitespace - handle both strings and numbers
        try:
            phone_str = str(phone_raw).strip()
        except AttributeError:
            # If it's already a number type, just convert to string
            phone_str = str(phone_raw)
        
        if not phone_str:
            return None
        
        # Extract only digits from the phone number
        digits = re.sub(r'[^\d]', '', phone_str)
        
        # Log the parsing process for debugging
        logger.debug(f"Parsing phone: '{phone_str}' -> digits: '{digits}'")
        
        if not digits:
            logger.debug(f"No digits found in phone number: '{phone_str}'")
            return None
        
        # Format phone number with +1 prefix
        if digits.startswith('1') and len(digits) == 11:
            # Already has country code
            formatted_phone = "+" + digits
        elif len(digits) == 10:
            # Add US country code
            formatted_phone = "+1" + digits
        else:
            # Invalid length
            logger.debug(f"Invalid phone number length ({len(digits)} digits): '{phone_str}' -> '{digits}'")
            return None
        
        logger.debug(f"Successfully formatted phone: '{phone_str}' -> '{formatted_phone}'")
        return formatted_phone
    
    def _authenticate(self):
        """Authenticate with Google Sheets API using service account credentials."""
        try:
            if not os.path.exists(self.config.credentials_file):
                logger.error(f"Credentials file not found: {self.config.credentials_file}")
                raise FileNotFoundError(f"Credentials file not found: {self.config.credentials_file}")
            
            creds = Credentials.from_service_account_file(
                self.config.credentials_file, scopes=SCOPES)
            self.client = gspread.authorize(creds)
            logger.info("Successfully authenticated with Google Sheets API")
        except Exception as e:
            logger.error(f"Error authenticating with Google Sheets API: {e}")
            raise
    
    def is_enabled(self) -> bool:
        """Check if Google Sheets integration is enabled and properly configured."""
        return (self.config.enabled and 
                self.client is not None and 
                self.config.input_spreadsheet_id is not None)
    
    async def read_contacts(self, contact_filter: Optional[ContactFilter] = None) -> List[Contact]:
        """
        Read contacts from the input spreadsheet.
        
        Args:
            contact_filter: Optional filter criteria for contacts
            
        Returns:
            A list of Contact objects containing contact information
        """
        if not self.is_enabled():
            logger.warning("Google Sheets integration is disabled or not configured")
            return []
        
        try:
            # Open the spreadsheet by ID
            sheet = self.client.open_by_key(self.config.input_spreadsheet_id).sheet1
            
            # Get all records as dictionaries
            records = sheet.get_all_records()
            
            # Get headers to identify phone columns
            headers = sheet.row_values(1)
            phone_columns = [header for header in headers if header.startswith('Phone ') and header.replace('Phone ', '').isdigit()]
            phone_columns.sort(key=lambda x: int(x.replace('Phone ', '')))  # Sort by number
            
            logger.info(f"Found phone columns: {phone_columns}")
            
            # Filter and format records
            contacts = []
            for record in records:
                # Check if the record has the required basic fields
                name_raw = record.get(self.config.name_column, '')
                address_raw = record.get(self.config.address_column, '')
                
                # Convert to string and strip safely
                name = self._safe_str_strip(name_raw)
                address = self._safe_str_strip(address_raw)
                
                if not (name and address):
                    continue
                
                # Apply name filter if specified
                if (contact_filter and contact_filter.name_contains and 
                    contact_filter.name_contains.lower() not in name.lower()):
                    continue
                
                # Apply address filter if specified
                if (contact_filter and contact_filter.address_contains and 
                    contact_filter.address_contains.lower() not in address.lower()):
                    continue
                
                # Try each phone column in order until we find a valid phone number
                phone_found = False
                for phone_col in phone_columns:
                    if phone_col in record and record[phone_col]:
                        # Parse and format the phone number
                        phone_number = self._parse_phone_number(record[phone_col])
                        
                        if phone_number:
                            # Apply phone filter if specified
                            if (contact_filter and contact_filter.phone_number and 
                                contact_filter.phone_number != phone_number):
                                continue
                            
                            # Extract bedroom and bathroom information if available
                            bedrooms = ""
                            bathrooms = ""
                            
                            for bed_col in self.config.bedrooms_columns:
                                if bed_col in record and record[bed_col]:
                                    bedrooms = self._safe_str_strip(record[bed_col])
                                    break
                            
                            for bath_col in self.config.bathrooms_columns:
                                if bath_col in record and record[bath_col]:
                                    bathrooms = self._safe_str_strip(record[bath_col])
                                    break
                            
                            # Extract additional fields
                            owner_last_name = self._safe_str_strip(record.get('Owner 1 Last Name', ''))
                            city = self._safe_str_strip(record.get('City', ''))
                            state = self._safe_str_strip(record.get('State', ''))
                            zip_code = self._safe_str_strip(record.get('Zip', record.get('ZIP', '')))
                            property_type = self._safe_str_strip(record.get('Property Type', ''))
                            estimated_value = self._safe_str_strip(record.get('Estimated Value', ''))
                            
                            contact = Contact(
                                name=name,
                                phone_number=phone_number,
                                property_address=address,
                                phone_column=phone_col,
                                bedrooms=bedrooms,
                                bathrooms=bathrooms,
                                owner_last_name=owner_last_name,
                                city=city,
                                state=state,
                                zip_code=zip_code,
                                property_type=property_type,
                                estimated_value=estimated_value
                            )
                            
                            contacts.append(contact)
                            phone_found = True
                            logger.debug(f"Using {phone_col} for {name}: {record[phone_col]} -> {phone_number}")
                            break  # Use the first valid phone number found
                
                if not phone_found:
                    logger.debug(f"No valid phone number found for {name} at {address}")
            
            # Apply limit if specified
            if contact_filter and contact_filter.limit and contact_filter.limit > 0:
                contacts = contacts[:contact_filter.limit]
            
            logger.info(f"Read {len(contacts)} contacts from the spreadsheet")
            return contacts
        except Exception as e:
            logger.error(f"Error reading contacts from spreadsheet: {e}")
            return []
    
    async def write_call_results(self, results: List[CallResult]) -> SheetsOperationResult:
        """
        Write call results to the output spreadsheet.
        
        Args:
            results: A list of CallResult objects containing call results
            
        Returns:
            SheetsOperationResult indicating success/failure
        """
        if not self.is_enabled() or not self.config.output_spreadsheet_id:
            return SheetsOperationResult(
                success=False,
                message="Google Sheets integration is disabled or output spreadsheet not configured",
                records_affected=0
            )
        
        try:
            # Open the spreadsheet by ID
            sheet = self.client.open_by_key(self.config.output_spreadsheet_id).sheet1
            
            # Check if the sheet has headers, if not add them
            headers = sheet.row_values(1)
            if not headers:
                sheet.append_row(self.config.output_headers)
                logger.info("Added headers to output spreadsheet")
            
            # Append results to the sheet
            rows_added = 0
            for result in results:
                row = [
                    result.name,
                    result.phone_number,
                    result.property_address,
                    result.call_date,
                    result.selling_timeline,
                    result.bedrooms,
                    result.bathrooms,
                    result.asking_price,
                    result.call_summary,
                    result.qualification_score,
                    result.interested,
                    result.follow_up_needed,
                    result.appointment_scheduled,
                    result.notes
                ]
                sheet.append_row(row)
                rows_added += 1
            
            logger.info(f"Successfully wrote {rows_added} results to the spreadsheet")
            return SheetsOperationResult(
                success=True,
                message=f"Successfully wrote {rows_added} call results",
                records_affected=rows_added
            )
        except Exception as e:
            logger.error(f"Error writing results to spreadsheet: {e}")
            return SheetsOperationResult(
                success=False,
                message=f"Error writing results: {str(e)}",
                error=str(e),
                records_affected=0
            )
    
    async def get_contact_by_phone(self, phone_number: str) -> Optional[Contact]:
        """
        Get contact information by phone number.
        
        Args:
            phone_number: The phone number to search for
            
        Returns:
            Contact object or None if not found
        """
        contacts = await self.read_contacts(
            ContactFilter(phone_number=phone_number, limit=1)
        )
        return contacts[0] if contacts else None
    
    async def update_contact_status(self, contact_status: ContactStatus) -> SheetsOperationResult:
        """
        Update the status of a contact in the input spreadsheet.
        
        Args:
            contact_status: ContactStatus object with phone number and new status
            
        Returns:
            SheetsOperationResult indicating success/failure
        """
        if not self.is_enabled():
            return SheetsOperationResult(
                success=False,
                message="Google Sheets integration is disabled or not configured",
                records_affected=0
            )
        
        try:
            # Open the spreadsheet by ID
            sheet = self.client.open_by_key(self.config.input_spreadsheet_id).sheet1
            
            # Get all records and headers
            records = sheet.get_all_records()
            headers = sheet.row_values(1)
            
            # Get all phone columns
            phone_columns = [header for header in headers if header.startswith('Phone ') and header.replace('Phone ', '').isdigit()]
            phone_columns.sort(key=lambda x: int(x.replace('Phone ', '')))  # Sort by number
            
            # Find the row with the matching phone number
            for i, record in enumerate(records):
                phone_match_found = False
                
                # Check all phone columns for a match
                for phone_col in phone_columns:
                    if phone_col in record and record[phone_col]:
                        # Parse phone number the same way as in read_contacts
                        formatted_phone = self._parse_phone_number(record[phone_col])
                        
                        if formatted_phone and formatted_phone == contact_status.phone_number:
                            phone_match_found = True
                            break
                
                if phone_match_found:
                    # Row index is i+2 (1-based index + header row)
                    row_idx = i + 2
                    
                    # Update Status column
                    if 'Status' in headers:
                        status_col_idx = headers.index('Status') + 1
                    else:
                        # Add a Status column if it doesn't exist
                        status_col_idx = len(headers) + 1
                        sheet.update_cell(1, status_col_idx, 'Status')
                        headers.append('Status')  # Update local headers list
                        logger.info("Added Status column to spreadsheet")
                    
                    # Update the status
                    sheet.update_cell(row_idx, status_col_idx, contact_status.status)
                    
                    # Update Call ID if provided
                    if contact_status.call_id:
                        if 'Call ID' in headers:
                            call_id_col_idx = headers.index('Call ID') + 1
                        else:
                            call_id_col_idx = len(headers) + 1
                            sheet.update_cell(1, call_id_col_idx, 'Call ID')
                            headers.append('Call ID')
                            logger.info("Added Call ID column to spreadsheet")
                        
                        sheet.update_cell(row_idx, call_id_col_idx, contact_status.call_id)
                    
                    # Update Timestamp if provided
                    if contact_status.timestamp:
                        if 'Last Called' in headers:
                            timestamp_col_idx = headers.index('Last Called') + 1
                        else:
                            timestamp_col_idx = len(headers) + 1
                            sheet.update_cell(1, timestamp_col_idx, 'Last Called')
                            headers.append('Last Called')
                            logger.info("Added Last Called column to spreadsheet")
                        
                        sheet.update_cell(row_idx, timestamp_col_idx, contact_status.timestamp)
                    
                    logger.info(f"Updated call status for {contact_status.phone_number} to {contact_status.status}")
                    return SheetsOperationResult(
                        success=True,
                        message=f"Updated status for {contact_status.phone_number}",
                        records_affected=1
                    )
            
            logger.warning(f"Contact with phone number {contact_status.phone_number} not found")
            return SheetsOperationResult(
                success=False,
                message=f"Contact with phone number {contact_status.phone_number} not found",
                records_affected=0
            )
        except Exception as e:
            logger.error(f"Error updating contact status: {e}")
            return SheetsOperationResult(
                success=False,
                message=f"Error updating contact status: {str(e)}",
                error=str(e),
                records_affected=0
            )
    
    async def get_contacts_to_call(self, limit: Optional[int] = None) -> List[Contact]:
        """
        Get contacts that haven't been called yet, up to the specified limit.
        
        Args:
            limit: Maximum number of contacts to return (None for all)
            
        Returns:
            List of Contact objects ready to be called
        """
        if not self.is_enabled():
            logger.warning("Google Sheets integration is disabled or not configured")
            return []
        
        try:
            # Get all contacts that have valid name and address
            all_contacts = await self.read_contacts()
            
            if not all_contacts:
                logger.info("No contacts found in spreadsheet")
                return []
            
            # Open the spreadsheet to check if status column exists
            sheet = self.client.open_by_key(self.config.input_spreadsheet_id).sheet1
            headers = sheet.row_values(1)
            
            # If there's no Status column, all valid contacts are available to call
            if 'Status' not in headers:
                logger.info("No Status column found - all valid contacts available to call")
                contacts_to_call = all_contacts
            else:
                # Get all records to check status
                records = sheet.get_all_records()
                
                # Filter out contacts that have already been called
                contacts_to_call = []
                for contact in all_contacts:
                    # Find the corresponding record for this contact
                    contact_called = False
                    for record in records:
                        # Match by phone number (most reliable)
                        record_phones = []
                        phone_columns = [header for header in headers if header.startswith('Phone ') and header.replace('Phone ', '').isdigit()]
                        
                        for phone_col in phone_columns:
                            if phone_col in record and record[phone_col]:
                                formatted_phone = self._parse_phone_number(record[phone_col])
                                if formatted_phone:
                                    record_phones.append(formatted_phone)
                        
                        # Check if this record matches our contact
                        if contact.phone_number in record_phones:
                            status_raw = record.get('Status', '')
                            status = self._safe_str_strip(status_raw).lower()
                            # Consider contact as called if status contains 'called', 'completed', or 'done'
                            if status and any(keyword in status for keyword in ['called', 'completed', 'done', 'finished']):
                                contact_called = True
                                logger.debug(f"Skipping {contact.name} - already called (status: {status})")
                            break
                    
                    if not contact_called:
                        contacts_to_call.append(contact)
            
            # Apply limit if specified
            if limit is not None and limit > 0:
                contacts_to_call = contacts_to_call[:limit]
            
            logger.info(f"Found {len(contacts_to_call)} contacts available to call (limit: {limit})")
            
            # Debug: Log details about the contacts found
            for i, contact in enumerate(contacts_to_call[:3]):  # Log first 3 contacts
                logger.info(f"Contact {i+1}: {contact.name} - {contact.phone_number} - {contact.property_address}")
            
            return contacts_to_call
            
        except Exception as e:
            logger.error(f"Error getting contacts to call: {e}")
            return []


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_client():
        client = GoogleSheetsClient()
        if client.is_enabled():
            contacts = await client.read_contacts(ContactFilter(limit=5))
            print(f"Read {len(contacts)} contacts")
            for contact in contacts:
                print(f"- {contact.name}: {contact.phone_number}")
        else:
            print("Google Sheets integration is not enabled or configured")
    
    asyncio.run(test_client())
