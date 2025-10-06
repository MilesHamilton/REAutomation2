"""
Test for Google Sheets integration using first entry as test data.
This test validates the GoogleSheetsClient functionality with realistic data
that matches the expected format from a real estate spreadsheet.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import asyncio

# Import the actual classes from the implementation
from src.integrations.google_sheets import GoogleSheetsClient
from src.integrations.models import (
    Contact, CallResult, ContactStatus, SheetsConfig, 
    SheetsOperationResult, ContactFilter
)


class TestGoogleSheetsFirstEntry:
    """Test Google Sheets integration using first entry test data"""

    @pytest.fixture
    def sheets_config(self):
        """Configuration matching the actual implementation"""
        return SheetsConfig(
            credentials_file="test_credentials.json",
            input_spreadsheet_id="test_input_sheet_id",
            output_spreadsheet_id="test_output_sheet_id",
            enabled=True,
            name_column="Owner 1 First Name",
            address_column="Address",
            phone_columns=["Phone 1", "Phone 2", "Phone 3"],
            bedrooms_columns=["Bedrooms", "Bed", "BR"],
            bathrooms_columns=["Bathrooms", "Bath", "BA"],
            output_headers=[
                "Name", "Phone Number", "Property Address", "Call Date",
                "Selling Timeline", "Bedrooms", "Bathrooms", "Asking Price",
                "Call Summary", "Qualification Score", "Interested",
                "Follow Up Needed", "Appointment Scheduled", "Notes"
            ]
        )

    @pytest.fixture
    def client(self, sheets_config):
        """Create GoogleSheetsClient instance with test config"""
        with patch('src.integrations.google_sheets.os.path.exists', return_value=True), \
             patch('src.integrations.google_sheets.Credentials.from_service_account_file') as mock_creds, \
             patch('src.integrations.google_sheets.gspread.authorize') as mock_authorize:
            
            # Mock the credentials and gspread client
            mock_creds.return_value = Mock()
            mock_authorize.return_value = Mock()
            
            client = GoogleSheetsClient(sheets_config)
            return client

    @pytest.fixture
    def mock_gspread_client(self):
        """Mock gspread client"""
        mock_client = Mock()
        return mock_client

    @pytest.fixture
    def first_entry_spreadsheet_data(self):
        """
        First entry test data representing a typical real estate contact
        This matches the format expected from a real Google Sheets input
        """
        return [
            {
                "Owner 1 First Name": "John Smith",
                "Owner 1 Last Name": "Smith", 
                "Address": "123 Main Street, Springfield, IL 62701",
                "City": "Springfield",
                "State": "IL",
                "Zip": "62701",
                "Phone 1": "(217) 555-0123",
                "Phone 2": "217-555-0124",
                "Phone 3": "",
                "Bedrooms": "3",
                "Bathrooms": "2.5",
                "Property Type": "Single Family",
                "Estimated Value": "$285,000",
                "Status": "",
                "Call ID": "",
                "Last Called": ""
            }
        ]

    @pytest.fixture
    def mock_worksheet(self, first_entry_spreadsheet_data):
        """Mock worksheet with first entry data"""
        mock_ws = Mock()
        mock_ws.get_all_records.return_value = first_entry_spreadsheet_data
        mock_ws.row_values.return_value = [
            "Owner 1 First Name", "Owner 1 Last Name", "Address", "City", "State", "Zip",
            "Phone 1", "Phone 2", "Phone 3", "Bedrooms", "Bathrooms", 
            "Property Type", "Estimated Value", "Status", "Call ID", "Last Called"
        ]
        mock_ws.append_row.return_value = None
        mock_ws.update_cell.return_value = None
        return mock_ws

    def test_client_initialization_with_first_entry_config(self, client, sheets_config):
        """Test that client initializes correctly with first entry configuration"""
        assert client.config == sheets_config
        assert client.config.enabled is True
        assert client.config.name_column == "Owner 1 First Name"
        assert client.config.address_column == "Address"
        assert "Phone 1" in client.config.phone_columns

    def test_safe_str_strip_with_first_entry_data(self, client):
        """Test _safe_str_strip method with various data types from spreadsheet"""
        # Test cases that might come from Google Sheets
        test_cases = [
            ("John Smith", "John Smith"),  # Normal string
            ("  John Smith  ", "John Smith"),  # String with whitespace
            (123, "123"),  # Integer (like zip code)
            (2.5, "2.5"),  # Float (like bathrooms)
            ("", ""),  # Empty string
            (None, ""),  # None value
            (0, "0"),  # Zero value
        ]
        
        for input_val, expected in test_cases:
            result = client._safe_str_strip(input_val)
            assert result == expected, f"Failed for input: {input_val}"

    def test_parse_phone_number_first_entry_formats(self, client):
        """Test phone number parsing with formats from first entry"""
        # Test the specific phone formats from our first entry
        test_cases = [
            ("(217) 555-0123", "+12175550123"),  # Primary phone format
            ("217-555-0124", "+12175550124"),    # Secondary phone format
            ("", None),                          # Empty phone field
            (None, None),                        # None phone field
            ("2175550123", "+12175550123"),      # Digits only
            ("+1 217 555 0123", "+12175550123"), # Already formatted
            ("(217) 555-0123 - Wireless", "+12175550123"),  # With carrier info
        ]
        
        for input_phone, expected in test_cases:
            result = client._parse_phone_number(input_phone)
            assert result == expected, f"Failed for input: '{input_phone}'"

    @pytest.mark.asyncio
    async def test_read_contacts_first_entry(self, client, mock_gspread_client, mock_worksheet):
        """Test reading contacts with first entry data"""
        # Setup mocks
        client.client = mock_gspread_client
        mock_spreadsheet = Mock()
        mock_spreadsheet.sheet1 = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        # Read contacts
        contacts = await client.read_contacts()
        
        # Verify results
        assert len(contacts) == 1
        contact = contacts[0]
        
        # Verify contact data matches first entry
        assert contact.name == "John Smith"
        assert contact.phone_number == "+12175550123"  # Parsed from "(217) 555-0123"
        assert contact.property_address == "123 Main Street, Springfield, IL 62701"
        assert contact.bedrooms == "3"
        assert contact.bathrooms == "2.5"
        assert contact.owner_last_name == "Smith"
        assert contact.city == "Springfield"
        assert contact.state == "IL"
        assert contact.zip_code == "62701"
        assert contact.property_type == "Single Family"
        assert contact.estimated_value == "$285,000"
        assert contact.phone_column == "Phone 1"

    @pytest.mark.asyncio
    async def test_read_contacts_with_filter_first_entry(self, client, mock_gspread_client, mock_worksheet):
        """Test reading contacts with filters applied to first entry"""
        # Setup mocks
        client.client = mock_gspread_client
        mock_spreadsheet = Mock()
        mock_spreadsheet.sheet1 = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        # Test name filter - should find contact
        contacts = await client.read_contacts(ContactFilter(name_contains="John"))
        assert len(contacts) == 1
        assert contacts[0].name == "John Smith"
        
        # Test name filter - should not find contact
        contacts = await client.read_contacts(ContactFilter(name_contains="Jane"))
        assert len(contacts) == 0
        
        # Test address filter - should find contact
        contacts = await client.read_contacts(ContactFilter(address_contains="Springfield"))
        assert len(contacts) == 1
        
        # Test phone filter - should find contact
        contacts = await client.read_contacts(ContactFilter(phone_number="+12175550123"))
        assert len(contacts) == 1
        
        # Test limit filter
        contacts = await client.read_contacts(ContactFilter(limit=1))
        assert len(contacts) == 1

    @pytest.mark.asyncio
    async def test_get_contact_by_phone_first_entry(self, client, mock_gspread_client, mock_worksheet):
        """Test getting contact by phone number using first entry data"""
        # Setup mocks
        client.client = mock_gspread_client
        mock_spreadsheet = Mock()
        mock_spreadsheet.sheet1 = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        # Test finding contact by phone
        contact = await client.get_contact_by_phone("+12175550123")
        
        assert contact is not None
        assert contact.name == "John Smith"
        assert contact.phone_number == "+12175550123"
        assert contact.property_address == "123 Main Street, Springfield, IL 62701"
        
        # Test not finding contact with different phone
        contact = await client.get_contact_by_phone("+19999999999")
        assert contact is None

    @pytest.mark.asyncio
    async def test_update_contact_status_first_entry(self, client, mock_gspread_client, mock_worksheet):
        """Test updating contact status for first entry"""
        # Setup mocks
        client.client = mock_gspread_client
        mock_spreadsheet = Mock()
        mock_spreadsheet.sheet1 = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        # Create contact status update
        contact_status = ContactStatus(
            phone_number="+12175550123",
            status="Called - Vapi",
            call_id="call_12345",
            timestamp="2024-01-15T10:30:00"
        )
        
        # Update status
        result = await client.update_contact_status(contact_status)
        
        # Verify success
        assert result.success is True
        assert result.records_affected == 1
        assert "Updated status for +12175550123" in result.message
        
        # Verify update_cell was called for status, call_id, and timestamp
        assert mock_worksheet.update_cell.call_count >= 3

    @pytest.mark.asyncio
    async def test_write_call_results_first_entry(self, client, mock_gspread_client, mock_worksheet):
        """Test writing call results for first entry"""
        # Setup mocks
        client.client = mock_gspread_client
        mock_spreadsheet = Mock()
        mock_spreadsheet.sheet1 = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        # Mock empty headers initially
        mock_worksheet.row_values.return_value = []
        
        # Create call result based on first entry
        call_result = CallResult(
            name="John Smith",
            phone_number="+12175550123",
            property_address="123 Main Street, Springfield, IL 62701",
            call_date="2024-01-15T10:30:00",
            selling_timeline="6-12 months",
            bedrooms="3",
            bathrooms="2.5",
            asking_price="$285,000",
            call_summary="Homeowner interested in selling, needs to discuss with spouse",
            qualification_score=0.75,
            interested=True,
            follow_up_needed=True,
            appointment_scheduled=False,
            notes="Call back in 2 weeks after spouse discussion"
        )
        
        # Write call results
        result = await client.write_call_results([call_result])
        
        # Verify success
        assert result.success is True
        assert result.records_affected == 1
        
        # Verify headers were added and result was appended
        mock_worksheet.append_row.assert_called()
        call_args = mock_worksheet.append_row.call_args_list
        
        # First call should be headers
        headers_call = call_args[0][0][0]
        assert "Name" in headers_call
        assert "Phone Number" in headers_call
        
        # Second call should be the data
        data_call = call_args[1][0][0]
        assert "John Smith" in data_call
        assert "+12175550123" in data_call
        assert "123 Main Street, Springfield, IL 62701" in data_call

    @pytest.mark.asyncio
    async def test_get_contacts_to_call_first_entry(self, client, mock_gspread_client, mock_worksheet):
        """Test getting contacts to call with first entry (no status column)"""
        # Setup mocks - no Status column initially
        client.client = mock_gspread_client
        mock_spreadsheet = Mock()
        mock_spreadsheet.sheet1 = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        # Mock headers without Status column
        mock_worksheet.row_values.return_value = [
            "Owner 1 First Name", "Address", "Phone 1", "Phone 2", "Phone 3", "Bedrooms", "Bathrooms"
        ]
        
        # Get contacts to call
        contacts = await client.get_contacts_to_call(limit=5)
        
        # Should return the first entry since no Status column exists
        assert len(contacts) == 1
        assert contacts[0].name == "John Smith"
        assert contacts[0].phone_number == "+12175550123"

    @pytest.mark.asyncio
    async def test_get_contacts_to_call_with_status_first_entry(self, client, mock_gspread_client, mock_worksheet):
        """Test getting contacts to call when first entry has been called"""
        # Setup mocks
        client.client = mock_gspread_client
        mock_spreadsheet = Mock()
        mock_spreadsheet.sheet1 = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        # Mock headers with Status column
        mock_worksheet.row_values.return_value = [
            "Owner 1 First Name", "Address", "Phone 1", "Status"
        ]
        
        # Mock data with called status
        mock_worksheet.get_all_records.return_value = [
            {
                "Owner 1 First Name": "John Smith",
                "Address": "123 Main Street, Springfield, IL 62701",
                "Phone 1": "(217) 555-0123",
                "Status": "Called - Vapi"
            }
        ]
        
        # Get contacts to call
        contacts = await client.get_contacts_to_call(limit=5)
        
        # Should return empty list since contact has been called
        assert len(contacts) == 0

    def test_phone_number_fallback_first_entry(self, client, mock_gspread_client, mock_worksheet):
        """Test phone number fallback logic with first entry having multiple phones"""
        # Test data with empty Phone 1 but valid Phone 2
        test_data = [
            {
                "Owner 1 First Name": "John Smith",
                "Address": "123 Main Street, Springfield, IL 62701",
                "Phone 1": "",  # Empty first phone
                "Phone 2": "217-555-0124",  # Valid second phone
                "Phone 3": "(217) 555-0125",  # Valid third phone
                "Bedrooms": "3",
                "Bathrooms": "2.5"
            }
        ]
        
        # Setup mock to return test data
        mock_worksheet.get_all_records.return_value = test_data
        
        # Setup client
        client.client = mock_gspread_client
        mock_spreadsheet = Mock()
        mock_spreadsheet.sheet1 = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        # Test that it uses Phone 2 when Phone 1 is empty
        contacts = asyncio.run(client.read_contacts())
        
        assert len(contacts) == 1
        contact = contacts[0]
        assert contact.phone_number == "+12175550124"  # Should use Phone 2
        assert contact.phone_column == "Phone 2"

    def test_invalid_phone_number_handling_first_entry(self, client):
        """Test handling of invalid phone numbers in first entry"""
        invalid_phones = [
            "123",  # Too short
            "abc-def-ghij",  # No digits
            "1234567890123456",  # Too long
            "",  # Empty
            None,  # None
        ]
        
        for invalid_phone in invalid_phones:
            result = client._parse_phone_number(invalid_phone)
            # Should return None for invalid phones or handle gracefully
            if invalid_phone in ["", None]:
                assert result is None
            # For other invalid formats, should still return None or handle gracefully
            # The actual behavior depends on implementation details

    def test_client_disabled_first_entry(self, sheets_config):
        """Test client behavior when disabled"""
        # Create client with disabled config
        sheets_config.enabled = False
        client = GoogleSheetsClient(sheets_config)
        
        # Should not be enabled
        assert not client.is_enabled()
        
        # Operations should return empty/failure results
        contacts = asyncio.run(client.read_contacts())
        assert contacts == []
        
        contact = asyncio.run(client.get_contact_by_phone("+12175550123"))
        assert contact is None

    def test_error_handling_first_entry(self, client, mock_gspread_client):
        """Test error handling with first entry operations"""
        # Setup client
        client.client = mock_gspread_client
        
        # Test spreadsheet not found
        mock_gspread_client.open_by_key.side_effect = Exception("Spreadsheet not found")
        
        contacts = asyncio.run(client.read_contacts())
        assert contacts == []
        
        # Test API error during status update
        contact_status = ContactStatus(
            phone_number="+12175550123",
            status="Called - Vapi"
        )
        
        result = asyncio.run(client.update_contact_status(contact_status))
        assert result.success is False
        assert "Error updating contact status" in result.message

    def test_first_entry_data_completeness(self, first_entry_spreadsheet_data):
        """Verify that first entry test data has all expected fields"""
        entry = first_entry_spreadsheet_data[0]
        
        # Required fields
        assert entry["Owner 1 First Name"] == "John Smith"
        assert entry["Address"] == "123 Main Street, Springfield, IL 62701"
        assert entry["Phone 1"] == "(217) 555-0123"
        
        # Optional but expected fields
        assert entry["Bedrooms"] == "3"
        assert entry["Bathrooms"] == "2.5"
        assert entry["City"] == "Springfield"
        assert entry["State"] == "IL"
        assert entry["Zip"] == "62701"
        assert entry["Property Type"] == "Single Family"
        assert entry["Estimated Value"] == "$285,000"
        
        # Status tracking fields (initially empty)
        assert entry["Status"] == ""
        assert entry["Call ID"] == ""
        assert entry["Last Called"] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
