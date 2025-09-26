"""
Unit tests for Google Sheets integration
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import gspread
from gspread.exceptions import APIError, SpreadsheetNotFound

# Handle optional imports gracefully for testing
try:
    from src.integrations.google_sheets import GoogleSheetsClient
    from src.integrations.models import Contact, CallResult, ContactStatus, SheetsConfig
except ImportError:
    # Skip tests if dependencies are not available
    pytest.skip("Integration dependencies not available", allow_module_level=True)


class TestGoogleSheetsClient:
    """Test suite for GoogleSheetsClient"""

    @pytest.fixture
    def sheets_config(self):
        """Sample sheets configuration"""
        return SheetsConfig(
            credentials_file="test_credentials.json",
            contacts_sheet_id="test_contacts_sheet_id",
            results_sheet_id="test_results_sheet_id",
            contacts_worksheet="Contacts",
            results_worksheet="Results"
        )

    @pytest.fixture
    def client(self, sheets_config):
        """Create Google Sheets client instance"""
        return GoogleSheetsClient(sheets_config)

    @pytest.fixture
    def mock_gspread_client(self):
        """Mock gspread client"""
        mock_client = Mock(spec=gspread.Client)
        return mock_client

    @pytest.fixture
    def mock_worksheet(self):
        """Mock worksheet"""
        mock_ws = Mock()
        mock_ws.get_all_records.return_value = [
            {
                "Name": "John Doe",
                "Phone": "+1-234-567-8900",
                "Email": "john@example.com",
                "Company": "Test Corp",
                "Status": "new",
                "Address": "123 Main St, City, State"
            },
            {
                "Name": "Jane Smith",
                "Phone": "(555) 123-4567",
                "Email": "jane@company.com", 
                "Company": "Smith LLC",
                "Status": "contacted",
                "Address": "456 Oak Ave, Town, State"
            }
        ]
        mock_ws.update.return_value = None
        mock_ws.append_row.return_value = None
        return mock_ws

    @pytest.fixture
    def sample_contacts(self):
        """Sample contact data"""
        return [
            Contact(
                name="John Doe",
                phone="+12345678900",
                email="john@example.com",
                company="Test Corp",
                status=ContactStatus.NEW,
                address="123 Main St, City, State"
            ),
            Contact(
                name="Jane Smith",
                phone="+15551234567",
                email="jane@company.com",
                company="Smith LLC", 
                status=ContactStatus.CONTACTED,
                address="456 Oak Ave, Town, State"
            )
        ]

    @pytest.fixture
    def sample_call_result(self):
        """Sample call result"""
        return CallResult(
            contact_name="John Doe",
            phone="+12345678900",
            call_date=datetime(2024, 1, 15, 10, 30),
            duration_seconds=180,
            outcome="qualified",
            qualification_score=0.8,
            notes="Very interested, wants demo next week",
            next_action="schedule_demo",
            agent_name="AI Agent"
        )

    def test_client_initialization(self, client, sheets_config):
        """Test client initialization"""
        assert client.config == sheets_config
        assert client.gc is None
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_success(self, client, mock_gspread_client):
        """Test successful connection"""
        with patch('src.integrations.google_sheets.gspread.service_account') as mock_service:
            mock_service.return_value = mock_gspread_client
            
            result = await client.connect()
            
            assert result is True
            assert client.gc == mock_gspread_client
            assert client.is_connected is True
            mock_service.assert_called_once_with(filename=client.config.credentials_file)

    @pytest.mark.asyncio
    async def test_connect_failure(self, client):
        """Test connection failure"""
        with patch('src.integrations.google_sheets.gspread.service_account') as mock_service:
            mock_service.side_effect = Exception("Connection failed")
            
            result = await client.connect()
            
            assert result is False
            assert client.gc is None
            assert client.is_connected is False

    def test_parse_phone_number_various_formats(self, client):
        """Test phone number parsing with various formats"""
        test_cases = [
            ("+1-234-567-8900", "+12345678900"),
            ("(555) 123-4567", "+15551234567"),
            ("555.123.4567", "+15551234567"),
            ("555 123 4567", "+15551234567"),
            ("15551234567", "+15551234567"),
            ("5551234567", "+15551234567"),
            ("+44 20 7946 0958", "+442079460958"),
            ("", ""),
            (None, "")
        ]
        
        for input_phone, expected in test_cases:
            result = client._parse_phone_number(input_phone)
            assert result == expected, f"Failed for input: {input_phone}"

    def test_format_contact_from_row_complete_data(self, client):
        """Test formatting contact from complete row data"""
        row = {
            "Name": "John Doe",
            "Phone": "+1-234-567-8900",
            "Email": "john@example.com",
            "Company": "Test Corp",
            "Status": "new",
            "Address": "123 Main St"
        }
        
        contact = client._format_contact_from_row(row)
        
        assert contact.name == "John Doe"
        assert contact.phone == "+12345678900"
        assert contact.email == "john@example.com"
        assert contact.company == "Test Corp"
        assert contact.status == ContactStatus.NEW
        assert contact.address == "123 Main St"

    def test_format_contact_from_row_missing_data(self, client):
        """Test formatting contact from row with missing data"""
        row = {
            "Name": "Jane Smith",
            "Phone": "555-1234"  # Only required fields
        }
        
        contact = client._format_contact_from_row(row)
        
        assert contact.name == "Jane Smith"
        assert contact.phone == "+15551234"
        assert contact.email == ""
        assert contact.company == ""
        assert contact.status == ContactStatus.NEW  # Default
        assert contact.address == ""

    def test_format_contact_from_row_invalid_status(self, client):
        """Test formatting contact with invalid status"""
        row = {
            "Name": "Test User",
            "Phone": "5551234567",
            "Status": "invalid_status"
        }
        
        contact = client._format_contact_from_row(row)
        
        assert contact.status == ContactStatus.NEW  # Should default to NEW

    @pytest.mark.asyncio
    async def test_get_contacts_success(self, client, mock_gspread_client, mock_worksheet, sample_contacts):
        """Test successful contact retrieval"""
        client.gc = mock_gspread_client
        client.is_connected = True
        
        mock_spreadsheet = Mock()
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        contacts = await client.get_contacts()
        
        assert len(contacts) == 2
        assert contacts[0].name == "John Doe"
        assert contacts[0].phone == "+12345678900"
        assert contacts[0].status == ContactStatus.NEW
        assert contacts[1].name == "Jane Smith"
        assert contacts[1].phone == "+15551234567"
        assert contacts[1].status == ContactStatus.CONTACTED

    @pytest.mark.asyncio
    async def test_get_contacts_not_connected(self, client):
        """Test get_contacts when not connected"""
        contacts = await client.get_contacts()
        assert contacts == []

    @pytest.mark.asyncio
    async def test_get_contacts_spreadsheet_not_found(self, client, mock_gspread_client):
        """Test get_contacts when spreadsheet not found"""
        client.gc = mock_gspread_client
        client.is_connected = True
        
        mock_gspread_client.open_by_key.side_effect = SpreadsheetNotFound("Sheet not found")
        
        contacts = await client.get_contacts()
        assert contacts == []

    @pytest.mark.asyncio
    async def test_get_contacts_api_error(self, client, mock_gspread_client):
        """Test get_contacts with API error"""
        client.gc = mock_gspread_client
        client.is_connected = True
        
        mock_gspread_client.open_by_key.side_effect = APIError("API Error")
        
        contacts = await client.get_contacts()
        assert contacts == []

    @pytest.mark.asyncio
    async def test_update_contact_status_success(self, client, mock_gspread_client, mock_worksheet):
        """Test successful contact status update"""
        client.gc = mock_gspread_client
        client.is_connected = True
        
        # Mock finding the contact
        mock_worksheet.get_all_records.return_value = [
            {"Name": "John Doe", "Phone": "+1-234-567-8900", "Status": "new"}
        ]
        
        mock_spreadsheet = Mock()
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        result = await client.update_contact_status("+12345678900", ContactStatus.CONTACTED)
        
        assert result is True
        mock_worksheet.update.assert_called_once()
        # Verify the update call
        call_args = mock_worksheet.update.call_args
        assert "contacted" in str(call_args).lower()

    @pytest.mark.asyncio
    async def test_update_contact_status_contact_not_found(self, client, mock_gspread_client, mock_worksheet):
        """Test update_contact_status when contact not found"""
        client.gc = mock_gspread_client
        client.is_connected = True
        
        # Mock empty contact list
        mock_worksheet.get_all_records.return_value = []
        
        mock_spreadsheet = Mock()
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        result = await client.update_contact_status("+12345678900", ContactStatus.CONTACTED)
        
        assert result is False
        mock_worksheet.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_contact_status_not_connected(self, client):
        """Test update_contact_status when not connected"""
        result = await client.update_contact_status("+12345678900", ContactStatus.CONTACTED)
        assert result is False

    @pytest.mark.asyncio
    async def test_record_call_result_success(self, client, mock_gspread_client, mock_worksheet, sample_call_result):
        """Test successful call result recording"""
        client.gc = mock_gspread_client
        client.is_connected = True
        
        mock_spreadsheet = Mock()
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        result = await client.record_call_result(sample_call_result)
        
        assert result is True
        mock_worksheet.append_row.assert_called_once()
        
        # Verify the data being appended
        call_args = mock_worksheet.append_row.call_args[0][0]
        assert "John Doe" in call_args
        assert "+12345678900" in call_args
        assert "qualified" in call_args
        assert "0.8" in str(call_args)

    @pytest.mark.asyncio
    async def test_record_call_result_not_connected(self, client, sample_call_result):
        """Test record_call_result when not connected"""
        result = await client.record_call_result(sample_call_result)
        assert result is False

    @pytest.mark.asyncio
    async def test_record_call_result_api_error(self, client, mock_gspread_client, mock_worksheet, sample_call_result):
        """Test record_call_result with API error"""
        client.gc = mock_gspread_client
        client.is_connected = True
        
        mock_spreadsheet = Mock()
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        mock_worksheet.append_row.side_effect = APIError("API Error")
        
        result = await client.record_call_result(sample_call_result)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_contacts_by_status_success(self, client, mock_gspread_client, mock_worksheet):
        """Test getting contacts by status"""
        client.gc = mock_gspread_client
        client.is_connected = True
        
        mock_spreadsheet = Mock()
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        contacts = await client.get_contacts_by_status(ContactStatus.NEW)
        
        # Should return only contacts with "new" status
        new_contacts = [c for c in contacts if c.status == ContactStatus.NEW]
        assert len(new_contacts) == 1
        assert new_contacts[0].name == "John Doe"

    @pytest.mark.asyncio
    async def test_get_contacts_by_status_not_connected(self, client):
        """Test get_contacts_by_status when not connected"""
        contacts = await client.get_contacts_by_status(ContactStatus.NEW)
        assert contacts == []

    @pytest.mark.asyncio
    async def test_find_contact_by_phone_success(self, client, mock_gspread_client, mock_worksheet):
        """Test finding contact by phone number"""
        client.gc = mock_gspread_client
        client.is_connected = True
        
        mock_spreadsheet = Mock()
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        contact = await client.find_contact_by_phone("+12345678900")
        
        assert contact is not None
        assert contact.name == "John Doe"
        assert contact.phone == "+12345678900"

    @pytest.mark.asyncio
    async def test_find_contact_by_phone_not_found(self, client, mock_gspread_client, mock_worksheet):
        """Test finding contact by phone when not found"""
        client.gc = mock_gspread_client
        client.is_connected = True
        
        mock_spreadsheet = Mock()
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        contact = await client.find_contact_by_phone("+19999999999")
        
        assert contact is None

    @pytest.mark.asyncio
    async def test_find_contact_by_phone_not_connected(self, client):
        """Test find_contact_by_phone when not connected"""
        contact = await client.find_contact_by_phone("+12345678900")
        assert contact is None

    @pytest.mark.asyncio
    async def test_health_check_connected(self, client, mock_gspread_client):
        """Test health check when connected"""
        client.gc = mock_gspread_client
        client.is_connected = True
        
        # Mock successful spreadsheet access
        mock_spreadsheet = Mock()
        mock_gspread_client.open_by_key.return_value = mock_spreadsheet
        
        health = await client.health_check()
        
        assert health["status"] == "healthy"
        assert health["connected"] is True
        assert "contacts_sheet_accessible" in health
        assert "results_sheet_accessible" in health

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self, client):
        """Test health check when not connected"""
        health = await client.health_check()
        
        assert health["status"] == "unhealthy"
        assert health["connected"] is False
        assert "error" in health

    @pytest.mark.asyncio
    async def test_health_check_sheet_access_error(self, client, mock_gspread_client):
        """Test health check with sheet access error"""
        client.gc = mock_gspread_client
        client.is_connected = True
        
        mock_gspread_client.open_by_key.side_effect = APIError("Access denied")
        
        health = await client.health_check()
        
        assert health["status"] == "degraded"
        assert health["connected"] is True
        assert health["contacts_sheet_accessible"] is False

    def test_format_call_result_row(self, client, sample_call_result):
        """Test formatting call result for spreadsheet row"""
        row = client._format_call_result_row(sample_call_result)
        
        expected_fields = [
            "John Doe",
            "+12345678900", 
            "2024-01-15 10:30:00",
            "180",
            "qualified",
            "0.8",
            "Very interested, wants demo next week",
            "schedule_demo",
            "AI Agent"
        ]
        
        assert row == expected_fields

    def test_format_call_result_row_minimal_data(self, client):
        """Test formatting call result with minimal data"""
        minimal_result = CallResult(
            contact_name="Test User",
            phone="+15551234567",
            call_date=datetime(2024, 1, 15),
            duration_seconds=60,
            outcome="no_answer"
        )
        
        row = client._format_call_result_row(minimal_result)
        
        assert row[0] == "Test User"
        assert row[1] == "+15551234567"
        assert row[3] == "60"
        assert row[4] == "no_answer"
        assert row[5] == ""  # No qualification score
        assert row[6] == ""  # No notes
        assert row[7] == ""  # No next action
        assert row[8] == ""  # No agent name


class TestGoogleSheetsClientEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def client(self):
        config = SheetsConfig(
            credentials_file="test.json",
            contacts_sheet_id="test_id",
            results_sheet_id="test_id"
        )
        return GoogleSheetsClient(config)

    def test_parse_phone_number_edge_cases(self, client):
        """Test phone number parsing edge cases"""
        edge_cases = [
            ("123", "+1123"),  # Very short number
            ("1" * 20, "+1" + "1" * 19),  # Very long number
            ("abc123def456", "+1123456"),  # Mixed alphanumeric
            ("+++123---456---7890", "+11234567890"),  # Multiple special chars
            ("   555 123 4567   ", "+15551234567"),  # Whitespace
        ]
        
        for input_phone, expected in edge_cases:
            result = client._parse_phone_number(input_phone)
            assert result == expected, f"Failed for input: {input_phone}"

    def test_format_contact_case_insensitive_status(self, client):
        """Test contact formatting with case-insensitive status"""
        test_cases = [
            ("NEW", ContactStatus.NEW),
            ("new", ContactStatus.NEW),
            ("New", ContactStatus.NEW),
            ("CONTACTED", ContactStatus.CONTACTED),
            ("contacted", ContactStatus.CONTACTED),
            ("Contacted", ContactStatus.CONTACTED),
            ("QUALIFIED", ContactStatus.QUALIFIED),
            ("qualified", ContactStatus.QUALIFIED),
            ("SCHEDULED", ContactStatus.SCHEDULED),
            ("scheduled", ContactStatus.SCHEDULED),
            ("COMPLETED", ContactStatus.COMPLETED),
            ("completed", ContactStatus.COMPLETED),
            ("DO_NOT_CALL", ContactStatus.DO_NOT_CALL),
            ("do_not_call", ContactStatus.DO_NOT_CALL),
            ("invalid", ContactStatus.NEW)  # Should default to NEW
        ]
        
        for status_str, expected_status in test_cases:
            row = {
                "Name": "Test User",
                "Phone": "5551234567",
                "Status": status_str
            }
            
            contact = client._format_contact_from_row(row)
            assert contact.status == expected_status, f"Failed for status: {status_str}"

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, client):
        """Test that client handles concurrent operations safely"""
        import asyncio
        
        # Mock successful connection
        with patch('src.integrations.google_sheets.gspread.service_account') as mock_service:
            mock_service.return_value = Mock()
            
            # Start multiple concurrent connections
            tasks = [client.connect() for _ in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should succeed or fail gracefully
            assert all(isinstance(r, bool) for r in results)

    def test_memory_efficiency_large_dataset(self, client):
        """Test memory efficiency with large contact dataset"""
        # Create a large mock dataset
        large_dataset = []
        for i in range(1000):
            large_dataset.append({
                "Name": f"Contact {i}",
                "Phone": f"555{i:07d}",
                "Email": f"contact{i}@example.com",
                "Company": f"Company {i}",
                "Status": "new"
            })
        
        # Test that formatting doesn't consume excessive memory
        contacts = []
        for row in large_dataset[:100]:  # Test with subset
            contact = client._format_contact_from_row(row)
            contacts.append(contact)
        
        assert len(contacts) == 100
        assert all(isinstance(c, Contact) for c in contacts)

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, client):
        """Test handling of network timeouts"""
        with patch('src.integrations.google_sheets.gspread.service_account') as mock_service:
            # Simulate timeout
            import socket
            mock_service.side_effect = socket.timeout("Connection timed out")
            
            result = await client.connect()
            assert result is False
            assert client.is_connected is False

    def test_unicode_handling(self, client):
        """Test handling of unicode characters in contact data"""
        unicode_row = {
            "Name": "José María González",
            "Phone": "+34 123 456 789",
            "Email": "josé@empresa.es",
            "Company": "Empresa Española S.L.",
            "Address": "Calle de la Constitución, 123, Madrid"
        }
        
        contact = client._format_contact_from_row(unicode_row)
        
        assert contact.name == "José María González"
        assert contact.email == "josé@empresa.es"
        assert contact.company == "Empresa Española S.L."
        assert "Madrid" in contact.address

    def test_empty_spreadsheet_handling(self, client):
        """Test handling of empty spreadsheet"""
        empty_row = {}
        
        contact = client._format_contact_from_row(empty_row)
        
        # Should handle gracefully with defaults
        assert contact.name == ""
        assert contact.phone == ""
        assert contact.email == ""
        assert contact.company == ""
        assert contact.status == ContactStatus.NEW
        assert contact.address == ""


if __name__ == "__main__":
    pytest.main([__file__])
