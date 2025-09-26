"""
Standalone test to verify our testing framework works
This bypasses the complex dependency imports
"""
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_basic_functionality():
    """Test that basic testing works"""
    assert True

def test_math_operations():
    """Test basic math operations"""
    assert 2 + 2 == 4
    assert 10 * 5 == 50
    assert 100 / 4 == 25.0

@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality works"""
    import asyncio
    
    async def async_function():
        await asyncio.sleep(0.01)
        return "success"
    
    result = await async_function()
    assert result == "success"

def test_mock_functionality():
    """Test that mocking works"""
    from unittest.mock import Mock, patch
    
    # Test basic mocking
    mock_obj = Mock()
    mock_obj.method.return_value = "mocked"
    
    assert mock_obj.method() == "mocked"
    mock_obj.method.assert_called_once()

def test_google_sheets_models():
    """Test Google Sheets models work independently"""
    try:
        from integrations.models import Contact, ContactStatus, SheetsConfig
        
        # Test ContactStatus enum
        assert ContactStatus.NEW == "new"
        assert ContactStatus.CONTACTED == "contacted"
        
        # Test Contact model
        contact = Contact(
            name="Test User",
            phone="+1234567890",
            email="test@example.com",
            company="Test Corp",
            status=ContactStatus.NEW
        )
        
        assert contact.name == "Test User"
        assert contact.phone == "+1234567890"
        assert contact.status == ContactStatus.NEW
        
        # Test SheetsConfig model
        config = SheetsConfig(
            credentials_file="test.json",
            contacts_sheet_id="sheet1",
            results_sheet_id="sheet2"
        )
        
        assert config.credentials_file == "test.json"
        assert config.contacts_sheet_id == "sheet1"
        
        print("✅ Google Sheets models work correctly")
        
    except ImportError as e:
        print(f"⚠️ Google Sheets models not available: {e}")
        pytest.skip("Google Sheets models not available")

def test_agent_models():
    """Test agent models work independently"""
    try:
        from agents.models import AgentType, WorkflowState
        
        # Test AgentType enum
        assert AgentType.CONVERSATION == "conversation"
        assert AgentType.QUALIFICATION == "qualification"
        
        # Test WorkflowState enum
        assert WorkflowState.GREETING == "greeting"
        assert WorkflowState.QUALIFYING == "qualifying"
        
        print("✅ Agent models work correctly")
        
    except ImportError as e:
        print(f"⚠️ Agent models not available: {e}")
        pytest.skip("Agent models not available")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
