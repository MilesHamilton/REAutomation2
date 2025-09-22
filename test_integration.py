#!/usr/bin/env python3
"""
Test script for Google Sheets integration

This script tests the Google Sheets integration without requiring the full API server.
Run this to verify that your Google Sheets credentials and configuration are working.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from integrations.google_sheets import GoogleSheetsClient
from integrations.models import ContactFilter
from config.settings import settings


async def test_google_sheets_integration():
    """Test the Google Sheets integration."""
    print("🔍 Testing Google Sheets Integration")
    print("=" * 50)
    
    # Check configuration
    print(f"📋 Configuration Check:")
    print(f"   Credentials file: {settings.google_sheets_credentials_file}")
    print(f"   Input spreadsheet ID: {settings.input_spreadsheet_id}")
    print(f"   Output spreadsheet ID: {settings.output_spreadsheet_id}")
    print(f"   Sheets enabled: {settings.sheets_enabled}")
    
    # Check if credentials file exists
    if not os.path.exists(settings.google_sheets_credentials_file):
        print(f"❌ Credentials file not found: {settings.google_sheets_credentials_file}")
        print("   Please ensure you have downloaded your Google Service Account credentials")
        print("   and placed them in the correct location.")
        return False
    else:
        print(f"✅ Credentials file found")
    
    # Check if spreadsheet IDs are configured
    if not settings.input_spreadsheet_id:
        print("❌ INPUT_SPREADSHEET_ID not configured in environment variables")
        return False
    else:
        print(f"✅ Input spreadsheet ID configured")
    
    print("\n🔗 Testing Connection...")
    
    try:
        # Initialize the client
        client = GoogleSheetsClient()
        
        if not client.is_enabled():
            print("❌ Google Sheets client is not enabled or properly configured")
            return False
        
        print("✅ Google Sheets client initialized successfully")
        
        # Test reading contacts
        print("\n📖 Testing contact reading...")
        contacts = await client.read_contacts(ContactFilter(limit=5))
        
        if not contacts:
            print("⚠️  No contacts found in the spreadsheet")
            print("   This might be normal if your spreadsheet is empty or")
            print("   if contacts don't have valid names, addresses, and phone numbers")
        else:
            print(f"✅ Successfully read {len(contacts)} contacts")
            print("\n📋 Sample contacts:")
            for i, contact in enumerate(contacts[:3], 1):
                print(f"   {i}. {contact.name}")
                print(f"      Phone: {contact.phone_number}")
                print(f"      Address: {contact.property_address}")
                if contact.bedrooms or contact.bathrooms:
                    print(f"      Property: {contact.bedrooms} bed, {contact.bathrooms} bath")
                print()
        
        # Test getting contacts to call
        print("📞 Testing contacts to call...")
        contacts_to_call = await client.get_contacts_to_call(limit=3)
        
        if not contacts_to_call:
            print("⚠️  No contacts available to call")
            print("   This might mean all contacts have already been called")
            print("   or no contacts meet the criteria for calling")
        else:
            print(f"✅ Found {len(contacts_to_call)} contacts ready to call")
        
        print("\n🎉 Google Sheets integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False


async def test_environment_variables():
    """Test that all required environment variables are set."""
    print("\n🔧 Environment Variables Check")
    print("=" * 50)
    
    required_vars = [
        "INPUT_SPREADSHEET_ID",
        "GOOGLE_SHEETS_CREDENTIALS_FILE"
    ]
    
    optional_vars = [
        "OUTPUT_SPREADSHEET_ID",
        "SHEETS_ENABLED"
    ]
    
    all_good = True
    
    print("Required variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            display_value = value[:10] + "..." if len(value) > 10 else value
            print(f"   ✅ {var}: {display_value}")
        else:
            print(f"   ❌ {var}: Not set")
            all_good = False
    
    print("\nOptional variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            display_value = value[:10] + "..." if len(value) > 10 else value
            print(f"   ✅ {var}: {display_value}")
        else:
            print(f"   ⚠️  {var}: Not set (using default)")
    
    return all_good


def print_setup_instructions():
    """Print setup instructions for Google Sheets integration."""
    print("\n📚 Setup Instructions")
    print("=" * 50)
    print("To set up Google Sheets integration:")
    print()
    print("1. Create a Google Cloud Project:")
    print("   - Go to https://console.cloud.google.com/")
    print("   - Create a new project or select an existing one")
    print()
    print("2. Enable Google Sheets API:")
    print("   - Go to APIs & Services > Library")
    print("   - Search for 'Google Sheets API' and enable it")
    print("   - Also enable 'Google Drive API'")
    print()
    print("3. Create Service Account:")
    print("   - Go to APIs & Services > Credentials")
    print("   - Click 'Create Credentials' > 'Service Account'")
    print("   - Download the JSON key file")
    print()
    print("4. Share your Google Sheet:")
    print("   - Open your Google Sheet")
    print("   - Click 'Share' and add the service account email")
    print("   - Give it 'Editor' permissions")
    print()
    print("5. Configure environment variables:")
    print("   - Copy the spreadsheet ID from the URL")
    print("   - Set INPUT_SPREADSHEET_ID in your .env file")
    print("   - Set GOOGLE_SHEETS_CREDENTIALS_FILE to the path of your JSON file")
    print()
    print("Example .env configuration:")
    print("INPUT_SPREADSHEET_ID=1abc123def456ghi789jkl...")
    print("OUTPUT_SPREADSHEET_ID=1xyz987wvu654tsr321qpo...")
    print("GOOGLE_SHEETS_CREDENTIALS_FILE=credentials.json")
    print("SHEETS_ENABLED=true")


async def main():
    """Main test function."""
    print("🚀 Google Sheets Integration Test")
    print("=" * 50)
    
    # Test environment variables first
    env_ok = await test_environment_variables()
    
    if not env_ok:
        print("\n❌ Environment variables are not properly configured")
        print_setup_instructions()
        return
    
    # Test the actual integration
    success = await test_google_sheets_integration()
    
    if not success:
        print("\n❌ Integration test failed")
        print_setup_instructions()
    else:
        print("\n✅ All tests passed! Google Sheets integration is working correctly.")
        print("\nYou can now use the integration in your application:")
        print("- Start the API server: python -m src.api.main")
        print("- Visit http://localhost:8000/docs to see the integration endpoints")
        print("- Use /integrations/contacts/to-call to get contacts ready to call")
        print("- Use /integrations/contacts/mark-called to update call status")


if __name__ == "__main__":
    asyncio.run(main())
