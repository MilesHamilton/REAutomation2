# Google Sheets Integration

This document describes how to set up and use the Google Sheets integration in the REAutomation2 project. The integration allows you to read contact data from Google Sheets and write call results back to spreadsheets.

## Overview

The Google Sheets integration provides:

- **Contact Management**: Read contact information (names, phone numbers, addresses) from Google Sheets
- **Call Status Tracking**: Update contact status when calls are made
- **Results Recording**: Write call results and analytics to output spreadsheets
- **Filtering**: Find contacts that haven't been called yet
- **API Endpoints**: RESTful API for all operations

## Setup Instructions

### 1. Google Cloud Project Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the following APIs:
   - Google Sheets API
   - Google Drive API

### 2. Service Account Creation

1. Go to **APIs & Services > Credentials**
2. Click **Create Credentials > Service Account**
3. Fill in the service account details
4. Click **Create and Continue**
5. Skip the optional steps and click **Done**
6. Click on the created service account
7. Go to the **Keys** tab
8. Click **Add Key > Create New Key**
9. Choose **JSON** format and download the file
10. Save the file as `credentials.json` in your project root

### 3. Google Sheets Setup

1. Create or open your Google Sheet with contact data
2. Ensure your sheet has the following columns:

   - `Owner 1 First Name` (required)
   - `Address` (required)
   - `Phone 1`, `Phone 2`, `Phone 3`, etc. (at least one required)
   - `Bedrooms`, `Bed`, or `BR` (optional)
   - `Bathrooms`, `Bath`, or `BA` (optional)

3. Share the spreadsheet:

   - Click the **Share** button
   - Add the service account email (found in your credentials.json)
   - Give it **Editor** permissions

4. Copy the spreadsheet ID from the URL:
   ```
   https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit
   ```

### 4. Environment Configuration

Add the following to your `.env` file:

```env
# Google Sheets Configuration
INPUT_SPREADSHEET_ID=your_input_spreadsheet_id_here
OUTPUT_SPREADSHEET_ID=your_output_spreadsheet_id_here
GOOGLE_SHEETS_CREDENTIALS_FILE=credentials.json
SHEETS_ENABLED=true
```

## Testing the Integration

Run the test script to verify everything is working:

```bash
python test_integration.py
```

This will:

- Check your configuration
- Test the connection to Google Sheets
- Read sample contacts
- Verify the integration is working properly

## Usage

### API Endpoints

Once the integration is set up, you can use these API endpoints:

#### Get Integration Status

```http
GET /integrations/status
```

#### Get Contacts

```http
GET /integrations/contacts?limit=10&exclude_called=true
```

#### Get Contacts Ready to Call

```http
GET /integrations/contacts/to-call?limit=5
```

#### Get Contact by Phone Number

```http
GET /integrations/contacts/phone/{phone_number}
```

#### Mark Contact as Called

```http
POST /integrations/contacts/mark-called
Content-Type: application/json

{
  "phone_number": "+15551234567",
  "call_id": "call_123",
  "status": "Called - Vapi"
}
```

#### Update Contact Status

```http
POST /integrations/contacts/update-status
Content-Type: application/json

{
  "phone_number": "+15551234567",
  "status": "Completed",
  "call_id": "call_123"
}
```

#### Test Connection

```http
GET /integrations/sheets/test-connection
```

### Python Code Usage

```python
from src.integrations.service import integration_service
from src.integrations.models import ContactFilter

# Get contacts ready to call
contacts = await integration_service.get_next_contacts_to_call(limit=10)

# Mark a contact as called
result = await integration_service.mark_contact_as_called(
    phone_number="+15551234567",
    call_id="call_123",
    status="Called - Vapi"
)

# Search for specific contacts
contacts = await integration_service.search_contacts(
    ContactFilter(name_contains="John", limit=5)
)
```

## Data Models

### Contact

```python
class Contact(BaseModel):
    name: str
    phone_number: str
    property_address: str
    bedrooms: Optional[str] = ""
    bathrooms: Optional[str] = ""
    phone_column: Optional[str] = None
    owner_last_name: Optional[str] = ""
    city: Optional[str] = ""
    state: Optional[str] = ""
    zip_code: Optional[str] = ""
    property_type: Optional[str] = ""
    estimated_value: Optional[str] = ""
```

### Call Result

```python
class CallResult(BaseModel):
    name: str
    phone_number: str
    property_address: str
    call_date: str
    call_summary: str
    selling_timeline: Optional[str] = ""
    bedrooms: Optional[str] = ""
    bathrooms: Optional[str] = ""
    asking_price: Optional[str] = ""
    call_duration: Optional[str] = ""
    qualification_score: Optional[float] = None
    interested: Optional[bool] = None
    follow_up_needed: Optional[bool] = None
    appointment_scheduled: Optional[bool] = None
    appointment_date: Optional[str] = ""
    notes: Optional[str] = ""
```

## Spreadsheet Format

### Input Spreadsheet (Contacts)

Your input spreadsheet should have these columns:

| Owner 1 First Name | Address     | Phone 1  | Phone 2  | Bedrooms | Bathrooms | Status        | Call ID  | Last Called         |
| ------------------ | ----------- | -------- | -------- | -------- | --------- | ------------- | -------- | ------------------- |
| John Smith         | 123 Main St | 555-1234 | 555-5678 | 3        | 2         |               |          |                     |
| Jane Doe           | 456 Oak Ave | 555-9876 |          | 4        | 3         | Called - Vapi | call_123 | 2024-01-15T10:30:00 |

### Output Spreadsheet (Call Results)

The output spreadsheet will have these columns:

| Name | Phone Number | Property Address | Call Date | Selling Timeline | Bedrooms | Bathrooms | Asking Price | Call Summary | Qualification Score | Interested | Follow Up Needed | Appointment Scheduled | Notes |
| ---- | ------------ | ---------------- | --------- | ---------------- | -------- | --------- | ------------ | ------------ | ------------------- | ---------- | ---------------- | --------------------- | ----- |

## Configuration Options

### Column Mapping

You can customize column names in the `SheetsConfig`:

```python
config = SheetsConfig(
    name_column="Owner 1 First Name",
    address_column="Address",
    phone_columns=["Phone 1", "Phone 2", "Phone 3"],
    bedrooms_columns=["Bedrooms", "Bed", "BR"],
    bathrooms_columns=["Bathrooms", "Bath", "BA"]
)
```

### Phone Number Formats

The integration handles various phone number formats:

- `(555) 123-4567`
- `555-123-4567`
- `5551234567`
- `+1 555 123 4567`
- `(555) 123-4567 - Wireless`

All formats are normalized to `+1XXXXXXXXXX` format.

## Error Handling

The integration includes comprehensive error handling:

- **Authentication errors**: Invalid credentials or permissions
- **Network errors**: Connection issues with Google Sheets API
- **Data validation errors**: Invalid phone numbers or missing required fields
- **Rate limiting**: Automatic retry with exponential backoff

## Troubleshooting

### Common Issues

1. **"Credentials file not found"**

   - Ensure `credentials.json` exists in the project root
   - Check the `GOOGLE_SHEETS_CREDENTIALS_FILE` environment variable

2. **"Permission denied"**

   - Make sure you shared the spreadsheet with the service account email
   - Verify the service account has Editor permissions

3. **"Spreadsheet not found"**

   - Check the `INPUT_SPREADSHEET_ID` in your `.env` file
   - Ensure the spreadsheet ID is correct (from the URL)

4. **"No contacts found"**
   - Verify your spreadsheet has data in the expected columns
   - Check that contacts have valid names, addresses, and phone numbers

### Debug Mode

Enable debug logging to see detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing Individual Components

```python
# Test authentication only
client = GoogleSheetsClient()
print(f"Enabled: {client.is_enabled()}")

# Test reading without filters
contacts = await client.read_contacts()
print(f"Total contacts: {len(contacts)}")

# Test specific phone number lookup
contact = await client.get_contact_by_phone("+15551234567")
print(f"Found: {contact.name if contact else 'Not found'}")
```

## Security Considerations

1. **Credentials Protection**

   - Never commit `credentials.json` to version control
   - Add `credentials.json` to your `.gitignore`
   - Use environment variables for sensitive configuration

2. **Spreadsheet Permissions**

   - Only share spreadsheets with the service account
   - Use the minimum required permissions (Editor for full functionality)
   - Regularly audit spreadsheet access

3. **Data Privacy**
   - Be aware that contact data is stored in Google Sheets
   - Ensure compliance with privacy regulations (GDPR, CCPA, etc.)
   - Consider data retention policies

## Performance Optimization

1. **Batch Operations**

   - Use filters to limit the number of contacts retrieved
   - Process contacts in batches rather than one at a time

2. **Caching**

   - The integration includes built-in caching for frequently accessed data
   - Consider implementing additional caching layers for high-volume usage

3. **Rate Limiting**
   - Google Sheets API has rate limits (100 requests per 100 seconds per user)
   - The integration includes automatic retry logic with exponential backoff

## Integration with Voice System

The Google Sheets integration is designed to work seamlessly with the voice calling system:

1. **Pre-call**: Get contacts ready to call
2. **During call**: Update status to "Called - Vapi"
3. **Post-call**: Save call results and analytics

Example workflow:

```python
# Get next contacts to call
contacts = await integration_service.get_next_contacts_to_call(limit=5)

for contact in contacts:
    # Start call
    call_id = start_voice_call(contact.phone_number)

    # Mark as called
    await integration_service.mark_contact_as_called(
        phone_number=contact.phone_number,
        call_id=call_id
    )

    # After call completion
    await integration_service.save_call_results(
        workflow_context=context,
        analytics=metrics
    )
```
