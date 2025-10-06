# Two-Tier TTS Outbound Lead Generation System

An intelligent voice-based lead generation system using LangChain/LangGraph orchestration with dual-tier TTS for cost-effective pre-screening and premium voice for qualified leads.

## Features

- **Local LLM Infrastructure**: Ollama with Llama 3.1 8B for cost-effective processing
- **Dual-Tier Voice System**: Local TTS for pre-screening, 11Labs for qualified leads
- **Multi-Agent Orchestration**: LangGraph-based agent workflows with comprehensive monitoring
- **Real-time Voice Processing**: Pipecat + Twilio WebRTC integration
- **Intelligent Lead Qualification**: ML-powered scoring and tier escalation
- **Advanced Cost Control**: Real-time budget management with multi-level alerts and enforcement
- **Production Monitoring**: LangSmith integration with circuit breaker and fallback mechanisms
- **Google Sheets Integration**: Complete contact management and results tracking
- **Cost-Optimized**: <$0.10 per call target with comprehensive budget controls

## Quick Start

### Prerequisites

- Python 3.12+ (tested and optimized for Python 3.12)
- Ollama (for local LLM)
- PostgreSQL 16+
- Redis
- GPU with 6GB+ VRAM (recommended)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/MilesHamilton/REAutomation2.git
cd REAutomation2
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install system dependencies (Ubuntu/Debian):

```bash
# Install PostgreSQL
sudo apt update
sudo apt install -y postgresql postgresql-contrib

# Install Ollama
sudo snap install ollama
```

4. Install Python dependencies:

```bash
pip install -r requirements.txt
```

5. Set up PostgreSQL database:

```bash
# Create database user and database
sudo -u postgres psql -c "CREATE USER reautomation_user WITH PASSWORD 'reautomation_pass' SUPERUSER CREATEDB CREATEROLE;"
sudo -u postgres createdb -O reautomation_user reautomation2

# Create system user for peer authentication (if needed)
sudo adduser --system --no-create-home reautomation_user
```

6. Set up environment variables:

```bash
cp .env.template .env
# The .env file is already configured with working defaults
# Update API keys and credentials as needed
```

7. Set up Ollama and download model:

```bash
# Pull the required model (4.7GB download)
ollama pull llama3.1:8b-instruct-q4_0

# Verify model is working
ollama run llama3.1:8b-instruct-q4_0 "Hello, can you confirm you're working?"
```

8. Start the database:

```bash
# Start PostgreSQL service
sudo systemctl start postgresql

# Enable PostgreSQL to start on boot (optional)
sudo systemctl enable postgresql

# Verify PostgreSQL is running
sudo systemctl status postgresql
```

9. Initialize database:

```bash
# Run database migrations to create all tables
alembic upgrade head
```

**What does `alembic upgrade head` do?**
- **Alembic** is a database migration tool that manages database schema changes
- **`upgrade head`** applies all pending migrations to create the database tables and indexes
- This command creates all the necessary tables for the application:
  - `calls` - Store call records and metrics
  - `contacts` - Lead and contact information
  - `conversation_history` - Chat messages and responses
  - `workflow_traces` - LangSmith monitoring data
  - `agent_executions` - Agent performance tracking
  - `cost_tracking` - Budget and cost management
  - And several other supporting tables
- **Why it's needed**: The application requires these database tables to store call data, track costs, monitor performance, and manage leads

10. Verify installation:

```bash
# Test database connection
python -c "from src.database.connection import get_database_url; print('Database URL configured:', get_database_url())"

# Check Ollama status
ollama list
```

## Database Management

### Starting the Database

```bash
# Start PostgreSQL service
sudo systemctl start postgresql

# Enable PostgreSQL to start automatically on boot
sudo systemctl enable postgresql
```

### Checking Database Status

```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Check which port PostgreSQL is running on
sudo netstat -tlnp | grep postgres

# Test database connection
python -c "from src.database.connection import db_manager; import asyncio; asyncio.run(db_manager.health_check())"
```

### Stopping the Database

```bash
# Stop PostgreSQL service
sudo systemctl stop postgresql

# Disable PostgreSQL from starting on boot
sudo systemctl disable postgresql
```

### Database Troubleshooting

```bash
# Restart PostgreSQL if having issues
sudo systemctl restart postgresql

# View PostgreSQL logs
sudo journalctl -u postgresql -f

# Connect to database directly
psql -U reautomation_user -d reautomation2 -h localhost
```

### Development

Run with auto-reload:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

**Note**: Some tests may have dependency issues due to missing optional packages. The core functionality tests should work.

Run tests:

```bash
# Run all tests (some may fail due to missing dependencies)
pytest

# Run only the standalone tests (these should work)
pytest tests/test_standalone.py -v

# Run tests with more verbose output
pytest -v --tb=short
```

Run with coverage:

```bash
pytest --cov=src --cov-report=html
```

**Common Test Issues:**
- Some tests require additional dependencies that may not be installed
- Voice processing tests need audio libraries that may have compatibility issues with Python 3.12
- Database tests require a running PostgreSQL instance

### Linting

Format code:

```bash
black src/ tests/
isort src/ tests/
```

Type checking:

```bash
mypy src/
```

Lint:

```bash
flake8 src/ tests/
```

## Architecture

### Phase 1: Foundation (Weeks 1-2)

- Local LLM setup with Ollama
- Pipecat voice pipeline integration
- LangGraph multi-agent orchestration

### Phase 2: Pre-Screening (Weeks 3-4)

- Conversation qualification logic
- Lead scoring algorithms
- Call management system

### Phase 3: Premium Voice (Weeks 5-7)

- 11Labs integration
- Seamless tier switching
- Advanced conversation handling

### Phase 4: Learning & Optimization (Weeks 8-10)

- ML-driven improvements
- Script evolution
- Performance optimization

## Configuration

Key configuration parameters in `.env`:

- **LLM_MAX_CONCURRENT**: Maximum concurrent LLM requests (default: 5)
- **MAX_CONCURRENT_CALLS**: Maximum simultaneous calls (default: 5)
- **QUALIFICATION_THRESHOLD**: Lead qualification score threshold (default: 0.8)
- **DAILY_BUDGET**: Daily spending limit (default: $50.00)
- **COST_PER_CALL_LIMIT**: Maximum cost per call (default: $0.10)

## API Endpoints

### Core Operations

- `GET /health` - System health check and component status
- `POST /calls/start` - Start outbound call with lead data
- `GET /calls/{call_id}/status` - Get real-time call status and metrics

### Lead Management

- `POST /leads/import` - Import leads from Google Sheets
- `GET /leads/{lead_id}` - Get lead information and call history
- `PUT /leads/{lead_id}/status` - Update lead status

### Integration Services

- `GET /integrations/sheets/contacts` - List Google Sheets contacts
- `POST /integrations/sheets/results` - Write call results to sheets
- `GET /integrations/health` - Integration service health status

### Monitoring & Analytics

- `GET /monitoring/traces` - Get workflow execution traces
- `GET /monitoring/agents/{agent_type}/performance` - Agent performance metrics
- `GET /analytics/performance` - Overall system performance metrics
- `GET /analytics/costs` - Cost analysis and budget status

### Cost Control

- `GET /costs/budget/status` - Current budget utilization
- `POST /costs/budget/update` - Update budget limits
- `GET /costs/trends` - Cost trends and projections

## Key Components

### LangSmith Monitoring

- **Circuit Breaker Protection**: Automatic failover for monitoring API calls
- **Batch Processing**: Efficient handling of high-volume trace data
- **Workflow Tracing**: Complete visibility into agent execution flows
- **Performance Analytics**: Real-time metrics on agent performance and costs

### Cost Control System

- **Multi-Level Budgets**: Daily, weekly, and monthly budget tracking
- **Real-Time Alerts**: WARNING, CRITICAL, and EMERGENCY alert levels
- **Automatic Enforcement**: Call blocking when budget thresholds exceeded
- **Cost Analytics**: Detailed cost breakdowns and trend analysis

### Google Sheets Integration

- **Contact Management**: Read and update contact information
- **Results Tracking**: Automatic call result recording
- **Status Updates**: Real-time lead status synchronization
- **Phone Number Processing**: Smart parsing of various phone formats

## Troubleshooting

### Common Installation Issues

#### Python 3.12 Dependency Issues
If you encounter dependency conflicts during installation:

```bash
# Ensure setuptools is installed for distutils compatibility
pip install --upgrade setuptools

# Install packages individually if needed
pip install langchain langgraph pipecat-ai transformers torchaudio
```

#### PostgreSQL Connection Issues
If Alembic migrations fail with connection errors:

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Restart PostgreSQL if needed
sudo systemctl restart postgresql

# Check which port PostgreSQL is running on
sudo netstat -tlnp | grep postgres

# Update .env file with correct port (usually 5432 or 5433)
DATABASE_URL=postgresql://reautomation_user:reautomation_pass@localhost:5433/reautomation2
```

#### Ollama Model Issues
If Ollama fails to load the model:

```bash
# Check Ollama service status
ollama list

# Restart Ollama service if needed
sudo systemctl restart snap.ollama.ollama

# Re-pull the model if corrupted
ollama pull llama3.1:8b-instruct-q4_0
```

#### Permission Issues
If you encounter permission errors:

```bash
# Ensure proper ownership of project files
sudo chown -R $USER:$USER /path/to/REAutomation2

# For PostgreSQL peer authentication issues, ensure system user exists
sudo adduser --system --no-create-home reautomation_user
```

### Performance Optimization

#### GPU Memory Management
For systems with limited GPU memory:

```bash
# Reduce GPU memory limit in .env
LLM_GPU_MEMORY_LIMIT=4096  # Reduce from 6144 to 4096MB
```

#### Concurrent Call Limits
Adjust based on system resources:

```bash
# In .env file
MAX_CONCURRENT_CALLS=3      # Reduce from 5 for lower-spec systems
LLM_MAX_CONCURRENT=3        # Match LLM concurrency to call limits
```

### Getting Help

- **Issues**: Report bugs on [GitHub Issues](https://github.com/MilesHamilton/REAutomation2/issues)
- **Documentation**: Check the `/docs` folder for detailed component documentation
- **Logs**: Check application logs in `/logs` directory for debugging information

## License

MIT License - see LICENSE file for details.
