# Two-Tier TTS Outbound Lead Generation System

An intelligent voice-based lead generation system using LangChain/LangGraph orchestration with dual-tier TTS for cost-effective pre-screening and premium voice for qualified leads.

## Features

- **Local LLM Infrastructure**: Ollama with Llama 3.1 8B for cost-effective processing
- **Dual-Tier Voice System**: Local TTS for pre-screening, 11Labs for qualified leads
- **Multi-Agent Orchestration**: LangGraph-based agent workflows
- **Real-time Voice Processing**: Pipecat + Twilio WebRTC integration
- **Intelligent Lead Qualification**: ML-powered scoring and tier escalation
- **Cost-Optimized**: <$0.10 per call target with budget controls

## Quick Start

### Prerequisites

- Python 3.9+
- Ollama (for local LLM)
- PostgreSQL
- Redis
- GPU with 6GB+ VRAM (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd REAutomation2
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Set up Ollama:
```bash
ollama pull llama3.1:8b-instruct-q4_0
```

6. Initialize database:
```bash
alembic upgrade head
```

### Development

Run with auto-reload:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

Run tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

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

- `GET /health` - Health check
- `POST /calls/start` - Start outbound call
- `GET /calls/{call_id}/status` - Get call status
- `POST /leads/import` - Import leads from Google Sheets
- `GET /analytics/performance` - Performance metrics

## License

MIT License - see LICENSE file for details.