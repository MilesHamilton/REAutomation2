# Tech Context: REAutomation2

## Core Technologies

### Backend Framework

- **FastAPI 0.104.1**: Modern, fast web framework for building APIs
- **Uvicorn 0.24.0**: ASGI server for running FastAPI applications
- **Pydantic 2.5.0**: Data validation and settings management using Python type annotations
- **Pydantic Settings 2.1.0**: Configuration management with environment variable support

### AI & Language Models

- **LangChain 0.1.0**: Framework for developing applications with language models
- **LangGraph 0.6.8**: Library for building stateful, multi-actor applications with LLMs (upgraded from 0.0.25)
- **LangSmith >=0.0.77,<0.1.0**: Production monitoring and tracing for LangChain applications
- **Ollama**: Local LLM inference server (Llama 3.1 8B model)
- **OpenAI Whisper 20231117**: Automatic speech recognition system

### Voice Processing

- **Pipecat AI 0.0.14**: Real-time voice conversation AI framework
- **Twilio 8.10.0**: Cloud communications platform for voice calls
- **Piper TTS 1.2.0**: Local text-to-speech synthesis
- **Coqui TTS 0.19.0**: Open-source text-to-speech toolkit
- **11Labs API**: Premium text-to-speech service (via API)

### Audio Processing

- **LibROSA 0.10.1**: Audio and music signal analysis
- **SoundFile 1.0.14**: Audio file I/O operations
- **NumPy 1.24.3**: Numerical computing for audio data
- **SciPy 1.11.4**: Scientific computing and signal processing
- **Speex DSP >=1.4.0** (speexdsp-python): Acoustic echo cancellation and audio preprocessing
- **NoiseReduce >=3.0.0**: Statistical noise reduction for speech clarity

#### Real-Time Audio Processing Pipeline

- **AudioBufferManager**: Circular buffer for 20ms chunk management with latency tracking
- **AdaptiveJitterBuffer**: Network jitter handling with 40-200ms adaptive delay range
- **PacketLossConcealment**: Three methods (simple, linear, spectral) for lost packet recovery
- **EchoCancellationProcessor**: Speex AEC engine with NLMS adaptive filter fallback
- **NoiseReductionProcessor**: noisereduce library with spectral subtraction fallback
- **AudioProcessor**: Unified pipeline orchestrating all audio components
- **Performance**: <200ms latency target, <50ms processing overhead, >5x real-time throughput

### Machine Learning

- **PyTorch >=2.0.0**: Deep learning framework
- **TorchAudio >=2.0.0**: Audio processing with PyTorch
- **Transformers >=4.30.0**: Hugging Face transformers library

### Database & Caching

- **SQLAlchemy 2.0.23**: SQL toolkit and Object-Relational Mapping
- **Alembic 1.12.1**: Database migration tool for SQLAlchemy
- **PostgreSQL**: Primary database (via psycopg2-binary 2.9.9)
- **Redis 5.0.1**: In-memory data structure store for caching and queues

### Development & Testing

- **Pytest 7.4.3**: Testing framework
- **Pytest-AsyncIO 0.21.1**: Async testing support
- **HTTPX 0.25.2**: HTTP client for testing API endpoints
- **Python-dotenv 1.0.0**: Environment variable management

### Utilities

- **WebSockets 12.0**: Real-time communication support
- **Python-multipart 0.0.6**: Multipart form data parsing
- **PyYAML 6.0.1**: YAML configuration file support
- **AIOFiles 23.2.1**: Async file operations
- **Jinja2 3.1.2**: Template engine
- **Requests 2.31.0**: HTTP library for external API calls
- **psutil >=5.9.0**: System and process monitoring utilities
- **backoff >=2.2.1**: Exponential backoff and retry functionality
- **icalendar >=5.0.0**: Calendar data parsing and generation
- **pytz >=2023.3**: Timezone handling and conversion
- **cartesia ~=2.0.3**: Cartesia TTS service integration (via pipecat-ai[cartesia])

## Development Setup

### System Requirements

- **Python**: 3.9+ (recommended 3.11)
- **GPU**: 6GB+ VRAM for local LLM inference
- **RAM**: 16GB+ recommended for concurrent operations
- **Storage**: 10GB+ for models and dependencies
- **OS**: Windows 11, macOS, or Linux

### External Dependencies

- **Ollama Server**: Local LLM inference
  - Model: `llama3.1:8b-instruct-q4_0`
  - Host: `http://localhost:11434` (default)
- **PostgreSQL**: Database server
- **Redis**: Cache and queue server
- **Twilio Account**: Voice communication services
- **11Labs Account**: Premium TTS services (optional)

### Environment Configuration

Key environment variables in `.env`:

```bash
# LLM Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4_0
LLM_MAX_CONCURRENT=5
LLM_GPU_MEMORY_LIMIT=6144

# Twilio Configuration
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_phone_number

# 11Labs Configuration (Optional)
ELEVENLABS_API_KEY=your_api_key
ELEVENLABS_MODEL=eleven_turbo_v2
ELEVENLABS_VOICE=your_voice_id

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/reautomation2
REDIS_URL=redis://localhost:6379/0

# Cost Controls
DAILY_BUDGET=50.00
COST_PER_CALL_LIMIT=0.10
QUALIFICATION_THRESHOLD=0.8
```

## Technical Constraints

### Performance Requirements

- **Response Time**: <500ms for LLM inference
- **Audio Latency**: <200ms total for real-time conversation
  - Jitter Buffer: 40-200ms adaptive delay
  - Echo Cancellation: <10ms processing overhead
  - Noise Reduction: <20ms processing overhead
  - Buffer Management: <5ms per operation
- **Concurrent Calls**: Up to 5 simultaneous calls
- **Memory Usage**: <8GB per instance under normal load
- **Audio Throughput**: >5x real-time (handle 100+ packets in <5s)

### Resource Limitations

- **GPU Memory**: 6GB VRAM limit for local LLM
- **CPU Usage**: Efficient async processing to handle I/O bound operations
- **Network Bandwidth**: Optimized for voice streaming and API calls
- **Storage**: Efficient model loading and conversation history management

### Integration Constraints

- **Twilio WebRTC**: Real-time audio streaming requirements
- **Ollama API**: Local inference server dependency
- **11Labs Rate Limits**: API quota management for premium TTS
- **Database Connections**: Connection pooling for concurrent access

## Architecture Decisions

### Async/Await Pattern

- **Rationale**: Handle multiple concurrent calls efficiently
- **Implementation**: All I/O operations use async/await
- **Benefits**: Better resource utilization, improved scalability

### Microservice-Ready Design

- **Stateless Components**: Agents designed for horizontal scaling
- **External State**: Context stored in Redis for multi-instance access
- **API-First**: All functionality exposed through REST endpoints

### Plugin Architecture

- **Agent System**: Modular agents for different conversation phases
- **TTS Providers**: Pluggable TTS engines (local/premium)
- **LLM Backends**: Configurable LLM providers (Ollama/cloud)

## Development Tools

### Code Quality

- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Linting and style checking
- **MyPy**: Static type checking

### Testing Strategy

- **Unit Tests**: Individual component testing (137 audio processing tests)
  - Audio Buffer Management: 48 tests (391 lines)
  - Jitter Buffer: 39 tests (385 lines)
  - Audio Processing Components: 50 tests (470 lines)
- **Integration Tests**: API endpoint testing
- **Load Tests**: Concurrent call simulation
- **Voice Tests**: Audio pipeline validation
- **Performance Tests**: Latency and throughput benchmarks
  - Processing latency: <50ms per packet target
  - Throughput: 100 packets in <5s target

### Monitoring & Debugging

- **FastAPI Debug Mode**: Development server with auto-reload
- **Logging**: Structured logging with configurable levels
- **Health Checks**: System component status monitoring
- **Metrics Collection**: Performance and usage analytics

## Deployment Considerations

### Container Support

- **Docker**: Containerized deployment support
- **GPU Access**: NVIDIA Docker for GPU-accelerated inference
- **Multi-stage Builds**: Optimized container images

### Scaling Strategy

- **Horizontal Scaling**: Multiple FastAPI instances behind load balancer
- **Database Scaling**: Read replicas for analytics queries
- **Cache Scaling**: Redis cluster for high availability

### Security

- **API Authentication**: Token-based authentication for API access
- **Environment Isolation**: Secure environment variable management
- **Network Security**: VPC and firewall configuration for production
