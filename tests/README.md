# REAutomation2 Testing Framework

This directory contains comprehensive unit and integration tests for the REAutomation2 voice AI system.

## 📁 Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── requirements.txt            # Test-specific dependencies
├── README.md                  # This file
├── unit/                      # Unit tests
│   ├── agents/               # Agent system tests
│   │   ├── test_base_agent.py
│   │   ├── test_conversation_agent.py
│   │   ├── test_qualification_agent.py
│   │   ├── test_objection_handler.py
│   │   ├── test_scheduler_agent.py
│   │   ├── test_analytics_agent.py
│   │   └── test_orchestrator.py
│   ├── integrations/         # Integration tests
│   │   ├── test_google_sheets.py
│   │   └── test_service.py
│   ├── llm/                  # LLM service tests
│   │   ├── test_ollama_client.py
│   │   ├── test_cache.py
│   │   ├── test_queue_manager.py
│   │   └── test_service.py
│   └── api/                  # API endpoint tests
│       ├── test_main.py
│       ├── test_calls.py
│       ├── test_health.py
│       └── test_integrations.py
├── integration/              # End-to-end integration tests
├── performance/              # Performance and load tests
└── voice/                    # Voice pipeline tests (existing)
```

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
# Install test dependencies
pip install -r tests/requirements.txt

# Or use the test runner
python run_tests.py --install-deps
```

### 2. Run Tests

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --unit          # Unit tests only
python run_tests.py --agents        # Agent tests only
python run_tests.py --sheets        # Google Sheets tests only
python run_tests.py --coverage      # With coverage analysis
python run_tests.py --parallel      # Parallel execution

# Run tests directly with pytest
pytest tests/unit/                  # Unit tests
pytest tests/unit/agents/           # Agent tests
pytest -m "agent"                   # Tests marked as agent tests
pytest -k "test_conversation"       # Tests matching pattern
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)

**Agent System Tests:**

- `test_base_agent.py` - BaseAgent abstract class functionality
- `test_conversation_agent.py` - Conversation flow and greeting handling
- `test_qualification_agent.py` - Lead qualification logic and scoring
- `test_objection_handler.py` - Objection detection and response
- `test_scheduler_agent.py` - Appointment scheduling functionality
- `test_analytics_agent.py` - Call analysis and metrics collection
- `test_orchestrator.py` - Multi-agent workflow orchestration

**Integration Tests:**

- `test_google_sheets.py` - Google Sheets API integration
- `test_service.py` - Integration service layer

**LLM Service Tests:**

- `test_ollama_client.py` - Ollama client functionality
- `test_cache.py` - Response caching system
- `test_queue_manager.py` - Request queue management
- `test_service.py` - LLM service orchestration

**API Tests:**

- `test_main.py` - FastAPI application setup
- `test_calls.py` - Call management endpoints
- `test_health.py` - Health check endpoints
- `test_integrations.py` - Integration API endpoints

### Integration Tests (`tests/integration/`)

End-to-end tests that verify complete workflows:

- Full conversation flows
- Agent handoffs and state transitions
- External service integrations
- Error handling and recovery

### Performance Tests (`tests/performance/`)

Load and performance testing:

- Concurrent call handling
- Response time benchmarks
- Memory usage analysis
- Scalability testing

## 🔧 Test Configuration

### Pytest Configuration (`pytest.ini`)

Key settings:

- **Coverage**: 80% minimum coverage requirement
- **Async Support**: Automatic async test detection
- **Markers**: Organized test categorization
- **Reports**: HTML and XML coverage reports
- **Parallel**: Support for parallel test execution

### Test Markers

Use markers to categorize and filter tests:

```python
@pytest.mark.unit
def test_agent_initialization():
    """Unit test for agent initialization"""
    pass

@pytest.mark.integration
def test_full_conversation_flow():
    """Integration test for complete conversation"""
    pass

@pytest.mark.performance
def test_concurrent_calls():
    """Performance test for concurrent call handling"""
    pass
```

Available markers:

- `unit` - Unit tests
- `integration` - Integration tests
- `performance` - Performance tests
- `slow` - Slow running tests
- `agent` - Agent-related tests
- `llm` - LLM service tests
- `api` - API endpoint tests
- `sheets` - Google Sheets integration tests

## 🛠 Test Utilities

### Fixtures (`conftest.py`)

Shared fixtures available in all tests:

```python
# Agent fixtures
def test_with_agent(agent, sample_workflow_context):
    response = await agent.process(sample_workflow_context, "Hello")
    assert response.agent_type == AgentType.CONVERSATION

# LLM service fixtures
def test_with_llm_service(mock_llm_service):
    mock_llm_service.generate_response.return_value = mock_response
    # Test LLM integration
```

Key fixtures:

- `mock_llm_service` - Mocked LLM service
- `sample_workflow_context` - Sample conversation context
- `sample_lead_data` - Sample lead information
- `mock_integration_service` - Mocked integration service
- `sample_objection_scenarios` - Objection test cases
- `sample_scheduling_slots` - Scheduling test data

### Mock Helpers

```python
# LLM response helpers
MockLLMResponse.conversation_response("Hello there!")
MockLLMResponse.qualification_response(0.8, {"intent": 0.9})
MockLLMResponse.objection_response("price", "I understand your concern...")

# Test helpers
TestHelpers.create_agent_message(AgentType.CONVERSATION, "Hello")
TestHelpers.create_workflow_context("test-call", WorkflowState.GREETING)
```

## 📊 Coverage and Reporting

### Coverage Reports

Tests generate multiple coverage reports:

```bash
# HTML report (interactive)
htmlcov/index.html

# Terminal report
pytest --cov-report=term-missing

# XML report (for CI/CD)
coverage.xml
```

### Test Reports

```bash
# HTML test report
reports/test_report.html

# JUnit XML (for CI/CD)
pytest --junitxml=reports/junit.xml
```

## 🚀 Running Tests in CI/CD

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements.txt

      - name: Run tests
        run: python run_tests.py --coverage --parallel

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### Local Development

```bash
# Quick test run during development
pytest tests/unit/agents/test_conversation_agent.py -v

# Test specific functionality
pytest -k "test_greeting" -v

# Run tests with coverage
python run_tests.py --coverage

# Run tests in parallel (faster)
python run_tests.py --parallel
```

## 🐛 Debugging Tests

### Verbose Output

```bash
# Detailed test output
pytest -v -s

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb
```

### Test-Specific Debugging

```python
import pytest

def test_debug_example():
    # Use pytest's built-in debugging
    pytest.set_trace()

    # Or use standard pdb
    import pdb; pdb.set_trace()
```

## 📝 Writing New Tests

### Test Structure

```python
"""
Unit tests for NewComponent
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.components.new_component import NewComponent


class TestNewComponent:
    """Test suite for NewComponent"""

    @pytest.fixture
    def component(self):
        """Create component instance"""
        return NewComponent()

    @pytest.mark.asyncio
    async def test_async_method(self, component):
        """Test async method"""
        result = await component.async_method()
        assert result is not None

    def test_sync_method(self, component):
        """Test synchronous method"""
        result = component.sync_method()
        assert result == expected_value

    @pytest.mark.integration
    def test_integration_scenario(self, component):
        """Integration test"""
        # Test complete workflow
        pass
```

### Best Practices

1. **Test Organization**: Group related tests in classes
2. **Descriptive Names**: Use clear, descriptive test names
3. **Fixtures**: Use fixtures for common setup
4. **Mocking**: Mock external dependencies
5. **Assertions**: Use specific assertions with clear messages
6. **Markers**: Tag tests appropriately
7. **Documentation**: Include docstrings for complex tests

### Async Testing

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async functionality"""
    result = await async_function()
    assert result is not None
```

### Mocking External Services

```python
@patch('src.module.external_service')
async def test_with_mocked_service(mock_service):
    """Test with mocked external service"""
    mock_service.return_value = expected_response
    result = await function_using_service()
    assert result == expected_result
```

## 🔍 Test Maintenance

### Regular Tasks

1. **Update Dependencies**: Keep test dependencies current
2. **Review Coverage**: Maintain >80% coverage
3. **Performance Monitoring**: Track test execution times
4. **Cleanup**: Remove obsolete tests
5. **Documentation**: Keep test documentation updated

### Monitoring Test Health

```bash
# Check test performance
pytest --benchmark-only

# Analyze slow tests
pytest --durations=10

# Check for flaky tests
pytest --lf  # Run last failed tests
```

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Factory Boy](https://factoryboy.readthedocs.io/) (for test data generation)

## 🤝 Contributing

When adding new tests:

1. Follow the existing test structure
2. Add appropriate markers
3. Include both positive and negative test cases
4. Test error conditions and edge cases
5. Update this README if adding new test categories
6. Ensure tests pass in CI/CD pipeline

## 📞 Support

For questions about the testing framework:

- Check existing test examples
- Review pytest documentation
- Ask in team discussions
- Update documentation for future developers
