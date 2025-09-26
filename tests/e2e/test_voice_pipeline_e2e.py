"""End-to-end tests for voice pipeline"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.voice.pipecat_integration import PipecatVoicePipeline
from src.voice.models import VoicePipelineConfig, AudioConfig, TTSProvider
from src.cost_control.cost_service import CostControlService
from src.database.service import DatabaseService


class TestVoicePipelineE2E:
    """End-to-end tests for complete voice pipeline"""

    @pytest.fixture
    async def e2e_config(self):
        """Configuration for E2E testing"""
        return VoicePipelineConfig(
            audio_config=AudioConfig(
                sample_rate=16000,
                channels=1,
                chunk_size=1024
            ),
            tts_provider=TTSProvider.LOCAL_PIPER,
            stt_provider="whisper-local",
            llm_model="llama3.1:8b"
        )

    @pytest.fixture
    async def mock_services(self):
        """Mock all external services for E2E testing"""
        services = {
            "database": AsyncMock(),
            "llm": AsyncMock(),
            "cost_control": AsyncMock(),
            "twilio": AsyncMock()
        }

        # Configure database service
        services["database"].is_ready.return_value = True
        services["database"].create_call.return_value = "test_call_id"

        # Configure LLM service
        services["llm"].generate_response.return_value = MagicMock(
            content="Hello! I'm calling from ABC Realty. How are you today?",
            usage_tokens=25,
            response_time_ms=150.0
        )

        # Configure cost control service
        services["cost_control"].initialize_call_costs.return_value = None
        services["cost_control"].check_budget_status.return_value = MagicMock(
            can_continue=True,
            remaining_budget=5.50
        )

        return services

    @pytest.mark.asyncio
    async def test_complete_call_flow(self, e2e_config, mock_services, mock_websocket, mock_audio_data):
        """Test complete call flow from start to finish"""
        # Initialize voice pipeline
        pipeline = PipecatVoicePipeline(e2e_config)

        with patch.multiple(
            'src.voice.pipecat_integration',
            Pipeline=MagicMock(),
            DatabaseService=lambda: mock_services["database"],
            LLMService=lambda: mock_services["llm"],
            CostControlService=lambda: mock_services["cost_control"]
        ):
            # Initialize pipeline
            await pipeline.initialize()

            # Start call
            call_id = "e2e_test_call_001"
            await pipeline.start_call(call_id, mock_websocket)

            # Simulate incoming audio (user speaking)
            await pipeline.send_audio_data(mock_audio_data["raw_audio"])

            # Wait a moment for processing
            await asyncio.sleep(0.1)

            # Simulate conversation turns
            conversation_turns = [
                "Hello, is this about real estate?",
                "Yes, I'm interested in commercial properties.",
                "What size property are you looking for?",
                "Around 5000 square feet for our office.",
                "That sounds perfect. Can we schedule a meeting?"
            ]

            for i, user_input in enumerate(conversation_turns):
                # Simulate user speech-to-text result
                with patch.object(pipeline, '_handle_user_speech') as mock_handler:
                    mock_handler.return_value = f"Assistant response to: {user_input}"
                    await pipeline._handle_user_speech(user_input)

                # Brief pause between turns
                await asyncio.sleep(0.05)

            # End call
            await pipeline.stop_call()

            # Verify call flow
            assert pipeline._current_call_id is None
            mock_services["database"].create_call.assert_called()
            assert mock_services["llm"].generate_response.call_count >= len(conversation_turns)

    @pytest.mark.asyncio
    async def test_tier_switching_during_call(self, e2e_config, mock_services, mock_websocket):
        """Test tier switching during an active call"""
        pipeline = PipecatVoicePipeline(e2e_config)

        # Mock tier decision engine
        with patch('src.voice.pipecat_integration.TierDecisionEngine') as mock_tier_engine:
            tier_instance = MagicMock()
            tier_instance.should_switch_tier.return_value = MagicMock(
                should_switch=True,
                target_tier=TTSProvider.ELEVENLABS,
                confidence=0.85,
                reasoning="High qualification score detected"
            )
            mock_tier_engine.return_value = tier_instance

            with patch.multiple(
                'src.voice.pipecat_integration',
                Pipeline=MagicMock(),
                DatabaseService=lambda: mock_services["database"],
                LLMService=lambda: mock_services["llm"],
                CostControlService=lambda: mock_services["cost_control"]
            ):
                await pipeline.initialize()
                await pipeline.start_call("tier_switch_test", mock_websocket)

                # Initial tier
                assert pipeline._config.tts_provider == TTSProvider.LOCAL_PIPER

                # Simulate qualification improvement during call
                qualification_data = {
                    "score": 0.85,
                    "conversation_length": 8,
                    "lead_indicators": {"company_size": "enterprise"}
                }

                # Trigger tier evaluation
                await pipeline._evaluate_tier_switch(qualification_data)

                # Verify tier was switched
                assert pipeline._config.tts_provider == TTSProvider.ELEVENLABS

                await pipeline.stop_call()

    @pytest.mark.asyncio
    async def test_cost_monitoring_and_limits(self, e2e_config, mock_services, mock_websocket):
        """Test cost monitoring and budget limit enforcement"""
        pipeline = PipecatVoicePipeline(e2e_config)

        # Configure cost service to hit limits
        cost_service = mock_services["cost_control"]

        # First few calls are under budget
        budget_responses = [
            MagicMock(can_continue=True, remaining_budget=2.50),
            MagicMock(can_continue=True, remaining_budget=1.20),
            MagicMock(can_continue=True, remaining_budget=0.80),
            MagicMock(can_continue=False, remaining_budget=0.05)  # Budget exceeded
        ]

        cost_service.check_budget_status.side_effect = budget_responses

        with patch.multiple(
            'src.voice.pipecat_integration',
            Pipeline=MagicMock(),
            DatabaseService=lambda: mock_services["database"],
            LLMService=lambda: mock_services["llm"],
            CostControlService=lambda: cost_service
        ):
            await pipeline.initialize()

            # Make multiple calls to exhaust budget
            for i in range(4):
                call_id = f"budget_test_call_{i}"

                if i < 3:
                    # First 3 calls should succeed
                    await pipeline.start_call(call_id, mock_websocket)
                    await pipeline.stop_call()
                else:
                    # 4th call should be rejected due to budget
                    with pytest.raises(Exception, match="Budget limit exceeded"):
                        await pipeline.start_call(call_id, mock_websocket)

    @pytest.mark.asyncio
    async def test_error_recovery_and_fallback(self, e2e_config, mock_services, mock_websocket):
        """Test error recovery and fallback mechanisms"""
        pipeline = PipecatVoicePipeline(e2e_config)

        # Configure LLM to fail initially, then recover
        llm_service = mock_services["llm"]
        llm_responses = [
            Exception("LLM service temporarily unavailable"),
            Exception("LLM service temporarily unavailable"),
            MagicMock(content="I apologize for the delay. How can I help you?", usage_tokens=20)
        ]
        llm_service.generate_response.side_effect = llm_responses

        with patch.multiple(
            'src.voice.pipecat_integration',
            Pipeline=MagicMock(),
            DatabaseService=lambda: mock_services["database"],
            LLMService=lambda: llm_service,
            CostControlService=lambda: mock_services["cost_control"]
        ):
            await pipeline.initialize()
            await pipeline.start_call("error_recovery_test", mock_websocket)

            # Simulate conversation that initially fails
            with patch.object(pipeline, '_handle_llm_failure') as mock_fallback:
                mock_fallback.return_value = "I'm experiencing some technical difficulties. Please hold on."

                # First attempt should trigger fallback
                response = await pipeline._generate_response("Hello")
                assert "technical difficulties" in response

                # After retries, should work normally
                response = await pipeline._generate_response("How are you?")
                assert response is not None

            await pipeline.stop_call()

    @pytest.mark.asyncio
    async def test_real_time_audio_processing(self, e2e_config, mock_services, mock_websocket, mock_audio_data):
        """Test real-time audio processing performance"""
        pipeline = PipecatVoicePipeline(e2e_config)

        processing_times = []

        def track_processing_time(*args, **kwargs):
            import time
            start = time.time()
            # Simulate processing
            time.sleep(0.01)  # 10ms processing
            end = time.time()
            processing_times.append((end - start) * 1000)  # Convert to ms
            return mock_audio_data["tts_audio"]

        with patch.multiple(
            'src.voice.pipecat_integration',
            Pipeline=MagicMock(),
            DatabaseService=lambda: mock_services["database"],
            LLMService=lambda: mock_services["llm"],
            CostControlService=lambda: mock_services["cost_control"]
        ):
            with patch.object(pipeline, '_process_audio_chunk', side_effect=track_processing_time):
                await pipeline.initialize()
                await pipeline.start_call("performance_test", mock_websocket)

                # Send multiple audio chunks rapidly
                for i in range(10):
                    await pipeline.send_audio_data(mock_audio_data["raw_audio"])
                    await asyncio.sleep(0.02)  # 20ms intervals

                await pipeline.stop_call()

                # Verify processing times are within acceptable limits
                avg_processing_time = sum(processing_times) / len(processing_times)
                assert avg_processing_time < 50.0  # Should process within 50ms average
                assert max(processing_times) < 100.0  # No single chunk should take over 100ms

    @pytest.mark.asyncio
    async def test_concurrent_calls_handling(self, e2e_config, mock_services):
        """Test handling multiple concurrent calls"""
        # Create multiple pipeline instances
        pipelines = []
        call_tasks = []

        for i in range(3):
            pipeline = PipecatVoicePipeline(e2e_config)
            pipelines.append(pipeline)

        with patch.multiple(
            'src.voice.pipecat_integration',
            Pipeline=MagicMock(),
            DatabaseService=lambda: mock_services["database"],
            LLMService=lambda: mock_services["llm"],
            CostControlService=lambda: mock_services["cost_control"]
        ):
            # Initialize all pipelines
            for pipeline in pipelines:
                await pipeline.initialize()

            # Start concurrent calls
            for i, pipeline in enumerate(pipelines):
                mock_ws = MagicMock()
                task = pipeline.start_call(f"concurrent_call_{i}", mock_ws)
                call_tasks.append(task)

            await asyncio.gather(*call_tasks)

            # Verify all calls are active
            active_calls = [p._current_call_id for p in pipelines]
            assert len([cid for cid in active_calls if cid is not None]) == 3

            # Stop all calls
            stop_tasks = [pipeline.stop_call() for pipeline in pipelines]
            await asyncio.gather(*stop_tasks)

            # Verify all calls are stopped
            active_calls = [p._current_call_id for p in pipelines]
            assert all(cid is None for cid in active_calls)

    @pytest.mark.asyncio
    async def test_integration_with_twilio_webhook(self, e2e_config, mock_services):
        """Test integration with Twilio webhook events"""
        pipeline = PipecatVoicePipeline(e2e_config)

        # Mock Twilio webhook payloads
        webhook_events = [
            {
                "event_type": "call_initiated",
                "call_sid": "CA123456789",
                "from": "+1234567890",
                "to": "+1555123456",
                "call_status": "initiated"
            },
            {
                "event_type": "call_ringing",
                "call_sid": "CA123456789",
                "call_status": "ringing"
            },
            {
                "event_type": "call_answered",
                "call_sid": "CA123456789",
                "call_status": "in-progress"
            },
            {
                "event_type": "call_ended",
                "call_sid": "CA123456789",
                "call_status": "completed",
                "call_duration": "00:02:30"
            }
        ]

        with patch.multiple(
            'src.voice.pipecat_integration',
            Pipeline=MagicMock(),
            DatabaseService=lambda: mock_services["database"],
            LLMService=lambda: mock_services["llm"],
            CostControlService=lambda: mock_services["cost_control"]
        ):
            await pipeline.initialize()

            # Process webhook events
            for event in webhook_events:
                await pipeline._handle_twilio_webhook(event)

            # Verify database was updated with call events
            assert mock_services["database"].create_call.called
            assert mock_services["database"].update_call_status.call_count >= 3

    @pytest.mark.asyncio
    async def test_full_analytics_pipeline(self, e2e_config, mock_services, mock_websocket):
        """Test complete analytics data collection"""
        pipeline = PipecatVoicePipeline(e2e_config)

        # Configure services to collect metrics
        database_service = mock_services["database"]
        cost_service = mock_services["cost_control"]

        analytics_data = []

        def capture_analytics(data):
            analytics_data.append(data)

        database_service.record_metric.side_effect = capture_analytics

        with patch.multiple(
            'src.voice.pipecat_integration',
            Pipeline=MagicMock(),
            DatabaseService=lambda: database_service,
            LLMService=lambda: mock_services["llm"],
            CostControlService=lambda: cost_service
        ):
            await pipeline.initialize()
            await pipeline.start_call("analytics_test", mock_websocket)

            # Simulate full conversation with metrics
            conversation_events = [
                {"type": "speech_started", "timestamp": datetime.now()},
                {"type": "speech_ended", "timestamp": datetime.now(), "duration": 3.2},
                {"type": "stt_completed", "timestamp": datetime.now(), "confidence": 0.95},
                {"type": "llm_started", "timestamp": datetime.now()},
                {"type": "llm_completed", "timestamp": datetime.now(), "tokens": 45},
                {"type": "tts_started", "timestamp": datetime.now()},
                {"type": "tts_completed", "timestamp": datetime.now(), "characters": 180}
            ]

            for event in conversation_events:
                await pipeline._record_analytics_event(event)

            await pipeline.stop_call()

            # Verify analytics were collected
            assert len(analytics_data) >= len(conversation_events)

            # Verify specific metrics were captured
            metric_types = [data.get("metric_type") for data in analytics_data]
            assert "speech_duration" in metric_types
            assert "stt_confidence" in metric_types
            assert "llm_tokens" in metric_types
            assert "tts_characters" in metric_types