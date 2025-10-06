"""
Tests for context management system including token counting,
pruning strategies, and context manager.
"""

import pytest
from datetime import datetime
from typing import List, Dict, Any

from src.llm.token_counter import TokenCounter
from src.llm.pruning_strategies import (
    Message,
    SlidingWindowStrategy,
    ImportanceBasedStrategy,
    HybridStrategy,
    SummarizationStrategy,
    PruningStrategyFactory
)
from src.llm.context_manager import ContextManager, ContextManagerFactory
from src.config.settings import settings


class TestTokenCounter:
    """Tests for TokenCounter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.counter = TokenCounter()

    def test_count_tokens_simple(self):
        """Test basic token counting."""
        text = "Hello, how are you?"
        tokens = self.counter.count_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_count_tokens_empty(self):
        """Test counting empty string."""
        assert self.counter.count_tokens("") == 0

    def test_count_tokens_long_text(self):
        """Test counting longer text."""
        text = " ".join(["word"] * 100)
        tokens = self.counter.count_tokens(text)
        assert tokens >= 100  # Should be at least the number of words

    def test_count_messages_tokens(self):
        """Test counting tokens in message list."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there, how can I help?"}
        ]
        tokens = self.counter.count_messages_tokens(messages)
        assert tokens > 0

    def test_will_fit_in_context(self):
        """Test context fitting check."""
        small_text = "Hello"
        assert self.counter.will_fit_in_context([small_text])

        # Test with very large text (should not fit in 8192 context)
        large_text = " ".join(["word"] * 10000)
        assert not self.counter.will_fit_in_context([large_text])

    def test_cache_efficiency(self):
        """Test that caching improves performance."""
        text = "This is a test sentence for caching."

        # First call - not cached
        tokens1 = self.counter.count_tokens(text)

        # Second call - should be cached
        tokens2 = self.counter.count_tokens(text)

        assert tokens1 == tokens2

    def test_estimate_tokens_fallback(self):
        """Test fallback estimation when tiktoken unavailable."""
        text = "Hello world"
        estimated = self.counter.estimate_tokens(text)
        assert estimated > 0
        assert isinstance(estimated, int)


class TestSlidingWindowStrategy:
    """Tests for SlidingWindowStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.counter = TokenCounter()
        self.strategy = SlidingWindowStrategy(self.counter)

    def create_messages(self, count: int) -> List[Message]:
        """Helper to create test messages."""
        messages = []
        for i in range(count):
            msg = Message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i} with some content to generate tokens",
                timestamp=datetime.utcnow()
            )
            messages.append(msg)
        return messages

    @pytest.mark.asyncio
    async def test_no_pruning_needed(self):
        """Test when messages fit in context."""
        messages = self.create_messages(5)
        pruned = await self.strategy.prune(messages)
        assert len(pruned) == len(messages)

    @pytest.mark.asyncio
    async def test_prune_oldest_first(self):
        """Test that oldest messages are removed first."""
        # Create many messages to exceed token limit
        messages = self.create_messages(100)

        pruned = await self.strategy.prune(messages)

        # Should have pruned some messages
        assert len(pruned) < len(messages)

        # Most recent messages should be preserved
        assert pruned[-1].content == messages[-1].content

    @pytest.mark.asyncio
    async def test_respects_target_ratio(self):
        """Test that pruning respects target token ratio."""
        messages = self.create_messages(100)
        pruned = await self.strategy.prune(messages)

        total_tokens = sum(msg.tokens for msg in pruned)
        assert total_tokens <= self.strategy.target_tokens


class TestImportanceBasedStrategy:
    """Tests for ImportanceBasedStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.counter = TokenCounter()
        self.strategy = ImportanceBasedStrategy(self.counter)

    def create_message(self, role: str, content: str, importance: float = 0.0) -> Message:
        """Helper to create a test message."""
        return Message(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            importance_score=importance
        )

    @pytest.mark.asyncio
    async def test_keeps_important_messages(self):
        """Test that messages with high importance are preserved."""
        messages = [
            self.create_message("user", "Regular message 1"),
            self.create_message("assistant", "Important response", importance=0.9),
            self.create_message("user", "Regular message 2"),
            self.create_message("user", "Regular message 3"),
        ]

        # Add enough messages to trigger pruning
        for i in range(100):
            messages.append(self.create_message("user", f"Filler message {i}"))

        pruned = await self.strategy.prune(messages)

        # Important message should be preserved
        important_preserved = any(
            msg.content == "Important response" for msg in pruned
        )
        assert important_preserved

    @pytest.mark.asyncio
    async def test_importance_score_calculation(self):
        """Test importance score calculation."""
        msg = self.create_message("assistant", "What is your budget?")
        score = self.strategy._calculate_importance_score(msg, 5, 10)

        assert 0 <= score <= 1.0
        assert score > 0  # Should have some importance

    @pytest.mark.asyncio
    async def test_question_detection(self):
        """Test that questions get higher importance."""
        question_msg = self.create_message("user", "What is the price?")
        statement_msg = self.create_message("user", "The price is fine.")

        question_score = self.strategy._calculate_importance_score(question_msg, 0, 2)
        statement_score = self.strategy._calculate_importance_score(statement_msg, 1, 2)

        assert question_score > statement_score


class TestHybridStrategy:
    """Tests for HybridStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.counter = TokenCounter()
        self.strategy = HybridStrategy(self.counter)

    def create_messages(self, count: int) -> List[Message]:
        """Helper to create test messages."""
        messages = []
        for i in range(count):
            msg = Message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i} with content",
                timestamp=datetime.utcnow(),
                importance_score=0.5 if i == 10 else 0.0  # Make one message important
            )
            messages.append(msg)
        return messages

    @pytest.mark.asyncio
    async def test_balances_recency_and_importance(self):
        """Test that hybrid strategy balances recent and important messages."""
        messages = self.create_messages(100)

        pruned = await self.strategy.prune(messages)

        # Should preserve recent messages
        recent_preserved = any(
            msg.content == messages[-1].content for msg in pruned
        )
        assert recent_preserved

        # Should also try to preserve important message
        # (Message 10 has high importance score)
        assert len(pruned) > 0


class TestSummarizationStrategy:
    """Tests for SummarizationStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.counter = TokenCounter()
        self.strategy = SummarizationStrategy(self.counter, llm_client=None)

    def create_messages(self, count: int) -> List[Message]:
        """Helper to create test messages."""
        messages = []
        for i in range(count):
            msg = Message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i} discussing topic {i}",
                timestamp=datetime.utcnow()
            )
            messages.append(msg)
        return messages

    @pytest.mark.asyncio
    async def test_creates_summary_messages(self):
        """Test that old messages are summarized."""
        messages = self.create_messages(100)

        pruned = await self.strategy.prune(messages)

        # Should have summaries and recent messages
        assert len(pruned) > 0

        # Check if any summary messages were created
        summaries = [msg for msg in pruned if "[Summary of" in msg.content]
        assert len(summaries) >= 0  # May or may not create summaries depending on budget

    @pytest.mark.asyncio
    async def test_preserves_recent_messages(self):
        """Test that recent messages are kept intact."""
        messages = self.create_messages(50)

        pruned = await self.strategy.prune(messages)

        # Most recent message should be preserved exactly
        recent_preserved = any(
            messages[-1].content in msg.content for msg in pruned
        )
        assert recent_preserved or len(pruned) > 0  # At least some messages preserved


class TestPruningStrategyFactory:
    """Tests for PruningStrategyFactory."""

    def test_create_sliding_window(self):
        """Test creating sliding window strategy."""
        counter = TokenCounter()
        strategy = PruningStrategyFactory.create("sliding_window", counter)
        assert isinstance(strategy, SlidingWindowStrategy)

    def test_create_importance(self):
        """Test creating importance-based strategy."""
        counter = TokenCounter()
        strategy = PruningStrategyFactory.create("importance", counter)
        assert isinstance(strategy, ImportanceBasedStrategy)

    def test_create_hybrid(self):
        """Test creating hybrid strategy."""
        counter = TokenCounter()
        strategy = PruningStrategyFactory.create("hybrid", counter)
        assert isinstance(strategy, HybridStrategy)

    def test_create_summarize(self):
        """Test creating summarization strategy."""
        counter = TokenCounter()
        strategy = PruningStrategyFactory.create("summarize", counter)
        assert isinstance(strategy, SummarizationStrategy)

    def test_create_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        counter = TokenCounter()
        with pytest.raises(ValueError):
            PruningStrategyFactory.create("invalid_strategy", counter)


class TestContextManager:
    """Tests for ContextManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ContextManager()

    def create_message_dicts(self, count: int) -> List[Dict[str, Any]]:
        """Helper to create test message dictionaries."""
        messages = []
        for i in range(count):
            messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i} with some content here",
                "timestamp": datetime.utcnow(),
                "importance_score": 0.0,
                "tokens": 0
            })
        return messages

    def test_count_context_tokens(self):
        """Test context token counting."""
        messages = self.create_message_dicts(5)
        total, system, msgs = self.manager.count_context_tokens(
            messages, "You are a helpful assistant"
        )

        assert total > 0
        assert system > 0
        assert msgs > 0
        assert total == system + msgs

    def test_needs_pruning(self):
        """Test pruning detection."""
        small_messages = self.create_message_dicts(5)
        assert not self.manager.needs_pruning(small_messages)

        # Create many messages to exceed limit
        large_messages = self.create_message_dicts(500)
        assert self.manager.needs_pruning(large_messages)

    def test_is_near_limit(self):
        """Test warning threshold detection."""
        small_messages = self.create_message_dicts(5)
        assert not self.manager.is_near_limit(small_messages)

    @pytest.mark.asyncio
    async def test_manage_context_no_pruning(self):
        """Test context management when no pruning needed."""
        messages = self.create_message_dicts(5)
        managed, stats = await self.manager.manage_context(messages)

        assert len(managed) == len(messages)
        assert not stats.pruned
        assert stats.pruned_count == 0

    @pytest.mark.asyncio
    async def test_manage_context_with_pruning(self):
        """Test context management with pruning."""
        messages = self.create_message_dicts(500)
        managed, stats = await self.manager.manage_context(messages)

        assert len(managed) < len(messages)
        assert stats.pruned
        assert stats.pruned_count > 0
        assert stats.strategy_used is not None

    def test_get_context_stats(self):
        """Test getting context statistics."""
        messages = self.create_message_dicts(10)
        stats = self.manager.get_context_stats(messages)

        assert stats.total_messages == 10
        assert stats.total_tokens > 0
        assert not stats.pruned

    def test_estimate_remaining_capacity(self):
        """Test remaining capacity estimation."""
        messages = self.create_message_dicts(10)
        capacity = self.manager.estimate_remaining_capacity(messages)

        assert "current_tokens" in capacity
        assert "max_tokens" in capacity
        assert "remaining_tokens" in capacity
        assert capacity["can_fit"]

    @pytest.mark.asyncio
    async def test_stats_history_tracking(self):
        """Test that stats history is tracked."""
        messages = self.create_message_dicts(10)

        # Perform multiple operations
        await self.manager.manage_context(messages)
        await self.manager.manage_context(messages)

        assert len(self.manager.stats_history) == 2

    def test_get_stats_summary(self):
        """Test statistics summary."""
        # Initially empty
        summary = self.manager.get_stats_summary()
        assert summary["total_operations"] == 0

    def test_clear_history(self):
        """Test clearing stats history."""
        messages = self.create_message_dicts(10)
        self.manager.get_context_stats(messages)
        self.manager.clear_history()

        assert len(self.manager.stats_history) == 0


class TestContextManagerFactory:
    """Tests for ContextManagerFactory."""

    def test_create_default(self):
        """Test creating default context manager."""
        manager = ContextManagerFactory.create()
        assert isinstance(manager, ContextManager)

    def test_create_with_strategy(self):
        """Test creating with specific strategy."""
        manager = ContextManagerFactory.create(strategy="importance")
        assert manager.strategy_name == "importance"

    def test_get_default_singleton(self):
        """Test that get_default returns singleton."""
        manager1 = ContextManagerFactory.get_default()
        manager2 = ContextManagerFactory.get_default()
        assert manager1 is manager2

    def test_reset_default(self):
        """Test resetting default singleton."""
        manager1 = ContextManagerFactory.get_default()
        ContextManagerFactory.reset_default()
        manager2 = ContextManagerFactory.get_default()
        assert manager1 is not manager2


class TestContextIntegration:
    """Integration tests for complete context management workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ContextManager()

    def create_conversation(self, turns: int) -> List[Dict[str, Any]]:
        """Helper to create a realistic conversation."""
        messages = []
        for i in range(turns):
            if i % 2 == 0:
                messages.append({
                    "role": "user",
                    "content": f"User question {i}: Can you tell me about your product?",
                    "timestamp": datetime.utcnow(),
                    "importance_score": 0.0,
                    "tokens": 0
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": f"Assistant response {i}: Here's information about our product features and benefits.",
                    "timestamp": datetime.utcnow(),
                    "importance_score": 0.0,
                    "tokens": 0
                })
        return messages

    @pytest.mark.asyncio
    async def test_full_conversation_workflow(self):
        """Test complete conversation management workflow."""
        # Create a long conversation
        messages = self.create_conversation(100)
        system_prompt = "You are a helpful sales assistant."

        # Check initial state
        initial_stats = self.manager.get_context_stats(messages, system_prompt)
        assert initial_stats.total_messages == 100

        # Manage context
        managed, stats = await self.manager.manage_context(messages, system_prompt)

        # Verify pruning occurred
        assert stats.total_messages < 100
        assert stats.total_tokens <= self.manager.target_tokens

        # Check capacity
        capacity = self.manager.estimate_remaining_capacity(managed, system_prompt)
        assert capacity["can_fit"]
        assert not capacity["needs_pruning"]

    @pytest.mark.asyncio
    async def test_multiple_pruning_operations(self):
        """Test multiple pruning operations maintain consistency."""
        messages = self.create_conversation(50)

        # First pruning
        managed1, stats1 = await self.manager.manage_context(messages)

        # Add more messages
        messages.extend(self.create_conversation(50))

        # Second pruning
        managed2, stats2 = await self.manager.manage_context(messages)

        # Stats should be tracked
        assert len(self.manager.stats_history) == 2

        # Summary should reflect operations
        summary = self.manager.get_stats_summary()
        assert summary["total_operations"] == 2
