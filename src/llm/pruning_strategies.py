"""
Context pruning strategies for managing conversation history.

This module implements various strategies for pruning conversation context
when it exceeds token limits, including sliding window, importance-based,
hybrid, and summarization approaches.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.llm.token_counter import TokenCounter
from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a conversation message with metadata."""
    role: str
    content: str
    timestamp: datetime
    importance_score: float = 0.0
    tokens: int = 0


class PruningStrategy(ABC):
    """Abstract base class for pruning strategies."""

    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter
        self.max_tokens = settings.LLM_CONTEXT_WINDOW
        self.target_tokens = int(self.max_tokens * settings.LLM_CONTEXT_TARGET_RATIO)

    @abstractmethod
    async def prune(self, messages: List[Message], system_prompt: Optional[str] = None) -> List[Message]:
        """
        Prune messages to fit within target token limit.

        Args:
            messages: List of messages to prune
            system_prompt: Optional system prompt (always preserved)

        Returns:
            Pruned list of messages
        """
        pass

    def _calculate_total_tokens(self, messages: List[Message], system_prompt: Optional[str] = None) -> int:
        """Calculate total tokens for messages and system prompt."""
        total = sum(msg.tokens for msg in messages)
        if system_prompt:
            total += self.token_counter.count_tokens(system_prompt)
        return total

    def _ensure_message_tokens(self, messages: List[Message]) -> None:
        """Ensure all messages have token counts."""
        for msg in messages:
            if msg.tokens == 0:
                msg.tokens = self.token_counter.count_tokens(msg.content)


class SlidingWindowStrategy(PruningStrategy):
    """
    Sliding window strategy - keeps the most recent messages.

    This is the simplest strategy that maintains conversation recency
    by removing the oldest messages first.
    """

    async def prune(self, messages: List[Message], system_prompt: Optional[str] = None) -> List[Message]:
        """Keep most recent messages within token limit."""
        self._ensure_message_tokens(messages)

        system_tokens = self.token_counter.count_tokens(system_prompt) if system_prompt else 0
        available_tokens = self.target_tokens - system_tokens

        # Start from most recent and work backwards
        pruned = []
        current_tokens = 0

        for msg in reversed(messages):
            if current_tokens + msg.tokens <= available_tokens:
                pruned.insert(0, msg)
                current_tokens += msg.tokens
            else:
                break

        logger.info(
            f"Sliding window pruning: {len(messages)} -> {len(pruned)} messages "
            f"({current_tokens + system_tokens}/{self.target_tokens} tokens)"
        )

        return pruned


class ImportanceBasedStrategy(PruningStrategy):
    """
    Importance-based strategy - keeps messages with highest importance scores.

    Messages are scored based on:
    - Recency (more recent = higher score)
    - Role (system/assistant messages weighted higher)
    - Content indicators (questions, key information)
    - User-assigned importance (if available)
    """

    def _calculate_importance_score(self, msg: Message, position: int, total: int) -> float:
        """
        Calculate importance score for a message.

        Score components:
        - Recency: 0-0.4 (position-based, recent = higher)
        - Role weight: 0-0.3 (assistant = 0.3, user = 0.2, system = 0.1)
        - Content features: 0-0.2 (questions, keywords)
        - User score: 0-0.1 (if provided)
        """
        score = 0.0

        # Recency score (most recent gets highest)
        recency = position / total if total > 0 else 0
        score += recency * 0.4

        # Role weight
        role_weights = {
            'assistant': 0.3,  # AI responses often contain key information
            'user': 0.2,       # User queries are important
            'system': 0.1      # System messages less critical
        }
        score += role_weights.get(msg.role, 0.1)

        # Content features
        content_lower = msg.content.lower()
        if any(q in content_lower for q in ['?', 'how', 'what', 'why', 'when', 'where']):
            score += 0.1  # Questions are important
        if any(kw in content_lower for kw in ['error', 'important', 'critical', 'issue']):
            score += 0.1  # Key indicators

        # User-assigned importance
        if msg.importance_score > 0:
            score += min(msg.importance_score, 0.1)

        return min(score, 1.0)

    async def prune(self, messages: List[Message], system_prompt: Optional[str] = None) -> List[Message]:
        """Keep messages with highest importance scores within token limit."""
        self._ensure_message_tokens(messages)

        system_tokens = self.token_counter.count_tokens(system_prompt) if system_prompt else 0
        available_tokens = self.target_tokens - system_tokens

        # Calculate importance scores
        total_messages = len(messages)
        for i, msg in enumerate(messages):
            if msg.importance_score == 0.0:
                msg.importance_score = self._calculate_importance_score(msg, i, total_messages)

        # Sort by importance (descending) and select messages
        sorted_messages = sorted(messages, key=lambda m: m.importance_score, reverse=True)

        pruned = []
        current_tokens = 0

        for msg in sorted_messages:
            if current_tokens + msg.tokens <= available_tokens:
                pruned.append(msg)
                current_tokens += msg.tokens
            else:
                break

        # Re-sort by timestamp to maintain conversation order
        pruned.sort(key=lambda m: m.timestamp)

        logger.info(
            f"Importance-based pruning: {len(messages)} -> {len(pruned)} messages "
            f"({current_tokens + system_tokens}/{self.target_tokens} tokens)"
        )

        return pruned


class HybridStrategy(PruningStrategy):
    """
    Hybrid strategy - combines sliding window with importance weighting.

    Maintains recent context while preserving important older messages.
    Uses a 70/30 split: 70% for recent messages, 30% for important older ones.
    """

    async def prune(self, messages: List[Message], system_prompt: Optional[str] = None) -> List[Message]:
        """Combine recency with importance for balanced pruning."""
        self._ensure_message_tokens(messages)

        system_tokens = self.token_counter.count_tokens(system_prompt) if system_prompt else 0
        available_tokens = self.target_tokens - system_tokens

        # Split budget: 70% for recent, 30% for important
        recent_budget = int(available_tokens * 0.7)
        important_budget = available_tokens - recent_budget

        # Phase 1: Keep recent messages (sliding window)
        recent_messages = []
        recent_tokens = 0

        for msg in reversed(messages):
            if recent_tokens + msg.tokens <= recent_budget:
                recent_messages.insert(0, msg)
                recent_tokens += msg.tokens
            else:
                break

        # Phase 2: From remaining messages, keep important ones
        remaining_messages = [msg for msg in messages if msg not in recent_messages]

        if remaining_messages:
            # Calculate importance scores for remaining
            total_remaining = len(remaining_messages)
            for i, msg in enumerate(remaining_messages):
                if msg.importance_score == 0.0:
                    # Use ImportanceBasedStrategy's scoring
                    importance_strategy = ImportanceBasedStrategy(self.token_counter)
                    msg.importance_score = importance_strategy._calculate_importance_score(
                        msg, i, total_remaining
                    )

            # Sort by importance and select
            remaining_messages.sort(key=lambda m: m.importance_score, reverse=True)

            important_messages = []
            important_tokens = 0

            for msg in remaining_messages:
                if important_tokens + msg.tokens <= important_budget:
                    important_messages.append(msg)
                    important_tokens += msg.tokens
                else:
                    break

            # Combine and sort by timestamp
            pruned = recent_messages + important_messages
            pruned.sort(key=lambda m: m.timestamp)

            total_tokens = recent_tokens + important_tokens
        else:
            pruned = recent_messages
            total_tokens = recent_tokens

        logger.info(
            f"Hybrid pruning: {len(messages)} -> {len(pruned)} messages "
            f"({total_tokens + system_tokens}/{self.target_tokens} tokens, "
            f"recent={len(recent_messages)}, important={len(pruned) - len(recent_messages)})"
        )

        return pruned


class SummarizationStrategy(PruningStrategy):
    """
    Summarization strategy - summarizes older messages to compress context.

    Keeps recent messages intact and summarizes older chunks of conversation.
    This preserves more information than simple pruning at the cost of an LLM call.
    """

    def __init__(self, token_counter: TokenCounter, llm_client=None):
        super().__init__(token_counter)
        self.llm_client = llm_client
        self.summary_ratio = settings.LLM_CONTEXT_SUMMARY_RATIO

    async def _summarize_messages(self, messages: List[Message]) -> Message:
        """
        Summarize a chunk of messages into a single summary message.

        Args:
            messages: Messages to summarize

        Returns:
            Summary message
        """
        if not self.llm_client:
            # Fallback: create simple concatenated summary
            content = "\n".join([f"{msg.role}: {msg.content[:100]}..." for msg in messages])
            summary_content = f"[Summary of {len(messages)} messages]\n{content}"
        else:
            # Use LLM to create intelligent summary
            conversation_text = "\n".join([
                f"{msg.role.upper()}: {msg.content}"
                for msg in messages
            ])

            prompt = f"""Summarize the following conversation segment concisely, preserving key information and decisions:

{conversation_text}

Provide a brief summary (2-3 sentences) capturing the main points."""

            try:
                response = await self.llm_client.generate(prompt, max_tokens=150)
                summary_content = f"[Summary of {len(messages)} messages]: {response}"
            except Exception as e:
                logger.error(f"Summarization failed: {e}, using fallback")
                summary_content = f"[Summary of {len(messages)} messages - conversation about various topics]"

        # Create summary message
        return Message(
            role="system",
            content=summary_content,
            timestamp=messages[-1].timestamp,
            importance_score=0.5,  # Medium importance
            tokens=self.token_counter.count_tokens(summary_content)
        )

    async def prune(self, messages: List[Message], system_prompt: Optional[str] = None) -> List[Message]:
        """Keep recent messages and summarize older ones."""
        self._ensure_message_tokens(messages)

        system_tokens = self.token_counter.count_tokens(system_prompt) if system_prompt else 0
        available_tokens = self.target_tokens - system_tokens

        # Calculate how many recent messages to keep intact
        recent_budget = int(available_tokens * (1 - self.summary_ratio))
        summary_budget = available_tokens - recent_budget

        # Phase 1: Keep recent messages
        recent_messages = []
        recent_tokens = 0

        for msg in reversed(messages):
            if recent_tokens + msg.tokens <= recent_budget:
                recent_messages.insert(0, msg)
                recent_tokens += msg.tokens
            else:
                break

        # Phase 2: Summarize older messages
        older_messages = [msg for msg in messages if msg not in recent_messages]

        summarized_messages = []
        if older_messages:
            # Split older messages into chunks and summarize
            chunk_size = max(5, len(older_messages) // 3)  # At least 5 messages per chunk

            for i in range(0, len(older_messages), chunk_size):
                chunk = older_messages[i:i + chunk_size]
                summary = await self._summarize_messages(chunk)

                # Only add summary if we have budget
                if summary.tokens <= summary_budget:
                    summarized_messages.append(summary)
                    summary_budget -= summary.tokens
                else:
                    break

        # Combine summaries and recent messages
        pruned = summarized_messages + recent_messages
        total_tokens = sum(msg.tokens for msg in pruned)

        logger.info(
            f"Summarization pruning: {len(messages)} -> {len(pruned)} messages "
            f"({total_tokens + system_tokens}/{self.target_tokens} tokens, "
            f"summaries={len(summarized_messages)}, recent={len(recent_messages)})"
        )

        return pruned


class PruningStrategyFactory:
    """Factory for creating pruning strategy instances."""

    @staticmethod
    def create(strategy_name: str, token_counter: TokenCounter, llm_client=None) -> PruningStrategy:
        """
        Create a pruning strategy instance.

        Args:
            strategy_name: Name of strategy ('sliding_window', 'importance', 'hybrid', 'summarize')
            token_counter: TokenCounter instance
            llm_client: Optional LLM client for summarization strategy

        Returns:
            PruningStrategy instance

        Raises:
            ValueError: If strategy name is unknown
        """
        strategies = {
            'sliding_window': SlidingWindowStrategy,
            'importance': ImportanceBasedStrategy,
            'hybrid': HybridStrategy,
            'summarize': SummarizationStrategy
        }

        strategy_class = strategies.get(strategy_name)
        if not strategy_class:
            raise ValueError(
                f"Unknown pruning strategy: {strategy_name}. "
                f"Available: {', '.join(strategies.keys())}"
            )

        if strategy_name == 'summarize':
            return strategy_class(token_counter, llm_client)
        else:
            return strategy_class(token_counter)
