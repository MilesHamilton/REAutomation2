"""
Context window management for conversation history.

This module provides the ContextManager class that orchestrates token counting,
pruning strategies, and context optimization to keep conversations within
model token limits.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

from src.llm.token_counter import TokenCounter
from src.llm.pruning_strategies import (
    Message,
    PruningStrategy,
    PruningStrategyFactory
)
from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ContextStats:
    """Statistics about context usage."""
    total_messages: int
    total_tokens: int
    system_tokens: int
    messages_tokens: int
    utilization: float  # Percentage of context window used
    pruned: bool
    pruned_count: int
    strategy_used: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ContextManager:
    """
    Manages conversation context to stay within token limits.

    Features:
    - Token counting with accurate tiktoken-based counting
    - Multiple pruning strategies (sliding window, importance, hybrid, summarization)
    - Automatic pruning when context exceeds limits
    - Context statistics and monitoring
    - Integration with LLM service
    """

    def __init__(
        self,
        token_counter: Optional[TokenCounter] = None,
        strategy: Optional[str] = None,
        llm_client=None
    ):
        """
        Initialize context manager.

        Args:
            token_counter: Optional TokenCounter instance (creates default if None)
            strategy: Pruning strategy name (uses config default if None)
            llm_client: Optional LLM client for summarization strategy
        """
        self.token_counter = token_counter or TokenCounter()
        self.llm_client = llm_client

        # Configuration
        self.max_tokens = settings.LLM_CONTEXT_WINDOW
        self.target_ratio = settings.LLM_CONTEXT_TARGET_RATIO
        self.target_tokens = int(self.max_tokens * self.target_ratio)
        self.warning_threshold = int(self.max_tokens * settings.LLM_CONTEXT_WARNING_THRESHOLD)

        # Pruning strategy
        strategy_name = strategy or settings.LLM_CONTEXT_PRUNING_STRATEGY
        self.strategy: PruningStrategy = PruningStrategyFactory.create(
            strategy_name,
            self.token_counter,
            llm_client
        )
        self.strategy_name = strategy_name

        # Statistics
        self.stats_history: List[ContextStats] = []

        logger.info(
            f"ContextManager initialized: max_tokens={self.max_tokens}, "
            f"target_tokens={self.target_tokens}, strategy={self.strategy_name}"
        )

    def _convert_to_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Message]:
        """
        Convert message dictionaries to Message objects.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            List of Message objects
        """
        converted = []
        for msg in messages:
            converted.append(Message(
                role=msg.get('role', 'user'),
                content=msg.get('content', ''),
                timestamp=msg.get('timestamp', datetime.utcnow()),
                importance_score=msg.get('importance_score', 0.0),
                tokens=msg.get('tokens', 0)
            ))
        return converted

    def _convert_from_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert Message objects back to dictionaries.

        Args:
            messages: List of Message objects

        Returns:
            List of message dictionaries
        """
        return [
            {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp,
                'importance_score': msg.importance_score,
                'tokens': msg.tokens
            }
            for msg in messages
        ]

    def count_context_tokens(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Tuple[int, int, int]:
        """
        Count tokens in conversation context.

        Args:
            messages: List of message dictionaries
            system_prompt: Optional system prompt

        Returns:
            Tuple of (total_tokens, system_tokens, messages_tokens)
        """
        system_tokens = 0
        if system_prompt:
            system_tokens = self.token_counter.count_tokens(system_prompt)

        messages_tokens = 0
        for msg in messages:
            if 'tokens' in msg and msg['tokens'] > 0:
                messages_tokens += msg['tokens']
            else:
                msg_tokens = self.token_counter.count_tokens(msg.get('content', ''))
                msg['tokens'] = msg_tokens
                messages_tokens += msg_tokens

        total_tokens = system_tokens + messages_tokens

        return total_tokens, system_tokens, messages_tokens

    def needs_pruning(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> bool:
        """
        Check if context needs pruning.

        Args:
            messages: List of message dictionaries
            system_prompt: Optional system prompt

        Returns:
            True if pruning is needed
        """
        total_tokens, _, _ = self.count_context_tokens(messages, system_prompt)
        return total_tokens > self.target_tokens

    def is_near_limit(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> bool:
        """
        Check if context is approaching token limit.

        Args:
            messages: List of message dictionaries
            system_prompt: Optional system prompt

        Returns:
            True if context is above warning threshold
        """
        total_tokens, _, _ = self.count_context_tokens(messages, system_prompt)
        return total_tokens > self.warning_threshold

    async def manage_context(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        force_prune: bool = False
    ) -> Tuple[List[Dict[str, Any]], ContextStats]:
        """
        Manage conversation context, pruning if necessary.

        Args:
            messages: List of message dictionaries
            system_prompt: Optional system prompt
            force_prune: Force pruning even if under limit

        Returns:
            Tuple of (managed_messages, context_stats)
        """
        # Count tokens
        total_tokens, system_tokens, messages_tokens = self.count_context_tokens(
            messages, system_prompt
        )

        utilization = (total_tokens / self.max_tokens) * 100

        # Check if pruning is needed
        should_prune = force_prune or total_tokens > self.target_tokens

        if should_prune:
            logger.info(
                f"Context pruning triggered: {total_tokens}/{self.target_tokens} tokens "
                f"({utilization:.1f}% utilization)"
            )

            # Convert to Message objects
            message_objs = self._convert_to_messages(messages)
            original_count = len(message_objs)

            # Apply pruning strategy
            pruned_objs = await self.strategy.prune(message_objs, system_prompt)

            # Convert back to dictionaries
            pruned_messages = self._convert_from_messages(pruned_objs)

            # Recalculate tokens
            pruned_tokens, _, pruned_msg_tokens = self.count_context_tokens(
                pruned_messages, system_prompt
            )
            pruned_utilization = (pruned_tokens / self.max_tokens) * 100

            # Create stats
            stats = ContextStats(
                total_messages=len(pruned_messages),
                total_tokens=pruned_tokens,
                system_tokens=system_tokens,
                messages_tokens=pruned_msg_tokens,
                utilization=pruned_utilization,
                pruned=True,
                pruned_count=original_count - len(pruned_messages),
                strategy_used=self.strategy_name
            )

            logger.info(
                f"Context pruned: {original_count} -> {len(pruned_messages)} messages, "
                f"{total_tokens} -> {pruned_tokens} tokens ({pruned_utilization:.1f}% utilization)"
            )

            self.stats_history.append(stats)
            return pruned_messages, stats

        else:
            # No pruning needed
            stats = ContextStats(
                total_messages=len(messages),
                total_tokens=total_tokens,
                system_tokens=system_tokens,
                messages_tokens=messages_tokens,
                utilization=utilization,
                pruned=False,
                pruned_count=0,
                strategy_used=None
            )

            if self.is_near_limit(messages, system_prompt):
                logger.warning(
                    f"Context approaching limit: {total_tokens}/{self.max_tokens} tokens "
                    f"({utilization:.1f}% utilization)"
                )

            self.stats_history.append(stats)
            return messages, stats

    def get_context_stats(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> ContextStats:
        """
        Get current context statistics without pruning.

        Args:
            messages: List of message dictionaries
            system_prompt: Optional system prompt

        Returns:
            ContextStats object
        """
        total_tokens, system_tokens, messages_tokens = self.count_context_tokens(
            messages, system_prompt
        )
        utilization = (total_tokens / self.max_tokens) * 100

        return ContextStats(
            total_messages=len(messages),
            total_tokens=total_tokens,
            system_tokens=system_tokens,
            messages_tokens=messages_tokens,
            utilization=utilization,
            pruned=False,
            pruned_count=0,
            strategy_used=None
        )

    def clear_history(self) -> None:
        """Clear statistics history."""
        self.stats_history.clear()

    def get_stats_summary(self) -> Dict[str, Any]:
        """
        Get summary of context management statistics.

        Returns:
            Dictionary with aggregated statistics
        """
        if not self.stats_history:
            return {
                'total_operations': 0,
                'pruning_operations': 0,
                'pruning_rate': 0.0,
                'avg_utilization': 0.0,
                'avg_messages': 0,
                'avg_tokens': 0
            }

        pruning_ops = sum(1 for s in self.stats_history if s.pruned)
        avg_util = sum(s.utilization for s in self.stats_history) / len(self.stats_history)
        avg_msgs = sum(s.total_messages for s in self.stats_history) / len(self.stats_history)
        avg_tokens = sum(s.total_tokens for s in self.stats_history) / len(self.stats_history)

        return {
            'total_operations': len(self.stats_history),
            'pruning_operations': pruning_ops,
            'pruning_rate': pruning_ops / len(self.stats_history) if self.stats_history else 0.0,
            'avg_utilization': avg_util,
            'avg_messages': avg_msgs,
            'avg_tokens': avg_tokens,
            'strategy': self.strategy_name
        }

    def estimate_remaining_capacity(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Estimate remaining context capacity.

        Args:
            messages: List of message dictionaries
            system_prompt: Optional system prompt

        Returns:
            Dictionary with capacity information
        """
        total_tokens, system_tokens, messages_tokens = self.count_context_tokens(
            messages, system_prompt
        )

        remaining_tokens = self.max_tokens - total_tokens
        remaining_target = self.target_tokens - total_tokens

        return {
            'current_tokens': total_tokens,
            'max_tokens': self.max_tokens,
            'target_tokens': self.target_tokens,
            'remaining_tokens': remaining_tokens,
            'remaining_target': remaining_target,
            'can_fit': remaining_tokens > 0,
            'needs_pruning': total_tokens > self.target_tokens,
            'utilization_percent': (total_tokens / self.max_tokens) * 100
        }


class ContextManagerFactory:
    """Factory for creating ContextManager instances."""

    _default_instance: Optional[ContextManager] = None

    @classmethod
    def create(
        cls,
        token_counter: Optional[TokenCounter] = None,
        strategy: Optional[str] = None,
        llm_client=None
    ) -> ContextManager:
        """
        Create a new ContextManager instance.

        Args:
            token_counter: Optional TokenCounter instance
            strategy: Optional pruning strategy name
            llm_client: Optional LLM client

        Returns:
            ContextManager instance
        """
        return ContextManager(token_counter, strategy, llm_client)

    @classmethod
    def get_default(cls) -> ContextManager:
        """
        Get or create default ContextManager instance (singleton).

        Returns:
            Default ContextManager instance
        """
        if cls._default_instance is None:
            cls._default_instance = cls.create()
        return cls._default_instance

    @classmethod
    def reset_default(cls) -> None:
        """Reset default instance (for testing)."""
        cls._default_instance = None
