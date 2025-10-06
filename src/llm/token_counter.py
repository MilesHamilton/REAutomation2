"""
Token Counter for LLM Context Management

Provides accurate token counting using tiktoken for context window management.
Supports multiple models and provides fallback estimation when tiktoken unavailable.
"""

import logging
from typing import List, Optional, Dict
from functools import lru_cache

from .models import Message, MessageRole

logger = logging.getLogger(__name__)

# Try to import tiktoken
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using approximate token counting")


class TokenCounter:
    """
    Token counter for LLM requests

    Uses tiktoken for accurate counting when available,
    falls back to estimation based on word/character count.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize token counter

        Args:
            model_name: Model name for tiktoken encoding (default: gpt-3.5-turbo)
                       Works reasonably well for most models including Llama
        """
        self.model_name = model_name
        self.encoding = None

        if TIKTOKEN_AVAILABLE:
            try:
                # Try to get encoding for specific model
                self.encoding = tiktoken.encoding_for_model(model_name)
                logger.info(f"Token counter initialized with tiktoken for {model_name}")
            except KeyError:
                # Fallback to cl100k_base (GPT-3.5/4 encoding)
                self.encoding = tiktoken.get_encoding("cl100k_base")
                logger.info("Token counter initialized with cl100k_base encoding")
        else:
            logger.warning("Token counter using estimation (tiktoken not available)")

    @lru_cache(maxsize=1000)
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        if self.encoding:
            # Use tiktoken for accurate counting
            try:
                tokens = self.encoding.encode(text)
                return len(tokens)
            except Exception as e:
                logger.warning(f"tiktoken encoding failed: {e}, using estimation")
                return self._estimate_tokens(text)
        else:
            # Fallback to estimation
            return self._estimate_tokens(text)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count without tiktoken

        Rule of thumb for English text:
        - 1 token ≈ 4 characters
        - 1 token ≈ 0.75 words

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Use character-based estimation (more accurate for varied text)
        char_count = len(text)
        estimated = char_count / 4.0

        # Round up to nearest integer
        return max(1, int(estimated + 0.5))

    def count_message_tokens(
        self,
        message: Message,
        include_role_tokens: bool = True
    ) -> int:
        """
        Count tokens in a message

        Accounts for:
        - Message content
        - Role prefix tokens
        - Formatting tokens

        Args:
            message: Message to count tokens for
            include_role_tokens: Include tokens for role formatting (default: True)

        Returns:
            Total token count for message
        """
        # Count content tokens
        content_tokens = self.count_tokens(message.content)

        if not include_role_tokens:
            return content_tokens

        # Add tokens for role and formatting
        # Format: "<role>: <content>\n"
        # Rough estimate: role name + colon + space + newline ≈ 3-5 tokens
        role_tokens = 4

        return content_tokens + role_tokens

    def count_messages_tokens(
        self,
        messages: List[Message],
        include_system_prompt: bool = True
    ) -> int:
        """
        Count total tokens across multiple messages

        Args:
            messages: List of messages
            include_system_prompt: Count system messages (default: True)

        Returns:
            Total token count
        """
        total_tokens = 0

        for message in messages:
            # Skip system messages if requested
            if not include_system_prompt and message.role == MessageRole.SYSTEM:
                continue

            total_tokens += self.count_message_tokens(message)

        # Add overhead for message formatting
        # Each message has wrapper tokens: ~3 tokens per message
        overhead = len(messages) * 3

        return total_tokens + overhead

    def count_request_tokens(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        max_tokens: int = 0
    ) -> Dict[str, int]:
        """
        Count all tokens in a complete LLM request

        Args:
            messages: Conversation messages
            system_prompt: System prompt (if separate)
            max_tokens: Maximum tokens for response

        Returns:
            Dict with token breakdown:
            - messages_tokens: Tokens in messages
            - system_tokens: Tokens in system prompt
            - max_response_tokens: Maximum response tokens
            - total_input_tokens: Total input to model
            - total_tokens: Total including response
        """
        # Count message tokens
        messages_tokens = self.count_messages_tokens(
            messages,
            include_system_prompt=False
        )

        # Count system prompt if provided separately
        system_tokens = 0
        if system_prompt:
            system_tokens = self.count_tokens(system_prompt) + 4  # +4 for formatting

        # Calculate totals
        total_input = messages_tokens + system_tokens
        total_with_response = total_input + max_tokens

        return {
            "messages_tokens": messages_tokens,
            "system_tokens": system_tokens,
            "max_response_tokens": max_tokens,
            "total_input_tokens": total_input,
            "total_tokens": total_with_response
        }

    def will_fit_in_context(
        self,
        messages: List[Message],
        context_window: int,
        system_prompt: Optional[str] = None,
        max_tokens: int = 150,
        reserve_tokens: int = 100
    ) -> tuple[bool, int]:
        """
        Check if messages will fit in context window

        Args:
            messages: Messages to check
            context_window: Model's context window size
            system_prompt: System prompt if separate
            max_tokens: Maximum tokens for response
            reserve_tokens: Extra tokens to reserve

        Returns:
            Tuple of (will_fit: bool, tokens_over: int)
            tokens_over is 0 if fits, otherwise number of tokens to remove
        """
        token_counts = self.count_request_tokens(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )

        total_needed = token_counts["total_tokens"] + reserve_tokens
        tokens_over = total_needed - context_window

        will_fit = tokens_over <= 0

        return will_fit, max(0, tokens_over)

    def get_stats(self) -> Dict[str, any]:
        """Get token counter statistics"""
        return {
            "tiktoken_available": TIKTOKEN_AVAILABLE,
            "model_name": self.model_name,
            "encoding": self.encoding.name if self.encoding else "estimation",
            "cache_size": self.count_tokens.cache_info().currsize if TIKTOKEN_AVAILABLE else 0,
            "cache_hits": self.count_tokens.cache_info().hits if TIKTOKEN_AVAILABLE else 0
        }


# Global token counter instance
token_counter = TokenCounter(model_name="gpt-3.5-turbo")
