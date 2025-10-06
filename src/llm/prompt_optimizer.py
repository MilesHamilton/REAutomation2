"""
Prompt Optimizer for LLM Inference

This module provides prompt compression, optimization, and management
capabilities to reduce token usage and improve inference speed.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from src.llm.token_counter import TokenCounter
from src.config.settings import settings

logger = logging.getLogger(__name__)


class PromptType(str, Enum):
    """Types of prompts used in the system."""
    CONVERSATION = "conversation"
    QUALIFICATION = "qualification"
    OBJECTION_HANDLING = "objection_handling"
    STRUCTURED_OUTPUT = "structured_output"
    SUMMARIZATION = "summarization"


@dataclass
class PromptTemplate:
    """Optimized prompt template with metadata."""
    name: str
    type: PromptType
    template: str
    tokens: int = 0
    version: str = "1.0"
    variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return self.template


@dataclass
class OptimizationMetrics:
    """Metrics for prompt optimization."""
    original_tokens: int
    optimized_tokens: int
    reduction_percent: float
    techniques_applied: List[str]


class PromptOptimizer:
    """
    Optimizes prompts to reduce tokens while maintaining effectiveness.

    Techniques:
    - Remove redundant whitespace and formatting
    - Replace verbose phrases with concise alternatives
    - Remove unnecessary explanations
    - Use abbreviations where appropriate
    - Optimize for specific model (Llama 3.1 8B)
    """

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        self.token_counter = token_counter or TokenCounter()
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_optimized_templates()

    def _load_optimized_templates(self):
        """Load pre-optimized prompt templates."""

        # Conversation template (optimized for Llama 3.1)
        conversation_template = """You're an AI sales agent for {company}. Be friendly, concise (1-2 sentences), ask one question at a time.

Lead: {lead_context}
State: {conversation_state}"""

        self.templates["conversation"] = PromptTemplate(
            name="conversation",
            type=PromptType.CONVERSATION,
            template=conversation_template,
            tokens=self.token_counter.count_tokens(conversation_template),
            variables=["company", "lead_context", "conversation_state"]
        )

        # Qualification template (optimized)
        qualification_template = """Analyze conversation for lead qualification. Score 0-1 on:
- Intent, Budget, Timeline, Authority, Needs

JSON format:
{{
  "qualification_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "factors": {{"intent": 0.0-1.0, "budget": 0.0-1.0, "timeline": 0.0-1.0, "authority": 0.0-1.0, "needs": 0.0-1.0}},
  "reasoning": "Brief explanation",
  "recommended_action": "continue|escalate|disqualify"
}}"""

        self.templates["qualification"] = PromptTemplate(
            name="qualification",
            type=PromptType.QUALIFICATION,
            template=qualification_template,
            tokens=self.token_counter.count_tokens(qualification_template),
            variables=[]
        )

        # Objection handling template (optimized)
        objection_template = """Handle objection: "{objection_type}"
Keep response empathetic, brief (1-2 sentences), address concern without being pushy."""

        self.templates["objection"] = PromptTemplate(
            name="objection",
            type=PromptType.OBJECTION_HANDLING,
            template=objection_template,
            tokens=self.token_counter.count_tokens(objection_template),
            variables=["objection_type"]
        )

        # Summarization template (optimized)
        summarization_template = """Summarize this conversation segment in 2-3 sentences, preserving key points:

{conversation_text}"""

        self.templates["summarization"] = PromptTemplate(
            name="summarization",
            type=PromptType.SUMMARIZATION,
            template=summarization_template,
            tokens=self.token_counter.count_tokens(summarization_template),
            variables=["conversation_text"]
        )

        logger.info(f"Loaded {len(self.templates)} optimized prompt templates")

    def optimize_prompt(self, prompt: str, aggressive: bool = False) -> Tuple[str, OptimizationMetrics]:
        """
        Optimize a prompt to reduce tokens while maintaining meaning.

        Args:
            prompt: Original prompt text
            aggressive: Apply more aggressive optimizations

        Returns:
            Tuple of (optimized_prompt, metrics)
        """
        original_tokens = self.token_counter.count_tokens(prompt)
        optimized = prompt
        techniques = []

        # 1. Remove excessive whitespace
        optimized = self._remove_excess_whitespace(optimized)
        if optimized != prompt:
            techniques.append("whitespace_removal")

        # 2. Remove formatting characters
        optimized = self._remove_formatting(optimized)
        if len(optimized) < len(prompt):
            techniques.append("formatting_removal")

        # 3. Replace verbose phrases
        optimized, replaced = self._replace_verbose_phrases(optimized)
        if replaced:
            techniques.append("phrase_optimization")

        # 4. Remove filler words
        optimized, removed = self._remove_filler_words(optimized)
        if removed:
            techniques.append("filler_removal")

        if aggressive:
            # 5. Aggressive abbreviations
            optimized = self._apply_abbreviations(optimized)
            techniques.append("abbreviations")

            # 6. Remove unnecessary explanations
            optimized = self._remove_explanations(optimized)
            techniques.append("explanation_removal")

        optimized_tokens = self.token_counter.count_tokens(optimized)
        reduction = ((original_tokens - optimized_tokens) / original_tokens * 100) if original_tokens > 0 else 0

        metrics = OptimizationMetrics(
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            reduction_percent=reduction,
            techniques_applied=techniques
        )

        logger.debug(
            f"Prompt optimized: {original_tokens} -> {optimized_tokens} tokens "
            f"({reduction:.1f}% reduction)"
        )

        return optimized, metrics

    def _remove_excess_whitespace(self, text: str) -> str:
        """Remove excessive whitespace."""
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        # Remove multiple newlines (keep max 2)
        text = re.sub(r'\n\n\n+', '\n\n', text)
        # Remove trailing whitespace
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text.strip()

    def _remove_formatting(self, text: str) -> str:
        """Remove unnecessary formatting characters."""
        # Remove markdown bold/italic (keep content)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        # Remove excessive punctuation
        text = re.sub(r'\.\.\.+', '...', text)
        text = re.sub(r'!!!+', '!', text)
        return text

    def _replace_verbose_phrases(self, text: str) -> Tuple[str, bool]:
        """Replace verbose phrases with concise alternatives."""
        replacements = {
            # Common verbose patterns
            r'\bplease be aware that\b': '',
            r'\bit is important to note that\b': '',
            r'\bin order to\b': 'to',
            r'\bdue to the fact that\b': 'because',
            r'\bat this point in time\b': 'now',
            r'\bin the event that\b': 'if',
            r'\bfor the purpose of\b': 'to',
            r'\bwith regard to\b': 'about',
            r'\bin a timely manner\b': 'quickly',

            # Sales-specific verbose patterns
            r'\bI would like to\b': "I'll",
            r'\bwould you be interested in\b': 'interested in',
            r'\bdo you have any questions\b': 'questions?',
            r'\blook forward to hearing from you\b': 'talk soon',
            r'\bthank you for your time\b': 'thanks',

            # Instruction verbose patterns
            r'\byou should try to\b': 'try to',
            r'\bmake sure that you\b': 'ensure you',
            r'\bit is necessary to\b': 'you must',
        }

        original = text
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text, text != original

    def _remove_filler_words(self, text: str) -> Tuple[str, bool]:
        """Remove unnecessary filler words."""
        fillers = [
            r'\bjust\b', r'\breally\b', r'\bactually\b', r'\bbasically\b',
            r'\bliterally\b', r'\bobviously\b', r'\bclearly\b', r'\bsimply\b',
            r'\bvery\b', r'\bquite\b', r'\brather\b'
        ]

        original = text
        for filler in fillers:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)

        # Clean up any double spaces created
        text = re.sub(r' +', ' ', text)

        return text, text != original

    def _apply_abbreviations(self, text: str) -> str:
        """Apply common abbreviations (aggressive mode)."""
        abbreviations = {
            r'\byou are\b': "you're",
            r'\bwe are\b': "we're",
            r'\bI am\b': "I'm",
            r'\bthey are\b': "they're",
            r'\bdo not\b': "don't",
            r'\bcannot\b': "can't",
            r'\bwill not\b': "won't",
            r'\bshould not\b': "shouldn't",
            r'\band\b': '&',  # Only in non-conversational contexts
        }

        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _remove_explanations(self, text: str) -> str:
        """Remove unnecessary explanatory text (aggressive mode)."""
        # Remove parenthetical explanations
        text = re.sub(r'\([^)]+\)', '', text)
        # Remove "Note:" sections
        text = re.sub(r'Note:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove "Remember:" sections
        text = re.sub(r'Remember:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)

        return text

    def get_template(self, name: str, **kwargs) -> str:
        """
        Get and format an optimized template.

        Args:
            name: Template name
            **kwargs: Variables to format template with

        Returns:
            Formatted template string
        """
        if name not in self.templates:
            logger.warning(f"Template '{name}' not found, returning empty string")
            return ""

        template = self.templates[name]
        return template.format(**kwargs)

    def register_template(self, template: PromptTemplate) -> None:
        """Register a new optimized template."""
        template.tokens = self.token_counter.count_tokens(template.template)
        self.templates[template.name] = template
        logger.info(f"Registered template '{template.name}' ({template.tokens} tokens)")

    def compare_prompts(self, original: str, optimized: str) -> Dict[str, Any]:
        """
        Compare two prompts and return detailed metrics.

        Args:
            original: Original prompt
            optimized: Optimized prompt

        Returns:
            Dictionary with comparison metrics
        """
        original_tokens = self.token_counter.count_tokens(original)
        optimized_tokens = self.token_counter.count_tokens(optimized)

        reduction = ((original_tokens - optimized_tokens) / original_tokens * 100) if original_tokens > 0 else 0

        return {
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "tokens_saved": original_tokens - optimized_tokens,
            "reduction_percent": reduction,
            "original_length": len(original),
            "optimized_length": len(optimized),
            "compression_ratio": len(optimized) / len(original) if len(original) > 0 else 1.0
        }

    def get_all_templates(self) -> Dict[str, PromptTemplate]:
        """Get all registered templates."""
        return self.templates.copy()

    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about all templates."""
        total_templates = len(self.templates)
        total_tokens = sum(t.tokens for t in self.templates.values())
        avg_tokens = total_tokens / total_templates if total_templates > 0 else 0

        by_type = {}
        for template in self.templates.values():
            type_name = template.type.value
            if type_name not in by_type:
                by_type[type_name] = {"count": 0, "total_tokens": 0}
            by_type[type_name]["count"] += 1
            by_type[type_name]["total_tokens"] += template.tokens

        return {
            "total_templates": total_templates,
            "total_tokens": total_tokens,
            "average_tokens": avg_tokens,
            "by_type": by_type
        }
