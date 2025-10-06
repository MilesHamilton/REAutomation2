"""Add context management fields

Revision ID: 003_context_management
Revises: 002_voice_agent_integration
Create Date: 2025-01-26 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003_context_management'
down_revision: Union[str, None] = '002_voice_agent_integration'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add context management fields to track LLM context window optimization."""

    # Add context management fields to calls table
    op.add_column('calls', sa.Column('context_pruned', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('calls', sa.Column('pruning_count', sa.Integer(), server_default='0', nullable=False))
    op.add_column('calls', sa.Column('total_context_tokens', sa.Integer(), server_default='0', nullable=False))

    # Create indexes for context management queries
    op.create_index('ix_calls_context_pruned', 'calls', ['context_pruned'])
    op.create_index('ix_calls_pruning_count', 'calls', ['pruning_count'])
    op.create_index('ix_calls_total_context_tokens', 'calls', ['total_context_tokens'])

    # Composite index for context analytics queries
    op.create_index(
        'ix_calls_context_pruned_tokens',
        'calls',
        ['context_pruned', 'total_context_tokens']
    )

    # Add context management fields to conversation_history table
    op.add_column('conversation_history', sa.Column('importance_score', sa.Float(), server_default='0.0', nullable=False))
    op.add_column('conversation_history', sa.Column('token_count', sa.Integer(), server_default='0', nullable=False))

    # Create indexes for message importance and token tracking
    op.create_index('ix_conversation_history_importance_score', 'conversation_history', ['importance_score'])
    op.create_index('ix_conversation_history_token_count', 'conversation_history', ['token_count'])

    # Composite index for pruning strategy queries
    op.create_index(
        'ix_conversation_history_call_importance',
        'conversation_history',
        ['call_id', 'importance_score']
    )


def downgrade() -> None:
    """Remove context management fields."""

    # Drop indexes from conversation_history
    op.drop_index('ix_conversation_history_call_importance', table_name='conversation_history')
    op.drop_index('ix_conversation_history_token_count', table_name='conversation_history')
    op.drop_index('ix_conversation_history_importance_score', table_name='conversation_history')

    # Drop columns from conversation_history
    op.drop_column('conversation_history', 'token_count')
    op.drop_column('conversation_history', 'importance_score')

    # Drop indexes from calls
    op.drop_index('ix_calls_context_pruned_tokens', table_name='calls')
    op.drop_index('ix_calls_total_context_tokens', table_name='calls')
    op.drop_index('ix_calls_pruning_count', table_name='calls')
    op.drop_index('ix_calls_context_pruned', table_name='calls')

    # Drop columns from calls
    op.drop_column('calls', 'total_context_tokens')
    op.drop_column('calls', 'pruning_count')
    op.drop_column('calls', 'context_pruned')
