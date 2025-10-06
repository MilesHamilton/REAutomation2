"""Add voice agent integration fields

Revision ID: 002_voice_agent_integration
Revises: 001_add_langsmith_monitoring_tables
Create Date: 2025-10-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002_voice_agent_integration'
down_revision: Union[str, Sequence[str], None] = '001_langsmith_monitoring'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add voice agent integration fields and tables."""

    # Add workflow integration fields to calls table
    op.add_column('calls', sa.Column('workflow_context_id', sa.String(length=255), nullable=True))
    op.add_column('calls', sa.Column('current_agent', sa.String(length=100), nullable=True))
    op.add_column('calls', sa.Column('agent_transition_count', sa.Integer(), default=0))
    op.add_column('calls', sa.Column('tier_escalation_trigger', sa.String(length=100), nullable=True))

    # Create indexes for new columns
    op.create_index('ix_calls_workflow_context', 'calls', ['workflow_context_id'])
    op.create_index('ix_calls_current_agent', 'calls', ['current_agent'])

    # Create agent_transitions table
    op.create_table('agent_transitions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('call_id', sa.String(length=255), nullable=False),
        sa.Column('from_agent', sa.String(length=100), nullable=False),
        sa.Column('to_agent', sa.String(length=100), nullable=False),
        sa.Column('timestamp', sa.Float(), nullable=False),
        sa.Column('trigger', sa.String(length=100), nullable=False),
        sa.Column('context_preserved', sa.Boolean(), default=True),
        sa.Column('transition_duration_ms', sa.Float(), default=0.0),
        sa.Column('transition_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for agent_transitions table
    op.create_index('ix_agent_transitions_call_id', 'agent_transitions', ['call_id'])
    op.create_index('ix_agent_transitions_timestamp', 'agent_transitions', ['timestamp'])
    op.create_index('ix_agent_transitions_from_agent', 'agent_transitions', ['from_agent'])
    op.create_index('ix_agent_transitions_to_agent', 'agent_transitions', ['to_agent'])

    # Create tier_escalation_events table
    op.create_table('tier_escalation_events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('call_id', sa.String(length=255), nullable=False),
        sa.Column('from_tier', sa.String(length=50), nullable=False),
        sa.Column('to_tier', sa.String(length=50), nullable=False),
        sa.Column('trigger', sa.String(length=100), nullable=False),
        sa.Column('qualification_score', sa.Float(), nullable=True),
        sa.Column('budget_available', sa.Boolean(), default=True),
        sa.Column('escalation_approved', sa.Boolean(), default=True),
        sa.Column('timestamp', sa.Float(), nullable=False),
        sa.Column('cost_impact', sa.Float(), nullable=True),
        sa.Column('event_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for tier_escalation_events table
    op.create_index('ix_tier_escalation_call_id', 'tier_escalation_events', ['call_id'])
    op.create_index('ix_tier_escalation_timestamp', 'tier_escalation_events', ['timestamp'])
    op.create_index('ix_tier_escalation_trigger', 'tier_escalation_events', ['trigger'])


def downgrade() -> None:
    """Remove voice agent integration fields and tables."""

    # Drop tables
    op.drop_table('tier_escalation_events')
    op.drop_table('agent_transitions')

    # Drop indexes from calls table
    op.drop_index('ix_calls_current_agent', table_name='calls')
    op.drop_index('ix_calls_workflow_context', table_name='calls')

    # Drop columns from calls table
    op.drop_column('calls', 'tier_escalation_trigger')
    op.drop_column('calls', 'agent_transition_count')
    op.drop_column('calls', 'current_agent')
    op.drop_column('calls', 'workflow_context_id')
