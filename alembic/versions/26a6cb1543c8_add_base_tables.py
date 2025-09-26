"""Add base tables

Revision ID: 26a6cb1543c8
Revises: 
Create Date: 2025-09-26 12:03:21.978341

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '26a6cb1543c8'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create base tables."""
    # Create calls table
    op.create_table('calls',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('call_id', sa.String(length=255), nullable=False),
        sa.Column('phone_number', sa.String(length=20), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('lead_data', sa.JSON(), nullable=True),
        sa.Column('qualification_score', sa.Float(), nullable=True),
        sa.Column('qualified', sa.Boolean(), nullable=True),
        sa.Column('initial_tier', sa.String(length=50), nullable=False),
        sa.Column('final_tier', sa.String(length=50), nullable=True),
        sa.Column('tier_switches', sa.Integer(), nullable=True),
        sa.Column('escalation_trigger', sa.String(length=100), nullable=True),
        sa.Column('total_cost', sa.Float(), nullable=True),
        sa.Column('llm_cost', sa.Float(), nullable=True),
        sa.Column('tts_cost', sa.Float(), nullable=True),
        sa.Column('stt_cost', sa.Float(), nullable=True),
        sa.Column('avg_response_time_ms', sa.Float(), nullable=True),
        sa.Column('llm_latency_ms', sa.Float(), nullable=True),
        sa.Column('tts_latency_ms', sa.Float(), nullable=True),
        sa.Column('stt_latency_ms', sa.Float(), nullable=True),
        sa.Column('error_occurred', sa.Boolean(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_code', sa.String(length=50), nullable=True),
        sa.Column('conversation_summary', sa.Text(), nullable=True),
        sa.Column('total_messages', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_calls_call_id', 'calls', ['call_id'], unique=True)
    op.create_index('ix_calls_cost_created', 'calls', ['total_cost', 'created_at'])
    op.create_index('ix_calls_id', 'calls', ['id'])
    op.create_index('ix_calls_phone_created', 'calls', ['phone_number', 'created_at'])
    op.create_index('ix_calls_phone_number', 'calls', ['phone_number'])
    op.create_index('ix_calls_qualification_score', 'calls', ['qualification_score'])
    op.create_index('ix_calls_qualified', 'calls', ['qualified'])
    op.create_index('ix_calls_qualified_created', 'calls', ['qualified', 'created_at'])
    op.create_index('ix_calls_status', 'calls', ['status'])
    op.create_index('ix_calls_status_created', 'calls', ['status', 'created_at'])

    # Create conversation_history table
    op.create_table('conversation_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('call_id', sa.String(length=255), nullable=False),
        sa.Column('message_order', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(length=20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('agent_type', sa.String(length=50), nullable=True),
        sa.Column('agent_state', sa.JSON(), nullable=True),
        sa.Column('llm_tokens_used', sa.Integer(), nullable=True),
        sa.Column('tts_characters', sa.Integer(), nullable=True),
        sa.Column('processing_cost', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['call_id'], ['calls.call_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_conv_call_order', 'conversation_history', ['call_id', 'message_order'])
    op.create_index('ix_conv_call_role', 'conversation_history', ['call_id', 'role'])
    op.create_index('ix_conversation_history_call_id', 'conversation_history', ['call_id'])
    op.create_index('ix_conversation_history_id', 'conversation_history', ['id'])

    # Create other base tables
    op.create_table('tier_switches',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('call_id', sa.String(length=255), nullable=False),
        sa.Column('from_tier', sa.String(length=50), nullable=False),
        sa.Column('to_tier', sa.String(length=50), nullable=False),
        sa.Column('trigger', sa.String(length=100), nullable=False),
        sa.Column('qualification_score_at_switch', sa.Float(), nullable=True),
        sa.Column('switched_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('cost_before_switch', sa.Float(), nullable=True),
        sa.Column('estimated_cost_savings', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['call_id'], ['calls.call_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_tier_switches_call_id', 'tier_switches', ['call_id'])
    op.create_index('ix_tier_switches_id', 'tier_switches', ['id'])

    # Create contacts table
    op.create_table('contacts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('phone_number', sa.String(length=20), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('email', sa.String(length=255), nullable=True),
        sa.Column('company', sa.String(length=255), nullable=True),
        sa.Column('lead_score', sa.Float(), nullable=True),
        sa.Column('lead_source', sa.String(length=100), nullable=True),
        sa.Column('lead_status', sa.String(length=50), nullable=True),
        sa.Column('preferred_contact_time', sa.String(length=50), nullable=True),
        sa.Column('timezone', sa.String(length=50), nullable=True),
        sa.Column('do_not_call', sa.Boolean(), nullable=True),
        sa.Column('custom_fields', sa.JSON(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('campaign_id', sa.String(length=100), nullable=True),
        sa.Column('campaign_name', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_contacted', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_calls', sa.Integer(), nullable=True),
        sa.Column('successful_calls', sa.Integer(), nullable=True),
        sa.Column('qualified_calls', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_contacts_campaign_id', 'contacts', ['campaign_id'])
    op.create_index('ix_contacts_do_not_call', 'contacts', ['do_not_call'])
    op.create_index('ix_contacts_id', 'contacts', ['id'])
    op.create_index('ix_contacts_lead_score', 'contacts', ['lead_score'])
    op.create_index('ix_contacts_lead_status', 'contacts', ['lead_status'])
    op.create_index('ix_contacts_phone_number', 'contacts', ['phone_number'], unique=True)

    # Create system_metrics table
    op.create_table('system_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('metric_type', sa.String(length=50), nullable=False),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('metric_unit', sa.String(length=20), nullable=True),
        sa.Column('call_id', sa.String(length=255), nullable=True),
        sa.Column('phone_number', sa.String(length=20), nullable=True),
        sa.Column('agent_type', sa.String(length=50), nullable=True),
        sa.Column('tier', sa.String(length=50), nullable=True),
        sa.Column('meta_data', sa.JSON(), nullable=True),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_metrics_call_time', 'system_metrics', ['call_id', 'recorded_at'])
    op.create_index('ix_metrics_name_time', 'system_metrics', ['metric_name', 'recorded_at'])
    op.create_index('ix_metrics_type_time', 'system_metrics', ['metric_type', 'recorded_at'])
    op.create_index('ix_system_metrics_call_id', 'system_metrics', ['call_id'])
    op.create_index('ix_system_metrics_id', 'system_metrics', ['id'])
    op.create_index('ix_system_metrics_metric_name', 'system_metrics', ['metric_name'])
    op.create_index('ix_system_metrics_metric_type', 'system_metrics', ['metric_type'])
    op.create_index('ix_system_metrics_recorded_at', 'system_metrics', ['recorded_at'])

    # Create cost_tracking table
    op.create_table('cost_tracking',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('cost_type', sa.String(length=50), nullable=False),
        sa.Column('cost_amount', sa.Float(), nullable=False),
        sa.Column('cost_currency', sa.String(length=3), nullable=True),
        sa.Column('units_consumed', sa.Float(), nullable=True),
        sa.Column('unit_type', sa.String(length=50), nullable=True),
        sa.Column('unit_cost', sa.Float(), nullable=True),
        sa.Column('call_id', sa.String(length=255), nullable=True),
        sa.Column('service_provider', sa.String(length=100), nullable=True),
        sa.Column('tier', sa.String(length=50), nullable=True),
        sa.Column('daily_date', sa.String(length=10), nullable=False),
        sa.Column('monthly_period', sa.String(length=7), nullable=False),
        sa.Column('incurred_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_cost_call_type', 'cost_tracking', ['call_id', 'cost_type'])
    op.create_index('ix_cost_tier_date', 'cost_tracking', ['tier', 'daily_date'])
    op.create_index('ix_cost_tracking_call_id', 'cost_tracking', ['call_id'])
    op.create_index('ix_cost_tracking_cost_type', 'cost_tracking', ['cost_type'])
    op.create_index('ix_cost_tracking_daily_date', 'cost_tracking', ['daily_date'])
    op.create_index('ix_cost_tracking_id', 'cost_tracking', ['id'])
    op.create_index('ix_cost_tracking_incurred_at', 'cost_tracking', ['incurred_at'])
    op.create_index('ix_cost_tracking_monthly_period', 'cost_tracking', ['monthly_period'])
    op.create_index('ix_cost_tracking_tier', 'cost_tracking', ['tier'])
    op.create_index('ix_cost_type_date', 'cost_tracking', ['cost_type', 'daily_date'])

    # Create scheduled_calls table
    op.create_table('scheduled_calls',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('phone_number', sa.String(length=20), nullable=False),
        sa.Column('lead_data', sa.JSON(), nullable=True),
        sa.Column('priority', sa.Integer(), nullable=True),
        sa.Column('scheduled_for', sa.DateTime(timezone=True), nullable=False),
        sa.Column('timezone', sa.String(length=50), nullable=True),
        sa.Column('campaign_id', sa.String(length=100), nullable=True),
        sa.Column('campaign_name', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('attempts', sa.Integer(), nullable=True),
        sa.Column('max_attempts', sa.Integer(), nullable=True),
        sa.Column('next_retry_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('retry_reason', sa.String(length=255), nullable=True),
        sa.Column('call_id', sa.String(length=255), nullable=True),
        sa.Column('completion_status', sa.String(length=50), nullable=True),
        sa.Column('qualification_result', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_scheduled_calls_campaign_id', 'scheduled_calls', ['campaign_id'])
    op.create_index('ix_scheduled_calls_id', 'scheduled_calls', ['id'])
    op.create_index('ix_scheduled_calls_next_retry_at', 'scheduled_calls', ['next_retry_at'])
    op.create_index('ix_scheduled_calls_phone_number', 'scheduled_calls', ['phone_number'])
    op.create_index('ix_scheduled_calls_priority', 'scheduled_calls', ['priority'])
    op.create_index('ix_scheduled_calls_scheduled_for', 'scheduled_calls', ['scheduled_for'])
    op.create_index('ix_scheduled_calls_status', 'scheduled_calls', ['status'])
    op.create_index('ix_scheduled_campaign_status', 'scheduled_calls', ['campaign_id', 'status'])
    op.create_index('ix_scheduled_priority_time', 'scheduled_calls', ['priority', 'scheduled_for'])
    op.create_index('ix_scheduled_retry_time', 'scheduled_calls', ['next_retry_at'])
    op.create_index('ix_scheduled_status_time', 'scheduled_calls', ['status', 'scheduled_for'])


def downgrade() -> None:
    """Drop base tables."""
    op.drop_table('scheduled_calls')
    op.drop_table('cost_tracking')
    op.drop_table('system_metrics')
    op.drop_table('contacts')
    op.drop_table('tier_switches')
    op.drop_table('conversation_history')
    op.drop_table('calls')
