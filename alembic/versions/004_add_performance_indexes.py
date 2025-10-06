"""Add performance optimization indexes

Revision ID: 004_performance_indexes
Revises: 003_context_management
Create Date: 2025-01-26 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '004_performance_indexes'
down_revision: Union[str, None] = '003_context_management'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add indexes for streaming, GPU monitoring, and performance optimization queries."""

    # ============================================
    # Calls Table - Performance Query Indexes
    # ============================================

    # Covering index for dashboard queries (status, created_at, total_cost)
    op.create_index(
        'ix_calls_status_created_cost',
        'calls',
        ['status', 'created_at', 'total_cost'],
        postgresql_using='btree'
    )

    # Index for qualified leads analytics
    op.create_index(
        'ix_calls_qualified_score_created',
        'calls',
        ['qualified', 'qualification_score', 'created_at']
    )

    # Index for cost analysis by tier
    op.create_index(
        'ix_calls_tier_cost',
        'calls',
        ['final_tier', 'total_cost']
    )

    # Partial index for context-pruned calls only (reduces index size)
    op.execute("""
        CREATE INDEX ix_calls_context_pruned_only
        ON calls(total_context_tokens, pruning_count)
        WHERE context_pruned = true
    """)

    # ============================================
    # Conversation History - Query Optimization
    # ============================================

    # Covering index for message retrieval (call_id, role, message_order)
    op.create_index(
        'ix_conversation_history_call_role_order',
        'conversation_history',
        ['call_id', 'role', 'message_order'],
        postgresql_using='btree'
    )

    # Index for token usage analysis
    op.create_index(
        'ix_conversation_history_tokens_used',
        'conversation_history',
        ['llm_tokens_used', 'created_at']
    )

    # Partial index for high-importance messages
    op.execute("""
        CREATE INDEX ix_conversation_history_important_only
        ON conversation_history(call_id, importance_score)
        WHERE importance_score > 0.5
    """)

    # ============================================
    # System Metrics - Time-Series Optimization
    # ============================================

    # Covering index for metric queries (name, recorded_at DESC, value)
    op.create_index(
        'ix_system_metrics_name_recorded_desc',
        'system_metrics',
        ['metric_name', sa.text('recorded_at DESC'), 'metric_value']
    )

    # Index for GPU metrics queries
    op.execute("""
        CREATE INDEX ix_system_metrics_gpu_metrics
        ON system_metrics(metric_name, recorded_at DESC, metric_value)
        WHERE metric_name LIKE 'gpu_%'
    """)

    # Index for streaming metrics queries
    op.execute("""
        CREATE INDEX ix_system_metrics_streaming_metrics
        ON system_metrics(metric_name, recorded_at DESC, metric_value)
        WHERE metric_name LIKE 'streaming_%'
    """)

    # ============================================
    # Cost Tracking - Analytics Optimization
    # ============================================

    # Covering index for cost analytics
    op.create_index(
        'ix_cost_tracking_type_date_amount',
        'cost_tracking',
        ['cost_type', 'daily_date', 'cost_amount']
    )

    # Index for tier-based cost queries
    op.create_index(
        'ix_cost_tracking_tier_date_amount',
        'cost_tracking',
        ['tier', 'daily_date', 'cost_amount']
    )

    # Index for monthly cost rollups
    op.create_index(
        'ix_cost_tracking_monthly_type',
        'cost_tracking',
        ['monthly_period', 'cost_type', 'cost_amount']
    )

    # ============================================
    # Workflow Traces - Monitoring Optimization
    # ============================================

    # Index for recent trace queries
    op.create_index(
        'ix_workflow_traces_recent',
        'workflow_traces',
        [sa.text('start_time DESC'), 'status']
    )

    # Index for failed traces
    op.execute("""
        CREATE INDEX ix_workflow_traces_errors
        ON workflow_traces(start_time DESC, error_type)
        WHERE error_occurred = true
    """)

    # ============================================
    # Agent Executions - Performance Tracking
    # ============================================

    # Index for agent performance analysis
    op.create_index(
        'ix_agent_executions_type_duration',
        'agent_executions',
        ['agent_type', 'duration_ms', 'start_time']
    )

    # Partial index for slow executions (duration > 2000ms)
    op.execute("""
        CREATE INDEX ix_agent_executions_slow
        ON agent_executions(agent_type, duration_ms, start_time)
        WHERE duration_ms > 2000
    """)


def downgrade() -> None:
    """Remove performance optimization indexes."""

    # Agent Executions
    op.drop_index('ix_agent_executions_slow', table_name='agent_executions')
    op.drop_index('ix_agent_executions_type_duration', table_name='agent_executions')

    # Workflow Traces
    op.drop_index('ix_workflow_traces_errors', table_name='workflow_traces')
    op.drop_index('ix_workflow_traces_recent', table_name='workflow_traces')

    # Cost Tracking
    op.drop_index('ix_cost_tracking_monthly_type', table_name='cost_tracking')
    op.drop_index('ix_cost_tracking_tier_date_amount', table_name='cost_tracking')
    op.drop_index('ix_cost_tracking_type_date_amount', table_name='cost_tracking')

    # System Metrics
    op.drop_index('ix_system_metrics_streaming_metrics', table_name='system_metrics')
    op.drop_index('ix_system_metrics_gpu_metrics', table_name='system_metrics')
    op.drop_index('ix_system_metrics_name_recorded_desc', table_name='system_metrics')

    # Conversation History
    op.drop_index('ix_conversation_history_important_only', table_name='conversation_history')
    op.drop_index('ix_conversation_history_tokens_used', table_name='conversation_history')
    op.drop_index('ix_conversation_history_call_role_order', table_name='conversation_history')

    # Calls
    op.drop_index('ix_calls_context_pruned_only', table_name='calls')
    op.drop_index('ix_calls_tier_cost', table_name='calls')
    op.drop_index('ix_calls_qualified_score_created', table_name='calls')
    op.drop_index('ix_calls_status_created_cost', table_name='calls')
