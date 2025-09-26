"""Add LangSmith monitoring tables

Revision ID: 001_langsmith_monitoring
Revises:
Create Date: 2025-01-26 10:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_langsmith_monitoring'
down_revision: Union[str, None] = '26a6cb1543c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create workflow_traces table
    op.create_table('workflow_traces',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('trace_id', sa.String(length=255), nullable=False),
        sa.Column('call_id', sa.String(length=255), nullable=False),
        sa.Column('workflow_name', sa.String(length=100), nullable=False),
        sa.Column('workflow_version', sa.String(length=20), nullable=True),
        sa.Column('parent_trace_id', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_ms', sa.Float(), nullable=True),
        sa.Column('initial_context', sa.JSON(), nullable=True),
        sa.Column('final_context', sa.JSON(), nullable=True),
        sa.Column('workflow_state', sa.JSON(), nullable=True),
        sa.Column('error_occurred', sa.Boolean(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_type', sa.String(length=100), nullable=True),
        sa.Column('total_agents_executed', sa.Integer(), nullable=True),
        sa.Column('total_llm_calls', sa.Integer(), nullable=True),
        sa.Column('total_cost', sa.Float(), nullable=True),
        sa.Column('langsmith_run_id', sa.String(length=255), nullable=True),
        sa.Column('langsmith_project', sa.String(length=100), nullable=True),
        sa.Column('langsmith_url', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['call_id'], ['calls.call_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_workflow_traces_call_id', 'workflow_traces', ['call_id'])
    op.create_index('ix_workflow_traces_call_status', 'workflow_traces', ['call_id', 'status'])
    op.create_index('ix_workflow_traces_id', 'workflow_traces', ['id'])
    op.create_index('ix_workflow_traces_langsmith_run_id', 'workflow_traces', ['langsmith_run_id'])
    op.create_index('ix_workflow_traces_parent_trace_id', 'workflow_traces', ['parent_trace_id'])
    op.create_index('ix_workflow_traces_start_time', 'workflow_traces', ['start_time'])
    op.create_index('ix_workflow_traces_status', 'workflow_traces', ['status'])
    op.create_index('ix_workflow_traces_status_time', 'workflow_traces', ['status', 'start_time'])
    op.create_index('ix_workflow_traces_trace_id', 'workflow_traces', ['trace_id'], unique=True)
    op.create_index('ix_workflow_traces_workflow_name', 'workflow_traces', ['workflow_name'])
    op.create_index('ix_workflow_traces_workflow_time', 'workflow_traces', ['workflow_name', 'start_time'])

    # Create agent_executions table
    op.create_table('agent_executions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('execution_id', sa.String(length=255), nullable=False),
        sa.Column('trace_id', sa.String(length=255), nullable=False),
        sa.Column('agent_type', sa.String(length=100), nullable=False),
        sa.Column('agent_name', sa.String(length=100), nullable=False),
        sa.Column('agent_version', sa.String(length=20), nullable=True),
        sa.Column('execution_order', sa.Integer(), nullable=False),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_ms', sa.Float(), nullable=True),
        sa.Column('input_data', sa.JSON(), nullable=True),
        sa.Column('output_data', sa.JSON(), nullable=True),
        sa.Column('agent_state_before', sa.JSON(), nullable=True),
        sa.Column('agent_state_after', sa.JSON(), nullable=True),
        sa.Column('decision_rationale', sa.Text(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('routing_decision', sa.String(length=100), nullable=True),
        sa.Column('llm_calls_made', sa.Integer(), nullable=True),
        sa.Column('tokens_consumed', sa.Integer(), nullable=True),
        sa.Column('processing_cost', sa.Float(), nullable=True),
        sa.Column('error_occurred', sa.Boolean(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=True),
        sa.Column('langsmith_run_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['trace_id'], ['workflow_traces.trace_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_agent_exec_duration', 'agent_executions', ['duration_ms'])
    op.create_index('ix_agent_exec_trace_order', 'agent_executions', ['trace_id', 'execution_order'])
    op.create_index('ix_agent_exec_type_time', 'agent_executions', ['agent_type', 'start_time'])
    op.create_index('ix_agent_executions_agent_type', 'agent_executions', ['agent_type'])
    op.create_index('ix_agent_executions_execution_id', 'agent_executions', ['execution_id'])
    op.create_index('ix_agent_executions_id', 'agent_executions', ['id'])
    op.create_index('ix_agent_executions_langsmith_run_id', 'agent_executions', ['langsmith_run_id'])
    op.create_index('ix_agent_executions_start_time', 'agent_executions', ['start_time'])
    op.create_index('ix_agent_executions_trace_id', 'agent_executions', ['trace_id'])

    # Create performance_metrics table
    op.create_table('performance_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('metric_id', sa.String(length=255), nullable=False),
        sa.Column('metric_category', sa.String(length=50), nullable=False),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('metric_unit', sa.String(length=20), nullable=True),
        sa.Column('call_id', sa.String(length=255), nullable=True),
        sa.Column('trace_id', sa.String(length=255), nullable=True),
        sa.Column('agent_type', sa.String(length=100), nullable=True),
        sa.Column('workflow_name', sa.String(length=100), nullable=True),
        sa.Column('aggregation_level', sa.String(length=50), nullable=False),
        sa.Column('time_window', sa.String(length=20), nullable=False),
        sa.Column('min_value', sa.Float(), nullable=True),
        sa.Column('max_value', sa.Float(), nullable=True),
        sa.Column('avg_value', sa.Float(), nullable=True),
        sa.Column('percentile_50', sa.Float(), nullable=True),
        sa.Column('percentile_95', sa.Float(), nullable=True),
        sa.Column('percentile_99', sa.Float(), nullable=True),
        sa.Column('threshold_warning', sa.Float(), nullable=True),
        sa.Column('threshold_critical', sa.Float(), nullable=True),
        sa.Column('threshold_breached', sa.Boolean(), nullable=True),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('time_bucket', sa.DateTime(timezone=True), nullable=False),
        sa.Column('additional_metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_perf_metrics_agent_time', 'performance_metrics', ['agent_type', 'time_bucket'])
    op.create_index('ix_perf_metrics_call_time', 'performance_metrics', ['call_id', 'recorded_at'])
    op.create_index('ix_perf_metrics_category_time', 'performance_metrics', ['metric_category', 'time_bucket'])
    op.create_index('ix_perf_metrics_name_time', 'performance_metrics', ['metric_name', 'time_bucket'])
    op.create_index('ix_performance_metrics_agent_type', 'performance_metrics', ['agent_type'])
    op.create_index('ix_performance_metrics_aggregation_level', 'performance_metrics', ['aggregation_level'])
    op.create_index('ix_performance_metrics_call_id', 'performance_metrics', ['call_id'])
    op.create_index('ix_performance_metrics_id', 'performance_metrics', ['id'])
    op.create_index('ix_performance_metrics_metric_category', 'performance_metrics', ['metric_category'])
    op.create_index('ix_performance_metrics_metric_id', 'performance_metrics', ['metric_id'])
    op.create_index('ix_performance_metrics_metric_name', 'performance_metrics', ['metric_name'])
    op.create_index('ix_performance_metrics_recorded_at', 'performance_metrics', ['recorded_at'])
    op.create_index('ix_performance_metrics_threshold_breached', 'performance_metrics', ['threshold_breached'])
    op.create_index('ix_performance_metrics_time_bucket', 'performance_metrics', ['time_bucket'])
    op.create_index('ix_performance_metrics_time_window', 'performance_metrics', ['time_window'])
    op.create_index('ix_performance_metrics_trace_id', 'performance_metrics', ['trace_id'])

    # Create cost_breakdown table
    op.create_table('cost_breakdown',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('cost_id', sa.String(length=255), nullable=False),
        sa.Column('call_id', sa.String(length=255), nullable=True),
        sa.Column('trace_id', sa.String(length=255), nullable=True),
        sa.Column('execution_id', sa.String(length=255), nullable=True),
        sa.Column('cost_type', sa.String(length=50), nullable=False),
        sa.Column('service_provider', sa.String(length=100), nullable=False),
        sa.Column('cost_amount', sa.Float(), nullable=False),
        sa.Column('cost_currency', sa.String(length=3), nullable=True),
        sa.Column('resource_type', sa.String(length=50), nullable=True),
        sa.Column('resource_quantity', sa.Float(), nullable=True),
        sa.Column('unit_cost', sa.Float(), nullable=True),
        sa.Column('agent_type', sa.String(length=100), nullable=True),
        sa.Column('workflow_step', sa.String(length=100), nullable=True),
        sa.Column('tier_used', sa.String(length=50), nullable=True),
        sa.Column('cost_incurred_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('billing_period', sa.String(length=10), nullable=False),
        sa.Column('daily_budget_impact', sa.Float(), nullable=True),
        sa.Column('monthly_budget_impact', sa.Float(), nullable=True),
        sa.Column('budget_category', sa.String(length=50), nullable=True),
        sa.Column('cost_efficiency_score', sa.Float(), nullable=True),
        sa.Column('optimization_opportunity', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_cost_breakdown_agent_period', 'cost_breakdown', ['agent_type', 'billing_period'])
    op.create_index('ix_cost_breakdown_agent_type', 'cost_breakdown', ['agent_type'])
    op.create_index('ix_cost_breakdown_billing_period', 'cost_breakdown', ['billing_period'])
    op.create_index('ix_cost_breakdown_budget_category', 'cost_breakdown', ['budget_category'])
    op.create_index('ix_cost_breakdown_call_id', 'cost_breakdown', ['call_id'])
    op.create_index('ix_cost_breakdown_call_time', 'cost_breakdown', ['call_id', 'cost_incurred_at'])
    op.create_index('ix_cost_breakdown_cost_id', 'cost_breakdown', ['cost_id'])
    op.create_index('ix_cost_breakdown_cost_incurred_at', 'cost_breakdown', ['cost_incurred_at'])
    op.create_index('ix_cost_breakdown_cost_type', 'cost_breakdown', ['cost_type'])
    op.create_index('ix_cost_breakdown_id', 'cost_breakdown', ['id'])
    op.create_index('ix_cost_breakdown_provider_period', 'cost_breakdown', ['service_provider', 'billing_period'])
    op.create_index('ix_cost_breakdown_service_provider', 'cost_breakdown', ['service_provider'])
    op.create_index('ix_cost_breakdown_tier_used', 'cost_breakdown', ['tier_used'])
    op.create_index('ix_cost_breakdown_type_period', 'cost_breakdown', ['cost_type', 'billing_period'])

    # Create alert_history table
    op.create_table('alert_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('alert_id', sa.String(length=255), nullable=False),
        sa.Column('alert_type', sa.String(length=50), nullable=False),
        sa.Column('alert_level', sa.String(length=20), nullable=False),
        sa.Column('alert_title', sa.String(length=255), nullable=False),
        sa.Column('alert_message', sa.Text(), nullable=False),
        sa.Column('call_id', sa.String(length=255), nullable=True),
        sa.Column('trace_id', sa.String(length=255), nullable=True),
        sa.Column('metric_name', sa.String(length=100), nullable=True),
        sa.Column('threshold_value', sa.Float(), nullable=True),
        sa.Column('actual_value', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('acknowledged_by', sa.String(length=100), nullable=True),
        sa.Column('acknowledged_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('notification_channels', sa.JSON(), nullable=True),
        sa.Column('notifications_sent', sa.Integer(), nullable=True),
        sa.Column('last_notification_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rule_name', sa.String(length=100), nullable=True),
        sa.Column('rule_condition', sa.Text(), nullable=True),
        sa.Column('alert_frequency', sa.String(length=20), nullable=True),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('auto_resolved', sa.Boolean(), nullable=True),
        sa.Column('triggered_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_alert_history_alert_id', 'alert_history', ['alert_id'])
    op.create_index('ix_alert_history_alert_level', 'alert_history', ['alert_level'])
    op.create_index('ix_alert_history_alert_type', 'alert_history', ['alert_type'])
    op.create_index('ix_alert_history_call_id', 'alert_history', ['call_id'])
    op.create_index('ix_alert_history_call_time', 'alert_history', ['call_id', 'triggered_at'])
    op.create_index('ix_alert_history_id', 'alert_history', ['id'])
    op.create_index('ix_alert_history_status', 'alert_history', ['status'])
    op.create_index('ix_alert_history_status_time', 'alert_history', ['status', 'triggered_at'])
    op.create_index('ix_alert_history_triggered_at', 'alert_history', ['triggered_at'])
    op.create_index('ix_alert_history_type_level', 'alert_history', ['alert_type', 'alert_level'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('alert_history')
    op.drop_table('cost_breakdown')
    op.drop_table('performance_metrics')
    op.drop_table('agent_executions')
    op.drop_table('workflow_traces')
