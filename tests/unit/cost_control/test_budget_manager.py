"""Tests for budget manager"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.cost_control.budget_manager import BudgetManager, BudgetStatus, BudgetAlert, AlertLevel


class TestBudgetManager:
    """Test BudgetManager functionality"""

    def test_initialize_budget_manager(self, budget_manager):
        """Test budget manager initialization"""
        assert budget_manager.daily_budget == 10.0
        assert budget_manager.weekly_budget == 50.0
        assert budget_manager.monthly_budget == 200.0
        assert budget_manager.cost_per_call_limit == 0.50
        assert budget_manager.emergency_stop_threshold == 0.90

    def test_check_daily_budget_under_threshold(self, budget_manager):
        """Test daily budget check under warning threshold"""
        daily_costs = {"2024-01-15": 3.0}

        status = budget_manager.check_daily_budget(daily_costs, "2024-01-15")

        assert status.is_over_budget is False
        assert status.utilization_percentage == 0.3  # 3.0 / 10.0
        assert status.remaining_budget == 7.0
        assert len(status.alerts) == 0

    def test_check_daily_budget_warning_threshold(self, budget_manager):
        """Test daily budget check at warning threshold"""
        daily_costs = {"2024-01-15": 7.5}

        status = budget_manager.check_daily_budget(daily_costs, "2024-01-15")

        assert status.is_over_budget is False
        assert status.utilization_percentage == 0.75
        assert len(status.alerts) == 1
        assert status.alerts[0].level == AlertLevel.WARNING

    def test_check_daily_budget_critical_threshold(self, budget_manager):
        """Test daily budget check at critical threshold"""
        daily_costs = {"2024-01-15": 8.5}

        status = budget_manager.check_daily_budget(daily_costs, "2024-01-15")

        assert status.is_over_budget is False
        assert status.utilization_percentage == 0.85
        assert len(status.alerts) == 2
        alert_levels = [alert.level for alert in status.alerts]
        assert AlertLevel.WARNING in alert_levels
        assert AlertLevel.CRITICAL in alert_levels

    def test_check_daily_budget_over_budget(self, budget_manager):
        """Test daily budget check when over budget"""
        daily_costs = {"2024-01-15": 12.0}

        status = budget_manager.check_daily_budget(daily_costs, "2024-01-15")

        assert status.is_over_budget is True
        assert status.utilization_percentage == 1.2
        assert status.remaining_budget == -2.0
        assert len(status.alerts) == 3
        alert_levels = [alert.level for alert in status.alerts]
        assert AlertLevel.EMERGENCY in alert_levels

    def test_check_call_cost_limit_under_limit(self, budget_manager):
        """Test call cost check under limit"""
        call_cost = 0.25

        status = budget_manager.check_call_cost_limit(call_cost, "test_call_123")

        assert status.is_over_budget is False
        assert len(status.alerts) == 0

    def test_check_call_cost_limit_over_limit(self, budget_manager):
        """Test call cost check over limit"""
        call_cost = 0.75

        status = budget_manager.check_call_cost_limit(call_cost, "test_call_123")

        assert status.is_over_budget is True
        assert len(status.alerts) == 1
        assert status.alerts[0].level == AlertLevel.EMERGENCY
        assert status.alerts[0].call_id == "test_call_123"

    def test_should_stop_service_normal_usage(self, budget_manager):
        """Test service stop check under normal usage"""
        daily_costs = {"2024-01-15": 5.0}

        should_stop = budget_manager.should_stop_service(daily_costs, "2024-01-15")

        assert should_stop is False

    def test_should_stop_service_emergency_threshold(self, budget_manager):
        """Test service stop check at emergency threshold"""
        daily_costs = {"2024-01-15": 9.5}  # 95% of daily budget

        should_stop = budget_manager.should_stop_service(daily_costs, "2024-01-15")

        assert should_stop is True

    def test_get_budget_summary(self, budget_manager):
        """Test budget summary generation"""
        daily_costs = {"2024-01-15": 7.0, "2024-01-14": 5.0, "2024-01-13": 8.0}
        weekly_costs = {"2024-W03": 35.0, "2024-W02": 28.0}
        monthly_costs = {"2024-01": 125.0}

        summary = budget_manager.get_budget_summary(
            daily_costs=daily_costs,
            weekly_costs=weekly_costs,
            monthly_costs=monthly_costs,
            current_date="2024-01-15"
        )

        assert "daily" in summary
        assert "weekly" in summary
        assert "monthly" in summary
        assert summary["daily"]["utilization_percentage"] == 0.7
        assert summary["weekly"]["utilization_percentage"] == 0.7  # 35.0 / 50.0
        assert summary["monthly"]["utilization_percentage"] == 0.625  # 125.0 / 200.0

    def test_estimate_remaining_capacity_daily(self, budget_manager):
        """Test estimating remaining daily capacity"""
        current_costs = 6.0
        avg_cost_per_call = 0.15

        capacity = budget_manager.estimate_remaining_capacity(
            current_costs, budget_manager.daily_budget, avg_cost_per_call
        )

        # (10.0 - 6.0) / 0.15 = 26.67 -> 26 calls
        assert capacity == 26

    def test_estimate_remaining_capacity_over_budget(self, budget_manager):
        """Test estimating capacity when over budget"""
        current_costs = 12.0
        avg_cost_per_call = 0.15

        capacity = budget_manager.estimate_remaining_capacity(
            current_costs, budget_manager.daily_budget, avg_cost_per_call
        )

        assert capacity == 0

    def test_update_alert_history(self, budget_manager):
        """Test alert history management"""
        alert1 = BudgetAlert(
            level=AlertLevel.WARNING,
            message="Test warning",
            budget_type="daily",
            current_spend=7.0,
            budget_limit=10.0,
            utilization_percentage=70.0,
            timestamp=datetime.now()
        )

        budget_manager.update_alert_history(alert1)
        assert len(budget_manager.alert_history) == 1

        # Add more alerts
        for i in range(15):
            alert = BudgetAlert(
                level=AlertLevel.WARNING,
                message=f"Alert {i}",
                budget_type="daily",
                current_spend=7.0,
                budget_limit=10.0,
                utilization_percentage=70.0,
                timestamp=datetime.now() - timedelta(minutes=i)
            )
            budget_manager.update_alert_history(alert)

        # Should keep only last 10
        assert len(budget_manager.alert_history) == 10

    def test_get_recent_alerts(self, budget_manager):
        """Test getting recent alerts"""
        # Add some alerts
        now = datetime.now()
        for i in range(5):
            alert = BudgetAlert(
                level=AlertLevel.WARNING,
                message=f"Alert {i}",
                budget_type="daily",
                current_spend=7.0,
                budget_limit=10.0,
                utilization_percentage=70.0,
                timestamp=now - timedelta(minutes=i * 10)
            )
            budget_manager.update_alert_history(alert)

        # Get alerts from last hour
        recent_alerts = budget_manager.get_recent_alerts(hours=1)
        assert len(recent_alerts) == 5

        # Get alerts from last 30 minutes
        recent_alerts_30min = budget_manager.get_recent_alerts(hours=0.5)
        assert len(recent_alerts_30min) == 4  # 0, 10, 20, 30 minute old alerts

    def test_calculate_projected_spend(self, budget_manager):
        """Test projected spend calculation"""
        current_costs = {"2024-01-15": 8.0}
        current_hour = 14  # 2 PM

        projected = budget_manager.calculate_projected_spend(current_costs, "2024-01-15", current_hour)

        # 8.0 / 14 * 24 = 13.71 (projected daily spend)
        assert abs(projected - 13.71) < 0.01

    def test_weekly_budget_check(self, budget_manager):
        """Test weekly budget checking"""
        weekly_costs = {"2024-W03": 35.0}

        status = budget_manager.check_weekly_budget(weekly_costs, "2024-W03")

        assert status.utilization_percentage == 0.7  # 35.0 / 50.0
        assert len(status.alerts) == 1
        assert status.alerts[0].level == AlertLevel.WARNING

    def test_monthly_budget_check(self, budget_manager):
        """Test monthly budget checking"""
        monthly_costs = {"2024-01": 180.0}

        status = budget_manager.check_monthly_budget(monthly_costs, "2024-01")

        assert status.utilization_percentage == 0.9  # 180.0 / 200.0
        assert len(status.alerts) == 2
        alert_levels = [alert.level for alert in status.alerts]
        assert AlertLevel.WARNING in alert_levels
        assert AlertLevel.CRITICAL in alert_levels