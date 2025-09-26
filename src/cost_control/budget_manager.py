import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class BudgetAlert:
    """Budget alert notification"""
    level: AlertLevel
    message: str
    budget_type: str  # daily, weekly, monthly
    current_spend: float
    budget_limit: float
    utilization_percentage: float
    timestamp: datetime
    call_id: Optional[str] = None


@dataclass
class BudgetStatus:
    """Current budget status"""
    daily_budget: float
    daily_spent: float
    daily_remaining: float
    daily_utilization: float

    weekly_budget: Optional[float] = None
    weekly_spent: Optional[float] = None
    weekly_remaining: Optional[float] = None
    weekly_utilization: Optional[float] = None

    monthly_budget: Optional[float] = None
    monthly_spent: Optional[float] = None
    monthly_remaining: Optional[float] = None
    monthly_utilization: Optional[float] = None

    over_budget: bool = False
    alerts: List[BudgetAlert] = None

    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []


class BudgetManager:
    """Budget management and enforcement system"""

    def __init__(
        self,
        daily_budget: float = 50.0,
        weekly_budget: Optional[float] = None,
        monthly_budget: Optional[float] = None,
        cost_per_call_limit: float = 0.10,
        emergency_stop_threshold: float = 0.95
    ):
        self.daily_budget = daily_budget
        self.weekly_budget = weekly_budget or (daily_budget * 7)
        self.monthly_budget = monthly_budget or (daily_budget * 30)
        self.cost_per_call_limit = cost_per_call_limit
        self.emergency_stop_threshold = emergency_stop_threshold

        # Tracking
        self.daily_costs: Dict[str, float] = {}  # date -> cost
        self.weekly_costs: Dict[str, float] = {}  # week -> cost
        self.monthly_costs: Dict[str, float] = {}  # month -> cost
        self.call_costs: Dict[str, float] = {}  # call_id -> cost

        # Alert thresholds
        self.alert_thresholds = {
            AlertLevel.WARNING: 0.70,  # 70% of budget
            AlertLevel.CRITICAL: 0.85,  # 85% of budget
            AlertLevel.EMERGENCY: 0.95  # 95% of budget
        }

        # Alert history
        self.recent_alerts: List[BudgetAlert] = []
        self.max_alert_history = 100

        logger.info(f"Budget Manager initialized: Daily=${daily_budget}, Weekly=${self.weekly_budget}, Monthly=${self.monthly_budget}")

    def record_cost(self, call_id: str, cost: float, timestamp: Optional[datetime] = None) -> List[BudgetAlert]:
        """Record a cost and check for budget violations"""
        if timestamp is None:
            timestamp = datetime.now()

        alerts = []

        try:
            # Update cost tracking
            self.call_costs[call_id] = self.call_costs.get(call_id, 0.0) + cost

            # Get date keys
            daily_key = timestamp.strftime("%Y-%m-%d")
            weekly_key = timestamp.strftime("%Y-W%U")
            monthly_key = timestamp.strftime("%Y-%m")

            # Update period costs
            self.daily_costs[daily_key] = self.daily_costs.get(daily_key, 0.0) + cost
            self.weekly_costs[weekly_key] = self.weekly_costs.get(weekly_key, 0.0) + cost
            self.monthly_costs[monthly_key] = self.monthly_costs.get(monthly_key, 0.0) + cost

            # Check for budget violations and generate alerts
            alerts.extend(self._check_budget_violations(call_id, timestamp))

            # Check per-call limit
            call_cost = self.call_costs[call_id]
            if call_cost > self.cost_per_call_limit:
                alert = BudgetAlert(
                    level=AlertLevel.CRITICAL,
                    message=f"Call {call_id} exceeded per-call limit: ${call_cost:.4f} > ${self.cost_per_call_limit}",
                    budget_type="per_call",
                    current_spend=call_cost,
                    budget_limit=self.cost_per_call_limit,
                    utilization_percentage=(call_cost / self.cost_per_call_limit) * 100,
                    timestamp=timestamp,
                    call_id=call_id
                )
                alerts.append(alert)

            # Store alerts
            self.recent_alerts.extend(alerts)
            self._trim_alert_history()

            logger.debug(f"Recorded cost ${cost:.4f} for call {call_id}")

            if alerts:
                for alert in alerts:
                    logger.warning(f"Budget alert: {alert.message}")

            return alerts

        except Exception as e:
            logger.error(f"Error recording cost for call {call_id}: {e}")
            return []

    def _check_budget_violations(self, call_id: str, timestamp: datetime) -> List[BudgetAlert]:
        """Check for budget violations and generate alerts"""
        alerts = []

        try:
            # Get date keys
            daily_key = timestamp.strftime("%Y-%m-%d")
            weekly_key = timestamp.strftime("%Y-W%U")
            monthly_key = timestamp.strftime("%Y-%m")

            # Get current spending
            daily_spent = self.daily_costs.get(daily_key, 0.0)
            weekly_spent = self.weekly_costs.get(weekly_key, 0.0)
            monthly_spent = self.monthly_costs.get(monthly_key, 0.0)

            # Check daily budget
            daily_utilization = daily_spent / self.daily_budget
            alerts.extend(self._generate_threshold_alerts(
                "daily", daily_spent, self.daily_budget, daily_utilization, timestamp, call_id
            ))

            # Check weekly budget
            if self.weekly_budget:
                weekly_utilization = weekly_spent / self.weekly_budget
                alerts.extend(self._generate_threshold_alerts(
                    "weekly", weekly_spent, self.weekly_budget, weekly_utilization, timestamp, call_id
                ))

            # Check monthly budget
            if self.monthly_budget:
                monthly_utilization = monthly_spent / self.monthly_budget
                alerts.extend(self._generate_threshold_alerts(
                    "monthly", monthly_spent, self.monthly_budget, monthly_utilization, timestamp, call_id
                ))

            return alerts

        except Exception as e:
            logger.error(f"Error checking budget violations: {e}")
            return []

    def _generate_threshold_alerts(
        self,
        budget_type: str,
        current_spend: float,
        budget_limit: float,
        utilization: float,
        timestamp: datetime,
        call_id: str
    ) -> List[BudgetAlert]:
        """Generate alerts based on threshold violations"""
        alerts = []

        for level, threshold in self.alert_thresholds.items():
            if utilization >= threshold and not self._recent_alert_exists(budget_type, level):
                alert = BudgetAlert(
                    level=level,
                    message=f"{budget_type.title()} budget {level.value}: ${current_spend:.2f}/{budget_limit:.2f} ({utilization*100:.1f}%)",
                    budget_type=budget_type,
                    current_spend=current_spend,
                    budget_limit=budget_limit,
                    utilization_percentage=utilization * 100,
                    timestamp=timestamp,
                    call_id=call_id
                )
                alerts.append(alert)

        return alerts

    def _recent_alert_exists(self, budget_type: str, level: AlertLevel, minutes: int = 5) -> bool:
        """Check if a recent alert of the same type and level exists"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        for alert in self.recent_alerts:
            if (alert.budget_type == budget_type and
                alert.level == level and
                alert.timestamp >= cutoff_time):
                return True

        return False

    def _trim_alert_history(self):
        """Trim alert history to maximum size"""
        if len(self.recent_alerts) > self.max_alert_history:
            self.recent_alerts = self.recent_alerts[-self.max_alert_history:]

    def should_block_call(self, estimated_cost: float = 0.0, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Determine if a new call should be blocked due to budget constraints"""
        if timestamp is None:
            timestamp = datetime.now()

        try:
            # Get current budget status
            status = self.get_budget_status(timestamp)

            # Check if over daily budget
            if status.daily_utilization >= self.emergency_stop_threshold:
                return {
                    "should_block": True,
                    "reason": "daily_budget_exceeded",
                    "message": f"Daily budget utilization {status.daily_utilization*100:.1f}% exceeds emergency threshold {self.emergency_stop_threshold*100:.1f}%",
                    "current_spend": status.daily_spent,
                    "budget_limit": status.daily_budget
                }

            # Check if estimated cost would exceed daily budget
            projected_daily_spend = status.daily_spent + estimated_cost
            projected_daily_utilization = projected_daily_spend / status.daily_budget

            if projected_daily_utilization >= self.emergency_stop_threshold:
                return {
                    "should_block": True,
                    "reason": "projected_daily_budget_exceeded",
                    "message": f"Estimated call cost ${estimated_cost:.4f} would exceed daily budget",
                    "projected_spend": projected_daily_spend,
                    "budget_limit": status.daily_budget
                }

            # Check per-call limit
            if estimated_cost > self.cost_per_call_limit:
                return {
                    "should_block": True,
                    "reason": "per_call_limit_exceeded",
                    "message": f"Estimated call cost ${estimated_cost:.4f} exceeds per-call limit ${self.cost_per_call_limit}",
                    "estimated_cost": estimated_cost,
                    "call_limit": self.cost_per_call_limit
                }

            return {
                "should_block": False,
                "reason": "within_budget",
                "message": "Call approved within budget limits",
                "estimated_cost": estimated_cost,
                "remaining_daily_budget": status.daily_remaining
            }

        except Exception as e:
            logger.error(f"Error checking call blocking status: {e}")
            return {
                "should_block": True,
                "reason": "error",
                "message": f"Budget check failed: {e}"
            }

    def get_budget_status(self, timestamp: Optional[datetime] = None) -> BudgetStatus:
        """Get current budget status"""
        if timestamp is None:
            timestamp = datetime.now()

        try:
            # Get date keys
            daily_key = timestamp.strftime("%Y-%m-%d")
            weekly_key = timestamp.strftime("%Y-W%U")
            monthly_key = timestamp.strftime("%Y-%m")

            # Calculate daily status
            daily_spent = self.daily_costs.get(daily_key, 0.0)
            daily_remaining = max(0.0, self.daily_budget - daily_spent)
            daily_utilization = daily_spent / self.daily_budget

            # Calculate weekly status
            weekly_spent = self.weekly_costs.get(weekly_key, 0.0) if self.weekly_budget else None
            weekly_remaining = max(0.0, self.weekly_budget - weekly_spent) if self.weekly_budget else None
            weekly_utilization = weekly_spent / self.weekly_budget if self.weekly_budget else None

            # Calculate monthly status
            monthly_spent = self.monthly_costs.get(monthly_key, 0.0) if self.monthly_budget else None
            monthly_remaining = max(0.0, self.monthly_budget - monthly_spent) if self.monthly_budget else None
            monthly_utilization = monthly_spent / self.monthly_budget if self.monthly_budget else None

            # Check if over budget
            over_budget = daily_utilization >= 1.0

            # Get recent alerts
            recent_alerts = [alert for alert in self.recent_alerts
                           if alert.timestamp >= timestamp - timedelta(hours=1)]

            return BudgetStatus(
                daily_budget=self.daily_budget,
                daily_spent=daily_spent,
                daily_remaining=daily_remaining,
                daily_utilization=daily_utilization,
                weekly_budget=self.weekly_budget,
                weekly_spent=weekly_spent,
                weekly_remaining=weekly_remaining,
                weekly_utilization=weekly_utilization,
                monthly_budget=self.monthly_budget,
                monthly_spent=monthly_spent,
                monthly_remaining=monthly_remaining,
                monthly_utilization=monthly_utilization,
                over_budget=over_budget,
                alerts=recent_alerts
            )

        except Exception as e:
            logger.error(f"Error getting budget status: {e}")
            return BudgetStatus(
                daily_budget=self.daily_budget,
                daily_spent=0.0,
                daily_remaining=self.daily_budget,
                daily_utilization=0.0,
                over_budget=False
            )

    def update_budgets(
        self,
        daily_budget: Optional[float] = None,
        weekly_budget: Optional[float] = None,
        monthly_budget: Optional[float] = None,
        cost_per_call_limit: Optional[float] = None
    ):
        """Update budget limits"""
        if daily_budget is not None:
            self.daily_budget = daily_budget
            logger.info(f"Updated daily budget to ${daily_budget}")

        if weekly_budget is not None:
            self.weekly_budget = weekly_budget
            logger.info(f"Updated weekly budget to ${weekly_budget}")

        if monthly_budget is not None:
            self.monthly_budget = monthly_budget
            logger.info(f"Updated monthly budget to ${monthly_budget}")

        if cost_per_call_limit is not None:
            self.cost_per_call_limit = cost_per_call_limit
            logger.info(f"Updated per-call limit to ${cost_per_call_limit}")

    def reset_daily_budget(self, date: Optional[str] = None):
        """Reset daily budget for a specific date"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        if date in self.daily_costs:
            old_cost = self.daily_costs[date]
            del self.daily_costs[date]
            logger.info(f"Reset daily budget for {date}, was ${old_cost:.2f}")

    def get_cost_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get cost trends over the last N days"""
        try:
            end_date = datetime.now()
            trends = {}

            for i in range(days):
                date = end_date - timedelta(days=i)
                date_key = date.strftime("%Y-%m-%d")
                daily_cost = self.daily_costs.get(date_key, 0.0)
                trends[date_key] = daily_cost

            # Calculate statistics
            costs = list(trends.values())
            avg_daily_cost = sum(costs) / len(costs) if costs else 0.0
            max_daily_cost = max(costs) if costs else 0.0
            min_daily_cost = min(costs) if costs else 0.0

            return {
                "daily_trends": trends,
                "average_daily_cost": avg_daily_cost,
                "max_daily_cost": max_daily_cost,
                "min_daily_cost": min_daily_cost,
                "total_period_cost": sum(costs),
                "days_over_budget": sum(1 for cost in costs if cost > self.daily_budget),
                "budget_compliance_rate": (sum(1 for cost in costs if cost <= self.daily_budget) / len(costs)) * 100 if costs else 100.0
            }

        except Exception as e:
            logger.error(f"Error getting cost trends: {e}")
            return {"error": str(e)}