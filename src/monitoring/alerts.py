"""
Alert system and notifications for REAutomation2 monitoring
"""
import asyncio
import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from ..database.core import get_db
from ..database.monitoring_models import AlertHistory
from ..config.settings import settings
from .models import AlertSeverity, AlertType, AlertRule, AlertNotification
from .performance import performance_monitor

logger = logging.getLogger(__name__)


class AlertStatus(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(str, Enum):
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"
    CONSOLE = "console"


@dataclass
class ActiveAlert:
    """Active alert instance"""
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    triggered_at: datetime
    value: Optional[float] = None
    threshold: Optional[float] = None
    call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledgments: List[Dict[str, Any]] = field(default_factory=list)
    last_notification_sent: Optional[datetime] = None


@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    type: str
    config: Dict[str, Any]
    enabled: bool = True


class AlertManager:
    """Centralized alert management system"""

    def __init__(self):
        self.enabled = True
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, ActiveAlert] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.alert_counters: Dict[str, int] = {}

        # Background tasks
        self._monitoring_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        # Rate limiting
        self.rate_limits: Dict[str, datetime] = {}
        self.default_rate_limit = timedelta(minutes=5)

        # Initialize default rules
        self._initialize_default_rules()
        self._initialize_notification_channels()

    async def initialize(self):
        """Initialize alert manager"""
        try:
            if not self.enabled:
                return

            logger.info("Initializing alert manager...")

            # Start background monitoring tasks
            self._monitoring_tasks = [
                asyncio.create_task(self._alert_evaluator()),
                asyncio.create_task(self._notification_processor()),
                asyncio.create_task(self._alert_cleanup())
            ]

            logger.info("Alert manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize alert manager: {e}")
            self.enabled = False

    async def shutdown(self):
        """Shutdown alert manager"""
        try:
            self._shutdown_event.set()

            # Cancel background tasks
            for task in self._monitoring_tasks:
                task.cancel()

            # Wait for tasks to finish
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)

            logger.info("Alert manager shutdown complete")

        except Exception as e:
            logger.error(f"Error during alert manager shutdown: {e}")

    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            # System performance alerts
            AlertRule(
                name="high_cpu_usage",
                condition="system_cpu_percent > 80",
                severity=AlertSeverity.WARNING,
                description="CPU usage is above 80%",
                threshold=80.0,
                evaluation_window_seconds=300,
                cooldown_seconds=600
            ),
            AlertRule(
                name="high_memory_usage",
                condition="system_memory_percent > 85",
                severity=AlertSeverity.WARNING,
                description="Memory usage is above 85%",
                threshold=85.0,
                evaluation_window_seconds=300,
                cooldown_seconds=600
            ),
            AlertRule(
                name="critical_memory_usage",
                condition="system_memory_percent > 95",
                severity=AlertSeverity.CRITICAL,
                description="Memory usage is critically high (>95%)",
                threshold=95.0,
                evaluation_window_seconds=60,
                cooldown_seconds=300
            ),

            # Agent performance alerts
            AlertRule(
                name="high_agent_response_time",
                condition="agent_execution_duration_ms > 10000",
                severity=AlertSeverity.WARNING,
                description="Agent response time is above 10 seconds",
                threshold=10000.0,
                evaluation_window_seconds=300,
                cooldown_seconds=900
            ),
            AlertRule(
                name="agent_failure_rate",
                condition="agent_executions_failure / (agent_executions_success + agent_executions_failure) > 0.1",
                severity=AlertSeverity.WARNING,
                description="Agent failure rate is above 10%",
                threshold=0.1,
                evaluation_window_seconds=600,
                cooldown_seconds=1200
            ),

            # Cost alerts
            AlertRule(
                name="high_llm_cost",
                condition="llm_cost > 10.0",
                severity=AlertSeverity.WARNING,
                description="LLM cost per call is above $10",
                threshold=10.0,
                evaluation_window_seconds=300,
                cooldown_seconds=600
            ),
            AlertRule(
                name="daily_cost_limit",
                condition="daily_total_cost > 500.0",
                severity=AlertSeverity.CRITICAL,
                description="Daily cost limit exceeded ($500)",
                threshold=500.0,
                evaluation_window_seconds=3600,
                cooldown_seconds=3600
            ),

            # Error rate alerts
            AlertRule(
                name="high_error_rate",
                condition="errors_total / total_calls > 0.05",
                severity=AlertSeverity.WARNING,
                description="Error rate is above 5%",
                threshold=0.05,
                evaluation_window_seconds=600,
                cooldown_seconds=900
            ),

            # LangSmith connectivity
            AlertRule(
                name="langsmith_connection_failure",
                condition="langsmith_connection_errors > 5",
                severity=AlertSeverity.WARNING,
                description="Multiple LangSmith connection failures",
                threshold=5.0,
                evaluation_window_seconds=600,
                cooldown_seconds=1800
            )
        ]

        for rule in default_rules:
            self.rules[rule.name] = rule

    def _initialize_notification_channels(self):
        """Initialize notification channels"""
        # Email notification
        if hasattr(settings, 'smtp_host') and settings.smtp_host:
            self.notification_channels["email"] = NotificationChannel(
                type="email",
                config={
                    "smtp_host": getattr(settings, 'smtp_host', ''),
                    "smtp_port": getattr(settings, 'smtp_port', 587),
                    "smtp_username": getattr(settings, 'smtp_username', ''),
                    "smtp_password": getattr(settings, 'smtp_password', ''),
                    "from_email": getattr(settings, 'alert_from_email', ''),
                    "to_emails": getattr(settings, 'alert_to_emails', [])
                }
            )

        # Webhook notification
        if hasattr(settings, 'alert_webhook_url') and settings.alert_webhook_url:
            self.notification_channels["webhook"] = NotificationChannel(
                type="webhook",
                config={
                    "url": settings.alert_webhook_url,
                    "headers": getattr(settings, 'alert_webhook_headers', {}),
                    "timeout": 30
                }
            )

        # Log notification (always available)
        self.notification_channels["log"] = NotificationChannel(
            type="log",
            config={
                "logger_name": "alerts",
                "level": "WARNING"
            }
        )

        # Console notification for development
        if settings.debug:
            self.notification_channels["console"] = NotificationChannel(
                type="console",
                config={}
            )

    # Alert Evaluation
    async def _alert_evaluator(self):
        """Background task to evaluate alert rules"""
        while not self._shutdown_event.is_set():
            try:
                await self._evaluate_all_rules()
                await asyncio.sleep(30)  # Evaluate every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert evaluator: {e}")
                await asyncio.sleep(60)

    async def _evaluate_all_rules(self):
        """Evaluate all alert rules"""
        current_time = datetime.utcnow()

        for rule_name, rule in self.rules.items():
            try:
                # Check if rule is in cooldown
                if rule_name in self.rate_limits:
                    if current_time < self.rate_limits[rule_name]:
                        continue

                # Evaluate rule condition
                should_trigger = await self._evaluate_rule_condition(rule)

                if should_trigger:
                    await self._trigger_alert(rule, current_time)

            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")

    async def _evaluate_rule_condition(self, rule: AlertRule) -> bool:
        """Evaluate a specific alert rule condition"""
        try:
            # Get current performance stats
            stats = performance_monitor.get_current_stats()

            # Simple condition evaluation - in production, use a proper expression evaluator
            condition = rule.condition

            # Replace metric names with actual values
            for metric_name, value in stats.get("call_counters", {}).items():
                condition = condition.replace(metric_name, str(value))

            for metric_name, value in stats.get("avg_response_times", {}).items():
                condition = condition.replace(f"{metric_name}_duration_ms", str(value))

            # Get system metrics from database (recent values)
            await self._get_recent_metric_values(condition)

            # For now, use a simplified evaluation
            # In production, implement a proper expression parser
            return self._simple_condition_evaluation(rule, stats)

        except Exception as e:
            logger.error(f"Error evaluating condition for rule {rule.name}: {e}")
            return False

    def _simple_condition_evaluation(self, rule: AlertRule, stats: Dict[str, Any]) -> bool:
        """Simplified condition evaluation (replace with proper parser in production)"""
        # High CPU usage
        if rule.name == "high_cpu_usage":
            # Would check actual system CPU - for now, simulate
            return False

        # High memory usage
        elif rule.name == "high_memory_usage":
            # Would check actual system memory
            return False

        # Agent response time
        elif rule.name == "high_agent_response_time":
            avg_times = stats.get("avg_response_times", {})
            for operation, avg_time in avg_times.items():
                if "agent" in operation.lower() and avg_time > rule.threshold:
                    return True
            return False

        # Error rate
        elif rule.name == "high_error_rate":
            total_calls = sum(stats.get("call_counters", {}).values())
            total_errors = sum(stats.get("error_counters", {}).values())
            if total_calls > 0:
                error_rate = total_errors / total_calls
                return error_rate > rule.threshold
            return False

        return False

    async def _get_recent_metric_values(self, condition: str):
        """Get recent metric values from database for condition evaluation"""
        # This would query the database for recent metric values
        # Implementation depends on specific metrics needed
        pass

    async def _trigger_alert(self, rule: AlertRule, triggered_at: datetime):
        """Trigger an alert"""
        try:
            # Create unique alert ID
            alert_id = f"{rule.name}_{int(triggered_at.timestamp())}"

            # Check if this alert is already active
            if alert_id in self.active_alerts:
                return

            # Create active alert
            alert = ActiveAlert(
                alert_id=alert_id,
                rule_name=rule.name,
                severity=rule.severity,
                message=rule.description,
                triggered_at=triggered_at,
                threshold=rule.threshold,
                metadata={
                    "rule": rule.dict(),
                    "evaluation_window": rule.evaluation_window_seconds,
                    "cooldown": rule.cooldown_seconds
                }
            )

            # Add to active alerts
            self.active_alerts[alert_id] = alert

            # Increment counter
            self.alert_counters[rule.name] = self.alert_counters.get(rule.name, 0) + 1

            # Set rate limit
            cooldown_until = triggered_at + timedelta(seconds=rule.cooldown_seconds)
            self.rate_limits[rule.name] = cooldown_until

            # Queue for notification
            await self._queue_notification(alert)

            # Store in database
            await self._store_alert_in_database(alert)

            logger.warning(f"Alert triggered: {rule.name} - {rule.description}")

        except Exception as e:
            logger.error(f"Error triggering alert for rule {rule.name}: {e}")

    # Notification Processing
    async def _notification_processor(self):
        """Background task to process alert notifications"""
        while not self._shutdown_event.is_set():
            try:
                await self._process_pending_notifications()
                await asyncio.sleep(10)  # Process every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in notification processor: {e}")
                await asyncio.sleep(30)

    async def _process_pending_notifications(self):
        """Process pending alert notifications"""
        current_time = datetime.utcnow()

        for alert_id, alert in list(self.active_alerts.items()):
            try:
                # Check if we should send notification
                if (alert.status == AlertStatus.ACTIVE and
                    (alert.last_notification_sent is None or
                     current_time - alert.last_notification_sent > timedelta(minutes=15))):

                    await self._send_notifications(alert)
                    alert.last_notification_sent = current_time

            except Exception as e:
                logger.error(f"Error processing notification for alert {alert_id}: {e}")

    async def _queue_notification(self, alert: ActiveAlert):
        """Queue alert for notification"""
        # In this implementation, we process immediately
        # In production, might want to use a proper queue
        await self._send_notifications(alert)

    async def _send_notifications(self, alert: ActiveAlert):
        """Send notifications through all configured channels"""
        notification_data = {
            "alert_id": alert.alert_id,
            "rule_name": alert.rule_name,
            "severity": alert.severity.value,
            "message": alert.message,
            "triggered_at": alert.triggered_at.isoformat(),
            "value": alert.value,
            "threshold": alert.threshold,
            "call_id": alert.call_id,
            "metadata": alert.metadata
        }

        for channel_name, channel in self.notification_channels.items():
            if channel.enabled:
                try:
                    await self._send_notification_to_channel(channel, notification_data)
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel_name}: {e}")

    async def _send_notification_to_channel(self, channel: NotificationChannel, data: Dict[str, Any]):
        """Send notification to a specific channel"""
        if channel.type == "email":
            await self._send_email_notification(channel, data)
        elif channel.type == "webhook":
            await self._send_webhook_notification(channel, data)
        elif channel.type == "log":
            self._send_log_notification(channel, data)
        elif channel.type == "console":
            self._send_console_notification(channel, data)

    async def _send_email_notification(self, channel: NotificationChannel, data: Dict[str, Any]):
        """Send email notification"""
        try:
            config = channel.config

            # Create message
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"REAutomation Alert: {data['severity'].upper()} - {data['rule_name']}"

            # Email body
            body = f"""
            Alert Triggered: {data['rule_name']}
            Severity: {data['severity']}
            Message: {data['message']}
            Triggered At: {data['triggered_at']}

            Details:
            - Alert ID: {data['alert_id']}
            - Value: {data['value']}
            - Threshold: {data['threshold']}
            - Call ID: {data['call_id']}

            Please investigate and acknowledge this alert.
            """

            msg.attach(MIMEText(body, 'plain'))

            # Send email
            with smtplib.SMTP(config['smtp_host'], config['smtp_port']) as server:
                server.starttls()
                if config['smtp_username'] and config['smtp_password']:
                    server.login(config['smtp_username'], config['smtp_password'])
                server.send_message(msg)

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    async def _send_webhook_notification(self, channel: NotificationChannel, data: Dict[str, Any]):
        """Send webhook notification"""
        try:
            import aiohttp

            config = channel.config

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config['url'],
                    json=data,
                    headers=config.get('headers', {}),
                    timeout=aiohttp.ClientTimeout(total=config.get('timeout', 30))
                ) as response:
                    if response.status >= 400:
                        logger.error(f"Webhook notification failed with status {response.status}")

        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")

    def _send_log_notification(self, channel: NotificationChannel, data: Dict[str, Any]):
        """Send log notification"""
        config = channel.config
        logger_name = config.get('logger_name', 'alerts')
        level = config.get('level', 'WARNING')

        alert_logger = logging.getLogger(logger_name)
        log_message = f"ALERT: {data['rule_name']} - {data['message']} (Severity: {data['severity']})"

        if level == 'CRITICAL':
            alert_logger.critical(log_message)
        elif level == 'ERROR':
            alert_logger.error(log_message)
        elif level == 'WARNING':
            alert_logger.warning(log_message)
        else:
            alert_logger.info(log_message)

    def _send_console_notification(self, channel: NotificationChannel, data: Dict[str, Any]):
        """Send console notification"""
        print(f"\n🚨 ALERT: {data['rule_name']}")
        print(f"   Severity: {data['severity']}")
        print(f"   Message: {data['message']}")
        print(f"   Triggered: {data['triggered_at']}")
        print(f"   Alert ID: {data['alert_id']}\n")

    # Alert Management
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str, notes: Optional[str] = None) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id not in self.active_alerts:
                return False

            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED

            acknowledgment = {
                "acknowledged_by": acknowledged_by,
                "acknowledged_at": datetime.utcnow().isoformat(),
                "notes": notes
            }
            alert.acknowledgments.append(acknowledgment)

            # Update in database
            await self._update_alert_in_database(alert)

            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True

        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False

    async def resolve_alert(self, alert_id: str, resolved_by: str, notes: Optional[str] = None) -> bool:
        """Resolve an alert"""
        try:
            if alert_id not in self.active_alerts:
                return False

            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.metadata["resolved_by"] = resolved_by
            alert.metadata["resolved_at"] = datetime.utcnow().isoformat()
            alert.metadata["resolution_notes"] = notes

            # Update in database
            await self._update_alert_in_database(alert)

            # Remove from active alerts
            del self.active_alerts[alert_id]

            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True

        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False

    # Database Operations
    async def _store_alert_in_database(self, alert: ActiveAlert):
        """Store alert in database"""
        try:
            async for db in get_db():
                db_alert = AlertHistory(
                    alert_id=alert.alert_id,
                    rule_name=alert.rule_name,
                    severity=alert.severity.value,
                    alert_type="system",  # Default type
                    message=alert.message,
                    triggered_at=alert.triggered_at,
                    call_id=alert.call_id,
                    metric_value=alert.value,
                    threshold_value=alert.threshold,
                    status=alert.status.value,
                    metadata=alert.metadata
                )

                db.add(db_alert)
                db.commit()
                break

        except Exception as e:
            logger.error(f"Error storing alert in database: {e}")

    async def _update_alert_in_database(self, alert: ActiveAlert):
        """Update alert in database"""
        try:
            async for db in get_db():
                db_alert = db.query(AlertHistory).filter(
                    AlertHistory.alert_id == alert.alert_id
                ).first()

                if db_alert:
                    db_alert.status = alert.status.value
                    db_alert.metadata = alert.metadata
                    db_alert.acknowledged_at = datetime.utcnow() if alert.status == AlertStatus.ACKNOWLEDGED else None
                    db_alert.resolved_at = datetime.utcnow() if alert.status == AlertStatus.RESOLVED else None

                    db.commit()
                break

        except Exception as e:
            logger.error(f"Error updating alert in database: {e}")

    # Cleanup
    async def _alert_cleanup(self):
        """Background task to clean up old alerts"""
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_old_alerts()
                await asyncio.sleep(3600)  # Clean up every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert cleanup: {e}")
                await asyncio.sleep(3600)

    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)

            # Remove old resolved alerts from active alerts
            to_remove = []
            for alert_id, alert in self.active_alerts.items():
                if (alert.status == AlertStatus.RESOLVED and
                    alert.triggered_at < cutoff_time):
                    to_remove.append(alert_id)

            for alert_id in to_remove:
                del self.active_alerts[alert_id]

            # Clean up rate limits
            current_time = datetime.utcnow()
            expired_limits = [
                rule_name for rule_name, limit_time in self.rate_limits.items()
                if current_time > limit_time
            ]

            for rule_name in expired_limits:
                del self.rate_limits[rule_name]

        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")

    # Query Methods
    def get_active_alerts(self) -> List[ActiveAlert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        return {
            "total_active": len(self.active_alerts),
            "by_severity": {
                severity.value: len([
                    alert for alert in self.active_alerts.values()
                    if alert.severity == severity
                ])
                for severity in AlertSeverity
            },
            "by_status": {
                status.value: len([
                    alert for alert in self.active_alerts.values()
                    if alert.status == AlertStatus(status.value)
                ])
                for status in AlertStatus
            },
            "alert_counts": dict(self.alert_counters),
            "active_rate_limits": len(self.rate_limits),
            "notification_channels": len([
                ch for ch in self.notification_channels.values() if ch.enabled
            ])
        }

    # Rule Management
    def add_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.rules[rule.name] = rule

    def remove_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]

    def get_rules(self) -> List[AlertRule]:
        """Get all alert rules"""
        return list(self.rules.values())


# Global alert manager instance
alert_manager = AlertManager()


# Utility functions for triggering alerts
async def trigger_custom_alert(
    name: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.WARNING,
    call_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Trigger a custom alert"""
    try:
        alert_id = f"custom_{name}_{int(datetime.utcnow().timestamp())}"

        alert = ActiveAlert(
            alert_id=alert_id,
            rule_name=f"custom_{name}",
            severity=severity,
            message=message,
            triggered_at=datetime.utcnow(),
            call_id=call_id,
            metadata=metadata or {}
        )

        alert_manager.active_alerts[alert_id] = alert
        await alert_manager._queue_notification(alert)
        await alert_manager._store_alert_in_database(alert)

        return alert_id

    except Exception as e:
        logger.error(f"Error triggering custom alert: {e}")
        return None


async def check_threshold_alert(
    metric_name: str,
    current_value: float,
    threshold: float,
    comparison: str = "greater_than",
    severity: AlertSeverity = AlertSeverity.WARNING,
    call_id: Optional[str] = None
):
    """Check if a metric crosses a threshold and trigger alert if needed"""
    try:
        should_alert = False

        if comparison == "greater_than" and current_value > threshold:
            should_alert = True
        elif comparison == "less_than" and current_value < threshold:
            should_alert = True
        elif comparison == "equals" and current_value == threshold:
            should_alert = True

        if should_alert:
            message = f"{metric_name} is {current_value} (threshold: {threshold})"
            return await trigger_custom_alert(
                f"threshold_{metric_name}",
                message,
                severity,
                call_id,
                {
                    "metric_name": metric_name,
                    "current_value": current_value,
                    "threshold": threshold,
                    "comparison": comparison
                }
            )

    except Exception as e:
        logger.error(f"Error checking threshold alert: {e}")

    return None