"""
Monitoring Module
==================
Author: Bhavin Kotak (kotak.bhavin@gmail.com | bhavin.kotak@ibm.com)

This module implements monitoring and alerting for ML observability:
- Metrics collection and aggregation
- Alert management with severity levels
- Dashboard data generation

Based on research paper: "Scalable ML Observability: A Multi-Layered Architecture
for Production Model Monitoring"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from collections import defaultdict
import threading
import time


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved: bool = False
    resolved_at: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'name': self.name,
            'severity': self.severity.value,
            'message': self.message,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at,
            'labels': self.labels
        }


@dataclass
class MetricValue:
    """Single metric value with timestamp"""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Rule for triggering alerts"""
    name: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    threshold: float
    severity: AlertSeverity
    message_template: str
    cooldown_seconds: int = 300  # Prevent alert storms
    labels_filter: Optional[Dict[str, str]] = None


class MetricsCollector:
    """
    Metrics collection and aggregation.
    
    Collects various types of metrics:
    - Model performance metrics (accuracy, latency, etc.)
    - Data quality metrics
    - Infrastructure metrics
    
    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_metric("prediction_latency", 0.05)
        >>> collector.record_metric("accuracy", 0.95)
        >>> summary = collector.get_summary("prediction_latency")
    """
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self.metric_types: Dict[str, MetricType] = {}
        self._lock = threading.Lock()
    
    def register_metric(
        self,
        name: str,
        metric_type: MetricType = MetricType.GAUGE,
        description: str = ""
    ) -> None:
        """Register a new metric"""
        self.metric_types[name] = metric_type
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
        timestamp: datetime = None
    ) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels for the metric
            timestamp: Optional timestamp (defaults to now)
        """
        with self._lock:
            metric = MetricValue(
                value=value,
                timestamp=timestamp or datetime.now(),
                labels=labels or {}
            )
            self.metrics[name].append(metric)
            
            # Cleanup old metrics
            self._cleanup_old_metrics(name)
    
    def _cleanup_old_metrics(self, name: str) -> None:
        """Remove metrics older than retention period"""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        self.metrics[name] = [
            m for m in self.metrics[name] 
            if m.timestamp > cutoff
        ]
    
    def get_metric_values(
        self,
        name: str,
        start_time: datetime = None,
        end_time: datetime = None,
        labels_filter: Dict[str, str] = None
    ) -> List[MetricValue]:
        """
        Get metric values with optional filtering.
        
        Args:
            name: Metric name
            start_time: Start of time range
            end_time: End of time range
            labels_filter: Filter by label values
            
        Returns:
            List of MetricValue objects
        """
        if name not in self.metrics:
            return []
        
        values = self.metrics[name]
        
        if start_time:
            values = [v for v in values if v.timestamp >= start_time]
        
        if end_time:
            values = [v for v in values if v.timestamp <= end_time]
        
        if labels_filter:
            values = [
                v for v in values
                if all(v.labels.get(k) == val for k, val in labels_filter.items())
            ]
        
        return values
    
    def get_summary(
        self,
        name: str,
        window_minutes: int = 60
    ) -> Dict[str, float]:
        """
        Get summary statistics for a metric.
        
        Args:
            name: Metric name
            window_minutes: Time window for summary
            
        Returns:
            Dict with count, mean, min, max, std, p50, p95, p99
        """
        start_time = datetime.now() - timedelta(minutes=window_minutes)
        values = self.get_metric_values(name, start_time=start_time)
        
        if not values:
            return {'count': 0}
        
        data = np.array([v.value for v in values])
        
        return {
            'count': len(data),
            'mean': float(np.mean(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'std': float(np.std(data)),
            'p50': float(np.percentile(data, 50)),
            'p95': float(np.percentile(data, 95)),
            'p99': float(np.percentile(data, 99))
        }
    
    def get_current_value(self, name: str) -> Optional[float]:
        """Get the most recent value for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return None
        return self.metrics[name][-1].value
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics"""
        return {name: self.get_summary(name) for name in self.metrics}
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        for name, values in self.metrics.items():
            if not values:
                continue
            
            current = values[-1]
            metric_type = self.metric_types.get(name, MetricType.GAUGE)
            
            # Type line
            lines.append(f"# TYPE {name} {metric_type.value}")
            
            # Value line
            labels_str = ""
            if current.labels:
                label_parts = [f'{k}="{v}"' for k, v in current.labels.items()]
                labels_str = "{" + ",".join(label_parts) + "}"
            
            lines.append(f"{name}{labels_str} {current.value}")
        
        return "\n".join(lines)


class AlertManager:
    """
    Alert management system.
    
    Manages alert rules, triggers, and notifications.
    
    Example:
        >>> manager = AlertManager()
        >>> manager.add_rule(AlertRule(
        ...     name="high_latency",
        ...     metric_name="prediction_latency",
        ...     condition="gt",
        ...     threshold=1.0,
        ...     severity=AlertSeverity.WARNING,
        ...     message_template="Latency {value}s exceeds threshold {threshold}s"
        ... ))
        >>> alerts = manager.check_metrics(metrics_collector)
    """
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self._last_fired: Dict[str, datetime] = {}
        self._alert_counter = 0
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule"""
        self.rules.append(rule)
    
    def remove_rule(self, name: str) -> None:
        """Remove an alert rule by name"""
        self.rules = [r for r in self.rules if r.name != name]
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        self._alert_counter += 1
        return f"alert_{self._alert_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def _check_condition(
        self,
        value: float,
        condition: str,
        threshold: float
    ) -> bool:
        """Check if condition is met"""
        conditions = {
            'gt': value > threshold,
            'lt': value < threshold,
            'gte': value >= threshold,
            'lte': value <= threshold,
            'eq': value == threshold
        }
        return conditions.get(condition, False)
    
    def _should_fire(self, rule: AlertRule) -> bool:
        """Check if rule should fire (respecting cooldown)"""
        if rule.name not in self._last_fired:
            return True
        
        elapsed = (datetime.now() - self._last_fired[rule.name]).total_seconds()
        return elapsed >= rule.cooldown_seconds
    
    def check_metrics(
        self,
        collector: MetricsCollector
    ) -> List[Alert]:
        """
        Check all rules against current metrics.
        
        Args:
            collector: MetricsCollector with current values
            
        Returns:
            List of triggered alerts
        """
        triggered = []
        
        for rule in self.rules:
            if not self._should_fire(rule):
                continue
            
            # Get metric values
            values = collector.get_metric_values(
                rule.metric_name,
                start_time=datetime.now() - timedelta(minutes=5),
                labels_filter=rule.labels_filter
            )
            
            if not values:
                continue
            
            current_value = values[-1].value
            
            if self._check_condition(current_value, rule.condition, rule.threshold):
                alert = Alert(
                    alert_id=self._generate_alert_id(),
                    name=rule.name,
                    severity=rule.severity,
                    message=rule.message_template.format(
                        value=current_value,
                        threshold=rule.threshold
                    ),
                    metric_name=rule.metric_name,
                    metric_value=current_value,
                    threshold=rule.threshold,
                    labels=values[-1].labels
                )
                
                triggered.append(alert)
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
                self._last_fired[rule.name] = datetime.now()
        
        return triggered
    
    def resolve_alert(self, alert_id: str) -> Optional[Alert]:
        """Mark an alert as resolved"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            alert.resolved = True
            alert.resolved_at = datetime.now().isoformat()
            return alert
        return None
    
    def get_active_alerts(
        self,
        severity: AlertSeverity = None
    ) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert state"""
        active = list(self.active_alerts.values())
        
        by_severity = defaultdict(int)
        for alert in active:
            by_severity[alert.severity.value] += 1
        
        return {
            'total_active': len(active),
            'by_severity': dict(by_severity),
            'total_historical': len(self.alert_history),
            'rules_count': len(self.rules)
        }


class MonitoringDashboard:
    """
    Dashboard data generator for monitoring visualization.
    
    Aggregates metrics and alerts into dashboard-ready formats.
    """
    
    def __init__(
        self,
        collector: MetricsCollector,
        alert_manager: AlertManager
    ):
        self.collector = collector
        self.alert_manager = alert_manager
    
    def get_overview(self) -> Dict[str, Any]:
        """Get overall system health overview"""
        alerts = self.alert_manager.get_active_alerts()
        
        # Determine health status
        if any(a.severity == AlertSeverity.EMERGENCY for a in alerts):
            health = "critical"
        elif any(a.severity == AlertSeverity.CRITICAL for a in alerts):
            health = "degraded"
        elif any(a.severity == AlertSeverity.WARNING for a in alerts):
            health = "warning"
        else:
            health = "healthy"
        
        return {
            'health_status': health,
            'active_alerts': len(alerts),
            'alert_summary': self.alert_manager.get_alert_summary(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metrics_dashboard(
        self,
        metric_names: List[str] = None,
        window_minutes: int = 60
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics data for dashboard.
        
        Args:
            metric_names: Specific metrics to include
            window_minutes: Time window
            
        Returns:
            Dict with metric summaries and trends
        """
        if metric_names is None:
            metric_names = list(self.collector.metrics.keys())
        
        dashboard_data = {}
        
        for name in metric_names:
            summary = self.collector.get_summary(name, window_minutes)
            
            # Get trend data
            values = self.collector.get_metric_values(
                name,
                start_time=datetime.now() - timedelta(minutes=window_minutes)
            )
            
            trend = []
            for v in values[-100:]:  # Limit for performance
                trend.append({
                    'timestamp': v.timestamp.isoformat(),
                    'value': v.value
                })
            
            dashboard_data[name] = {
                'summary': summary,
                'trend': trend,
                'current': summary.get('mean', 0)
            }
        
        return dashboard_data
    
    def get_alerts_panel(self) -> Dict[str, Any]:
        """Get alert data for dashboard panel"""
        active = self.alert_manager.get_active_alerts()
        recent = self.alert_manager.alert_history[-20:]  # Last 20
        
        return {
            'active_alerts': [a.to_dict() for a in active],
            'recent_alerts': [a.to_dict() for a in recent],
            'summary': self.alert_manager.get_alert_summary()
        }
    
    def export_dashboard_json(self) -> str:
        """Export complete dashboard state as JSON"""
        data = {
            'overview': self.get_overview(),
            'metrics': self.get_metrics_dashboard(),
            'alerts': self.get_alerts_panel(),
            'generated_at': datetime.now().isoformat()
        }
        return json.dumps(data, indent=2)


# Main execution for testing
if __name__ == "__main__":
    print("Testing Monitoring Module...")
    
    # Create collector
    collector = MetricsCollector()
    
    # Record some metrics
    for i in range(100):
        collector.record_metric("prediction_latency", np.random.exponential(0.05))
        collector.record_metric("accuracy", 0.95 + np.random.normal(0, 0.02))
        collector.record_metric("requests_per_second", np.random.uniform(80, 120))
    
    # Get summary
    print("\nMetric Summaries:")
    for metric in ["prediction_latency", "accuracy", "requests_per_second"]:
        summary = collector.get_summary(metric)
        print(f"  {metric}: mean={summary['mean']:.4f}, p95={summary['p95']:.4f}")
    
    # Create alert manager
    alert_manager = AlertManager()
    alert_manager.add_rule(AlertRule(
        name="high_latency",
        metric_name="prediction_latency",
        condition="gt",
        threshold=0.1,
        severity=AlertSeverity.WARNING,
        message_template="Latency {value:.3f}s exceeds threshold {threshold}s"
    ))
    alert_manager.add_rule(AlertRule(
        name="low_accuracy",
        metric_name="accuracy",
        condition="lt",
        threshold=0.90,
        severity=AlertSeverity.CRITICAL,
        message_template="Accuracy {value:.3f} below threshold {threshold}"
    ))
    
    # Check for alerts
    alerts = alert_manager.check_metrics(collector)
    print(f"\nTriggered alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  [{alert.severity.value}] {alert.name}: {alert.message}")
    
    # Create dashboard
    dashboard = MonitoringDashboard(collector, alert_manager)
    overview = dashboard.get_overview()
    print(f"\nDashboard Overview:")
    print(f"  Health Status: {overview['health_status']}")
    print(f"  Active Alerts: {overview['active_alerts']}")
    
    print("\nMonitoring module working correctly!")
