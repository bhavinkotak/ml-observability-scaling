"""
Tests for ML Observability Scaling
===================================
Author: Bhavin Kotak (kotak.bhavin@gmail.com | bhavin.kotak@ibm.com)

Comprehensive test suite for drift detection and monitoring.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from drift_detection import (
    DriftConfig, DriftResult, DriftType,
    PSICalculator, KLDivergenceCalculator, KSTestDetector,
    WassersteinDetector, DriftDetector
)
from monitoring import (
    MetricsCollector, AlertManager, MonitoringDashboard,
    Alert, AlertSeverity, AlertRule
)
from data_generators import DriftSimulator, MonitoringDataGenerator


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def drift_config():
    return DriftConfig()


@pytest.fixture
def reference_data():
    """Generate reference distribution"""
    np.random.seed(42)
    return np.random.normal(100, 15, 1000)


@pytest.fixture
def no_drift_data():
    """Data similar to reference"""
    np.random.seed(123)
    return np.random.normal(100, 15, 500)


@pytest.fixture
def drifted_data():
    """Data with significant drift"""
    np.random.seed(456)
    return np.random.normal(120, 25, 500)  # Mean and std shifted


@pytest.fixture
def reference_df():
    """Reference DataFrame"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': np.random.normal(100, 15, 1000),
        'feature_2': np.random.normal(50, 10, 1000),
        'feature_3': np.random.uniform(0, 100, 1000)
    })


@pytest.fixture
def metrics_collector():
    return MetricsCollector()


@pytest.fixture
def alert_manager():
    return AlertManager()


# ============================================================================
# PSI Calculator Tests
# ============================================================================

class TestPSICalculator:
    """Test PSI calculations"""
    
    def test_no_drift_low_psi(self, reference_data, no_drift_data):
        calc = PSICalculator()
        result = calc.calculate(reference_data, no_drift_data)
        
        assert result.drift_score < 0.1
        assert result.drift_detected == False
        assert result.severity == "none"
    
    def test_drift_high_psi(self, reference_data, drifted_data):
        calc = PSICalculator()
        result = calc.calculate(reference_data, drifted_data)
        
        assert result.drift_score > 0.1
        assert result.drift_detected == True
        assert result.severity in ["warning", "critical"]
    
    def test_identical_distributions_zero_psi(self, reference_data):
        calc = PSICalculator()
        result = calc.calculate(reference_data, reference_data)
        
        assert result.drift_score < 0.01
    
    def test_custom_bins(self, reference_data, drifted_data):
        calc = PSICalculator()
        
        result_10 = calc.calculate(reference_data, drifted_data, n_bins=10)
        result_20 = calc.calculate(reference_data, drifted_data, n_bins=20)
        
        # Both should detect drift
        assert result_10.drift_detected
        assert result_20.drift_detected
    
    def test_result_contains_details(self, reference_data, no_drift_data):
        calc = PSICalculator()
        result = calc.calculate(reference_data, no_drift_data)
        
        assert 'reference_samples' in result.details
        assert 'current_samples' in result.details
        assert result.details['reference_samples'] == len(reference_data)


# ============================================================================
# KL Divergence Tests
# ============================================================================

class TestKLDivergenceCalculator:
    """Test KL Divergence calculations"""
    
    def test_no_drift_low_divergence(self, reference_data, no_drift_data):
        calc = KLDivergenceCalculator()
        result = calc.calculate(reference_data, no_drift_data)
        
        assert result.drift_score < 0.1
    
    def test_drift_high_divergence(self, reference_data, drifted_data):
        calc = KLDivergenceCalculator()
        result = calc.calculate(reference_data, drifted_data)
        
        assert result.drift_score > 0.05
    
    def test_symmetric_vs_asymmetric(self, reference_data, drifted_data):
        calc = KLDivergenceCalculator()
        
        sym = calc.calculate(reference_data, drifted_data, symmetric=True)
        asym = calc.calculate(reference_data, drifted_data, symmetric=False)
        
        assert sym.details['method'] == 'jensen_shannon'
        assert asym.details['method'] == 'kl_divergence'


# ============================================================================
# KS Test Tests
# ============================================================================

class TestKSTestDetector:
    """Test Kolmogorov-Smirnov tests"""
    
    def test_no_drift_high_pvalue(self, reference_data, no_drift_data):
        calc = KSTestDetector()
        result = calc.calculate(reference_data, no_drift_data)
        
        assert result.p_value > 0.05
        assert result.drift_detected == False
    
    def test_drift_low_pvalue(self, reference_data, drifted_data):
        calc = KSTestDetector()
        result = calc.calculate(reference_data, drifted_data)
        
        assert result.p_value < 0.05
        assert result.drift_detected == True
    
    def test_ks_statistic_range(self, reference_data, drifted_data):
        calc = KSTestDetector()
        result = calc.calculate(reference_data, drifted_data)
        
        # KS statistic should be between 0 and 1
        assert 0 <= result.drift_score <= 1


# ============================================================================
# Wasserstein Distance Tests
# ============================================================================

class TestWassersteinDetector:
    """Test Wasserstein distance calculations"""
    
    def test_no_drift_small_distance(self, reference_data, no_drift_data):
        calc = WassersteinDetector()
        result = calc.calculate(reference_data, no_drift_data)
        
        assert result.drift_score < 0.2
        assert result.drift_detected == False
    
    def test_drift_large_distance(self, reference_data, drifted_data):
        calc = WassersteinDetector()
        result = calc.calculate(reference_data, drifted_data)
        
        assert result.drift_score > 0.1
    
    def test_normalized_vs_raw(self, reference_data, drifted_data):
        calc = WassersteinDetector()
        
        norm = calc.calculate(reference_data, drifted_data, normalize=True)
        raw = calc.calculate(reference_data, drifted_data, normalize=False)
        
        assert 'raw_distance' in norm.details
        assert norm.details['normalized'] == True


# ============================================================================
# Unified Drift Detector Tests
# ============================================================================

class TestDriftDetector:
    """Test unified drift detector"""
    
    def test_detect_feature_drift(self, reference_data, drifted_data):
        detector = DriftDetector()
        results = detector.detect_feature_drift(
            reference_data, drifted_data,
            feature_name="test_feature"
        )
        
        assert 'psi' in results
        assert 'kl' in results
        assert 'ks' in results
        assert 'wasserstein' in results
        
        # All methods should detect drift
        assert all(r.drift_detected for r in results.values())
    
    def test_detect_dataset_drift(self, reference_df):
        detector = DriftDetector()
        
        # Create drifted copy
        drifted_df = reference_df.copy()
        drifted_df['feature_1'] += 20  # Add drift
        
        results = detector.detect_drift(reference_df, drifted_df)
        
        assert 'feature_results' in results
        assert 'summary' in results
        assert results['summary']['total_features'] == 3
        assert 'feature_1' in results['summary']['drifted_feature_names']
    
    def test_get_drift_report(self, reference_df):
        detector = DriftDetector()
        
        drifted_df = reference_df.copy()
        drifted_df['feature_1'] += 20
        
        report = detector.get_drift_report(reference_df, drifted_df)
        
        assert isinstance(report, pd.DataFrame)
        assert 'feature' in report.columns
        assert 'any_drift' in report.columns
        assert len(report) == 3
    
    def test_monitor_drift_over_time(self, reference_df):
        detector = DriftDetector()
        sim = DriftSimulator(random_seed=42)
        
        batches = sim.generate_drift_sequence(
            reference_df, n_batches=5, drift_start_batch=3
        )
        
        trend = detector.monitor_drift_over_time(reference_df, batches)
        
        assert isinstance(trend, pd.DataFrame)
        assert len(trend) == 5
        assert 'batch_index' in trend.columns


# ============================================================================
# Metrics Collector Tests
# ============================================================================

class TestMetricsCollector:
    """Test metrics collection"""
    
    def test_record_and_retrieve_metric(self, metrics_collector):
        metrics_collector.record_metric("latency", 0.05)
        metrics_collector.record_metric("latency", 0.06)
        
        values = metrics_collector.get_metric_values("latency")
        assert len(values) == 2
        assert values[-1].value == 0.06
    
    def test_metric_summary(self, metrics_collector):
        for i in range(100):
            metrics_collector.record_metric("latency", np.random.exponential(0.05))
        
        summary = metrics_collector.get_summary("latency")
        
        assert 'count' in summary
        assert 'mean' in summary
        assert 'p95' in summary
        assert summary['count'] == 100
    
    def test_metric_with_labels(self, metrics_collector):
        metrics_collector.record_metric("latency", 0.05, labels={'endpoint': '/predict'})
        metrics_collector.record_metric("latency", 0.10, labels={'endpoint': '/batch'})
        
        values = metrics_collector.get_metric_values(
            "latency", 
            labels_filter={'endpoint': '/predict'}
        )
        
        assert len(values) == 1
        assert values[0].value == 0.05
    
    def test_current_value(self, metrics_collector):
        metrics_collector.record_metric("accuracy", 0.95)
        metrics_collector.record_metric("accuracy", 0.96)
        
        assert metrics_collector.get_current_value("accuracy") == 0.96
    
    def test_prometheus_export(self, metrics_collector):
        metrics_collector.record_metric("latency", 0.05)
        
        output = metrics_collector.export_prometheus_format()
        
        assert "latency" in output
        assert "0.05" in output


# ============================================================================
# Alert Manager Tests
# ============================================================================

class TestAlertManager:
    """Test alert management"""
    
    def test_add_and_check_rule(self, metrics_collector, alert_manager):
        # Add metric
        metrics_collector.record_metric("latency", 1.5)  # High latency
        
        # Add rule
        alert_manager.add_rule(AlertRule(
            name="high_latency",
            metric_name="latency",
            condition="gt",
            threshold=1.0,
            severity=AlertSeverity.WARNING,
            message_template="Latency {value}s exceeds {threshold}s"
        ))
        
        alerts = alert_manager.check_metrics(metrics_collector)
        
        assert len(alerts) == 1
        assert alerts[0].name == "high_latency"
        assert alerts[0].severity == AlertSeverity.WARNING
    
    def test_no_alert_when_below_threshold(self, metrics_collector, alert_manager):
        metrics_collector.record_metric("latency", 0.5)  # Normal latency
        
        alert_manager.add_rule(AlertRule(
            name="high_latency",
            metric_name="latency",
            condition="gt",
            threshold=1.0,
            severity=AlertSeverity.WARNING,
            message_template="Latency {value}s exceeds {threshold}s"
        ))
        
        alerts = alert_manager.check_metrics(metrics_collector)
        
        assert len(alerts) == 0
    
    def test_resolve_alert(self, metrics_collector, alert_manager):
        metrics_collector.record_metric("latency", 1.5)
        
        alert_manager.add_rule(AlertRule(
            name="high_latency",
            metric_name="latency",
            condition="gt",
            threshold=1.0,
            severity=AlertSeverity.WARNING,
            message_template="Test"
        ))
        
        alerts = alert_manager.check_metrics(metrics_collector)
        alert_id = alerts[0].alert_id
        
        assert len(alert_manager.get_active_alerts()) == 1
        
        alert_manager.resolve_alert(alert_id)
        
        assert len(alert_manager.get_active_alerts()) == 0
    
    def test_alert_summary(self, metrics_collector, alert_manager):
        metrics_collector.record_metric("error_rate", 0.15)
        
        alert_manager.add_rule(AlertRule(
            name="high_errors",
            metric_name="error_rate",
            condition="gt",
            threshold=0.1,
            severity=AlertSeverity.CRITICAL,
            message_template="Error rate {value}"
        ))
        
        alert_manager.check_metrics(metrics_collector)
        
        summary = alert_manager.get_alert_summary()
        
        assert summary['total_active'] == 1
        assert summary['by_severity']['critical'] == 1


# ============================================================================
# Monitoring Dashboard Tests
# ============================================================================

class TestMonitoringDashboard:
    """Test monitoring dashboard"""
    
    def test_overview_healthy(self, metrics_collector, alert_manager):
        dashboard = MonitoringDashboard(metrics_collector, alert_manager)
        
        overview = dashboard.get_overview()
        
        assert overview['health_status'] == 'healthy'
        assert overview['active_alerts'] == 0
    
    def test_overview_with_alerts(self, metrics_collector, alert_manager):
        metrics_collector.record_metric("latency", 5.0)
        
        alert_manager.add_rule(AlertRule(
            name="critical_latency",
            metric_name="latency",
            condition="gt",
            threshold=2.0,
            severity=AlertSeverity.CRITICAL,
            message_template="Critical latency"
        ))
        alert_manager.check_metrics(metrics_collector)
        
        dashboard = MonitoringDashboard(metrics_collector, alert_manager)
        overview = dashboard.get_overview()
        
        assert overview['health_status'] == 'degraded'
        assert overview['active_alerts'] == 1
    
    def test_metrics_dashboard(self, metrics_collector, alert_manager):
        for i in range(10):
            metrics_collector.record_metric("latency", 0.05 + i * 0.01)
        
        dashboard = MonitoringDashboard(metrics_collector, alert_manager)
        data = dashboard.get_metrics_dashboard(['latency'])
        
        assert 'latency' in data
        assert 'summary' in data['latency']
        assert 'trend' in data['latency']
    
    def test_export_json(self, metrics_collector, alert_manager):
        metrics_collector.record_metric("test", 1.0)
        
        dashboard = MonitoringDashboard(metrics_collector, alert_manager)
        json_output = dashboard.export_dashboard_json()
        
        assert isinstance(json_output, str)
        assert 'overview' in json_output


# ============================================================================
# Data Generator Tests
# ============================================================================

class TestDriftSimulator:
    """Test drift simulation"""
    
    def test_generate_reference(self):
        sim = DriftSimulator(random_seed=42)
        df = sim.generate_reference_distribution(n_samples=100, n_features=5)
        
        assert len(df) == 100
        assert len(df.columns) == 5
    
    def test_gradual_drift(self):
        sim = DriftSimulator(random_seed=42)
        ref = sim.generate_reference_distribution(n_samples=100, n_features=3)
        drifted = sim.apply_gradual_drift(ref, drift_magnitude=1.0)
        
        # Mean should be shifted
        for col in ref.columns:
            assert drifted[col].mean() > ref[col].mean()
    
    def test_sudden_drift(self):
        sim = DriftSimulator(random_seed=42)
        ref = sim.generate_reference_distribution(n_samples=100, n_features=3)
        drifted = sim.apply_sudden_drift(ref, drift_magnitude=2.0)
        
        # Significant mean shift
        for col in ref.columns:
            shift = abs(drifted[col].mean() - ref[col].mean())
            assert shift > ref[col].std()
    
    def test_variance_drift(self):
        sim = DriftSimulator(random_seed=42)
        ref = sim.generate_reference_distribution(n_samples=1000, n_features=3)
        drifted = sim.apply_variance_drift(ref, variance_multiplier=2.0)
        
        # Variance should approximately double
        for col in ref.columns:
            ratio = drifted[col].std() / ref[col].std()
            assert 1.3 < ratio < 1.6  # sqrt(2) ≈ 1.41
    
    def test_drift_sequence(self):
        sim = DriftSimulator(random_seed=42)
        ref = sim.generate_reference_distribution(n_samples=100, n_features=3)
        batches = sim.generate_drift_sequence(ref, n_batches=10, drift_start_batch=5)
        
        assert len(batches) == 10
        for batch in batches:
            assert len(batch) == len(ref)


class TestMonitoringDataGenerator:
    """Test monitoring data generation"""
    
    def test_latency_data(self):
        gen = MonitoringDataGenerator(random_seed=42)
        df = gen.generate_latency_data(n_samples=100)
        
        assert len(df) == 100
        assert 'latency_ms' in df.columns
        assert 'timestamp' in df.columns
    
    def test_model_metrics(self):
        gen = MonitoringDataGenerator(random_seed=42)
        df = gen.generate_model_metrics(n_samples=100)
        
        assert len(df) == 100
        assert 'accuracy' in df.columns
        assert 'precision' in df.columns
        assert 'recall' in df.columns
    
    def test_model_degradation(self):
        gen = MonitoringDataGenerator(random_seed=42)
        df = gen.generate_model_metrics(n_samples=100, degradation_point=50)
        
        # Accuracy should drop after degradation point
        early_acc = df['accuracy'].iloc[:50].mean()
        late_acc = df['accuracy'].iloc[50:].mean()
        
        assert late_acc < early_acc


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests"""
    
    def test_full_drift_detection_pipeline(self):
        # Setup
        sim = DriftSimulator(random_seed=42)
        detector = DriftDetector()
        
        # Generate data
        reference = sim.generate_reference_distribution(n_samples=1000, n_features=5)
        batches = sim.generate_drift_sequence(reference, n_batches=10, drift_start_batch=5)
        
        # Monitor drift
        trend = detector.monitor_drift_over_time(reference, batches)
        
        # Verify drift detected after start point
        assert trend.loc[4, 'drift_percentage'] < trend.loc[9, 'drift_percentage']
    
    def test_full_monitoring_pipeline(self):
        # Setup
        mon_gen = MonitoringDataGenerator(random_seed=42)
        collector = MetricsCollector()
        alert_manager = AlertManager()
        
        # Generate and record metrics
        latency_data = mon_gen.generate_latency_data(n_samples=100)
        for _, row in latency_data.iterrows():
            collector.record_metric("latency", row['latency_ms'] / 1000)
        
        # Add alert rules
        alert_manager.add_rule(AlertRule(
            name="high_latency",
            metric_name="latency",
            condition="gt",
            threshold=0.2,
            severity=AlertSeverity.WARNING,
            message_template="Latency {value:.3f}s"
        ))
        
        # Check alerts
        alerts = alert_manager.check_metrics(collector)
        
        # Create dashboard
        dashboard = MonitoringDashboard(collector, alert_manager)
        overview = dashboard.get_overview()
        
        assert 'health_status' in overview


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
