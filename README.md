# ML Observability Scaling Code Project

**Author:** Bhavin Kotak (kotak.bhavin@gmail.com | bhavin.kotak@ibm.com)

Implementation of concepts from the research paper: *"Scalable ML Observability: A Multi-Layered Architecture for Production Model Monitoring"*

## Overview

This code project demonstrates a comprehensive ML observability framework with:

- **Drift Detection**: Multiple statistical methods for detecting distribution drift
- **Metrics Collection**: Time-series metrics with aggregation and export
- **Alert Management**: Rule-based alerting with severity levels
- **Monitoring Dashboard**: Dashboard data generation for visualization

## Project Structure

```
ml-observability-scaling/
├── src/
│   ├── __init__.py
│   ├── drift_detection.py    # Drift detection algorithms
│   ├── monitoring.py         # Metrics and alerting
│   └── data_generators.py    # Synthetic data for testing
├── tests/
│   └── test_observability.py # Comprehensive test suite
├── data/                     # Generated data
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Drift Detection

```python
from src.drift_detection import DriftDetector
import pandas as pd
import numpy as np

# Create detector
detector = DriftDetector()

# Reference data (training distribution)
reference_df = pd.DataFrame({
    'feature_1': np.random.normal(100, 15, 1000),
    'feature_2': np.random.normal(50, 10, 1000)
})

# Current data (production)
current_df = pd.DataFrame({
    'feature_1': np.random.normal(110, 20, 500),  # Drifted!
    'feature_2': np.random.normal(50, 10, 500)    # No drift
})

# Detect drift
results = detector.detect_drift(reference_df, current_df)
print(f"Drifted features: {results['summary']['drifted_feature_names']}")

# Get detailed report
report = detector.get_drift_report(reference_df, current_df)
print(report)
```

### Individual Drift Methods

```python
from src.drift_detection import (
    PSICalculator, KLDivergenceCalculator, 
    KSTestDetector, WassersteinDetector
)

reference = np.random.normal(100, 15, 1000)
current = np.random.normal(110, 15, 500)

# PSI
psi = PSICalculator()
result = psi.calculate(reference, current)
print(f"PSI Score: {result.drift_score:.4f}")

# KS Test
ks = KSTestDetector()
result = ks.calculate(reference, current)
print(f"KS p-value: {result.p_value:.4f}")

# Wasserstein
ws = WassersteinDetector()
result = ws.calculate(reference, current)
print(f"Wasserstein Distance: {result.drift_score:.4f}")
```

### Monitoring and Alerting

```python
from src.monitoring import (
    MetricsCollector, AlertManager, 
    MonitoringDashboard, AlertRule, AlertSeverity
)

# Create collector
collector = MetricsCollector()

# Record metrics
collector.record_metric("prediction_latency", 0.05)
collector.record_metric("model_accuracy", 0.95)

# Get summary
summary = collector.get_summary("prediction_latency")
print(f"P95 Latency: {summary['p95']:.4f}s")

# Setup alerting
alert_manager = AlertManager()
alert_manager.add_rule(AlertRule(
    name="high_latency",
    metric_name="prediction_latency",
    condition="gt",
    threshold=0.1,
    severity=AlertSeverity.WARNING,
    message_template="Latency {value:.3f}s exceeds {threshold}s"
))

# Check for alerts
alerts = alert_manager.check_metrics(collector)
for alert in alerts:
    print(f"[{alert.severity.value}] {alert.message}")

# Create dashboard
dashboard = MonitoringDashboard(collector, alert_manager)
overview = dashboard.get_overview()
print(f"System Health: {overview['health_status']}")
```

## Drift Detection Methods

### Population Stability Index (PSI)

Measures how much a distribution has shifted:
- PSI < 0.1: No significant change
- 0.1 ≤ PSI < 0.25: Moderate change
- PSI ≥ 0.25: Significant change

### KL Divergence / Jensen-Shannon

Measures information-theoretic distance between distributions. Uses Jensen-Shannon divergence for symmetric comparison.

### Kolmogorov-Smirnov Test

Statistical test comparing cumulative distributions. Returns p-value for hypothesis testing.

### Wasserstein Distance

Earth Mover's Distance - the minimum "work" to transform one distribution into another.

## Alert Severity Levels

| Level | Description |
|-------|-------------|
| INFO | Informational, no action needed |
| WARNING | Investigate, may need attention |
| CRITICAL | Immediate attention required |
| EMERGENCY | System-wide impact, urgent |

## Running Tests

```bash
# Run all tests
pytest tests/test_observability.py -v

# Run with coverage
pytest tests/test_observability.py -v --cov=src

# Run specific test class
pytest tests/test_observability.py::TestDriftDetector -v
```

## Running the Demos

```bash
# Test drift detection
python src/drift_detection.py

# Test monitoring
python src/monitoring.py

# Test data generators
python src/data_generators.py
```

## Architecture Alignment

| Paper Concept | Implementation |
|--------------|----------------|
| Multi-method drift detection | `PSICalculator`, `KLDivergenceCalculator`, `KSTestDetector`, `WassersteinDetector` |
| Unified drift detector | `DriftDetector` class |
| Metrics collection | `MetricsCollector` with time-series support |
| Alert management | `AlertManager` with rules and severity |
| Dashboard generation | `MonitoringDashboard` class |

## License

MIT License - See research paper for full attribution.
