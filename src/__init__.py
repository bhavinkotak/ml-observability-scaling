# ML Observability Scaling Code Project
# Author: Bhavin Kotak (kotak.bhavin@gmail.com | bhavin.kotak@ibm.com)

from .drift_detection import (
    DriftDetector,
    DriftConfig,
    DriftType,
    PSICalculator,
    KLDivergenceCalculator,
    KSTestDetector,
    WassersteinDetector
)

from .monitoring import (
    MetricsCollector,
    AlertManager,
    Alert,
    AlertSeverity,
    MonitoringDashboard
)

from .data_generators import (
    DriftSimulator,
    MonitoringDataGenerator
)

__version__ = "1.0.0"
__author__ = "Bhavin Kotak"
__email__ = "kotak.bhavin@gmail.com"
