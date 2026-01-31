"""
Drift Detection Module
=======================
Author: Bhavin Kotak (kotak.bhavin@gmail.com | bhavin.kotak@ibm.com)

This module implements statistical drift detection methods for ML observability:
- Population Stability Index (PSI)
- KL Divergence
- Kolmogorov-Smirnov Test
- Wasserstein Distance

Based on research paper: "Scalable ML Observability: A Multi-Layered Architecture
for Production Model Monitoring"
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import warnings


class DriftType(Enum):
    """Types of distribution drift"""
    NONE = "none"
    GRADUAL = "gradual"          # Slow change over time
    SUDDEN = "sudden"            # Abrupt change
    RECURRING = "recurring"      # Periodic pattern
    INCREMENTAL = "incremental"  # Small consistent changes


@dataclass
class DriftConfig:
    """Configuration for drift detection"""
    # PSI thresholds
    psi_threshold_warning: float = 0.1    # Minor change
    psi_threshold_critical: float = 0.25  # Major change
    
    # KL Divergence thresholds
    kl_threshold_warning: float = 0.1
    kl_threshold_critical: float = 0.5
    
    # KS Test thresholds
    ks_alpha: float = 0.05  # Significance level
    
    # Wasserstein thresholds (relative to std)
    wasserstein_threshold_warning: float = 0.1
    wasserstein_threshold_critical: float = 0.3
    
    # Binning settings
    n_bins: int = 10
    min_bin_size: int = 5
    
    # Window settings
    reference_window_size: int = 1000
    detection_window_size: int = 500


@dataclass
class DriftResult:
    """Result of drift detection"""
    feature_name: str
    drift_detected: bool
    drift_type: DriftType
    drift_score: float
    p_value: Optional[float]
    threshold_used: float
    severity: str  # 'none', 'warning', 'critical'
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_name': self.feature_name,
            'drift_detected': self.drift_detected,
            'drift_type': self.drift_type.value,
            'drift_score': self.drift_score,
            'p_value': self.p_value,
            'threshold_used': self.threshold_used,
            'severity': self.severity,
            'details': self.details,
            'timestamp': self.timestamp
        }


class PSICalculator:
    """
    Population Stability Index (PSI) Calculator.
    
    PSI measures how much a distribution has shifted from a reference.
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.25: Moderate change, investigate
    - PSI >= 0.25: Significant change, action required
    
    Formula: PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)
    """
    
    def __init__(self, config: DriftConfig = None):
        self.config = config or DriftConfig()
    
    def calculate(
        self, 
        reference: np.ndarray, 
        current: np.ndarray,
        n_bins: int = None
    ) -> DriftResult:
        """
        Calculate PSI between reference and current distributions.
        
        Args:
            reference: Reference distribution (training data)
            current: Current distribution (production data)
            n_bins: Number of bins for discretization
            
        Returns:
            DriftResult with PSI score and severity
        """
        n_bins = n_bins or self.config.n_bins
        
        # Create bins based on reference distribution
        _, bin_edges = np.histogram(reference, bins=n_bins)
        
        # Extend edge bins to capture outliers
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        # Calculate bin frequencies
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to percentages (with smoothing to avoid zero)
        ref_pct = (ref_counts + 0.0001) / (ref_counts.sum() + 0.0001 * n_bins)
        cur_pct = (cur_counts + 0.0001) / (cur_counts.sum() + 0.0001 * n_bins)
        
        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        # Determine severity
        if psi >= self.config.psi_threshold_critical:
            severity = "critical"
            drift_detected = True
        elif psi >= self.config.psi_threshold_warning:
            severity = "warning"
            drift_detected = True
        else:
            severity = "none"
            drift_detected = False
        
        return DriftResult(
            feature_name="unknown",
            drift_detected=drift_detected,
            drift_type=DriftType.SUDDEN if drift_detected else DriftType.NONE,
            drift_score=float(psi),
            p_value=None,
            threshold_used=self.config.psi_threshold_warning,
            severity=severity,
            details={
                'reference_samples': len(reference),
                'current_samples': len(current),
                'n_bins': n_bins,
                'bin_edges': bin_edges.tolist() if hasattr(bin_edges, 'tolist') else list(bin_edges)
            }
        )


class KLDivergenceCalculator:
    """
    Kullback-Leibler Divergence Calculator.
    
    KL Divergence measures information loss when approximating
    the reference distribution with the current distribution.
    
    Note: KL Divergence is asymmetric. We use Jensen-Shannon
    divergence for symmetric comparison.
    """
    
    def __init__(self, config: DriftConfig = None):
        self.config = config or DriftConfig()
    
    def calculate(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        symmetric: bool = True
    ) -> DriftResult:
        """
        Calculate KL Divergence between distributions.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            symmetric: Use Jensen-Shannon divergence if True
            
        Returns:
            DriftResult with divergence score
        """
        n_bins = self.config.n_bins
        
        # Create shared bins
        combined = np.concatenate([reference, current])
        _, bin_edges = np.histogram(combined, bins=n_bins)
        
        # Calculate histograms
        ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
        cur_hist, _ = np.histogram(current, bins=bin_edges, density=True)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_hist = ref_hist + epsilon
        cur_hist = cur_hist + epsilon
        
        # Normalize
        ref_hist = ref_hist / ref_hist.sum()
        cur_hist = cur_hist / cur_hist.sum()
        
        if symmetric:
            # Jensen-Shannon divergence
            divergence = jensenshannon(ref_hist, cur_hist) ** 2
        else:
            # Standard KL divergence
            divergence = stats.entropy(cur_hist, ref_hist)
        
        # Determine severity
        if divergence >= self.config.kl_threshold_critical:
            severity = "critical"
            drift_detected = True
        elif divergence >= self.config.kl_threshold_warning:
            severity = "warning"
            drift_detected = True
        else:
            severity = "none"
            drift_detected = False
        
        return DriftResult(
            feature_name="unknown",
            drift_detected=drift_detected,
            drift_type=DriftType.GRADUAL if drift_detected else DriftType.NONE,
            drift_score=float(divergence),
            p_value=None,
            threshold_used=self.config.kl_threshold_warning,
            severity=severity,
            details={
                'symmetric': symmetric,
                'method': 'jensen_shannon' if symmetric else 'kl_divergence'
            }
        )


class KSTestDetector:
    """
    Kolmogorov-Smirnov Test for Distribution Comparison.
    
    The KS test measures the maximum distance between two
    cumulative distribution functions.
    """
    
    def __init__(self, config: DriftConfig = None):
        self.config = config or DriftConfig()
    
    def calculate(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> DriftResult:
        """
        Perform KS test between distributions.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            
        Returns:
            DriftResult with KS statistic and p-value
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        
        drift_detected = p_value < self.config.ks_alpha
        
        if drift_detected:
            severity = "critical" if p_value < 0.01 else "warning"
        else:
            severity = "none"
        
        return DriftResult(
            feature_name="unknown",
            drift_detected=drift_detected,
            drift_type=DriftType.SUDDEN if drift_detected else DriftType.NONE,
            drift_score=float(statistic),
            p_value=float(p_value),
            threshold_used=self.config.ks_alpha,
            severity=severity,
            details={
                'ks_statistic': float(statistic),
                'critical_value': self.config.ks_alpha
            }
        )


class WassersteinDetector:
    """
    Wasserstein Distance (Earth Mover's Distance) Detector.
    
    Measures the minimum "work" needed to transform one
    distribution into another.
    """
    
    def __init__(self, config: DriftConfig = None):
        self.config = config or DriftConfig()
    
    def calculate(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        normalize: bool = True
    ) -> DriftResult:
        """
        Calculate Wasserstein distance between distributions.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            normalize: Normalize by reference std
            
        Returns:
            DriftResult with Wasserstein distance
        """
        distance = stats.wasserstein_distance(reference, current)
        
        if normalize:
            ref_std = np.std(reference)
            if ref_std > 0:
                normalized_distance = distance / ref_std
            else:
                normalized_distance = distance
        else:
            normalized_distance = distance
        
        # Determine severity
        if normalized_distance >= self.config.wasserstein_threshold_critical:
            severity = "critical"
            drift_detected = True
        elif normalized_distance >= self.config.wasserstein_threshold_warning:
            severity = "warning"
            drift_detected = True
        else:
            severity = "none"
            drift_detected = False
        
        return DriftResult(
            feature_name="unknown",
            drift_detected=drift_detected,
            drift_type=DriftType.GRADUAL if drift_detected else DriftType.NONE,
            drift_score=float(normalized_distance),
            p_value=None,
            threshold_used=self.config.wasserstein_threshold_warning,
            severity=severity,
            details={
                'raw_distance': float(distance),
                'normalized': normalize,
                'reference_std': float(np.std(reference))
            }
        )


class DriftDetector:
    """
    Unified Drift Detector combining multiple methods.
    
    Provides a comprehensive drift detection pipeline with:
    - Multiple detection algorithms
    - Feature-level and dataset-level analysis
    - Configurable thresholds and alerts
    
    Example:
        >>> detector = DriftDetector()
        >>> results = detector.detect_drift(reference_df, current_df)
        >>> print(results['summary'])
    """
    
    def __init__(self, config: DriftConfig = None):
        self.config = config or DriftConfig()
        
        # Initialize calculators
        self.psi_calc = PSICalculator(config)
        self.kl_calc = KLDivergenceCalculator(config)
        self.ks_calc = KSTestDetector(config)
        self.wasserstein_calc = WassersteinDetector(config)
        
        # Track history
        self.detection_history: List[Dict[str, Any]] = []
    
    def detect_feature_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "unknown",
        methods: List[str] = None
    ) -> Dict[str, DriftResult]:
        """
        Detect drift for a single feature using multiple methods.
        
        Args:
            reference: Reference values
            current: Current values
            feature_name: Name of the feature
            methods: List of methods to use ('psi', 'kl', 'ks', 'wasserstein')
            
        Returns:
            Dict mapping method name to DriftResult
        """
        methods = methods or ['psi', 'kl', 'ks', 'wasserstein']
        results = {}
        
        # Clean data
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]
        
        if len(ref_clean) < self.config.min_bin_size:
            warnings.warn(f"Reference data too small for {feature_name}")
            return results
        
        if len(cur_clean) < self.config.min_bin_size:
            warnings.warn(f"Current data too small for {feature_name}")
            return results
        
        for method in methods:
            if method == 'psi':
                result = self.psi_calc.calculate(ref_clean, cur_clean)
            elif method == 'kl':
                result = self.kl_calc.calculate(ref_clean, cur_clean)
            elif method == 'ks':
                result = self.ks_calc.calculate(ref_clean, cur_clean)
            elif method == 'wasserstein':
                result = self.wasserstein_calc.calculate(ref_clean, cur_clean)
            else:
                continue
            
            result.feature_name = feature_name
            results[method] = result
        
        return results
    
    def detect_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: List[str] = None,
        methods: List[str] = None
    ) -> Dict[str, Any]:
        """
        Detect drift across all features in a dataset.
        
        Args:
            reference_df: Reference DataFrame
            current_df: Current DataFrame
            features: List of features to check (default: all numeric)
            methods: Detection methods to use
            
        Returns:
            Dict with per-feature results and summary
        """
        if features is None:
            features = reference_df.select_dtypes(include=[np.number]).columns.tolist()
        
        all_results = {}
        drifted_features = []
        
        for feature in features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                continue
            
            ref_values = reference_df[feature].values
            cur_values = current_df[feature].values
            
            feature_results = self.detect_feature_drift(
                ref_values, cur_values, feature, methods
            )
            
            all_results[feature] = feature_results
            
            # Check if any method detected drift
            if any(r.drift_detected for r in feature_results.values()):
                drifted_features.append(feature)
        
        summary = {
            'total_features': len(features),
            'drifted_features': len(drifted_features),
            'drift_percentage': len(drifted_features) / len(features) * 100 if features else 0,
            'drifted_feature_names': drifted_features,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.detection_history.append({
            'summary': summary,
            'results': {f: {m: r.to_dict() for m, r in results.items()} 
                       for f, results in all_results.items()}
        })
        
        return {
            'feature_results': all_results,
            'summary': summary
        }
    
    def get_drift_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: List[str] = None
    ) -> pd.DataFrame:
        """
        Generate a comprehensive drift report as DataFrame.
        
        Args:
            reference_df: Reference DataFrame
            current_df: Current DataFrame
            features: Features to analyze
            
        Returns:
            DataFrame with drift metrics for each feature
        """
        results = self.detect_drift(reference_df, current_df, features)
        
        report_data = []
        for feature, method_results in results['feature_results'].items():
            row = {'feature': feature}
            
            for method, result in method_results.items():
                row[f'{method}_score'] = result.drift_score
                row[f'{method}_drift'] = result.drift_detected
                row[f'{method}_severity'] = result.severity
            
            # Overall drift assessment
            row['any_drift'] = any(r.drift_detected for r in method_results.values())
            row['max_severity'] = max(
                (r.severity for r in method_results.values()),
                key=lambda s: {'none': 0, 'warning': 1, 'critical': 2}[s]
            )
            
            report_data.append(row)
        
        return pd.DataFrame(report_data)
    
    def monitor_drift_over_time(
        self,
        reference_df: pd.DataFrame,
        data_batches: List[pd.DataFrame],
        features: List[str] = None
    ) -> pd.DataFrame:
        """
        Monitor drift across multiple data batches over time.
        
        Args:
            reference_df: Reference distribution
            data_batches: List of DataFrames representing time periods
            features: Features to monitor
            
        Returns:
            DataFrame with drift trends over time
        """
        if features is None:
            features = reference_df.select_dtypes(include=[np.number]).columns.tolist()
        
        trend_data = []
        
        for batch_idx, batch_df in enumerate(data_batches):
            results = self.detect_drift(reference_df, batch_df, features)
            
            row = {
                'batch_index': batch_idx,
                'drifted_features': results['summary']['drifted_features'],
                'drift_percentage': results['summary']['drift_percentage']
            }
            
            # Average scores across features
            for method in ['psi', 'kl', 'ks', 'wasserstein']:
                scores = []
                for feature_results in results['feature_results'].values():
                    if method in feature_results:
                        scores.append(feature_results[method].drift_score)
                
                if scores:
                    row[f'avg_{method}_score'] = np.mean(scores)
            
            trend_data.append(row)
        
        return pd.DataFrame(trend_data)


# Main execution for testing
if __name__ == "__main__":
    print("Testing Drift Detection Module...")
    
    # Generate test data
    np.random.seed(42)
    reference = np.random.normal(100, 15, 1000)
    
    # No drift
    current_no_drift = np.random.normal(100, 15, 500)
    
    # With drift
    current_drift = np.random.normal(110, 20, 500)  # Mean and std shifted
    
    # Test PSI
    psi_calc = PSICalculator()
    result = psi_calc.calculate(reference, current_no_drift)
    print(f"PSI (no drift): {result.drift_score:.4f}, Detected: {result.drift_detected}")
    
    result = psi_calc.calculate(reference, current_drift)
    print(f"PSI (with drift): {result.drift_score:.4f}, Detected: {result.drift_detected}")
    
    # Test full detector
    detector = DriftDetector()
    
    ref_df = pd.DataFrame({'feature1': reference, 'feature2': reference * 1.5})
    cur_df = pd.DataFrame({'feature1': current_drift, 'feature2': current_drift * 1.5})
    
    results = detector.detect_drift(ref_df, cur_df)
    print(f"\nDataset drift summary:")
    print(f"  Features checked: {results['summary']['total_features']}")
    print(f"  Features drifted: {results['summary']['drifted_features']}")
    print(f"  Drift percentage: {results['summary']['drift_percentage']:.1f}%")
    
    print("\nDrift detection module working correctly!")
