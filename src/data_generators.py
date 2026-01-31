"""
Data Generators for ML Observability
=====================================
Author: Bhavin Kotak (kotak.bhavin@gmail.com | bhavin.kotak@ibm.com)

Generates synthetic data for testing drift detection and monitoring.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


class DriftSimulator:
    """
    Simulates various types of distribution drift.
    
    Used for testing and demonstrating drift detection algorithms.
    """
    
    def __init__(self, random_seed: int = 42):
        self._rng = np.random.RandomState(random_seed)
    
    def generate_reference_distribution(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        distribution: str = "normal"
    ) -> pd.DataFrame:
        """
        Generate a reference distribution.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            distribution: 'normal', 'uniform', 'exponential'
            
        Returns:
            DataFrame with reference data
        """
        data = {}
        
        for i in range(n_features):
            if distribution == "normal":
                data[f'feature_{i}'] = self._rng.normal(
                    loc=self._rng.uniform(-10, 10),
                    scale=self._rng.uniform(1, 5),
                    size=n_samples
                )
            elif distribution == "uniform":
                low = self._rng.uniform(-20, 0)
                data[f'feature_{i}'] = self._rng.uniform(
                    low=low,
                    high=low + self._rng.uniform(10, 30),
                    size=n_samples
                )
            elif distribution == "exponential":
                data[f'feature_{i}'] = self._rng.exponential(
                    scale=self._rng.uniform(1, 10),
                    size=n_samples
                )
        
        return pd.DataFrame(data)
    
    def apply_gradual_drift(
        self,
        reference: pd.DataFrame,
        drift_magnitude: float = 0.5,
        features: List[str] = None
    ) -> pd.DataFrame:
        """
        Apply gradual drift (mean shift).
        
        Args:
            reference: Reference DataFrame
            drift_magnitude: Amount of drift (in std units)
            features: Features to drift (default: all)
            
        Returns:
            Drifted DataFrame
        """
        drifted = reference.copy()
        features = features or reference.columns.tolist()
        
        for feature in features:
            if feature not in reference.columns:
                continue
            
            std = reference[feature].std()
            shift = drift_magnitude * std
            drifted[feature] = drifted[feature] + shift
        
        return drifted
    
    def apply_sudden_drift(
        self,
        reference: pd.DataFrame,
        drift_magnitude: float = 2.0,
        features: List[str] = None
    ) -> pd.DataFrame:
        """
        Apply sudden/abrupt drift.
        
        Args:
            reference: Reference DataFrame
            drift_magnitude: Amount of drift (in std units)
            features: Features to drift
            
        Returns:
            Drifted DataFrame
        """
        drifted = reference.copy()
        features = features or reference.columns.tolist()
        
        for feature in features:
            if feature not in reference.columns:
                continue
            
            std = reference[feature].std()
            mean = reference[feature].mean()
            
            # Regenerate with shifted parameters
            drifted[feature] = self._rng.normal(
                loc=mean + drift_magnitude * std,
                scale=std * 1.5,  # Also increase variance
                size=len(reference)
            )
        
        return drifted
    
    def apply_variance_drift(
        self,
        reference: pd.DataFrame,
        variance_multiplier: float = 2.0,
        features: List[str] = None
    ) -> pd.DataFrame:
        """
        Apply variance/spread change (no mean shift).
        
        Args:
            reference: Reference DataFrame
            variance_multiplier: Factor to multiply variance by
            features: Features to drift
            
        Returns:
            Drifted DataFrame
        """
        drifted = reference.copy()
        features = features or reference.columns.tolist()
        
        for feature in features:
            if feature not in reference.columns:
                continue
            
            mean = reference[feature].mean()
            drifted[feature] = mean + (drifted[feature] - mean) * np.sqrt(variance_multiplier)
        
        return drifted
    
    def apply_label_drift(
        self,
        reference: pd.DataFrame,
        target_column: str = "target",
        new_class_balance: float = 0.8
    ) -> pd.DataFrame:
        """
        Apply label/concept drift.
        
        Args:
            reference: Reference DataFrame
            target_column: Target column name
            new_class_balance: New proportion of positive class
            
        Returns:
            Drifted DataFrame with changed class balance
        """
        drifted = reference.copy()
        
        if target_column not in reference.columns:
            # Create target column
            drifted[target_column] = (self._rng.random(len(reference)) < new_class_balance).astype(int)
        else:
            # Resample to change balance
            n_positive = int(len(reference) * new_class_balance)
            new_target = np.zeros(len(reference))
            new_target[:n_positive] = 1
            self._rng.shuffle(new_target)
            drifted[target_column] = new_target.astype(int)
        
        return drifted
    
    def generate_drift_sequence(
        self,
        reference: pd.DataFrame,
        n_batches: int = 10,
        drift_start_batch: int = 5,
        drift_type: str = "gradual"
    ) -> List[pd.DataFrame]:
        """
        Generate sequence of batches with drift occurring mid-sequence.
        
        Args:
            reference: Reference distribution
            n_batches: Number of batches to generate
            drift_start_batch: When drift begins
            drift_type: 'gradual', 'sudden', 'variance'
            
        Returns:
            List of DataFrames representing time batches
        """
        batches = []
        batch_size = len(reference)
        
        for i in range(n_batches):
            if i < drift_start_batch:
                # No drift - resample from reference
                batch = reference.sample(n=batch_size, replace=True)
                batch = batch.reset_index(drop=True)
                # Add some natural variation
                for col in batch.columns:
                    batch[col] = batch[col] + self._rng.normal(0, batch[col].std() * 0.05, batch_size)
            else:
                # Apply drift
                drift_intensity = (i - drift_start_batch + 1) * 0.2  # Gradual increase
                
                if drift_type == "gradual":
                    batch = self.apply_gradual_drift(reference, drift_intensity)
                elif drift_type == "sudden":
                    batch = self.apply_sudden_drift(reference, drift_intensity * 2)
                elif drift_type == "variance":
                    batch = self.apply_variance_drift(reference, 1 + drift_intensity)
                else:
                    batch = reference.sample(n=batch_size, replace=True)
                
                batch = batch.reset_index(drop=True)
            
            batches.append(batch)
        
        return batches


class MonitoringDataGenerator:
    """
    Generate synthetic monitoring data for testing.
    """
    
    def __init__(self, random_seed: int = 42):
        self._rng = np.random.RandomState(random_seed)
    
    def generate_latency_data(
        self,
        n_samples: int = 1000,
        base_latency_ms: float = 50,
        include_spikes: bool = True
    ) -> pd.DataFrame:
        """Generate latency time series data"""
        
        timestamps = [
            datetime.now() - timedelta(minutes=i)
            for i in range(n_samples, 0, -1)
        ]
        
        # Base latency (exponential distribution is realistic)
        latency = self._rng.exponential(base_latency_ms, n_samples)
        
        if include_spikes:
            # Add occasional spikes
            spike_indices = self._rng.choice(n_samples, size=int(n_samples * 0.02), replace=False)
            latency[spike_indices] *= self._rng.uniform(5, 20, len(spike_indices))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'latency_ms': latency,
            'endpoint': self._rng.choice(['/predict', '/batch', '/health'], n_samples)
        })
    
    def generate_error_rate_data(
        self,
        n_samples: int = 1000,
        base_error_rate: float = 0.01
    ) -> pd.DataFrame:
        """Generate error rate time series data"""
        
        timestamps = [
            datetime.now() - timedelta(minutes=i)
            for i in range(n_samples, 0, -1)
        ]
        
        # Bernoulli trials for errors
        total_requests = self._rng.poisson(100, n_samples)
        errors = self._rng.binomial(total_requests, base_error_rate)
        error_rate = errors / np.maximum(total_requests, 1)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'total_requests': total_requests,
            'errors': errors,
            'error_rate': error_rate
        })
    
    def generate_model_metrics(
        self,
        n_samples: int = 1000,
        base_accuracy: float = 0.95,
        degradation_point: int = None
    ) -> pd.DataFrame:
        """
        Generate model performance metrics over time.
        
        Args:
            n_samples: Number of time points
            base_accuracy: Baseline accuracy
            degradation_point: When to start degrading (optional)
            
        Returns:
            DataFrame with accuracy, precision, recall, f1
        """
        timestamps = [
            datetime.now() - timedelta(hours=i)
            for i in range(n_samples, 0, -1)
        ]
        
        accuracy = np.ones(n_samples) * base_accuracy
        
        if degradation_point:
            # Add degradation after the point
            degradation = np.linspace(0, 0.15, n_samples - degradation_point)
            accuracy[degradation_point:] -= degradation
        
        # Add noise
        accuracy += self._rng.normal(0, 0.01, n_samples)
        accuracy = np.clip(accuracy, 0, 1)
        
        # Generate correlated metrics
        precision = accuracy + self._rng.normal(0.02, 0.01, n_samples)
        recall = accuracy - self._rng.normal(0.02, 0.01, n_samples)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'accuracy': accuracy,
            'precision': np.clip(precision, 0, 1),
            'recall': np.clip(recall, 0, 1),
            'f1_score': np.clip(f1, 0, 1)
        })
    
    def generate_feature_statistics(
        self,
        n_features: int = 10,
        n_time_points: int = 100
    ) -> pd.DataFrame:
        """Generate feature statistics over time"""
        
        data = []
        
        for t in range(n_time_points):
            timestamp = datetime.now() - timedelta(hours=t)
            
            for f in range(n_features):
                # Generate summary stats with some drift
                base_mean = self._rng.uniform(-10, 10)
                base_std = self._rng.uniform(1, 5)
                
                # Add gradual drift
                mean_drift = t * 0.01 * self._rng.normal(0, 1)
                
                data.append({
                    'timestamp': timestamp,
                    'feature_name': f'feature_{f}',
                    'mean': base_mean + mean_drift,
                    'std': base_std * (1 + t * 0.001),
                    'min': base_mean - 3 * base_std,
                    'max': base_mean + 3 * base_std,
                    'null_percentage': self._rng.uniform(0, 0.02),
                    'unique_values': self._rng.randint(100, 1000)
                })
        
        return pd.DataFrame(data)


# Main execution for testing
if __name__ == "__main__":
    print("Testing Data Generators...")
    
    # Test drift simulator
    drift_sim = DriftSimulator(random_seed=42)
    
    reference = drift_sim.generate_reference_distribution(n_samples=1000, n_features=5)
    print(f"Reference data shape: {reference.shape}")
    
    # Apply different drift types
    gradual = drift_sim.apply_gradual_drift(reference, drift_magnitude=0.5)
    sudden = drift_sim.apply_sudden_drift(reference, drift_magnitude=2.0)
    variance = drift_sim.apply_variance_drift(reference, variance_multiplier=2.0)
    
    print("\nDrift applied:")
    print(f"  Gradual: mean shift = {(gradual['feature_0'].mean() - reference['feature_0'].mean()):.2f}")
    print(f"  Sudden: mean shift = {(sudden['feature_0'].mean() - reference['feature_0'].mean()):.2f}")
    print(f"  Variance: std ratio = {(variance['feature_0'].std() / reference['feature_0'].std()):.2f}")
    
    # Generate drift sequence
    batches = drift_sim.generate_drift_sequence(reference, n_batches=10, drift_start_batch=5)
    print(f"\nGenerated {len(batches)} batches for drift sequence")
    
    # Test monitoring data generator
    mon_gen = MonitoringDataGenerator(random_seed=42)
    
    latency = mon_gen.generate_latency_data(n_samples=500)
    print(f"\nLatency data: {latency.shape}")
    print(f"  Mean latency: {latency['latency_ms'].mean():.2f}ms")
    
    metrics = mon_gen.generate_model_metrics(n_samples=200, degradation_point=150)
    print(f"\nModel metrics: {metrics.shape}")
    print(f"  Initial accuracy: {metrics['accuracy'].iloc[0]:.4f}")
    print(f"  Final accuracy: {metrics['accuracy'].iloc[-1]:.4f}")
    
    print("\nData generators working correctly!")
