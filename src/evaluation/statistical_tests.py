"""
Statistical Tests for Evaluating Synthetic Medical Data Quality
Built by Prashant Ambati
"""

import numpy as np
import pandas as pd
from scipy import stats
try:
    from scipy.stats import wasserstein_distance
except ImportError:
    # Fallback implementation for older scipy versions
    def wasserstein_distance(u_values, v_values):
        return stats.energy_distance(u_values, v_values)
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


class StatisticalEvaluator:
    """
    Comprehensive statistical evaluation of synthetic medical data.
    """
    
    def __init__(self):
        self.results = {}
    
    def kolmogorov_smirnov_test(self, real_data, synthetic_data, feature_names=None):
        """
        Perform Kolmogorov-Smirnov test for each feature.
        
        Args:
            real_data (np.ndarray): Real medical data
            synthetic_data (np.ndarray): Synthetic medical data
            feature_names (list): Names of features
            
        Returns:
            dict: KS test results for each feature
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(real_data.shape[1])]
        
        ks_results = {}
        
        for i, feature_name in enumerate(feature_names):
            real_feature = real_data[:, i]
            synthetic_feature = synthetic_data[:, i]
            
            # Perform KS test
            ks_statistic, p_value = stats.kstest(synthetic_feature, real_feature)
            
            ks_results[feature_name] = {
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        self.results['ks_test'] = ks_results
        return ks_results
    
    def wasserstein_distance_test(self, real_data, synthetic_data, feature_names=None):
        """
        Calculate Wasserstein distance for each feature.
        
        Args:
            real_data (np.ndarray): Real medical data
            synthetic_data (np.ndarray): Synthetic medical data
            feature_names (list): Names of features
            
        Returns:
            dict: Wasserstein distances for each feature
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(real_data.shape[1])]
        
        wd_results = {}
        
        for i, feature_name in enumerate(feature_names):
            real_feature = real_data[:, i]
            synthetic_feature = synthetic_data[:, i]
            
            # Calculate Wasserstein distance
            wd = wasserstein_distance(real_feature, synthetic_feature)
            wd_results[feature_name] = wd
        
        self.results['wasserstein_distance'] = wd_results
        return wd_results
    
    def correlation_analysis(self, real_data, synthetic_data, feature_names=None):
        """
        Compare correlation matrices between real and synthetic data.
        
        Args:
            real_data (np.ndarray): Real medical data
            synthetic_data (np.ndarray): Synthetic medical data
            feature_names (list): Names of features
            
        Returns:
            dict: Correlation analysis results
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(real_data.shape[1])]
        
        # Calculate correlation matrices
        real_corr = np.corrcoef(real_data.T)
        synthetic_corr = np.corrcoef(synthetic_data.T)
        
        # Calculate correlation difference
        corr_diff = np.abs(real_corr - synthetic_corr)
        
        # Calculate Frobenius norm of difference
        frobenius_norm = np.linalg.norm(corr_diff, 'fro')
        
        # Calculate mean absolute error
        mae = np.mean(corr_diff)
        
        correlation_results = {
            'real_correlation': real_corr,
            'synthetic_correlation': synthetic_corr,
            'correlation_difference': corr_diff,
            'frobenius_norm': frobenius_norm,
            'mean_absolute_error': mae,
            'feature_names': feature_names
        }
        
        self.results['correlation_analysis'] = correlation_results
        return correlation_results
    
    def distribution_comparison(self, real_data, synthetic_data, feature_names=None):
        """
        Compare statistical distributions of real and synthetic data.
        
        Args:
            real_data (np.ndarray): Real medical data
            synthetic_data (np.ndarray): Synthetic medical data
            feature_names (list): Names of features
            
        Returns:
            dict: Distribution comparison results
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(real_data.shape[1])]
        
        distribution_results = {}
        
        for i, feature_name in enumerate(feature_names):
            real_feature = real_data[:, i]
            synthetic_feature = synthetic_data[:, i]
            
            # Calculate basic statistics
            real_stats = {
                'mean': np.mean(real_feature),
                'std': np.std(real_feature),
                'min': np.min(real_feature),
                'max': np.max(real_feature),
                'median': np.median(real_feature),
                'q25': np.percentile(real_feature, 25),
                'q75': np.percentile(real_feature, 75)
            }
            
            synthetic_stats = {
                'mean': np.mean(synthetic_feature),
                'std': np.std(synthetic_feature),
                'min': np.min(synthetic_feature),
                'max': np.max(synthetic_feature),
                'median': np.median(synthetic_feature),
                'q25': np.percentile(synthetic_feature, 25),
                'q75': np.percentile(synthetic_feature, 75)
            }
            
            # Calculate differences
            stat_differences = {}
            for stat_name in real_stats.keys():
                stat_differences[stat_name] = abs(real_stats[stat_name] - synthetic_stats[stat_name])
            
            distribution_results[feature_name] = {
                'real_stats': real_stats,
                'synthetic_stats': synthetic_stats,
                'differences': stat_differences
            }
        
        self.results['distribution_comparison'] = distribution_results
        return distribution_results
    
    def privacy_metrics(self, real_data, synthetic_data, k=5):
        """
        Calculate privacy-related metrics.
        
        Args:
            real_data (np.ndarray): Real medical data
            synthetic_data (np.ndarray): Synthetic medical data
            k (int): Number of nearest neighbors for privacy analysis
            
        Returns:
            dict: Privacy metrics
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Fit nearest neighbors on real data
        nn_real = NearestNeighbors(n_neighbors=k+1)
        nn_real.fit(real_data)
        
        # Find distances from synthetic to real data
        distances, indices = nn_real.kneighbors(synthetic_data)
        
        # Calculate average minimum distance (privacy measure)
        min_distances = distances[:, 1]  # Exclude self (distance 0)
        avg_min_distance = np.mean(min_distances)
        
        # Calculate percentage of synthetic samples that are too close to real data
        threshold = np.percentile(min_distances, 5)  # 5th percentile as threshold
        privacy_violations = np.sum(min_distances < threshold) / len(min_distances)
        
        privacy_results = {
            'average_minimum_distance': avg_min_distance,
            'privacy_violation_rate': privacy_violations,
            'distance_threshold': threshold,
            'min_distances': min_distances
        }
        
        self.results['privacy_metrics'] = privacy_results
        return privacy_results
    
    def comprehensive_evaluation(self, real_data, synthetic_data, feature_names=None):
        """
        Perform comprehensive evaluation of synthetic data quality.
        
        Args:
            real_data (np.ndarray): Real medical data
            synthetic_data (np.ndarray): Synthetic medical data
            feature_names (list): Names of features
            
        Returns:
            dict: Complete evaluation results
        """
        print("Performing comprehensive evaluation...")
        
        # Run all tests
        ks_results = self.kolmogorov_smirnov_test(real_data, synthetic_data, feature_names)
        wd_results = self.wasserstein_distance_test(real_data, synthetic_data, feature_names)
        corr_results = self.correlation_analysis(real_data, synthetic_data, feature_names)
        dist_results = self.distribution_comparison(real_data, synthetic_data, feature_names)
        privacy_results = self.privacy_metrics(real_data, synthetic_data)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score()
        
        evaluation_summary = {
            'quality_score': quality_score,
            'ks_test_results': ks_results,
            'wasserstein_distances': wd_results,
            'correlation_analysis': corr_results,
            'distribution_comparison': dist_results,
            'privacy_metrics': privacy_results
        }
        
        return evaluation_summary
    
    def _calculate_quality_score(self):
        """
        Calculate an overall quality score based on all metrics.
        
        Returns:
            float: Quality score between 0 and 1
        """
        scores = []
        
        # KS test score (percentage of features that pass the test)
        if 'ks_test' in self.results:
            ks_pass_rate = sum(1 for result in self.results['ks_test'].values() 
                              if not result['significant']) / len(self.results['ks_test'])
            scores.append(ks_pass_rate)
        
        # Wasserstein distance score (normalized)
        if 'wasserstein_distance' in self.results:
            wd_scores = list(self.results['wasserstein_distance'].values())
            wd_score = 1 / (1 + np.mean(wd_scores))  # Inverse relationship
            scores.append(wd_score)
        
        # Correlation score
        if 'correlation_analysis' in self.results:
            corr_mae = self.results['correlation_analysis']['mean_absolute_error']
            corr_score = 1 / (1 + corr_mae)  # Inverse relationship
            scores.append(corr_score)
        
        # Privacy score
        if 'privacy_metrics' in self.results:
            privacy_score = 1 - self.results['privacy_metrics']['privacy_violation_rate']
            scores.append(privacy_score)
        
        return np.mean(scores) if scores else 0.0
    
    def generate_report(self, save_path=None):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            save_path (str): Path to save the report
            
        Returns:
            str: Evaluation report
        """
        report = "# Synthetic Medical Data Evaluation Report\n\n"
        
        if 'quality_score' in self.results:
            report += f"## Overall Quality Score: {self.results['quality_score']:.3f}\n\n"
        
        # Add detailed results for each test
        for test_name, results in self.results.items():
            if test_name != 'quality_score':
                report += f"## {test_name.replace('_', ' ').title()}\n"
                report += f"Results: {results}\n\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report