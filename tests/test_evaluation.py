import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.statistical_tests import StatisticalEvaluator


def test_ks_two_sample_identical():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 2000).reshape(-1, 1)
    evaluator = StatisticalEvaluator()
    res = evaluator.kolmogorov_smirnov_test(data, data.copy(), ['x'])
    assert 'x' in res
    assert 0 <= res['x']['p_value'] <= 1
    assert res['x']['p_value'] > 0.5


def test_wasserstein_distance_zero_for_identical():
    rng = np.random.default_rng(0)
    real = rng.normal(0, 1, 3000).reshape(-1, 1)
    synth = real.copy()
    evaluator = StatisticalEvaluator()
    wd = evaluator.wasserstein_distance_test(real, synth, ['x'])
    assert wd['x'] < 1e-6


def test_correlation_analysis_shapes_and_small_error():
    rng = np.random.default_rng(123)
    real = rng.normal(0, 1, (500, 5))
    synth = real.copy()
    evaluator = StatisticalEvaluator()
    corr = evaluator.correlation_analysis(real, synth, [f'f{i}' for i in range(5)])
    assert 'mean_absolute_error' in corr
    assert corr['correlation_difference'].shape == (5, 5)
    assert corr['mean_absolute_error'] < 1e-6


def test_privacy_metrics_structure():
    rng = np.random.default_rng(7)
    real = rng.normal(0, 1, (300, 4))
    synth = rng.normal(0, 1, (300, 4))
    evaluator = StatisticalEvaluator()
    pm = evaluator.privacy_metrics(real, synth, k=3)
    assert set(pm.keys()) == {
        'average_minimum_distance', 'privacy_violation_rate',
        'distance_threshold', 'min_distances'
    }
    assert pm['min_distances'].shape[0] == synth.shape[0]


def test_comprehensive_evaluation_returns_quality_score():
    rng = np.random.default_rng(21)
    real = rng.normal(0, 1, (400, 3))
    synth = rng.normal(0, 1, (400, 3))
    evaluator = StatisticalEvaluator()
    res = evaluator.comprehensive_evaluation(real, synth, [f'f{i}' for i in range(3)])
    assert 'quality_score' in res
