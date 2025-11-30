import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import MedicalDataLoader


def test_create_synthetic_medical_data_columns():
    dl = MedicalDataLoader()
    df = dl.create_synthetic_medical_data(n_samples=100)
    assert 'condition' in df.columns
    assert 'gender' in df.columns
    # 20 numerical features + 2 categorical
    numeric_cols = [c for c in df.columns if c not in ['condition', 'gender']]
    assert len(numeric_cols) == 20


def test_load_and_preprocess_shapes_and_encoder():
    dl = MedicalDataLoader()
    X_scaled, cond_onehot, df = dl.load_and_preprocess_data()
    feature_cols = [c for c in df.columns if c not in ['condition', 'gender']]
    assert X_scaled.shape[1] == len(feature_cols) == 20
    # onehot encoder exists and matches condition + gender categories
    assert dl.onehot_encoder is not None
    cats = dl.onehot_encoder.categories_
    assert sum(len(c) for c in cats) == cond_onehot.shape[1]


def test_inverse_transform_recovers_original_scale():
    dl = MedicalDataLoader()
    X_scaled, cond_onehot, df = dl.load_and_preprocess_data()
    X_inv = dl.inverse_transform_data(X_scaled)
    feature_cols = [c for c in df.columns if c not in ['condition', 'gender']]
    X = df[feature_cols].values
    assert X_inv.shape == X.shape
    assert np.allclose(X_inv.mean(axis=0), X.mean(axis=0), atol=1e-6)
