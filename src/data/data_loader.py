"""
Data Loading and Processing for Medical Data
Built by Prashant Ambati
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class MedicalDataset(Dataset):
    """
    Custom Dataset class for medical data.
    """
    
    def __init__(self, data, conditions, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data (np.ndarray): Medical data features
            conditions (np.ndarray): Condition labels
            transform (callable): Optional transform to be applied
        """
        self.data = torch.FloatTensor(data)
        self.conditions = torch.FloatTensor(conditions)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        condition = self.conditions[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, condition


class MedicalDataLoader:
    """
    Data loader for medical datasets with preprocessing capabilities.
    """
    
    def __init__(self, data_path=None, test_size=0.2, random_state=42):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to the medical data CSV file
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_synthetic_medical_data(self, n_samples=10000):
        """
        Create synthetic medical data for demonstration purposes.
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Synthetic medical dataset
        """
        np.random.seed(self.random_state)
        
        # Generate synthetic patient data
        data = {
            'age': np.random.normal(50, 15, n_samples).clip(18, 90),
            'bmi': np.random.normal(25, 5, n_samples).clip(15, 50),
            'blood_pressure_systolic': np.random.normal(120, 20, n_samples).clip(80, 200),
            'blood_pressure_diastolic': np.random.normal(80, 15, n_samples).clip(50, 120),
            'cholesterol': np.random.normal(200, 40, n_samples).clip(100, 400),
            'glucose': np.random.normal(100, 25, n_samples).clip(70, 300),
            'heart_rate': np.random.normal(70, 15, n_samples).clip(50, 120),
            'temperature': np.random.normal(98.6, 1, n_samples).clip(95, 105),
            'respiratory_rate': np.random.normal(16, 4, n_samples).clip(10, 30),
            'oxygen_saturation': np.random.normal(98, 2, n_samples).clip(85, 100),
            'white_blood_cells': np.random.normal(7000, 2000, n_samples).clip(3000, 15000),
            'red_blood_cells': np.random.normal(4.5, 0.5, n_samples).clip(3.5, 6.0),
            'hemoglobin': np.random.normal(14, 2, n_samples).clip(10, 18),
            'platelets': np.random.normal(250000, 50000, n_samples).clip(150000, 450000),
            'creatinine': np.random.normal(1.0, 0.3, n_samples).clip(0.5, 3.0),
            'sodium': np.random.normal(140, 5, n_samples).clip(130, 150),
            'potassium': np.random.normal(4.0, 0.5, n_samples).clip(3.0, 5.5),
            'chloride': np.random.normal(100, 5, n_samples).clip(90, 110),
            'co2': np.random.normal(24, 3, n_samples).clip(18, 30),
            'bun': np.random.normal(15, 5, n_samples).clip(5, 50),
        }
        
        # Add categorical conditions
        conditions = np.random.choice(['healthy', 'diabetes', 'hypertension', 'heart_disease'], 
                                    n_samples, p=[0.4, 0.2, 0.25, 0.15])
        data['condition'] = conditions
        
        # Add gender
        data['gender'] = np.random.choice(['M', 'F'], n_samples, p=[0.5, 0.5])
        
        return pd.DataFrame(data)
    
    def load_and_preprocess_data(self, data_path=None):
        """
        Load and preprocess medical data.
        
        Args:
            data_path (str): Path to data file (optional)
            
        Returns:
            tuple: Processed data and conditions
        """
        if data_path is None:
            # Create synthetic data for demonstration
            df = self.create_synthetic_medical_data()
        else:
            df = pd.read_csv(data_path)
        
        # Separate features and conditions
        condition_cols = ['condition', 'gender']
        feature_cols = [col for col in df.columns if col not in condition_cols]
        
        # Process numerical features
        X = df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Process categorical conditions
        conditions = []
        for col in condition_cols:
            if col in df.columns:
                le = LabelEncoder()
                encoded = le.fit_transform(df[col])
                self.label_encoders[col] = le
                conditions.append(encoded)
        
        # Combine conditions into one-hot encoding
        condition_matrix = np.column_stack(conditions) if conditions else np.zeros((len(df), 1))
        
        # Convert to one-hot for conditions
        from sklearn.preprocessing import OneHotEncoder
        onehot = OneHotEncoder(sparse=False)
        conditions_onehot = onehot.fit_transform(condition_matrix)
        
        return X_scaled, conditions_onehot, df
    
    def create_data_loaders(self, batch_size=64, data_path=None):
        """
        Create PyTorch data loaders.
        
        Args:
            batch_size (int): Batch size for training
            data_path (str): Path to data file
            
        Returns:
            tuple: Train and test data loaders
        """
        X, conditions, df = self.load_and_preprocess_data(data_path)
        
        # Split data
        X_train, X_test, cond_train, cond_test = train_test_split(
            X, conditions, test_size=self.test_size, random_state=self.random_state
        )
        
        # Create datasets
        train_dataset = MedicalDataset(X_train, cond_train)
        test_dataset = MedicalDataset(X_test, cond_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, df
    
    def inverse_transform_data(self, scaled_data):
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            scaled_data (np.ndarray): Scaled data
            
        Returns:
            np.ndarray: Original scale data
        """
        return self.scaler.inverse_transform(scaled_data)