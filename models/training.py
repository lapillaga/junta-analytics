import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from config import Config

def prepare_data_for_anomaly_detection(merged_data):
    """
    Prepare data for the anomaly detection model
    """
    if merged_data.empty:
        return None
    
    # Select features for the model
    features = ['avg_consumption', 'total_consumption']
    
    if 'total_rainfall' in merged_data.columns:
        features.append('total_rainfall')
    
    if 'avg_rainfall' in merged_data.columns:
        features.append('avg_rainfall')
    
    # Check if we have enough data
    if merged_data.shape[0] < 10:
        print("Not enough data for training anomaly detection model")
        return None
    
    # Prepare feature matrix
    X = merged_data[features].fillna(0)
    
    return X

def train_anomaly_detection_model(merged_data, save_model=True):
    """
    Train an Isolation Forest model to detect consumption anomalies
    """
    X = prepare_data_for_anomaly_detection(merged_data)
    
    if X is None:
        return None
    
    # Create and train the model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('isolation_forest', IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        ))
    ])
    
    model.fit(X)
    
    # Save the model
    if save_model:
        os.makedirs(os.path.dirname(Config.ANOMALY_DETECTION_MODEL_PATH), exist_ok=True)
        joblib.dump(model, Config.ANOMALY_DETECTION_MODEL_PATH)
        print(f"Anomaly detection model saved to {Config.ANOMALY_DETECTION_MODEL_PATH}")
    
    return model

def prepare_data_for_forecast(merged_data):
    """
    Prepare data for the consumption forecast model
    """
    if merged_data.empty:
        return None, None
    
    # Create lagged features
    data = merged_data.copy()
    
    # Create lag features (1, 2, and 3 periods ago)
    for lag in range(1, 4):
        data[f'consumption_lag_{lag}'] = data['total_consumption'].shift(lag)
        
        if 'total_rainfall' in data.columns:
            data[f'rainfall_lag_{lag}'] = data['total_rainfall'].shift(lag)
    
    # Drop rows with NaN values
    data = data.dropna()
    
    if data.shape[0] < 10:
        print("Not enough data for training forecast model after creating lags")
        return None, None
    
    # Select features and target
    features = [f'consumption_lag_{lag}' for lag in range(1, 4)]
    
    if 'total_rainfall' in data.columns:
        features.extend([f'rainfall_lag_{lag}' for lag in range(1, 4)])
        features.append('total_rainfall')
    
    X = data[features]
    y = data['total_consumption']
    
    return X, y

def train_forecast_model(merged_data, save_model=True):
    """
    Train a Random Forest model to forecast consumption
    """
    X, y = prepare_data_for_forecast(merged_data)
    
    if X is None or y is None:
        return None
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('random_forest', RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])
    
    model.fit(X_train, y_train)
    
    # Print model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Forecast model R² on train: {train_score:.4f}")
    print(f"Forecast model R² on test: {test_score:.4f}")
    
    # Save the model
    if save_model:
        os.makedirs(os.path.dirname(Config.FORECAST_MODEL_PATH), exist_ok=True)
        joblib.dump(model, Config.FORECAST_MODEL_PATH)
        print(f"Forecast model saved to {Config.FORECAST_MODEL_PATH}")
    
    return model