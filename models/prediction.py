import os
import pandas as pd
import numpy as np
import joblib
from config import Config
from utils.data_processing import is_consumption_anomaly

def load_anomaly_detection_model():
    """
    Load the trained anomaly detection model
    """
    model_path = Config.ANOMALY_DETECTION_MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"Anomaly detection model not found at {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading anomaly detection model: {e}")
        return None

def load_forecast_model():
    """
    Load the trained forecast model
    """
    model_path = Config.FORECAST_MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"Forecast model not found at {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading forecast model: {e}")
        return None

def detect_anomalies(data, model=None):
    """
    Detect anomalies in consumption data
    """
    if data.empty:
        return pd.DataFrame()
    
    # Try to load model if not provided
    if model is None:
        model = load_anomaly_detection_model()
    
    # If model is still None, use simple rule-based approach
    if model is None:
        data['is_anomaly'] = data['total_consumption'].apply(is_consumption_anomaly)
        return data
    
    # Prepare data for prediction
    features = ['avg_consumption', 'total_consumption']
    
    if 'total_rainfall' in data.columns:
        features.append('total_rainfall')
    
    if 'avg_rainfall' in data.columns:
        features.append('avg_rainfall')
    
    X = data[features].fillna(0)
    
    # Predict anomalies
    # Isolation Forest: -1 for anomalies, 1 for normal
    predictions = model.predict(X)
    
    # Convert to boolean (True for anomalies)
    data['is_anomaly'] = predictions == -1
    
    return data

def predict_single_consumption(consumption_value, rainfall_value=None):
    """
    Predict if a single consumption value is anomalous
    """
    # Try simple rule-based approach first
    rule_based_result = is_consumption_anomaly(consumption_value)
    
    # Try model-based approach if available
    model = load_anomaly_detection_model()
    
    if model is None:
        return {
            'is_anomaly': rule_based_result,
            'method': 'rule-based'
        }
    
    # Prepare features
    features = np.array([[consumption_value, consumption_value]])
    
    if rainfall_value is not None:
        features = np.array([[consumption_value, consumption_value, rainfall_value, rainfall_value]])
    
    # Predict
    prediction = model.predict(features)[0]
    
    return {
        'is_anomaly': prediction == -1,
        'method': 'model-based'
    }

def forecast_consumption(merged_data, forecast_periods=3):
    """
    Forecast future consumption based on historical data
    """
    model = load_forecast_model()
    
    if model is None or merged_data.empty:
        return pd.DataFrame()
    
    # Get the latest data
    latest_data = merged_data.copy().sort_values('dekad_period').tail(3)
    
    # Create features for prediction
    forecast_results = []
    
    # Create initial lag features from latest data
    lag_values = {
        f'consumption_lag_{i+1}': latest_data['total_consumption'].iloc[-(i+1)]
        for i in range(min(3, len(latest_data)))
    }
    
    # Add rainfall lag features if available
    if 'total_rainfall' in latest_data.columns:
        lag_values.update({
            f'rainfall_lag_{i+1}': latest_data['total_rainfall'].iloc[-(i+1)]
            for i in range(min(3, len(latest_data)))
        })
    
    # Get the last available rainfall value
    latest_rainfall = latest_data['total_rainfall'].iloc[-1] if 'total_rainfall' in latest_data.columns else 0
    
    # Generate forecasts for future periods
    for i in range(1, forecast_periods + 1):
        # Create feature dictionary for this forecast period
        features = lag_values.copy()
        
        # Add rainfall if available
        if 'total_rainfall' in latest_data.columns:
            features['total_rainfall'] = latest_rainfall
        
        # Convert to dataframe
        X = pd.DataFrame([features])
        
        # Predict consumption
        predicted = float(model.predict(X)[0])
        
        # Calculate bounds (simple approach with ±15%)
        lower_bound = predicted * 0.85
        upper_bound = predicted * 1.15
        
        # Generate next dekad period label
        last_period = latest_data['dekad_period'].iloc[-1]
        year, month, dekad = map(int, last_period.split('-'))
        
        # Move to next dekad
        dekad += 1
        if dekad > 3:
            dekad = 1
            month += 1
            if month > 12:
                month = 1
                year += 1
        
        next_period = f"{year}-{month:02d}-{dekad}"
        
        # Add to results
        forecast_results.append({
            'dekad_period': next_period,
            'predicted_consumption': predicted,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
        
        # Update lag values for next iteration
        lag_values['consumption_lag_3'] = lag_values.get('consumption_lag_2', 0)
        lag_values['consumption_lag_2'] = lag_values.get('consumption_lag_1', 0)
        lag_values['consumption_lag_1'] = predicted
        
        if 'rainfall_lag_1' in lag_values:
            lag_values['rainfall_lag_3'] = lag_values.get('rainfall_lag_2', 0)
            lag_values['rainfall_lag_2'] = lag_values.get('rainfall_lag_1', 0)
            lag_values['rainfall_lag_1'] = latest_rainfall
    
    return pd.DataFrame(forecast_results)