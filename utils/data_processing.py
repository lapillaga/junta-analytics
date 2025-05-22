import os
import pandas as pd
import numpy as np
from datetime import datetime
from config import Config

def load_rainfall_data(csv_path=None):
    """
    Load rainfall data from CSV file
    
    CSV should have columns:
    - date: Date in YYYY-MM-DD format
    - rainfall: Rainfall measurement in mm
    """
    if csv_path is None:
        csv_path = Config.DEFAULT_RAINFALL_CSV_PATH
    
    if not os.path.exists(csv_path):
        print(f"Warning: Rainfall data file not found at {csv_path}")
        return pd.DataFrame(columns=['date', 'rainfall'])
    
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Ensure date is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        return df
    except Exception as e:
        print(f"Error loading rainfall data: {e}")
        return pd.DataFrame(columns=['date', 'rainfall'])

def convert_to_dekad_period(date):
    """
    Convert a date to its dekad period (1-3 for each month)
    """
    day = date.day
    
    if day <= 10:
        dekad = 1
    elif day <= 20:
        dekad = 2
    else:
        dekad = 3
        
    return f"{date.year}-{date.month:02d}-{dekad}"

def aggregate_rainfall_by_dekad(rainfall_df):
    """
    Aggregate rainfall data by dekad periods
    """
    if rainfall_df.empty:
        return pd.DataFrame(columns=['dekad_period', 'total_rainfall', 'avg_rainfall'])
    
    # Add dekad period column
    rainfall_df['dekad_period'] = rainfall_df['date'].apply(convert_to_dekad_period)
    
    # Aggregate by dekad
    dekad_rainfall = rainfall_df.groupby('dekad_period').agg(
        total_rainfall=('rainfall', 'sum'),
        avg_rainfall=('rainfall', 'mean'),
        days_with_rain=('rainfall', lambda x: (x > 0).sum())
    ).reset_index()
    
    return dekad_rainfall

def merge_consumption_and_rainfall(consumption_df, rainfall_df):
    """
    Merge consumption and rainfall data by dekad period
    """
    # Ensure consumption data has dekad_period column
    if 'dekad_period' not in consumption_df.columns and 'reading_date' in consumption_df.columns:
        consumption_df['dekad_period'] = consumption_df['reading_date'].apply(convert_to_dekad_period)
    
    # Convert dekad_period to common format if needed
    if 'date' in rainfall_df.columns and 'dekad_period' not in rainfall_df.columns:
        rainfall_df['dekad_period'] = rainfall_df['date'].apply(convert_to_dekad_period)
    
    # Merge dataframes
    merged_df = pd.merge(consumption_df, rainfall_df, on='dekad_period', how='outer')
    
    # Fill missing values
    merged_df = merged_df.fillna({
        'total_consumption': 0,
        'avg_consumption': 0,
        'total_rainfall': 0,
        'avg_rainfall': 0
    })
    
    return merged_df

def normalize_data(df, columns):
    """
    Normalize specified columns to 0-1 range
    """
    result = df.copy()
    
    for column in columns:
        if column in df.columns:
            min_val = df[column].min()
            max_val = df[column].max()
            
            if max_val > min_val:
                result[f'{column}_normalized'] = (df[column] - min_val) / (max_val - min_val)
            else:
                result[f'{column}_normalized'] = 0
                
    return result

def is_consumption_anomaly(consumption_value):
    """
    Check if a consumption value is outside the expected range
    """
    min_val = Config.WATER_PARAMETERS['consumption']['min']
    max_val = Config.WATER_PARAMETERS['consumption']['max']
    
    return consumption_value < min_val or consumption_value > max_val