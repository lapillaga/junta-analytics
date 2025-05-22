from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import json

from utils.database import get_meter_readings, get_aggregate_consumption_by_dekad
from utils.data_processing import (
    load_rainfall_data, 
    aggregate_rainfall_by_dekad,
    merge_consumption_and_rainfall,
    is_consumption_anomaly
)
from utils.visualization import (
    create_consumption_rainfall_chart,
    create_consumption_range_chart,
    create_correlation_heatmap
)
from models.prediction import (
    detect_anomalies,
    predict_single_consumption,
    forecast_consumption
)

# Create blueprint
main = Blueprint('main', __name__)

@main.route('/')
def index():
    """Main dashboard page"""
    # Get consumption data
    consumption_data = get_aggregate_consumption_by_dekad()
    
    # Load rainfall data
    rainfall_data = load_rainfall_data()
    rainfall_by_dekad = aggregate_rainfall_by_dekad(rainfall_data)
    
    # Merge data
    if not consumption_data.empty and not rainfall_by_dekad.empty:
        merged_data = merge_consumption_and_rainfall(consumption_data, rainfall_by_dekad)
        
        # Detect anomalies
        merged_data = detect_anomalies(merged_data)
        
        # Generate forecasts
        forecast_data = forecast_consumption(merged_data)
        
        # Create charts
        consumption_rainfall_chart = create_consumption_rainfall_chart(merged_data)
        consumption_range_chart = create_consumption_range_chart(
            merged_data, 
            forecast_data
        )
        correlation_chart = create_correlation_heatmap(merged_data)
    else:
        consumption_rainfall_chart = '{}'
        consumption_range_chart = '{}'
        correlation_chart = '{}'
    
    return render_template(
        'index.html',
        consumption_rainfall_chart=consumption_rainfall_chart,
        consumption_range_chart=consumption_range_chart,
        correlation_chart=correlation_chart
    )

@main.route('/data')
def data_page():
    """Data exploration page"""
    # Get meter readings
    readings = get_meter_readings()
    
    # Load rainfall data
    rainfall_data = load_rainfall_data()
    
    readings_sample = readings.head(100).to_dict('records') if not readings.empty else []
    rainfall_sample = rainfall_data.head(100).to_dict('records') if not rainfall_data.empty else []
    
    return render_template(
        'data.html',
        readings=readings_sample,
        rainfall_data=rainfall_sample
    )

@main.route('/check-consumption', methods=['GET', 'POST'])
def check_consumption():
    """Form to check if a consumption value is anomalous"""
    result = None
    consumption = None
    rainfall = None
    
    if request.method == 'POST':
        try:
            consumption = float(request.form.get('consumption', 0))
            rainfall = float(request.form.get('rainfall', 0))
            
            # Predict if consumption is anomalous
            result = predict_single_consumption(consumption, rainfall)
            
        except ValueError:
            flash('Please enter valid numbers for consumption and rainfall', 'error')
            
    return render_template(
        'check_consumption.html',
        result=result,
        consumption=consumption,
        rainfall=rainfall
    )

@main.route('/api/meter-readings')
def api_meter_readings():
    """API endpoint for meter readings"""
    readings = get_meter_readings()
    return jsonify(readings.to_dict('records') if not readings.empty else [])

@main.route('/api/consumption-rainfall')
def api_consumption_rainfall():
    """API endpoint for consumption and rainfall data"""
    # Get consumption data
    consumption_data = get_aggregate_consumption_by_dekad()
    
    # Load rainfall data
    rainfall_data = load_rainfall_data()
    rainfall_by_dekad = aggregate_rainfall_by_dekad(rainfall_data)
    
    # Merge data
    if not consumption_data.empty and not rainfall_by_dekad.empty:
        merged_data = merge_consumption_and_rainfall(consumption_data, rainfall_by_dekad)
        merged_data = detect_anomalies(merged_data)
        return jsonify(merged_data.to_dict('records'))
    
    return jsonify([])

@main.route('/api/forecast')
def api_forecast():
    """API endpoint for consumption forecasts"""
    # Get consumption data
    consumption_data = get_aggregate_consumption_by_dekad()
    
    # Load rainfall data
    rainfall_data = load_rainfall_data()
    rainfall_by_dekad = aggregate_rainfall_by_dekad(rainfall_data)
    
    # Merge and generate forecasts
    if not consumption_data.empty and not rainfall_by_dekad.empty:
        merged_data = merge_consumption_and_rainfall(consumption_data, rainfall_by_dekad)
        forecast_data = forecast_consumption(merged_data)
        return jsonify(forecast_data.to_dict('records') if not forecast_data.empty else [])
    
    return jsonify([])

@main.route('/api/check-consumption', methods=['POST'])
def api_check_consumption():
    """API endpoint to check if a consumption value is anomalous"""
    data = request.json
    
    consumption = data.get('consumption')
    rainfall = data.get('rainfall')
    
    if consumption is None:
        return jsonify({'error': 'Consumption value is required'}), 400
    
    try:
        consumption = float(consumption)
        rainfall = float(rainfall) if rainfall is not None else None
        
        result = predict_single_consumption(consumption, rainfall)
        return jsonify(result)
        
    except ValueError:
        return jsonify({'error': 'Invalid consumption or rainfall value'}), 400