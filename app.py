import json
import logging
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from database.connection import DatabaseManager
from utils.data_processing import (
    DataProcessor,
    RainfallProcessor,
    ConsumptionAnomalyDetector,
)
from utils.visualization import VisualizationHelper
from ml_models.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
CORS(app)

# Global variables for data and models
db_manager = None
data_processor = None
viz_helper = None
model_manager = None
merged_data = None
consumption_data = None

def _to_serializable(obj):
    """Convierte tipos de NumPy a tipos nativos de Python."""
    if isinstance(obj, np.generic):       # bool_, int64, float64…
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Deja que json trate de serializar lo demás o lance error
    raise TypeError(f"{type(obj)} is not JSON serializable")


def initialize_app():
    """Initialize application components"""
    global db_manager, data_processor, viz_helper, model_manager
    global merged_data, consumption_data

    try:
        # Initialize database connection
        db_manager = DatabaseManager()
        logger.info("Database connection established")

        # Initialize rainfall processor
        rainfall_processor = RainfallProcessor(Config.RAINFALL_DATA_PATH)

        # Initialize data processor
        data_processor = DataProcessor(db_manager, rainfall_processor)

        # Initialize visualization helper
        viz_helper = VisualizationHelper()

        # Initialize model manager
        model_manager = ModelManager(
            Config.MLFLOW_TRACKING_URI,
            Config.MLFLOW_EXPERIMENT_NAME
        )

        # Load processed data if available
        try:
            merged_data_path = os.path.join(
                Config.PROCESSED_DATA_PATH,
                'merged_rainfall_consumption.csv',
            )
            consumption_data_path = os.path.join(
                Config.PROCESSED_DATA_PATH,
                'individual_consumption.csv',
            )

            if os.path.exists(merged_data_path):
                merged_data = pd.read_csv(merged_data_path)
                merged_data['period_dt'] = pd.to_datetime(
                    merged_data['period_dt'])
                logger.info(f"Loaded merged data: {len(merged_data)} records")

            if os.path.exists(consumption_data_path):
                consumption_data = pd.read_csv(consumption_data_path)
                consumption_data['created_at'] = pd.to_datetime(
                    consumption_data['created_at'])
                consumption_data['period_start'] = pd.to_datetime(
                    consumption_data['period_start'])
                logger.info(
                    f"Loaded consumption data: {len(consumption_data)} records")

        except Exception as e:
            logger.warning(f"Could not load processed data: {e}")
            logger.info("Please run the data integration notebook first")

        logger.info("Application initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        raise


@app.route('/')
def index():
    """Main dashboard route"""
    try:
        return render_template('dashboard.html')
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return render_template('error.html', error=str(e)), 500


@app.route('/api/dashboard-data')
def get_dashboard_data():
    """Get all data needed for dashboard"""
    try:
        if merged_data is None:
            return jsonify({
                'error': 'No processed data available. Please run data integration first.',
            }), 404

        # Get summary statistics
        stats = data_processor.get_summary_statistics()

        # Create visualizations
        timeline_chart = viz_helper.create_rainfall_consumption_timeline(
            merged_data)
        correlation_chart = viz_helper.create_correlation_scatter(merged_data)

        # Create KPI cards
        kpi_cards = viz_helper.create_kpi_cards(stats)

        # Sample prediction data (replace with actual predictions when available)
        prediction_chart = None
        if len(merged_data) >= 6:
            historical = merged_data.iloc[:-3]
            predictions = pd.DataFrame({
                'period_dt': merged_data.iloc[-3:]['period_dt'],
                'predicted_consumption': merged_data.iloc[-3:][
                                             'avg_consumption'] * 1.1
                # Sample predictions
            })
            prediction_chart = viz_helper.create_consumption_prediction_chart(
                historical, predictions)

        response_data = {
            'kpi_cards': kpi_cards,
            'charts': {
                'timeline': timeline_chart,
                'correlation': correlation_chart,
                'prediction': prediction_chart
            },
            'stats': stats,
            'last_updated': datetime.now().isoformat()
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-anomaly', methods=['POST'])
def detect_anomaly():
    """API endpoint for anomaly detection"""
    try:
        # Get request data
        data = request.get_json()

        required_fields = ['water_meter_id', 'current_reading',
                           'previous_reading']
        if not all(field in data for field in required_fields):
            return jsonify({
                'error': 'Missing required fields',
                'required': required_fields
            }), 400

        water_meter_id = int(data['water_meter_id'])
        current_reading = int(data['current_reading'])
        previous_reading = int(data['previous_reading'])
        days_billed = int(data.get('days_billed', 30))

        # Validate inputs
        if current_reading < previous_reading:
            return jsonify({
                'error': 'Current reading cannot be less than previous reading'
            }), 400

        # Check if we have a trained model
        try:
            print("Checking for anomaly detector model...")
            anomaly_detector = model_manager.get_model('anomaly_detector')
        except:
            print("Anomaly detector model not found")
            # If no trained model, use statistical approach
            if consumption_data is None:
                return jsonify({
                    'error': 'No historical data available for anomaly detection'
                }), 404

            # Use statistical anomaly detector
            stat_detector = ConsumptionAnomalyDetector(consumption_data)
            result = stat_detector.detect_anomaly(
                water_meter_id, current_reading, previous_reading, days_billed
            )
            cleaned = json.loads(json.dumps(result, default=_to_serializable))

            return jsonify(cleaned)

        # Use ML-based anomaly detector
        result = anomaly_detector.detect_single_reading(
            water_meter_id=water_meter_id,
            current_reading=current_reading,
            previous_reading=previous_reading,
            days_billed=days_billed,
            historical_data=consumption_data
        )

        cleaned = json.loads(json.dumps(result, default=_to_serializable))

        return jsonify(cleaned)

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        # Show traceback in logs
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/api/predict-consumption', methods=['POST'])
def predict_consumption():
    """API endpoint for consumption prediction"""
    try:
        data = request.get_json()

        if 'water_meter_id' not in data:
            return jsonify({'error': 'water_meter_id is required'}), 400

        water_meter_id = int(data['water_meter_id'])

        # Check if we have a trained model
        try:
            consumption_predictor = model_manager.get_model(
                'consumption_predictor')
        except:
            return jsonify({
                'error': 'No trained consumption prediction model available'
            }), 404

        if consumption_data is None:
            return jsonify({
                'error': 'No historical data available for prediction'
            }), 404

        # Get climate forecast if provided
        climate_forecast = data.get('climate_forecast', {})

        # Make prediction
        result = consumption_predictor.predict_next_period(
            meter_id=water_meter_id,
            historical_data=consumption_data,
            climate_forecast=climate_forecast
        )

        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error in consumption prediction: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/api/water-meters')
def get_water_meters():
    """Get list of water meters"""
    try:
        meters_data = db_manager.get_water_meters_data()

        # Convert to list of dictionaries
        meters_list = []
        for _, meter in meters_data.iterrows():
            meters_list.append({
                'id': meter['water_meter_id'],
                'number': meter['meter_number'],
                'neighborhood': meter['neighborhood_name'],
                'customer_name': meter['full_name'],
                'status': meter['meter_status']
            })

        return jsonify({
            'meters': meters_list,
            'total_count': len(meters_list)
        })

    except Exception as e:
        logger.error(f"Error getting water meters: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/neighborhoods')
def get_neighborhoods():
    """Get neighborhood statistics"""
    try:
        neighborhood_data = db_manager.get_neighborhoods_stats()

        neighborhoods_list = []
        for _, neighborhood in neighborhood_data.iterrows():
            neighborhoods_list.append({
                'id': neighborhood['neighborhood_id'],
                'name': neighborhood['neighborhood_name'],
                'total_meters': int(neighborhood['total_meters']),
                'total_customers': int(neighborhood['total_customers']),
                'avg_consumption': float(neighborhood['avg_consumption']),
                'total_consumption': float(neighborhood['total_consumption'])
            })

        return jsonify({
            'neighborhoods': neighborhoods_list,
            'total_count': len(neighborhoods_list)
        })

    except Exception as e:
        logger.error(f"Error getting neighborhoods: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/recent-anomalies')
def get_recent_anomalies():
    """Get recent anomaly detections"""
    try:
        # This would typically come from a database table storing anomaly results
        # For now, return sample data
        sample_anomalies = [
            {
                'id': 1,
                'water_meter_id': 123,
                'consumption': 85,
                'is_anomaly': True,
                'confidence': 0.82,
                'reason': 'High consumption detected',
                'detected_at': datetime.now().isoformat()
            },
            {
                'id': 2,
                'water_meter_id': 456,
                'consumption': 12,
                'is_anomaly': True,
                'confidence': 0.75,
                'reason': 'Low consumption detected',
                'detected_at': datetime.now().isoformat()
            }
        ]

        # Create anomaly chart
        anomaly_chart = viz_helper.create_anomaly_detection_chart(
            sample_anomalies)

        return jsonify({
            'anomalies': sample_anomalies,
            'chart': anomaly_chart,
            'total_count': len(sample_anomalies)
        })

    except Exception as e:
        logger.error(f"Error getting recent anomalies: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info')
def get_model_info():
    """Get information about trained models"""
    try:
        model_info = {
            'models': {},
            'mlflow_experiments': model_manager.list_experiments(),
            'recent_runs': model_manager.list_runs(max_results=5)
        }

        # Check which models are available
        for model_type in ['anomaly_detector', 'consumption_predictor']:
            try:
                model = model_manager.get_model(model_type)
                model_info['models'][model_type] = {
                    'is_trained': model.is_trained,
                    'available': True,
                    'features': getattr(model, 'feature_columns', [])
                }
            except:
                model_info['models'][model_type] = {
                    'is_trained': False,
                    'available': False,
                    'features': []
                }

        return jsonify(model_info)

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/train-models', methods=['POST'])
def train_models():
    """Trigger model training"""
    try:
        if consumption_data is None:
            return jsonify({
                'error': 'No training data available. Please run data integration first.'
            }), 404

        data = request.get_json() or {}
        model_types = data.get('model_types',
                               ['anomaly_detector', 'consumption_predictor'])

        training_results = {}

        # Train anomaly detector
        if 'anomaly_detector' in model_types:
            try:
                run_id = model_manager.train_anomaly_detector(
                    training_df=consumption_data,
                    climate_df=merged_data,
                    model_params=data.get('anomaly_params', {}),
                    use_synthetic=data.get('use_synthetic', True)
                )
                training_results['anomaly_detector'] = {
                    'status': 'success',
                    'run_id': run_id
                }
            except Exception as e:
                training_results['anomaly_detector'] = {
                    'status': 'failed',
                    'error': str(e)
                }

        # Train consumption predictor
        if 'consumption_predictor' in model_types:
            try:
                run_id = model_manager.train_consumption_predictor(
                    consumption_data,
                    climate_data=merged_data,
                    model_params=data.get('predictor_params', {})
                )
                training_results['consumption_predictor'] = {
                    'status': 'success',
                    'run_id': run_id
                }
            except Exception as e:
                training_results['consumption_predictor'] = {
                    'status': 'failed',
                    'error': str(e)
                }

        return jsonify({
            'message': 'Model training completed',
            'results': training_results
        })

    except Exception as e:
        logger.error(f"Error training models: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_status = "connected" if db_manager and db_manager.engine else "disconnected"

        # Check data availability
        data_status = {
            'merged_data': merged_data is not None,
            'consumption_data': consumption_data is not None
        }

        # Check model availability
        model_status = {}
        for model_type in ['anomaly_detector', 'consumption_predictor']:
            try:
                model = model_manager.get_model(model_type)
                model_status[model_type] = model.is_trained
            except:
                model_status[model_type] = False

        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': db_status,
            'data': data_status,
            'models': model_status
        })

    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# Template functions
@app.template_global()
def get_current_year():
    return datetime.now().year


if __name__ == '__main__':
    try:
        # Initialize the application
        initialize_app()

        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=8000,
            debug=Config.DEBUG
        )

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)
    finally:
        # Clean up
        if db_manager:
            db_manager.close()
