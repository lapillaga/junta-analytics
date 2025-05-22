import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File paths
    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASEDIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')
    MODELS_DIR = os.path.join(BASEDIR, 'models')
    
    # Model parameters
    ANOMALY_DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, 'anomaly_detector.pkl')
    FORECAST_MODEL_PATH = os.path.join(MODELS_DIR, 'forecast_model.pkl')
    
    # Upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    # Chart settings
    CHART_COLORS = {
        'primary': '#3498db',
        'secondary': '#2ecc71',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'info': '#1abc9c'
    }
    
    # Default parameters
    DEFAULT_RAINFALL_CSV_PATH = os.path.join(RAW_DATA_DIR, 'rainfall_data.csv')
    
    # Normal ranges for water parameters
    WATER_PARAMETERS = {
        'consumption': {
            'min': 5,  # minimum expected consumption in cubic meters
            'max': 50   # maximum expected consumption in cubic meters
        }
    }