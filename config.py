import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration class"""

    # Database Configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'junta_jeru_backend')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')

    # Construct database URI
    DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # MLflow Configuration
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', './mlflow_runs')
    MLFLOW_EXPERIMENT_NAME = os.getenv(
        'MLFLOW_EXPERIMENT_NAME',
        'junta_analytics',
    )

    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    # Data Paths
    RAINFALL_DATA_PATH = os.getenv(
        'RAINFALL_DATA_PATH',
        './data/raw/rainfall_ecuador.csv',
    )
    PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH', './data/processed/')
    MODELS_PATH = os.getenv('MODELS_PATH', './data/models/')
    
    # Model Paths
    ANOMALY_DETECTOR_MODEL_PATH = os.path.join(MODELS_PATH, 'anomaly_detector_v2.joblib')

    # Ensure directories exist
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(MLFLOW_TRACKING_URI, exist_ok=True)
