import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import mlflow
import mlflow.sklearn
import pandas as pd

from .anomaly_detector_v3 import AnomalyDetectorV3 as AnomalyDetector
from config import Config

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ML models with MLflow integration"""

    def __init__(self, mlflow_tracking_uri: str, experiment_name: str):
        """
        Initialize model manager

        Args:
            mlflow_tracking_uri: MLflow tracking URI
            experiment_name: MLflow experiment name
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name
        self.models = {}

        # Configure MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new MLflow experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name}")

            self.experiment_id = experiment_id
            mlflow.set_experiment(experiment_name)

        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            self.experiment_id = None

        # Load existing models
        self._load_models()

    def _load_models(self):
        """Load existing trained models from disk"""
        # Load AnomalyDetector
        print("Loading anomaly detector model...")
        try:
            if os.path.exists(Config.ANOMALY_DETECTOR_MODEL_PATH):
                anomaly_detector = AnomalyDetector()
                anomaly_detector.load(Config.ANOMALY_DETECTOR_MODEL_PATH)
                self.models['anomaly_detector'] = anomaly_detector
                logger.info("Loaded existing anomaly detector model")
            else:
                # Initialize empty model
                self.models['anomaly_detector'] = AnomalyDetector()
                logger.info("Initialized empty anomaly detector model")
        except Exception as e:
            logger.warning(f"Failed to load anomaly detector: {e}")
            self.models['anomaly_detector'] = AnomalyDetector()


    def train_anomaly_detector(
        self, 
        training_df: pd.DataFrame,
        climate_df: Optional[pd.DataFrame] = None,
        model_params: Optional[Dict] = None,
        use_synthetic: bool = True
    ) -> str:
        """
        Train anomaly detection model with MLflow tracking

        Args:
            training_df: Training data DataFrame
            climate_df: Optional climate data DataFrame
            model_params: Model parameters
            use_synthetic: Whether to use synthetic anomalies for training

        Returns:
            MLflow run ID
        """
        with mlflow.start_run(run_name="anomaly_detector_training") as run:
            try:
                # Initialize model
                params = model_params or {}
                anomaly_detector = AnomalyDetector(
                    contamination=params.get('contamination', 0.03),
                    random_state=params.get('random_state', 42)
                )

                # Log parameters
                mlflow.log_param("model_type", "anomaly_detector")
                mlflow.log_param("algorithm", "isolation_forest")
                mlflow.log_param("contamination", anomaly_detector.contamination)
                mlflow.log_param("random_state", anomaly_detector.random_state)
                mlflow.log_param("training_samples", len(training_df))
                mlflow.log_param("has_climate_data", climate_df is not None)
                mlflow.log_param("use_synthetic", use_synthetic)

                logger.info(f"Training anomaly detector with {len(training_df)} samples")
                logger.info(f"Training data columns: {list(training_df.columns)}")

                # Train model (V3 doesn't use climate_df or use_synthetic)
                metrics = anomaly_detector.train(training_df=training_df)

                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log feature names if available
                if anomaly_detector.feature_columns_:
                    mlflow.log_param("feature_names", ",".join(anomaly_detector.feature_columns_))
                    mlflow.log_param("num_features", len(anomaly_detector.feature_columns_))

                # Save model to configured path
                anomaly_detector.save(Config.ANOMALY_DETECTOR_MODEL_PATH)
                logger.info(f"Model saved to {Config.ANOMALY_DETECTOR_MODEL_PATH}")

                # Log model artifact in MLflow
                mlflow.log_artifact(Config.ANOMALY_DETECTOR_MODEL_PATH, "models")

                # Store model in memory
                self.models['anomaly_detector'] = anomaly_detector

                logger.info(f"Anomaly detector training completed. Run ID: {run.info.run_id}")
                return run.info.run_id

            except Exception as e:
                logger.error(f"Error training anomaly detector: {e}")
                mlflow.log_param("training_status", "failed")
                mlflow.log_param("error_message", str(e))
                raise


    def get_model(self, model_type: str) -> Any:
        """
        Get model from memory

        Args:
            model_type: Type of model to retrieve

        Returns:
            Model instance

        Raises:
            ValueError: If model type not found
        """
        logger.debug(f'Getting model of type: {model_type}')
        logger.debug(f'Models in memory: {list(self.models.keys())}')
        
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Train or load model first.")

        model = self.models[model_type]
        if not hasattr(model, 'is_trained') or not model.is_trained:
            raise ValueError(f"Model {model_type} is not trained yet.")

        return model

    def list_experiments(self) -> List[Dict]:
        """List all MLflow experiments"""
        try:
            experiments = mlflow.search_experiments()
            return [
                {
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'lifecycle_stage': exp.lifecycle_stage
                }
                for exp in experiments
            ]
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
            return []

    def list_runs(self, experiment_id: str = None, max_results: int = 10) -> List[Dict]:
        """
        List MLflow runs

        Args:
            experiment_id: Optional experiment ID to filter by
            max_results: Maximum number of results

        Returns:
            List of runs
        """
        try:
            exp_id = experiment_id or self.experiment_id
            if exp_id is None:
                return []
                
            runs = mlflow.search_runs(
                experiment_ids=[exp_id],
                max_results=max_results,
                order_by=["start_time DESC"]
            )

            return runs.to_dict('records') if not runs.empty else []

        except Exception as e:
            logger.error(f"Error listing runs: {e}")
            return []

    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        model_info = {}
        
        for model_type, model in self.models.items():
            info = {
                'is_trained': getattr(model, 'is_trained', False),
                'model_class': model.__class__.__name__
            }
            
            if hasattr(model, 'feature_columns_'):
                info['feature_columns'] = model.feature_columns_
                info['num_features'] = len(model.feature_columns_)
            
            if hasattr(model, 'contamination'):
                info['contamination'] = model.contamination
                
            if hasattr(model, 'score_threshold_'):
                info['score_threshold'] = model.score_threshold_
                
            model_info[model_type] = info
            
        return model_info

    def export_model_info(self, output_path: str):
        """
        Export model information to JSON file

        Args:
            output_path: Path to save the JSON file
        """
        try:
            model_info = {
                'experiment_name': self.experiment_name,
                'experiment_id': self.experiment_id,
                'models': self.get_model_info(),
                'recent_runs': self.list_runs(max_results=5),
                'exported_at': datetime.now().isoformat()
            }

            with open(output_path, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)

            logger.info(f"Model info exported to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting model info: {e}")
            raise