import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import mlflow
import mlflow.sklearn
import pandas as pd

from .anomaly_detector import AnomalyDetector
from .consumption_predictor import ConsumptionPredictor

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
                logger.info(
                    f"Created new MLflow experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(
                    f"Using existing MLflow experiment: {experiment_name}")

            self.experiment_id = experiment_id
            mlflow.set_experiment(experiment_name)

        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            self.experiment_id = None

    def train_anomaly_detector(self, training_data: pd.DataFrame,
                               climate_data: pd.DataFrame = None,
                               model_params: Dict = None) -> str:
        """
        Train anomaly detection model with MLflow tracking

        Args:
            training_data: Training data
            climate_data: Optional climate data
            model_params: Model parameters

        Returns:
            MLflow run ID
        """
        with mlflow.start_run(run_name="anomaly_detector_training") as run:
            try:
                # Initialize model
                params = model_params or {}
                anomaly_detector = AnomalyDetector(
                    contamination=params.get('contamination', 0.1),
                    random_state=params.get('random_state', 42)
                )

                # Log parameters
                mlflow.log_param("model_type", "anomaly_detector")
                mlflow.log_param("algorithm", "isolation_forest")
                mlflow.log_param("contamination",
                                 anomaly_detector.contamination)
                mlflow.log_param("random_state", anomaly_detector.random_state)
                mlflow.log_param("training_samples", len(training_data))
                mlflow.log_param("has_climate_data", climate_data is not None)

                # Train model
                results = anomaly_detector.train(
                    training_data,
                    climate_data,
                    use_synthetic_anomalies=True
                )

                # Log metrics
                mlflow.log_metric("detected_anomalies_train",
                                  results['detected_anomalies_train'])
                mlflow.log_metric("anomaly_rate_train",
                                  results['anomaly_rate_train'])
                mlflow.log_metric("features_used", results['features_used'])

                if 'precision' in results:
                    mlflow.log_metric("precision", results['precision'])
                    mlflow.log_metric("recall", results['recall'])
                    mlflow.log_metric("f1_score", results['f1_score'])
                    mlflow.log_metric("auc_score", results['auc_score'])

                # Log feature names
                mlflow.log_param("feature_names",
                                 ",".join(results['feature_names']))

                # Save and log model
                model_path = f"anomaly_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                anomaly_detector.save_model(model_path)
                mlflow.log_artifact(model_path, "models")

                # Log model with MLflow
                mlflow.sklearn.log_model(
                    anomaly_detector.model,
                    "isolation_forest_model",
                    signature=mlflow.models.infer_signature(
                        training_data[results['feature_names']].head()
                    )
                )

                # Store model in memory
                self.models['anomaly_detector'] = anomaly_detector

                logger.info(
                    f"Anomaly detector training completed. Run ID: {run.info.run_id}")
                return run.info.run_id

            except Exception as e:
                logger.error(f"Error training anomaly detector: {e}")
                mlflow.log_param("training_status", "failed")
                mlflow.log_param("error_message", str(e))
                raise

    def train_consumption_predictor(self, training_data: pd.DataFrame,
                                    climate_data: pd.DataFrame = None,
                                    model_params: Dict = None) -> str:
        """
        Train consumption prediction model with MLflow tracking

        Args:
            training_data: Training data
            climate_data: Optional climate data
            model_params: Model parameters

        Returns:
            MLflow run ID
        """
        with mlflow.start_run(
            run_name="consumption_predictor_training") as run:
            try:
                # Initialize model
                params = model_params or {}
                consumption_predictor = ConsumptionPredictor(
                    model_type=params.get('model_type', 'random_forest'),
                    random_state=params.get('random_state', 42)
                )

                # Log parameters
                mlflow.log_param("model_type", "consumption_predictor")
                mlflow.log_param("algorithm", consumption_predictor.model_type)
                mlflow.log_param("random_state",
                                 consumption_predictor.random_state)
                mlflow.log_param("training_samples", len(training_data))
                mlflow.log_param("has_climate_data", climate_data is not None)

                # Train model
                results = consumption_predictor.train(
                    training_data,
                    climate_data,
                    test_size=params.get('test_size', 0.2)
                )

                # Log metrics
                mlflow.log_metric("train_mae", results['train_mae'])
                mlflow.log_metric("train_rmse", results['train_rmse'])
                mlflow.log_metric("train_r2", results['train_r2'])
                mlflow.log_metric("test_mae", results['test_mae'])
                mlflow.log_metric("test_rmse", results['test_rmse'])
                mlflow.log_metric("test_r2", results['test_r2'])
                mlflow.log_metric("cv_mae", results['cv_mae'])
                mlflow.log_metric("cv_mae_std", results['cv_mae_std'])
                mlflow.log_metric("features_used", results['features_used'])

                # Log feature importance if available
                if results['feature_importance']:
                    # Log top 10 features
                    top_features = dict(
                        list(results['feature_importance'].items())[:10])
                    for feature, importance in top_features.items():
                        mlflow.log_metric(f"importance_{feature}", importance)

                # Log feature names
                mlflow.log_param("feature_names",
                                 ",".join(results['feature_names']))

                # Save and log model
                model_path = f"consumption_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                consumption_predictor.save_model(model_path)
                mlflow.log_artifact(model_path, "models")

                # Log model with MLflow
                mlflow.sklearn.log_model(
                    consumption_predictor.model,
                    "consumption_model",
                    signature=mlflow.models.infer_signature(
                        training_data[results['feature_names']].head()
                    )
                )

                # Store model in memory
                self.models['consumption_predictor'] = consumption_predictor

                logger.info(
                    f"Consumption predictor training completed. Run ID: {run.info.run_id}")
                return run.info.run_id

            except Exception as e:
                logger.error(f"Error training consumption predictor: {e}")
                mlflow.log_param("training_status", "failed")
                mlflow.log_param("error_message", str(e))
                raise

    def register_model(self, model_name: str, run_id: str,
                       stage: str = "Staging") -> str:
        """
        Register model in MLflow Model Registry

        Args:
            model_name: Name for the registered model
            run_id: MLflow run ID
            stage: Model stage (Staging, Production, etc.)

        Returns:
            Model version
        """
        try:
            # Register model
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(model_uri, model_name)

            # Transition to specified stage
            mlflow.transitions.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )

            logger.info(
                f"Model {model_name} version {model_version.version} registered in {stage}")
            return model_version.version

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def load_model(self, model_name: str, model_type: str,
                   stage: str = "Production") -> Any:
        """
        Load model from MLflow Model Registry

        Args:
            model_name: Registered model name
            model_type: Type of model ('anomaly_detector', 'consumption_predictor')
            stage: Model stage to load from

        Returns:
            Loaded model instance
        """
        try:
            # Load model from registry
            model_uri = f"models:/{model_name}/{stage}"
            loaded_model = mlflow.sklearn.load_model(model_uri)

            # Create appropriate model instance
            if model_type == 'anomaly_detector':
                model_instance = AnomalyDetector()
                model_instance.model = loaded_model
                model_instance.is_trained = True
            elif model_type == 'consumption_predictor':
                model_instance = ConsumptionPredictor()
                model_instance.model = loaded_model
                model_instance.is_trained = True
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Store in memory
            self.models[model_type] = model_instance

            logger.info(f"Model {model_name} loaded from {stage}")
            return model_instance

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_model(self, model_type: str) -> Any:
        """
        Get model from memory

        Args:
            model_type: Type of model to retrieve

        Returns:
            Model instance
        """
        if model_type not in self.models:
            raise ValueError(
                f"Model {model_type} not found. Train or load model first.")

        return self.models[model_type]

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

    def list_runs(self, experiment_id: str = None, max_results: int = 10) -> \
    List[Dict]:
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
            runs = mlflow.search_runs(
                experiment_ids=[exp_id],
                max_results=max_results,
                order_by=["metrics.test_r2 DESC", "start_time DESC"]
            )

            return runs.to_dict('records') if not runs.empty else []

        except Exception as e:
            logger.error(f"Error listing runs: {e}")
            return []

    def compare_models(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple model runs

        Args:
            run_ids: List of MLflow run IDs to compare

        Returns:
            Comparison dataframe
        """
        try:
            comparison_data = []

            for run_id in run_ids:
                run = mlflow.get_run(run_id)

                data = {
                    'run_id': run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                    'start_time': run.info.start_time,
                    'status': run.info.status
                }

                # Add parameters
                data.update(
                    {f"param_{k}": v for k, v in run.data.params.items()})

                # Add metrics
                data.update(
                    {f"metric_{k}": v for k, v in run.data.metrics.items()})

                comparison_data.append(data)

            return pd.DataFrame(comparison_data)

        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return pd.DataFrame()

    def get_best_model(self, model_type: str, metric: str = "test_r2") -> \
    Optional[str]:
        """
        Get best model run ID based on a metric

        Args:
            model_type: Type of model to search for
            metric: Metric to optimize for

        Returns:
            Best run ID or None
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"params.model_type = '{model_type}'",
                order_by=[f"metrics.{metric} DESC"],
                max_results=1
            )

            if not runs.empty:
                return runs.iloc[0]['run_id']

            return None

        except Exception as e:
            logger.error(f"Error finding best model: {e}")
            return None

    def cleanup_old_runs(self, keep_last_n: int = 10):
        """
        Clean up old MLflow runs (keep only the most recent)

        Args:
            keep_last_n: Number of recent runs to keep
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=["start_time DESC"]
            )

            if len(runs) > keep_last_n:
                old_runs = runs.iloc[keep_last_n:]

                for _, run in old_runs.iterrows():
                    mlflow.delete_run(run['run_id'])
                    logger.info(f"Deleted old run: {run['run_id']}")

                logger.info(f"Cleaned up {len(old_runs)} old runs")

        except Exception as e:
            logger.error(f"Error cleaning up runs: {e}")

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
                'models': {}
            }

            for model_type, model in self.models.items():
                if hasattr(model, 'feature_columns'):
                    model_info['models'][model_type] = {
                        'is_trained': model.is_trained,
                        'feature_columns': model.feature_columns,
                        'model_class': model.__class__.__name__
                    }

            # Get recent runs
            recent_runs = self.list_runs(max_results=5)
            model_info['recent_runs'] = recent_runs

            with open(output_path, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)

            logger.info(f"Model info exported to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting model info: {e}")
            raise
