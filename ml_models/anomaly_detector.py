import logging
import warnings
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Modelo de ML para detectar lecturas de consumo de agua anómalas.
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize anomaly detector

        Args:
            contamination: Expected proportion of anomalies in the dataset
            random_state: Random state for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.is_trained = False

    def prepare_features(
        self,
        data: pd.DataFrame,
        climate_data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Prepare features for anomaly detection

        Args:
            data: Consumption data
            climate_data: Optional climate context data

        Returns:
            Prepared feature dataframe
        """
        features_df = data.copy()

        # Basic consumption features
        features_df['consumption_per_day'] = (
            features_df['total_consumed'] / features_df['days_billed'].clip(
            lower=1)
        )

        # Mejora 2: Agregar transformación logarítmica para normalizar
        features_df['log_consumption'] = np.log1p(
            features_df['total_consumed'])
        features_df['log_consumption_per_day'] = np.log1p(
            features_df['consumption_per_day'])

        # Historical features per meter
        meter_stats = data.groupby('water_meter_id')['total_consumed'].agg([
            'mean', 'std', 'median', 'count', 'min', 'max',
            # Agregamos mediana
            lambda x: x.quantile(0.25),  # Q1
            lambda x: x.quantile(0.75),  # Q3
        ]).reset_index()

        meter_stats.columns = [
            'water_meter_id', 'meter_avg', 'meter_std', 'meter_median',
            'meter_count', 'meter_min', 'meter_max', 'meter_q1', 'meter_q3'
        ]

        # Mejora 3: IQR para detección más robusta
        meter_stats['meter_iqr'] = meter_stats['meter_q3'] - meter_stats[
            'meter_q1']

        # Handle missing std (when count = 1)
        meter_stats['meter_std'] = meter_stats['meter_std'].fillna(
            meter_stats['meter_median'] * 0.2  # Usar mediana en lugar de media
        )
        meter_stats['meter_iqr'] = meter_stats['meter_iqr'].fillna(
            meter_stats['meter_median'] * 0.5
        )

        features_df = features_df.merge(meter_stats, on='water_meter_id',
                                        how='left')

        # Mejora 4: Z-score robusto usando mediana
        features_df['consumption_robust_zscore'] = (
            (features_df['total_consumed'] - features_df['meter_median']) /
            (features_df['meter_std'] + 1)
        )

        # Mejora 5: Ratio desde mediana
        features_df['consumption_median_ratio'] = (
            features_df['total_consumed'] / (features_df['meter_median'] + 1)
        )

        # Consumption relative to meter history (mantener el original también)
        features_df['consumption_zscore'] = (
            (features_df['total_consumed'] - features_df['meter_avg']) /
            (features_df['meter_std'] + 1)
        )

        features_df['consumption_ratio'] = (
            features_df['total_consumed'] / (features_df['meter_avg'] + 1)
        )

        # Mejora 6: Percentil en la distribución del medidor
        features_df['meter_percentile'] = np.where(
            features_df['meter_iqr'] > 0,
            np.clip(
                (features_df['total_consumed'] - features_df['meter_q1']) /
                features_df['meter_iqr'],
                0, 1
            ),
            0.5
        )

        # Neighborhood context
        if 'neighborhood_id' in features_df.columns:
            neighborhood_stats = data.groupby('neighborhood_id')[
                'total_consumed'].agg([
                'mean', 'std', 'median'  # Agregar mediana
            ]).reset_index()
            neighborhood_stats.columns = [
                'neighborhood_id', 'neighborhood_avg', 'neighborhood_std',
                'neighborhood_median'
            ]
            neighborhood_stats['neighborhood_std'] = neighborhood_stats[
                'neighborhood_std'].fillna(
                neighborhood_stats['neighborhood_median'] * 0.3)

            features_df = features_df.merge(neighborhood_stats,
                                            on='neighborhood_id', how='left')

            # Neighborhood relative features
            features_df['neighborhood_zscore'] = (
                (features_df['total_consumed'] - features_df[
                    'neighborhood_median']) /
                (features_df['neighborhood_std'] + 1)
            )
        else:
            features_df['neighborhood_zscore'] = 0

        # Temporal features (mantener como estaba)
        features_df['period_start'] = pd.to_datetime(
            features_df['period_start'])
        features_df['month'] = features_df['period_start'].dt.month
        features_df['quarter'] = features_df['period_start'].dt.quarter
        features_df['month_sin'] = np.sin(
            2 * np.pi * features_df['month'] / 12)
        features_df['month_cos'] = np.cos(
            2 * np.pi * features_df['month'] / 12)

        # Climate context if available
        if climate_data is not None:
            features_df['period_month'] = features_df[
                'period_start'].dt.to_period('M').astype(str)
            climate_monthly = climate_data.groupby('period_str')[
                'avg_rainfall'].mean().reset_index()
            climate_monthly.columns = ['period_month', 'period_rainfall']

            features_df = features_df.merge(climate_monthly, on='period_month',
                                            how='left')
            features_df['period_rainfall'] = features_df[
                'period_rainfall'].fillna(
                features_df['period_rainfall'].median()
            )

            # Rainfall-consumption interaction
            features_df['rainfall_consumption_interaction'] = (
                features_df['consumption_robust_zscore'] * features_df[
                'period_rainfall'] / 100
            )

        # Mejora 7: Lista de características
        feature_columns = [
            # Características básicas
            'total_consumed', 'consumption_per_day',
            'log_consumption', 'log_consumption_per_day',

            # Características robustas
            'consumption_robust_zscore', 'consumption_median_ratio',
            'meter_percentile',  # Nuevas

            # Características originales
            'consumption_zscore', 'consumption_ratio',

            # Contexto
            'neighborhood_zscore', 'month_sin', 'month_cos',
            'days_billed', 'meter_count'
        ]

        if climate_data is not None:
            feature_columns.extend(
                ['period_rainfall', 'rainfall_consumption_interaction'])

        # Ensure all features exist
        for col in feature_columns:
            if col not in features_df.columns:
                logger.warning(f"Feature {col} not found, creating with zeros")
                features_df[col] = 0

        self.feature_columns = feature_columns
        return features_df[
            feature_columns + ['water_meter_id', 'measure_id']].fillna(0)

    def create_synthetic_anomalies(self, data: pd.DataFrame,
                                   anomaly_ratio: float = 0.03) -> Tuple[
        pd.DataFrame, np.ndarray]:
        """
        Create synthetic anomalies for training validation

        Args:
            data: Clean data
            anomaly_ratio: Proportion of synthetic anomalies to create

        Returns:
            Data with synthetic anomalies and labels
        """
        np.random.seed(self.random_state)
        data_with_anomalies = data.copy()

        # Number of anomalies to create
        n_anomalies = int(len(data) * anomaly_ratio)
        anomaly_indices = np.random.choice(len(data), n_anomalies,
                                           replace=False)

        # Create labels (0 = normal, 1 = anomaly)
        labels = np.zeros(len(data))
        labels[anomaly_indices] = 1

        for idx in anomaly_indices:
            anomaly_type = np.random.choice([
                'meter_malfunction',
                'leak',
                'meter_fraud',
                'billing_error'
            ], p=[0.4, 0.3, 0.2, 0.1])

            if anomaly_type == 'meter_malfunction':
                # Medidor que cuenta mal o se atasca
                if np.random.random() < 0.5:
                    # Muy bajo (medidor atascado)
                    multiplier = np.random.uniform(0.05, 0.2)
                else:
                    # Muy alto (medidor roto)
                    multiplier = np.random.uniform(4, 8)

                data_with_anomalies.loc[idx, 'total_consumed'] *= multiplier
                data_with_anomalies.loc[idx, 'log_consumption'] = np.log1p(
                    data_with_anomalies.loc[idx, 'total_consumed']
                )

            elif anomaly_type == 'leak':
                # Fuga = consumo consistentemente alto
                multiplier = np.random.uniform(2.5, 5.0)
                data_with_anomalies.loc[idx, 'total_consumed'] *= multiplier
                data_with_anomalies.loc[
                    idx, 'consumption_median_ratio'] *= multiplier

            elif anomaly_type == 'meter_fraud':
                # Manipulación = consumo sospechosamente bajo
                multiplier = np.random.uniform(0.1, 0.3)
                data_with_anomalies.loc[idx, 'total_consumed'] *= multiplier
                data_with_anomalies.loc[
                    idx, 'meter_percentile'] = np.random.uniform(0, 0.1)

            elif anomaly_type == 'billing_error':
                # Error en días facturados o lectura
                if np.random.random() < 0.5:
                    data_with_anomalies.loc[
                        idx, 'days_billed'] *= np.random.uniform(0.3, 3.0)
                else:
                    data_with_anomalies.loc[
                        idx, 'consumption_per_day'] *= np.random.uniform(0.2,
                                                                         4.0)

        logger.info(
            f"Created {n_anomalies} realistic synthetic anomalies ({anomaly_ratio * 100:.1f}%)")
        return data_with_anomalies, labels

    def train(self, training_data: pd.DataFrame,
              climate_data: pd.DataFrame = None,
              use_synthetic_anomalies: bool = True) -> Dict:
        """
        Train the anomaly detection model

        Args:
            training_data: Historical consumption data
            climate_data: Optional climate data for context
            use_synthetic_anomalies: Whether to create synthetic anomalies for validation

        Returns:
            Training metrics and results
        """
        logger.info("Starting anomaly detector training...")

        # Prepare features
        features_df = self.prepare_features(training_data, climate_data)

        if use_synthetic_anomalies:
            # Create synthetic anomalies for validation
            features_with_anomalies, true_labels = self.create_synthetic_anomalies(
                features_df, anomaly_ratio=0.03
            )
            X = features_with_anomalies[self.feature_columns].values

            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, true_labels, test_size=0.3, random_state=self.random_state,
                stratify=true_labels
            )
        else:
            X = features_df[self.feature_columns].values
            X_train = X

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
            max_samples=0.8,
            max_features=0.9,
            bootstrap=False,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled)
        self.is_trained = True

        # Calculate training metrics
        train_predictions = self.model.predict(X_train_scaled)
        train_anomalies = (train_predictions == -1).sum()
        train_anomaly_rate = train_anomalies / len(train_predictions)

        results = {
            'model_type': 'IsolationForest',
            'training_samples': len(X_train),
            'features_used': len(self.feature_columns),
            'contamination_rate': self.contamination,
            'detected_anomalies_train': train_anomalies,
            'anomaly_rate_train': train_anomaly_rate,
            'feature_names': self.feature_columns
        }

        # Validation metrics if synthetic anomalies were used
        if use_synthetic_anomalies:
            X_test_scaled = self.scaler.transform(X_test)
            test_predictions = self.model.predict(X_test_scaled)

            # Convert to binary labels (1 = anomaly, 0 = normal)
            test_predictions_binary = (test_predictions == -1).astype(int)

            # Calculate metrics
            precision = precision_score(y_test, test_predictions_binary)
            recall = recall_score(y_test, test_predictions_binary)
            f1 = f1_score(y_test, test_predictions_binary)

            # Anomaly scores for AUC
            anomaly_scores = self.model.decision_function(X_test_scaled)
            auc = roc_auc_score(y_test, -anomaly_scores)

            results.update({
                'validation_samples': len(X_test),
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc,
                'confusion_matrix': confusion_matrix(y_test,
                                                     test_predictions_binary).tolist()
            })

            logger.info(
                f"Validation metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

        logger.info(
            f"Training completed. Detected {train_anomalies} anomalies ({train_anomaly_rate:.1%})")
        return results

    def predict(self, data: pd.DataFrame,
                climate_data: pd.DataFrame = None) -> Dict:
        """Predict anomalies in new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare features
        features_df = self.prepare_features(data, climate_data)
        X = features_df[self.feature_columns].values

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict anomalies
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.decision_function(X_scaled)

        # Convert predictions to binary
        is_anomaly = (predictions == -1)

        # Calculate confidence scores (normalized)
        confidence_scores = np.abs(anomaly_scores)
        confidence_scores = (confidence_scores - confidence_scores.min()) / (
            confidence_scores.max() - confidence_scores.min() + 1e-8
        )

        results = {
            'predictions': is_anomaly.tolist(),
            'confidence_scores': confidence_scores.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'total_samples': len(data),
            'detected_anomalies': is_anomaly.sum(),
            'anomaly_rate': is_anomaly.mean()
        }

        return results

    def detect_single_reading(self, water_meter_id: int, current_reading: int,
                              previous_reading: int, days_billed: int = 30,
                              historical_data: pd.DataFrame = None) -> Dict:
        """Detect anomaly for a single reading"""
        if not self.is_trained:
            raise ValueError(
                "Model must be trained before detecting anomalies")

        # Calculate consumption
        consumption = max(0, current_reading - previous_reading)
        consumption_per_day = consumption / max(1, days_billed)

        # Create single record dataframe
        single_record = pd.DataFrame({
            'water_meter_id': [water_meter_id],
            'total_consumed': [consumption],
            'days_billed': [days_billed],
            'period_start': [pd.Timestamp.now()],
            'neighborhood_id': ['Unknown']
        })

        # Add historical context if available
        if historical_data is not None:
            meter_data = historical_data[
                historical_data['water_meter_id'] == water_meter_id]
            if len(meter_data) > 0:
                single_record['neighborhood_id'] = \
                meter_data['neighborhood_id'].iloc[0]

        # Prepare features
        if historical_data is not None:
            combined_data = pd.concat([historical_data, single_record],
                                      ignore_index=True)
            features_df = self.prepare_features(combined_data)
            features_df = features_df.tail(1)
        else:
            features_df = self.prepare_features(single_record)

        # Make prediction
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)[0]
        anomaly_score = self.model.decision_function(X_scaled)[0]

        is_anomaly = (prediction == -1)
        confidence = min(0.95, abs(anomaly_score) / 5.0)

        # Determine reason based on features
        reason = "Normal consumption pattern"
        recommendation = "No action required"

        if is_anomaly:
            meter_median = features_df.get('meter_median', consumption)
            consumption_ratio = consumption / (
                    meter_median + 1) if 'meter_median' in features_df.columns else 1

            if consumption_ratio > 3:
                reason = "Extremely high consumption detected"
                recommendation = "Check for leaks or meter malfunction"
            elif consumption_ratio < 0.1:
                reason = "Extremely low consumption detected"
                recommendation = "Verify meter reading accuracy"
            elif consumption_per_day > 50:
                reason = "High daily consumption rate"
                recommendation = "Monitor for unusual usage patterns"
            else:
                reason = "Unusual consumption pattern detected"
                recommendation = "Manual verification recommended"

        return {
            'is_anomaly': bool(is_anomaly),
            'confidence': round(confidence, 3),
            'anomaly_score': round(float(anomaly_score), 3),
            'reason': reason,
            'recommendation': recommendation,
            'consumption': consumption,
            'consumption_per_day': round(consumption_per_day, 2),
            'analysis_features': {
                col: round(float(features_df[col].iloc[0]), 3)
                for col in self.feature_columns if col in features_df.columns
            }
        }

    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("No trained model to save")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'contamination': self.contamination,
            'random_state': self.random_state,
            'model_version': '1.1_improved',
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from file"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.contamination = model_data['contamination']
        self.random_state = model_data['random_state']
        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")
