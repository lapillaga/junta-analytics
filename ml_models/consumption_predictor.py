import logging
import warnings
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ConsumptionPredictor:
    """Machine Learning model for predicting water consumption"""

    def __init__(self, model_type: str = 'random_forest',
                 random_state: int = 42):
        """
        Initialize consumption predictor

        Args:
            model_type: Type of model ('random_forest', 'linear_regression')
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False

        # Initialize model based on type
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            )
        elif model_type == "linear_regression":
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def prepare_features(
        self,
        consumption_data: pd.DataFrame,
        climate_data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Prepare features for consumption prediction

        Args:
            consumption_data: Historical consumption data
            climate_data: Climate data for additional features

        Returns:
            Prepared feature dataframe
        """
        df = consumption_data.copy()

        # Ensure datetime columns
        df["period_start"] = pd.to_datetime(df["period_start"])
        df["period_end"] = pd.to_datetime(df["period_end"])

        # Sort by meter and date
        df = df.sort_values(['water_meter_id', 'period_start'])

        # Basic temporal features
        df['month'] = df['period_start'].dt.month
        df['quarter'] = df['period_start'].dt.quarter
        df['year'] = df['period_start'].dt.year
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Lag features (previous consumption)
        df['prev_consumption_1'] = df.groupby('water_meter_id')[
            'total_consumed'].shift(1)
        df['prev_consumption_2'] = df.groupby('water_meter_id')[
            'total_consumed'].shift(2)
        df['prev_consumption_3'] = df.groupby('water_meter_id')[
            'total_consumed'].shift(3)

        # Rolling averages
        df['consumption_ma_3'] = df.groupby('water_meter_id')[
            'total_consumed'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['consumption_ma_6'] = df.groupby('water_meter_id')[
            'total_consumed'].transform(
            lambda x: x.rolling(window=6, min_periods=1).mean()
        )

        # Rolling standard deviations
        df['consumption_std_3'] = df.groupby('water_meter_id')[
            'total_consumed'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )

        # Meter-specific features
        meter_stats = df.groupby('water_meter_id')['total_consumed'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        meter_stats.columns = ['water_meter_id', 'meter_avg_historical',
                               'meter_std_historical', 'meter_reading_count']
        meter_stats['meter_std_historical'] = meter_stats[
            'meter_std_historical'].fillna(0)

        df = df.merge(meter_stats, on='water_meter_id', how='left')

        # Neighborhood features
        neighborhood_stats = df.groupby(['neighborhood_id', 'month'])[
            'total_consumed'].agg([
            'mean', 'std'
        ]).reset_index()
        neighborhood_stats.columns = ['neighborhood_id', 'month',
                                      'neighborhood_monthly_avg',
                                      'neighborhood_monthly_std']
        neighborhood_stats['neighborhood_monthly_std'] = neighborhood_stats[
            'neighborhood_monthly_std'].fillna(0)

        df = df.merge(neighborhood_stats, on=['neighborhood_id', 'month'],
                      how='left')

        # Climate features if available
        if climate_data is not None:
            # Create period column for merging
            df['period_month'] = df['period_start'].dt.to_period('M').astype(
                str)

            # Aggregate climate data monthly
            climate_monthly = climate_data.groupby('period_str').agg({
                'avg_rainfall': 'mean',
                'max_rainfall': 'max',
                'total_rainfall': 'sum'
            }).reset_index()
            climate_monthly.columns = ['period_month', 'avg_rainfall',
                                       'max_rainfall', 'total_rainfall']

            df = df.merge(climate_monthly, on='period_month', how='left')

            # Fill missing climate data
            df['avg_rainfall'] = df['avg_rainfall'].fillna(
                df['avg_rainfall'].median())
            df['max_rainfall'] = df['max_rainfall'].fillna(
                df['max_rainfall'].median())
            df['total_rainfall'] = df['total_rainfall'].fillna(
                df['total_rainfall'].median())

            # Climate lag features
            df['prev_rainfall_1'] = df['avg_rainfall'].shift(1)
            df['prev_rainfall_2'] = df['avg_rainfall'].shift(2)

            # Seasonal rainfall patterns
            seasonal_rainfall = df.groupby('month')[
                'avg_rainfall'].mean().reset_index()
            seasonal_rainfall.columns = ['month', 'seasonal_avg_rainfall']
            df = df.merge(seasonal_rainfall, on='month', how='left')

            df['rainfall_deviation'] = df['avg_rainfall'] - df[
                'seasonal_avg_rainfall']

        # Days billed normalization
        df['consumption_per_day'] = df['total_consumed'] / df[
            'days_billed'].clip(lower=1)

        # Growth rates
        df['consumption_growth'] = df.groupby('water_meter_id')[
            'total_consumed'].pct_change()
        df['consumption_growth'] = df['consumption_growth'].fillna(0)

        # Select features for modeling
        feature_columns = [
            'month_sin', 'month_cos', 'quarter', 'days_billed',
            'prev_consumption_1', 'prev_consumption_2', 'prev_consumption_3',
            'consumption_ma_3', 'consumption_ma_6', 'consumption_std_3',
            'meter_avg_historical', 'meter_std_historical',
            'meter_reading_count',
            'neighborhood_monthly_avg', 'neighborhood_monthly_std',
            'consumption_per_day', 'consumption_growth'
        ]

        # Add climate features if available
        if climate_data is not None:
            climate_features = [
                'avg_rainfall', 'max_rainfall', 'total_rainfall',
                'prev_rainfall_1', 'prev_rainfall_2', 'seasonal_avg_rainfall',
                'rainfall_deviation'
            ]
            feature_columns.extend(climate_features)

        # Ensure all features exist
        for col in feature_columns:
            if col not in df.columns:
                logger.warning(f"Feature {col} not found, creating with zeros")
                df[col] = 0

        # Fill remaining NaN values
        df[feature_columns] = df[feature_columns].fillna(0)

        self.feature_columns = feature_columns
        return df

    def train(self, training_data: pd.DataFrame,
              climate_data: pd.DataFrame = None,
              test_size: float = 0.2) -> Dict:
        """
        Train the consumption prediction model

        Args:
            training_data: Historical consumption data
            climate_data: Optional climate data
            test_size: Proportion of data for testing

        Returns:
            Training metrics and results
        """
        logger.info(
            f"Starting {self.model_type} training for consumption prediction...")

        # Prepare features
        features_df = self.prepare_features(training_data, climate_data)

        # Remove rows with missing target values
        features_df = features_df.dropna(subset=['total_consumed'])

        # Prepare X and y
        X = features_df[self.feature_columns].values
        y = features_df['total_consumed'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Scale features for linear regression
        if self.model_type == 'linear_regression':
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train,
                                    cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_mae_std = cv_scores.std()

        # Feature importance (for tree-based models)
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = {
                feature: importance
                for feature, importance in
                zip(self.feature_columns, self.model.feature_importances_)
            }
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(),
                                             key=lambda x: x[1], reverse=True))

        results = {
            'model_type': self.model_type,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': len(self.feature_columns),
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'cv_mae': cv_mae,
            'cv_mae_std': cv_mae_std,
            'feature_importance': feature_importance,
            'feature_names': self.feature_columns
        }

        logger.info(
            f"Training completed - Test R²: {test_r2:.3f}, Test MAE: {test_mae:.2f}")
        return results

    def predict(self, data: pd.DataFrame,
                climate_data: pd.DataFrame = None,
                return_confidence: bool = False) -> Dict:
        """
        Predict consumption for new data

        Args:
            data: New consumption data
            climate_data: Optional climate context data
            return_confidence: Whether to return prediction intervals

        Returns:
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare features
        features_df = self.prepare_features(data, climate_data)
        X = features_df[self.feature_columns].values

        # Scale features if needed
        if self.model_type == 'linear_regression':
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # Make predictions
        predictions = self.model.predict(X_scaled)

        results = {
            'predictions': predictions.tolist(),
            'total_samples': len(data),
            'mean_prediction': float(np.mean(predictions)),
            'std_prediction': float(np.std(predictions))
        }

        # Add confidence intervals for ensemble models
        if return_confidence and hasattr(self.model, 'estimators_'):
            # For Random Forest, use estimator predictions for intervals
            estimator_predictions = np.array([
                estimator.predict(X_scaled) for estimator in
                self.model.estimators_
            ])

            lower_bound = np.percentile(estimator_predictions, 5, axis=0)
            upper_bound = np.percentile(estimator_predictions, 95, axis=0)

            results.update({
                'lower_bound': lower_bound.tolist(),
                'upper_bound': upper_bound.tolist(),
                'confidence_interval': 90
            })

        return results

    def predict_next_period(self, meter_id: int, historical_data: pd.DataFrame,
                            climate_forecast: Dict = None) -> Dict:
        """
        Predict consumption for next period for a specific meter

        Args:
            meter_id: Water meter ID
            historical_data: Historical consumption data
            climate_forecast: Optional climate forecast data

        Returns:
            Prediction for next period
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Get meter's historical data
        meter_data = historical_data[
            historical_data['water_meter_id'] == meter_id].copy()

        if len(meter_data) == 0:
            return {
                'error': 'No historical data found for this meter',
                'meter_id': meter_id
            }

        # Get the latest period
        meter_data = meter_data.sort_values('period_start')
        latest_record = meter_data.iloc[-1].copy()

        # Create next period record
        next_period = latest_record.to_dict()

        # Update temporal features for next period
        next_period_start = pd.to_datetime(
            latest_record['period_start']) + pd.DateOffset(months=1)
        next_period['period_start'] = next_period_start
        next_period['month'] = next_period_start.month
        next_period['quarter'] = next_period_start.quarter
        next_period['year'] = next_period_start.year

        # Add climate forecast if available
        if climate_forecast:
            next_period.update(climate_forecast)

        # Create DataFrame for prediction
        next_period_df = pd.DataFrame([next_period])

        # Combine with historical data for proper feature calculation
        combined_data = pd.concat([meter_data, next_period_df],
                                  ignore_index=True)

        # Prepare features (this will calculate lag features correctly)
        features_df = self.prepare_features(combined_data)

        # Get only the last row (next period)
        next_features = features_df.tail(1)

        # Make prediction
        prediction_result = self.predict(next_features, return_confidence=True)

        return {
            'meter_id': meter_id,
            'predicted_consumption': prediction_result['predictions'][0],
            'prediction_date': next_period_start.strftime('%Y-%m-%d'),
            'confidence_interval': prediction_result.get(
                'confidence_interval'),
            'lower_bound': prediction_result.get('lower_bound', [None])[0],
            'upper_bound': prediction_result.get('upper_bound', [None])[0],
            'historical_average': float(meter_data['total_consumed'].mean()),
            'recent_trend': float(meter_data['total_consumed'].tail(3).mean())
        }

    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("No trained model to save")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'random_state': self.random_state
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from file"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.random_state = model_data['random_state']
        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")
