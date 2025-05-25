import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RainfallProcessor:
    """Processes rainfall data from CSV files"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None

    def load_data(self) -> pd.DataFrame:
        """Load rainfall data from CSV"""
        try:
            # Read the CSV file
            self.raw_data = pd.read_csv(self.file_path)
            logger.info(f"Loaded rainfall data: {len(self.raw_data)} records")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error loading rainfall data: {e}")
            raise

    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess rainfall data"""
        if self.raw_data is None:
            self.load_data()

        df = self.raw_data.copy()

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Create year-month column for aggregation
        df['year_month'] = df['date'].dt.to_period('M')

        # Handle missing values
        rainfall_columns = ['rfh', 'rfh_avg', 'r1h', 'r1h_avg', 'r3h',
                            'r3h_avg']
        for col in rainfall_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())

        # Create rainfall intensity categories
        df['rainfall_intensity'] = pd.cut(
            df['rfh_avg'] if 'rfh_avg' in df.columns else df['rfh'],
            bins=[0, 10, 50, 100, np.inf],
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        self.processed_data = df
        logger.info("Rainfall data cleaned successfully")
        return df

    def aggregate_monthly(self) -> pd.DataFrame:
        """Aggregate rainfall data by month"""
        if self.processed_data is None:
            self.clean_data()

        df = self.processed_data

        # Define aggregation functions based on your CSV structure
        agg_dict = {
            'rfh': ['mean', 'sum', 'max', 'min', 'std'],
            'rfh_avg': ['mean', 'sum', 'max', 'min', 'std'],
            'r1h': ['mean', 'sum', 'max', 'min'],
            'r1h_avg': ['mean', 'sum', 'max', 'min'],
            'r3h': ['mean', 'sum', 'max', 'min'],
            'r3h_avg': ['mean', 'sum', 'max', 'min']
        }

        # Add other columns if they exist
        if 'n_pixels' in df.columns:
            agg_dict['n_pixels'] = 'mean'
        if 'rfq' in df.columns:
            agg_dict['rfq'] = 'mean'
        if 'r1q' in df.columns:
            agg_dict['r1q'] = 'mean'
        if 'r3q' in df.columns:
            agg_dict['r3q'] = 'mean'

        # Aggregate by year-month
        monthly_data = df.groupby('year_month').agg(agg_dict).reset_index()

        # Flatten column names CORRECTLY
        flattened_columns = []
        for col in monthly_data.columns:
            if isinstance(col, tuple) and len(col) > 1 and col[1]:
                flattened_columns.append(f"{col[0]}_{col[1]}")
            else:
                flattened_columns.append(
                    col[0] if isinstance(col, tuple) else col)

        monthly_data.columns = flattened_columns

        # Create standard columns using rfh_avg as primary
        monthly_data['avg_rainfall'] = monthly_data['rfh_avg_mean']
        monthly_data['total_rainfall'] = monthly_data['rfh_avg_sum']
        monthly_data['max_rainfall'] = monthly_data['rfh_avg_max']
        monthly_data['min_rainfall'] = monthly_data['rfh_avg_min']
        monthly_data['rainfall_std'] = monthly_data['rfh_avg_std']

        # Convert period to string for merging (NO renaming to 'period')
        monthly_data['period_str'] = monthly_data['year_month'].astype(str)

        # Create categorical rainfall intensity
        monthly_data['monthly_intensity'] = pd.cut(
            monthly_data['avg_rainfall'],
            bins=[0, 20, 60, 120, np.inf],
            labels=['Dry', 'Normal', 'Wet', 'Very Wet']
        )

        logger.info(
            f"Monthly aggregation completed: {len(monthly_data)} periods")
        return monthly_data


class DataProcessor:
    """Processes and merges water consumption and rainfall data"""

    def __init__(self, db_manager, rainfall_processor: RainfallProcessor):
        self.db_manager = db_manager
        self.rainfall_processor = rainfall_processor
        self.consumption_data = None
        self.rainfall_data = None
        self.merged_data = None

    def load_consumption_data(self) -> pd.DataFrame:
        """Load and process consumption data from database"""
        try:
            # Get measures data
            measures_df = self.db_manager.get_measures_data()

            # Convert dates
            measures_df['created_at'] = pd.to_datetime(
                measures_df['created_at'])
            measures_df['period_start'] = pd.to_datetime(
                measures_df['period_start'])
            measures_df['period_end'] = pd.to_datetime(
                measures_df['period_end'])

            # Create period column for merging
            measures_df['period'] = measures_df['period_start'].dt.to_period(
                'M')
            measures_df['period_str'] = measures_df['period'].astype(str)

            # Handle missing consumption values
            measures_df['total_consumed'] = measures_df[
                'total_consumed'].fillna(0)
            measures_df['total_consumed'] = measures_df['total_consumed'].clip(
                lower=0)

            # Calculate derived features
            measures_df['consumption_per_day'] = (
                measures_df['total_consumed'] / measures_df['days_billed']
            ).fillna(0)

            self.consumption_data = measures_df
            logger.info(f"Loaded consumption data: {len(measures_df)} records")
            return measures_df

        except Exception as e:
            logger.error(f"Error loading consumption data: {e}")
            raise

    def aggregate_consumption_monthly(self) -> pd.DataFrame:
        """Aggregate consumption data by month"""
        if self.consumption_data is None:
            self.load_consumption_data()

        df = self.consumption_data

        # Aggregate consumption by period
        monthly_consumption = df.groupby(
            ['period_str', 'neighborhood_id', 'neighborhood_name']).agg({
            'total_consumed': ['sum', 'mean', 'count', 'std'],
            'consumption_per_day': ['mean'],
            'water_meter_id': 'nunique',
            'excess_consumption': 'sum'
        }).reset_index()

        # Flatten column names
        monthly_consumption.columns = [
            '_'.join(col).strip() if col[1] else col[0]
            for col in monthly_consumption.columns.values
        ]

        # Rename columns for clarity
        monthly_consumption.rename(columns={
            'period_str_': 'period',
            'neighborhood_id_': 'neighborhood_id',
            'neighborhood_name_': 'neighborhood_name',
            'total_consumed_sum': 'total_consumption',
            'total_consumed_mean': 'avg_consumption',
            'total_consumed_count': 'num_measures',
            'total_consumed_std': 'consumption_std',
            'consumption_per_day_mean': 'avg_daily_consumption',
            'water_meter_id_nunique': 'active_meters',
            'excess_consumption_sum': 'total_excess'
        }, inplace=True)

        # Fill NaN values
        monthly_consumption['consumption_std'] = monthly_consumption[
            'consumption_std'].fillna(0)

        # Overall monthly aggregation (all neighborhoods combined)
        overall_monthly = df.groupby('period_str').agg({
            'total_consumed': ['sum', 'mean', 'count', 'std'],
            'consumption_per_day': ['mean'],
            'water_meter_id': 'nunique',
            'excess_consumption': 'sum'
        }).reset_index()

        # Flatten column names
        overall_monthly.columns = [
            '_'.join(col).strip() if col[1] else col[0]
            for col in overall_monthly.columns.values
        ]

        # Rename columns
        overall_monthly.rename(columns={
            'period_str_': 'period',
            'total_consumed_sum': 'total_consumption',
            'total_consumed_mean': 'avg_consumption',
            'total_consumed_count': 'num_measures',
            'total_consumed_std': 'consumption_std',
            'consumption_per_day_mean': 'avg_daily_consumption',
            'water_meter_id_nunique': 'active_meters',
            'excess_consumption_sum': 'total_excess'
        }, inplace=True)

        # Fill NaN values
        overall_monthly['consumption_std'] = overall_monthly[
            'consumption_std'].fillna(0)

        logger.info(
            f"Monthly consumption aggregation completed: {len(overall_monthly)} periods")
        return overall_monthly, monthly_consumption

    def merge_data(self) -> pd.DataFrame:
        """Merge rainfall and consumption data temporally"""
        try:
            # Load and process rainfall data
            self.rainfall_data = self.rainfall_processor.aggregate_monthly()

            # Load and process consumption data
            overall_consumption, neighborhood_consumption = self.aggregate_consumption_monthly()

            # Merge on period
            merged = pd.merge(
                overall_consumption,
                self.rainfall_data,
                on='period_str',
                how='inner',
                suffixes=('_consumption', '_rainfall')
            )

            # Calculate derived features
            merged['rainfall_consumption_ratio'] = (
                merged['avg_rainfall'] / (merged['avg_consumption'] + 1)
            )

            merged['consumption_efficiency'] = (
                merged['total_consumption'] / merged['active_meters']
            )

            # Create seasonal indicators
            merged['period_dt'] = pd.to_datetime(merged['period_str'])
            merged['month'] = merged['period_dt'].dt.month
            merged['quarter'] = merged['period_dt'].dt.quarter
            merged['season'] = merged['quarter'].map({
                1: 'Dry Season',
                2: 'Rainy Season',
                3: 'Rainy Season',
                4: 'Dry Season'
            })

            # Create consumption categories
            merged['consumption_category'] = pd.cut(
                merged['avg_consumption'],
                bins=merged['avg_consumption'].quantile(
                    [0, 0.33, 0.66, 1]).values,
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )

            # Lag features for time series analysis
            merged = merged.sort_values('period_dt')
            merged['prev_month_consumption'] = merged['avg_consumption'].shift(
                1)
            merged['prev_month_rainfall'] = merged['avg_rainfall'].shift(1)

            # Calculate growth rates
            merged['consumption_growth'] = (
                (merged['avg_consumption'] - merged[
                    'prev_month_consumption']) /
                (merged['prev_month_consumption'] + 1) * 100
            )

            merged['rainfall_change'] = (
                merged['avg_rainfall'] - merged['prev_month_rainfall']
            )

            self.merged_data = merged
            logger.info(f"Data merge completed: {len(merged)} periods")
            return merged

        except Exception as e:
            logger.error(f"Error merging data: {e}")
            raise

    def create_features_for_ml(self) -> pd.DataFrame:
        """Create additional features for machine learning models"""
        if self.merged_data is None:
            self.merge_data()

        df = self.merged_data.copy()

        # Rolling averages
        df['consumption_ma_3'] = df['avg_consumption'].rolling(window=3).mean()
        df['rainfall_ma_3'] = df['avg_rainfall'].rolling(window=3).mean()

        # Seasonal decomposition features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Interaction features
        df['rainfall_season_interaction'] = (
            df['avg_rainfall'] * df['month_sin']
        )

        # Anomaly indicators (simple statistical approach)
        consumption_mean = df['avg_consumption'].mean()
        consumption_std = df['avg_consumption'].std()
        df['consumption_zscore'] = (
            (df['avg_consumption'] - consumption_mean) / consumption_std
        )

        rainfall_mean = df['avg_rainfall'].mean()
        rainfall_std = df['avg_rainfall'].std()
        df['rainfall_zscore'] = (
            (df['avg_rainfall'] - rainfall_mean) / rainfall_std
        )

        # Fill remaining NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')

        logger.info("ML features created successfully")
        return df

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of the merged dataset"""
        if self.merged_data is None:
            self.merge_data()

        df = self.merged_data

        stats = {
            'total_periods': len(df),
            'date_range': {
                'start': df['period_dt'].min().strftime('%Y-%m'),
                'end': df['period_dt'].max().strftime('%Y-%m')
            },
            'consumption_stats': {
                'mean': df['avg_consumption'].mean(),
                'std': df['avg_consumption'].std(),
                'min': df['avg_consumption'].min(),
                'max': df['avg_consumption'].max()
            },
            'rainfall_stats': {
                'mean': df['avg_rainfall'].mean(),
                'std': df['avg_rainfall'].std(),
                'min': df['avg_rainfall'].min(),
                'max': df['avg_rainfall'].max()
            },
            'correlation': {
                'rainfall_consumption': df['avg_rainfall'].corr(
                    df['avg_consumption'])
            },
            'seasonal_patterns': df.groupby('season')[
                ['avg_consumption', 'avg_rainfall']].mean().to_dict()
        }

        return stats

    def save_processed_data(self, output_path: str):
        """Save processed data to files"""
        if self.merged_data is None:
            self.merge_data()

        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)

        # Save main merged dataset
        self.merged_data.to_csv(
            output_path / 'merged_rainfall_consumption.csv', index=False)

        # Save features for ML
        ml_features = self.create_features_for_ml()
        ml_features.to_csv(output_path / 'ml_features.csv', index=False)

        # Save individual consumption data for anomaly detection
        if self.consumption_data is not None:
            self.consumption_data.to_csv(
                output_path / 'individual_consumption.csv', index=False)

        # Save summary statistics
        stats = self.get_summary_statistics()
        import json
        with open(output_path / 'summary_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info(f"Processed data saved to {output_path}")


class ConsumptionAnomalyDetector:
    """Detect anomalies in individual consumption readings"""

    def __init__(self, consumption_data: pd.DataFrame):
        self.data = consumption_data
        self.meter_profiles = None
        self._build_meter_profiles()

    def _build_meter_profiles(self):
        """Build consumption profiles for each water meter"""
        profiles = self.data.groupby('water_meter_id').agg({
            'total_consumed': ['mean', 'std', 'count', 'min', 'max'],
            'consumption_per_day': ['mean', 'std'],
            'neighborhood_id': 'first'
        }).reset_index()

        # Flatten column names
        profiles.columns = [
            '_'.join(col).strip() if col[1] else col[0]
            for col in profiles.columns.values
        ]

        # Rename for clarity
        profiles.rename(columns={
            'water_meter_id_': 'water_meter_id',
            'total_consumed_mean': 'avg_consumption',
            'total_consumed_std': 'consumption_std',
            'total_consumed_count': 'num_readings',
            'total_consumed_min': 'min_consumption',
            'total_consumed_max': 'max_consumption',
            'consumption_per_day_mean': 'avg_daily_consumption',
            'consumption_per_day_std': 'daily_consumption_std',
            'neighborhood_id_first': 'neighborhood_id'
        }, inplace=True)

        # Handle cases with insufficient data
        profiles['consumption_std'] = profiles['consumption_std'].fillna(
            profiles['avg_consumption'] * 0.3  # Assume 30% variation if no std
        )
        profiles['daily_consumption_std'] = profiles[
            'daily_consumption_std'].fillna(
            profiles['avg_daily_consumption'] * 0.3
        )

        self.meter_profiles = profiles
        logger.info(f"Built profiles for {len(profiles)} water meters")

    def detect_anomaly(self, water_meter_id: int, current_reading: int,
                       previous_reading: int, days_billed: int = 30) -> Dict:
        """Detect if a reading is anomalous for a specific meter"""

        # Calculate consumption
        consumption = current_reading - previous_reading
        daily_consumption = consumption / days_billed if days_billed > 0 else 0

        # Get meter profile
        meter_profile = self.meter_profiles[
            self.meter_profiles['water_meter_id'] == water_meter_id
            ]

        if len(meter_profile) == 0:
            return {
                'is_anomaly': False,
                'confidence': 0.0,
                'reason': 'No historical data available',
                'recommendation': 'Collect more data for this meter'
            }

        profile = meter_profile.iloc[0]

        # Calculate z-scores
        consumption_zscore = abs(
            (consumption - profile['avg_consumption']) /
            (profile['consumption_std'] + 1)
        )

        daily_zscore = abs(
            (daily_consumption - profile['avg_daily_consumption']) /
            (profile['daily_consumption_std'] + 1)
        )

        # Determine anomaly thresholds
        anomaly_threshold = 2.5  # 2.5 standard deviations
        high_anomaly_threshold = 3.5

        # Check for anomalies
        is_consumption_anomaly = consumption_zscore > anomaly_threshold
        is_daily_anomaly = daily_zscore > anomaly_threshold
        is_high_anomaly = (consumption_zscore > high_anomaly_threshold or
                           daily_zscore > high_anomaly_threshold)

        # Determine reason and confidence
        if is_high_anomaly:
            confidence = min(0.95, max(consumption_zscore, daily_zscore) / 10)
            if consumption > profile['avg_consumption'] * 3:
                reason = "Extremely high consumption - possible leak or meter error"
                recommendation = "Immediate inspection required"
            elif consumption < profile['avg_consumption'] * 0.1:
                reason = "Extremely low consumption - possible meter malfunction"
                recommendation = "Check meter functionality"
            else:
                reason = "Highly unusual consumption pattern"
                recommendation = "Manual verification recommended"
        elif is_consumption_anomaly or is_daily_anomaly:
            confidence = min(0.8, max(consumption_zscore, daily_zscore) / 5)
            if consumption > profile['avg_consumption'] * 2:
                reason = "High consumption detected"
                recommendation = "Check for leaks or unusual usage"
            elif consumption < profile['avg_consumption'] * 0.3:
                reason = "Low consumption detected"
                recommendation = "Verify meter reading"
            else:
                reason = "Unusual consumption pattern"
                recommendation = "Monitor next reading"
        else:
            confidence = 0.0
            reason = "Normal consumption pattern"
            recommendation = "No action required"

        return {
            'is_anomaly': is_consumption_anomaly or is_daily_anomaly,
            'confidence': round(confidence, 3),
            'reason': reason,
            'recommendation': recommendation,
            'consumption_zscore': round(consumption_zscore, 2),
            'daily_zscore': round(daily_zscore, 2),
            'consumption': consumption,
            'daily_consumption': round(daily_consumption, 2),
            'meter_avg_consumption': round(profile['avg_consumption'], 2),
            'meter_avg_daily': round(profile['avg_daily_consumption'], 2)
        }
