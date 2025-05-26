"""
AnomalyDetector class – version 2.3
Autor: Luis Pillaga • Mayo 2025

• Inyección de anomalías sintéticas realistas.
• Búsqueda de hiper-parámetros con ROC-AUC.
• Sin dependencias de MLflow.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Isolation-Forest basado en características robustas para lecturas de agua."""

    # ---------- init ----------
    def __init__(self, contamination: float = 0.03, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state

        self.pipeline_: Optional[Pipeline] = None
        self.feature_columns_: List[str] = []

        self.meter_stats_: Optional[pd.DataFrame] = None
        self.neigh_stats_: Optional[pd.DataFrame] = None
        self.rain_monthly_: Optional[pd.DataFrame] = None

        self.score_threshold_: float = 0.0
        self.is_trained: bool = False

    # ---------- API ----------
    def train(
        self,
        training_df: pd.DataFrame,
        climate_df: Optional[pd.DataFrame] = None,
        use_synthetic: bool = True,
    ) -> Dict[str, float]:
        """Entrena el modelo y devuelve métricas de validación (si se usan sintéticos)."""
        logger.info("Starting training …")

        # 1) estadísticas de referencia
        self._compute_reference_stats(training_df)
        if climate_df is not None:
            self.rain_monthly_ = (
                climate_df.groupby("period_str")["avg_rainfall"]
                .mean()
                .reset_index(name="period_rainfall")
                .rename(columns={"period_str": "period_month"})
            )
        else:
            self.rain_monthly_ = None

        # 2) features + etiquetas
        if use_synthetic:
            X_full, y_full = self._inject_synthetic(training_df, ratio=0.03)
            X_train, X_val, y_train, y_val = train_test_split(
                X_full,
                y_full,
                test_size=0.3,
                stratify=y_full,
                random_state=self.random_state,
            )
        else:
            X_full = self._build_features(training_df)
            X_train, X_val, y_train, y_val = X_full, None, None, None

        self.feature_columns_ = X_full.columns.tolist()

        # 3) búsqueda de hiper-parámetros
        pipe = Pipeline(
            [
                ("scaler", RobustScaler()),
                ("clf", IsolationForest(random_state=self.random_state)),
            ]
        )
        search = RandomizedSearchCV(
            pipe,
            {
                "clf__n_estimators": [200, 300, 400, 500],
                "clf__max_samples": ["auto", 0.8, 0.9],
                "clf__max_features": [1.0, 0.9, 0.8],
                "clf__contamination": [self.contamination],
            },
            n_iter=12,
            cv=3,
            scoring="roc_auc",
            random_state=self.random_state,
            verbose=0,
        )
        search.fit(X_train, y_train)
        self.pipeline_ = search.best_estimator_

        # 4) métricas
        metrics: Dict[str, float] = {}
        if use_synthetic:
            y_pred = (self.pipeline_.predict(X_val) == -1).astype(int)
            metrics.update(
                dict(
                    precision=precision_score(y_val, y_pred),
                    recall=recall_score(y_val, y_pred),
                    f1=f1_score(y_val, y_pred),
                    auc=roc_auc_score(
                        y_val, -self.pipeline_.decision_function(X_val)
                    ),
                )
            )

        # 5) umbral P90
        scores_train = -self.pipeline_.decision_function(X_full)
        self.score_threshold_ = float(np.percentile(scores_train, 90))
        self.is_trained = True
        metrics["threshold"] = self.score_threshold_

        return metrics

    def predict(self, df: pd.DataFrame) -> Dict:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        X = self._build_features(df)
        scores = -self.pipeline_.decision_function(X)
        anomalies = scores > self.score_threshold_
        confidence = np.clip(scores / (self.score_threshold_ + 1e-6), 0, 1)
        return {
            "anomaly": anomalies.tolist(),
            "score": scores.tolist(),
            "confidence": confidence.tolist(),
        }

    def save(self, path: str | Path):
        if not self.is_trained:
            raise RuntimeError("Train first")
        joblib.dump(
            {
                "pipeline": self.pipeline_,
                "feature_columns": self.feature_columns_,
                "meter_stats": self.meter_stats_,
                "neigh_stats": self.neigh_stats_,
                "rain_monthly": self.rain_monthly_,
                "threshold": self.score_threshold_,
                "created_at": datetime.now(),
                "version": "2.3",
            },
            path,
        )
        logger.info(f"Saved to {path}")

    def load(self, path: str | Path):
        art = joblib.load(path)
        self.pipeline_ = art["pipeline"]
        self.feature_columns_ = art["feature_columns"]
        self.meter_stats_ = art["meter_stats"]
        self.neigh_stats_ = art["neigh_stats"]
        self.rain_monthly_ = art["rain_monthly"]
        self.score_threshold_ = art["threshold"]
        self.is_trained = True
        logger.info(f"Model loaded from {path}")

    def detect_single_reading(
        self,
        water_meter_id: int,
        current_reading: int,
        previous_reading: int,
        days_billed: int,
        historical_data: pd.DataFrame,
        period_start: Optional[str] = None,
    ) -> Dict:
        """
        Detecta anomalías en una sola lectura usando el modelo entrenado.
        
        Args:
            water_meter_id: ID del medidor de agua
            current_reading: Lectura actual del medidor
            previous_reading: Lectura anterior del medidor
            days_billed: Días del período de facturación
            historical_data: DataFrame con datos históricos para inferir neighborhood_id
            period_start: Fecha de inicio del período (opcional)
        
        Returns:
            Dict con is_anomaly, score, confidence, reason
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Calcular consumo total
        total_consumed = current_reading - previous_reading
        
        if total_consumed < 0:
            return {
                "is_anomaly": True,
                "score": 1.0,
                "confidence": 1.0,
                "reason": "Lectura actual menor que anterior",
                "total_consumed": total_consumed,
                "consumption_per_day": 0
            }
        
        # Inferir neighborhood_id desde historical_data si está disponible
        neighborhood_id = None
        if 'neighborhood_id' in historical_data.columns:
            meter_neighborhoods = historical_data[
                historical_data['water_meter_id'] == water_meter_id
            ]['neighborhood_id'].unique()
            if len(meter_neighborhoods) > 0:
                neighborhood_id = meter_neighborhoods[0]
        
        # Crear DataFrame para la predicción
        if period_start is None:
            period_start = datetime.now().strftime('%Y-%m-%d')
        
        single_reading = pd.DataFrame({
            'water_meter_id': [water_meter_id],
            'total_consumed': [total_consumed],
            'days_billed': [days_billed],
            'period_start': [period_start],
            'neighborhood_id': [neighborhood_id] if neighborhood_id is not None else [0]
        })
        
        try:
            # Realizar predicción
            result = self.predict(single_reading)
            
            is_anomaly = bool(result['anomaly'][0])
            score = float(result['score'][0])
            confidence = float(result['confidence'][0])

            print("Anomaly detected:", is_anomaly)
            print("Score:", score)
            print("Confidence:", confidence)
            
            consumption_per_day = total_consumed / max(days_billed, 1)
            
            # Determinar razón de la anomalía
            reason = "Normal"
            if is_anomaly:
                if total_consumed == 0:
                    reason = "Consumo cero detectado"
                elif consumption_per_day > 50:  # Umbral alto
                    reason = "Consumo excesivamente alto"
                elif consumption_per_day < 0.1:  # Umbral bajo
                    reason = "Consumo excesivamente bajo"
                else:
                    reason = "Patrón de consumo anómalo detectado"
            
            return {
                "is_anomaly": is_anomaly,
                "score": score,
                "confidence": confidence,
                "reason": reason,
                "total_consumed": total_consumed,
                "consumption_per_day": consumption_per_day,
                "threshold": self.score_threshold_
            }
            
        except Exception as e:
            logger.error(f"Error in single reading detection: {e}")
            return {
                "is_anomaly": True,
                "score": 1.0,
                "confidence": 0.5,
                "reason": f"Error en predicción: {str(e)}",
                "total_consumed": total_consumed,
                "consumption_per_day": total_consumed / max(days_billed, 1)
            }

    # ---------- internals ----------
    def _inject_synthetic(self, df: pd.DataFrame, ratio: float = 0.03):
        """Genera anomalías sintéticas más realistas basadas en patrones reales"""
        df_syn = df.copy().reset_index(drop=True)
        n = int(len(df_syn) * ratio)
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(df_syn), n, replace=False)
        
        for i in idx:
            anomaly_type = rng.choice(['high', 'low', 'zero', 'spike'], p=[0.4, 0.3, 0.2, 0.1])
            
            if anomaly_type == 'high':
                # Consumo anormalmente alto (fuga, mal funcionamiento)
                mult = rng.uniform(3, 6)
                df_syn.at[i, "total_consumed"] *= mult
            elif anomaly_type == 'low':
                # Consumo anormalmente bajo (medidor atascado)
                mult = rng.uniform(0.05, 0.2)
                df_syn.at[i, "total_consumed"] *= mult
            elif anomaly_type == 'zero':
                # Sin consumo (medidor roto, casa vacía)
                df_syn.at[i, "total_consumed"] = 0
            elif anomaly_type == 'spike':
                # Pico repentino seguido de normalidad
                mult = rng.uniform(8, 15)
                df_syn.at[i, "total_consumed"] *= mult
        
        labels = np.zeros(len(df_syn), dtype=int)
        labels[idx] = 1
        return self._build_features(df_syn), labels

    def _compute_reference_stats(self, df: pd.DataFrame):
        df = df.copy()
        df["consumption_per_day"] = df["total_consumed"] / df["days_billed"].clip(lower=1)
        self.meter_stats_ = df.groupby("water_meter_id")["consumption_per_day"].agg(
            [
                ("meter_mean", "mean"),
                ("meter_std", "std"),
                ("meter_median", "median"),
                ("meter_q1", lambda x: x.quantile(0.25)),
                ("meter_q3", lambda x: x.quantile(0.75)),
            ]
        )
        self.meter_stats_["meter_iqr"] = (
            self.meter_stats_["meter_q3"] - self.meter_stats_["meter_q1"]
        )
        self.meter_stats_["meter_std"].fillna(
            self.meter_stats_["meter_median"] * 0.2, inplace=True
        )

        if "neighborhood_id" in df.columns:
            self.neigh_stats_ = df.groupby("neighborhood_id")["total_consumed"].agg(
                [
                    ("neigh_mean", "mean"),
                    ("neigh_std", "std"),
                    ("neigh_median", "median"),
                ]
            )
            self.neigh_stats_["neigh_std"].fillna(
                self.neigh_stats_["neigh_median"] * 0.3, inplace=True
            )
        else:
            self.neigh_stats_ = pd.DataFrame()

    def _build_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        if self.meter_stats_ is None:
            raise RuntimeError("Stats not ready")

        df = df_raw.copy()
        df["consumption_per_day"] = df["total_consumed"] / df["days_billed"].clip(lower=1)
        df["log_consumption"] = np.log1p(df["total_consumed"])
        df["log_consumption_per_day"] = np.log1p(df["consumption_per_day"])

        # Merge meter statistics
        df = df.merge(self.meter_stats_.reset_index(), on="water_meter_id", how="left")
        for c in ["meter_mean", "meter_std", "meter_median", "meter_iqr", "meter_q1", "meter_q3"]:
            df[c] = df[c].fillna(self.meter_stats_[c].median())

        # Características específicas del medidor
        df["z_meter"] = (df["consumption_per_day"] - df["meter_mean"]) / (df["meter_std"] + 1e-3)
        df["meter_percentile"] = np.clip(
            (df["consumption_per_day"] - df["meter_q1"]) / (df["meter_iqr"] + 1e-3),
            0,
            1,
        )
        
        # Nuevas características para mejor detección
        # Desviación absoluta de la mediana (MAD) - más robusta que z-score
        df["mad_meter"] = np.abs(df["consumption_per_day"] - df["meter_median"]) / (df["meter_iqr"] + 1e-3)
        
        # Ratio respecto al consumo típico del medidor
        df["consumption_ratio"] = df["consumption_per_day"] / (df["meter_median"] + 1e-3)
        
        # Detectar consumos extremos
        df["is_zero_consumption"] = (df["total_consumed"] == 0).astype(int)
        df["is_very_high"] = (df["consumption_per_day"] > (df["meter_q3"] + 3 * df["meter_iqr"])).astype(int)
        df["is_very_low"] = (df["consumption_per_day"] < (df["meter_q1"] - 3 * df["meter_iqr"])).astype(int)

        # Características de vecindario
        if "neighborhood_id" in df.columns and not self.neigh_stats_.empty:
            df = df.merge(self.neigh_stats_.reset_index(), on="neighborhood_id", how="left")
            df["neigh_z"] = (df["consumption_per_day"] - df["neigh_median"]) / (
                df["neigh_std"] + 1e-3
            )
            # Ratio respecto al vecindario
            df["neigh_ratio"] = df["consumption_per_day"] / (df["neigh_median"] + 1e-3)
        else:
            df["neigh_z"] = 0
            df["neigh_ratio"] = 1

        # Características temporales
        df["month"] = pd.to_datetime(df["period_start"], errors="coerce").dt.month.fillna(0)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        # Características de días facturados
        df["days_billed_norm"] = df["days_billed"] / 30.0  # Normalizado a mes estándar
        df["consumption_per_standard_month"] = df["total_consumed"] * (30.0 / df["days_billed"].clip(lower=1))

        # Características de lluvia
        if self.rain_monthly_ is not None:
            df["period_month"] = (
                pd.to_datetime(df["period_start"], errors="coerce").dt.to_period("M").astype(str)
            )
            df = df.merge(self.rain_monthly_, on="period_month", how="left")
            df["period_rainfall"] = df["period_rainfall"].fillna(0)
            # Interacción lluvia-consumo
            df["rain_consumption_interaction"] = df["period_rainfall"] * df["consumption_per_day"]
        else:
            df["period_rainfall"] = 0
            df["rain_consumption_interaction"] = 0

        feat_cols = [
            "total_consumed",
            "consumption_per_day",
            "log_consumption",
            "log_consumption_per_day",
            "z_meter",
            "meter_percentile",
            "mad_meter",
            "consumption_ratio",
            "is_zero_consumption",
            "is_very_high",
            "is_very_low",
            "neigh_z",
            "neigh_ratio",
            "month_sin",
            "month_cos",
            "days_billed",
            "days_billed_norm",
            "consumption_per_standard_month",
            "meter_median",
            "meter_iqr",
            "period_rainfall",
            "rain_consumption_interaction",
        ]
        return df[feat_cols].fillna(0)
