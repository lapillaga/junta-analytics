"""
AnomalyDetector v3.0 - Enfoque simplificado y efectivo
Autor: Luis Pillaga • Mayo 2025

Enfoque híbrido: Reglas estadísticas + One-Class SVM
Optimizado para detección de anomalías en medidores específicos
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class AnomalyDetectorV3:
    """Detector de anomalías híbrido para lecturas de medidores de agua"""

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        
        # Modelos
        self.isolation_forest = None
        self.one_class_svm = None
        self.scaler = StandardScaler()
        
        # Estadísticas por medidor
        self.meter_stats_: Optional[pd.DataFrame] = None
        self.global_stats_: Optional[Dict] = None
        
        # Estado
        self.is_trained: bool = False
        self.feature_columns_: List[str] = []

    def train(self, training_df: pd.DataFrame) -> Dict[str, float]:
        """
        Entrena el detector híbrido con datos históricos
        
        Args:
            training_df: DataFrame con columnas [water_meter_id, total_consumed, days_billed, period_start]
        
        Returns:
            Métricas de entrenamiento
        """
        logger.info("🔄 Iniciando entrenamiento del detector híbrido...")
        
        # 1. Calcular estadísticas por medidor
        self._compute_meter_statistics(training_df)
        
        # 2. Preparar características
        features_df = self._build_features(training_df)
        
        # 3. Entrenar modelos
        X = features_df.values
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200
        )
        self.isolation_forest.fit(X_scaled)
        
        # One-Class SVM
        self.one_class_svm = OneClassSVM(
            gamma='scale',
            nu=self.contamination
        )
        self.one_class_svm.fit(X_scaled)
        
        self.feature_columns_ = features_df.columns.tolist()
        self.is_trained = True
        
        # 4. Calcular métricas básicas
        if_scores = self.isolation_forest.decision_function(X_scaled)
        svm_scores = self.one_class_svm.decision_function(X_scaled)
        
        metrics = {
            'samples_trained': len(training_df),
            'meters_trained': training_df['water_meter_id'].nunique(),
            'features_used': len(self.feature_columns_),
            'if_score_mean': float(np.mean(if_scores)),
            'svm_score_mean': float(np.mean(svm_scores)),
            'contamination': self.contamination
        }
        
        logger.info(f"✅ Entrenamiento completado: {metrics['samples_trained']} muestras, {metrics['meters_trained']} medidores")
        return metrics

    def detect_single_reading(
        self,
        water_meter_id: int,
        current_reading: int,
        previous_reading: int,
        days_billed: int,
        historical_data: pd.DataFrame
    ) -> Dict:
        """
        Detecta anomalías en una lectura específica usando enfoque híbrido
        
        Returns:
            Dict con is_anomaly, score, confidence, reason, details
        """
        if not self.is_trained:
            raise RuntimeError("❌ Modelo no entrenado")
        
        # Calcular consumo
        total_consumed = current_reading - previous_reading
        consumption_per_day = total_consumed / max(days_billed, 1)
        
        # 1. Verificaciones básicas
        basic_check = self._basic_anomaly_checks(total_consumed, consumption_per_day)
        if basic_check['is_anomaly']:
            return basic_check
        
        # 2. Verificación estadística específica del medidor
        statistical_check = self._statistical_anomaly_check(
            water_meter_id, total_consumed, consumption_per_day, days_billed
        )
        
        # 3. Verificación con modelos ML
        ml_check = self._ml_anomaly_check(
            water_meter_id, total_consumed, consumption_per_day, days_billed
        )
        
        # 4. Combinar resultados
        final_result = self._combine_anomaly_results(
            basic_check, statistical_check, ml_check,
            total_consumed, consumption_per_day
        )
        
        return final_result

    def _basic_anomaly_checks(self, total_consumed: int, consumption_per_day: float) -> Dict:
        """Verificaciones básicas de anomalías obvias"""
        
        if total_consumed < 0:
            return {
                'is_anomaly': True,
                'score': 1.0,
                'confidence': 1.0,
                'reason': 'Lectura negativa: lectura actual menor que anterior',
                'category': 'error_lectura',
                'total_consumed': total_consumed,
                'consumption_per_day': 0
            }
        
        if total_consumed == 0:
            return {
                'is_anomaly': True,
                'score': 0.8,
                'confidence': 0.9,
                'reason': 'Consumo cero: posible medidor roto o casa vacía',
                'category': 'consumo_cero',
                'total_consumed': total_consumed,
                'consumption_per_day': consumption_per_day
            }
        
        if consumption_per_day > 100:  # Consumo extremadamente alto
            return {
                'is_anomaly': True,
                'score': 0.95,
                'confidence': 0.95,
                'reason': f'Consumo extremo: {consumption_per_day:.1f} unidades/día (posible fuga mayor)',
                'category': 'consumo_extremo',
                'total_consumed': total_consumed,
                'consumption_per_day': consumption_per_day
            }
        
        return {'is_anomaly': False}

    def _statistical_anomaly_check(
        self, water_meter_id: int, total_consumed: int, 
        consumption_per_day: float, days_billed: int
    ) -> Dict:
        """Verificación estadística basada en historial del medidor"""
        
        if self.meter_stats_ is None:
            return {'is_anomaly': False, 'score': 0, 'reason': 'Sin estadísticas'}
        
        # Buscar estadísticas del medidor
        meter_stats = self.meter_stats_[self.meter_stats_['water_meter_id'] == water_meter_id]
        
        if meter_stats.empty:
            # Usar estadísticas globales
            global_mean = self.global_stats_['global_mean']
            global_std = self.global_stats_['global_std']
            
            z_score = abs(consumption_per_day - global_mean) / (global_std + 1e-6)
            
            if z_score > 3:
                return {
                    'is_anomaly': True,
                    'score': min(0.7, z_score / 5),
                    'confidence': min(0.8, z_score / 4),
                    'reason': f'Medidor nuevo: consumo {z_score:.1f} desviaciones de la media global',
                    'category': 'estadistico_global',
                    'z_score': z_score
                }
        else:
            # Usar estadísticas específicas del medidor
            stats = meter_stats.iloc[0]
            
            # Z-score respecto al medidor
            z_score = abs(consumption_per_day - stats['mean_consumption']) / (stats['std_consumption'] + 1e-6)
            
            # IQR check
            q1, q3 = stats['q1_consumption'], stats['q3_consumption']
            iqr = q3 - q1
            is_outlier_iqr = (consumption_per_day < q1 - 1.5 * iqr) or (consumption_per_day > q3 + 1.5 * iqr)
            
            # Ratio check
            typical_consumption = stats['median_consumption']
            ratio = consumption_per_day / (typical_consumption + 1e-6)
            
            if z_score > 2.5 or is_outlier_iqr or ratio > 5 or ratio < 0.2:
                return {
                    'is_anomaly': True,
                    'score': min(0.8, z_score / 4),
                    'confidence': min(0.85, z_score / 3),
                    'reason': f'Anomalía estadística: {z_score:.1f}σ del patrón del medidor (ratio: {ratio:.1f}x)',
                    'category': 'estadistico_medidor',
                    'z_score': z_score,
                    'ratio': ratio
                }
        
        return {'is_anomaly': False, 'score': 0}

    def _ml_anomaly_check(
        self, water_meter_id: int, total_consumed: int,
        consumption_per_day: float, days_billed: int
    ) -> Dict:
        """Verificación usando modelos de ML"""
        
        # Crear DataFrame para predicción
        test_data = pd.DataFrame({
            'water_meter_id': [water_meter_id],
            'total_consumed': [total_consumed],
            'days_billed': [days_billed],
            'consumption_per_day': [consumption_per_day]
        })
        
        try:
            # Preparar características
            features_df = self._build_features(test_data)
            X = features_df.values
            X_scaled = self.scaler.transform(X)
            
            # Predicciones
            if_pred = self.isolation_forest.predict(X_scaled)[0]
            if_score = self.isolation_forest.decision_function(X_scaled)[0]
            
            svm_pred = self.one_class_svm.predict(X_scaled)[0]
            svm_score = self.one_class_svm.decision_function(X_scaled)[0]
            
            # Normalizar scores
            if_score_norm = max(0, min(1, (if_score + 0.5) / 1.0))
            svm_score_norm = max(0, min(1, (svm_score + 1) / 2.0))
            
            # Combinar modelos
            is_anomaly_if = if_pred == -1
            is_anomaly_svm = svm_pred == -1
            
            if is_anomaly_if or is_anomaly_svm:
                combined_score = (if_score_norm + svm_score_norm) / 2
                confidence = combined_score
                
                return {
                    'is_anomaly': True,
                    'score': combined_score,
                    'confidence': confidence,
                    'reason': f'Anomalía detectada por ML (IF: {is_anomaly_if}, SVM: {is_anomaly_svm})',
                    'category': 'ml_detection',
                    'if_score': if_score,
                    'svm_score': svm_score
                }
            
        except Exception as e:
            logger.warning(f"Error en verificación ML: {e}")
        
        return {'is_anomaly': False, 'score': 0}

    def _combine_anomaly_results(
        self, basic: Dict, statistical: Dict, ml: Dict,
        total_consumed: int, consumption_per_day: float
    ) -> Dict:
        """Combina resultados de todas las verificaciones"""
        
        # Si hay anomalía básica, esa tiene prioridad
        if basic.get('is_anomaly', False):
            return {
                **basic,
                'detection_method': 'basic_rules',
                'total_consumed': total_consumed,
                'consumption_per_day': consumption_per_day
            }
        
        # Combinar estadística y ML
        stat_anomaly = statistical.get('is_anomaly', False)
        ml_anomaly = ml.get('is_anomaly', False)
        
        if stat_anomaly or ml_anomaly:
            # Usar el score más alto
            stat_score = statistical.get('score', 0)
            ml_score = ml.get('score', 0)
            
            if stat_score > ml_score:
                final_result = statistical
                method = 'statistical'
            else:
                final_result = ml
                method = 'ml'
            
            return {
                **final_result,
                'detection_method': method,
                'total_consumed': total_consumed,
                'consumption_per_day': consumption_per_day,
                'stat_detected': stat_anomaly,
                'ml_detected': ml_anomaly
            }
        
        # No es anomalía
        return {
            'is_anomaly': False,
            'score': 0.0,
            'confidence': 0.0,
            'reason': 'Consumo normal dentro de parámetros esperados',
            'category': 'normal',
            'detection_method': 'none',
            'total_consumed': total_consumed,
            'consumption_per_day': consumption_per_day
        }

    def _compute_meter_statistics(self, df: pd.DataFrame):
        """Calcula estadísticas por medidor y globales"""
        
        # Calcular consumo por día
        df = df.copy()
        df['consumption_per_day'] = df['total_consumed'] / df['days_billed'].clip(lower=1)
        
        # Estadísticas por medidor
        self.meter_stats_ = df.groupby('water_meter_id')['consumption_per_day'].agg([
            'count', 'mean', 'std', 'median', 
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75),
            'min', 'max'
        ]).reset_index()
        
        self.meter_stats_.columns = [
            'water_meter_id', 'readings_count', 'mean_consumption', 'std_consumption',
            'median_consumption', 'q1_consumption', 'q3_consumption', 
            'min_consumption', 'max_consumption'
        ]
        
        # Llenar NaN en std
        self.meter_stats_['std_consumption'] = self.meter_stats_['std_consumption'].fillna(
            self.meter_stats_['median_consumption'] * 0.3
        )
        
        # Estadísticas globales
        self.global_stats_ = {
            'global_mean': df['consumption_per_day'].mean(),
            'global_std': df['consumption_per_day'].std(),
            'global_median': df['consumption_per_day'].median()
        }
        
        logger.info(f"📊 Estadísticas calculadas para {len(self.meter_stats_)} medidores")

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construye características simplificadas pero efectivas"""
        
        result = df.copy()
        
        # Características básicas
        result['consumption_per_day'] = result['total_consumed'] / result['days_billed'].clip(lower=1)
        result['log_consumption'] = np.log1p(result['total_consumed'])
        result['consumption_per_30days'] = result['total_consumed'] * (30 / result['days_billed'].clip(lower=1))
        
        # Merge con estadísticas del medidor
        if self.meter_stats_ is not None:
            result = result.merge(
                self.meter_stats_[['water_meter_id', 'mean_consumption', 'std_consumption', 'median_consumption']],
                on='water_meter_id', how='left'
            )
            
            # Llenar valores faltantes con estadísticas globales
            global_mean = self.global_stats_['global_mean']
            global_std = self.global_stats_['global_std']
            global_median = self.global_stats_['global_median']
            
            result['mean_consumption'] = result['mean_consumption'].fillna(global_mean)
            result['std_consumption'] = result['std_consumption'].fillna(global_std)
            result['median_consumption'] = result['median_consumption'].fillna(global_median)
            
            # Características relativas
            result['consumption_vs_mean'] = result['consumption_per_day'] / (result['mean_consumption'] + 1e-6)
            result['consumption_vs_median'] = result['consumption_per_day'] / (result['median_consumption'] + 1e-6)
            result['z_score_meter'] = (result['consumption_per_day'] - result['mean_consumption']) / (result['std_consumption'] + 1e-6)
        else:
            # Sin estadísticas previas
            result['consumption_vs_mean'] = 1.0
            result['consumption_vs_median'] = 1.0
            result['z_score_meter'] = 0.0
        
        # Características temporales básicas
        if 'period_start' in result.columns:
            result['period_start'] = pd.to_datetime(result['period_start'], errors='coerce')
            result['month'] = result['period_start'].dt.month.fillna(6)
            result['is_winter'] = ((result['month'] >= 6) & (result['month'] <= 8)).astype(int)
            result['is_summer'] = ((result['month'] >= 12) | (result['month'] <= 2)).astype(int)
        else:
            result['month'] = 6
            result['is_winter'] = 0
            result['is_summer'] = 0
        
        # Seleccionar características finales
        feature_cols = [
            'total_consumed', 'consumption_per_day', 'log_consumption', 'consumption_per_30days',
            'consumption_vs_mean', 'consumption_vs_median', 'z_score_meter',
            'days_billed', 'is_winter', 'is_summer'
        ]
        
        return result[feature_cols].fillna(0)

    def save(self, path: str | Path):
        """Guarda el modelo entrenado"""
        if not self.is_trained:
            raise RuntimeError("❌ Entrena el modelo primero")
        
        model_data = {
            'isolation_forest': self.isolation_forest,
            'one_class_svm': self.one_class_svm,
            'scaler': self.scaler,
            'meter_stats': self.meter_stats_,
            'global_stats': self.global_stats_,
            'feature_columns': self.feature_columns_,
            'contamination': self.contamination,
            'version': '3.0',
            'created_at': datetime.now()
        }
        
        joblib.dump(model_data, path)
        logger.info(f"💾 Modelo guardado en {path}")

    def load(self, path: str | Path):
        """Carga un modelo entrenado"""
        model_data = joblib.load(path)
        
        self.isolation_forest = model_data['isolation_forest']
        self.one_class_svm = model_data['one_class_svm']
        self.scaler = model_data['scaler']
        self.meter_stats_ = model_data['meter_stats']
        self.global_stats_ = model_data['global_stats']
        self.feature_columns_ = model_data['feature_columns']
        self.contamination = model_data.get('contamination', 0.1)
        self.is_trained = True
        
        logger.info(f"📂 Modelo cargado desde {path}")