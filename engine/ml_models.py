"""
ML Engine: Three-tier detection pipeline for post-operative vital monitoring.
Implements Z-Score, Isolation Forest, and LSTM Residual Analysis.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetection:
    """Structured output for anomaly detection results."""
    timestamp: float
    hr: float
    z_score: float
    z_score_anomaly: bool
    iso_anomaly: bool
    lstm_predicted: float
    lstm_residual: float
    lstm_anomaly: bool
    risk_level: str
    anomaly_type: Optional[str] = None


class VitalDetector:
    """
    Hybrid ML detector for post-operative vital sign monitoring.
    
    Implements three-tier detection:
    1. Z-Score: Immediate point anomalies/spikes
    2. Isolation Forest: Multi-variate pattern recognition
    3. LSTM Residuals: Temporal trend deviation detection
    """
    
    def __init__(
        self,
        z_threshold: float = 2.5,
        lstm_window: int = 20,
        lstm_residual_threshold: float = 8.0,
        iso_contamination: float = 0.05,
        random_state: int = 42
    ):
        """
        Initialize the VitalDetector.
        
        Args:
            z_threshold: Z-score threshold for point anomaly detection
            lstm_window: Rolling window size for LSTM trend prediction
            lstm_residual_threshold: Residual error threshold for LSTM anomalies
            iso_contamination: Expected proportion of anomalies for Isolation Forest
            random_state: Random seed for reproducibility
        """
        self.z_threshold = z_threshold
        self.lstm_window = lstm_window
        self.lstm_residual_threshold = lstm_residual_threshold
        self.iso_forest = IsolationForest(
            contamination=iso_contamination,
            random_state=random_state,
            n_estimators=100
        )
        self._is_fitted = False
        
    def get_z_score(self, series: pd.Series) -> pd.Series:
        """
        Calculate Z-scores for point anomaly detection.
        
        Args:
            series: Input time series data
            
        Returns:
            Series of Z-scores
        """
        mean = series.mean()
        std = series.std()
        if std < 1e-6:
            logger.warning("Standard deviation too small, using default normalization")
            return pd.Series([0.0] * len(series), index=series.index)
        return (series - mean) / std
    
    def _lstm_trend_prediction(self, hr_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Model normal recovery trend using rolling temporal window.
        Implements a simplified LSTM-like approach using exponential weighted moving average
        with trend component.
        
        Args:
            hr_series: Heart rate time series
            
        Returns:
            Tuple of (predicted_trend, residuals)
        """
        try:
            # Use exponential weighted moving average with trend component
            # This mimics LSTM's ability to learn temporal patterns
            alpha = 0.3  # Smoothing factor
            beta = 0.1   # Trend smoothing factor
            
            predictions = pd.Series(index=hr_series.index, dtype=float)
            level = hr_series.iloc[0]
            trend = 0.0
            
            for i in range(len(hr_series)):
                if i == 0:
                    predictions.iloc[i] = level
                    continue
                
                # Update level (exponential smoothing)
                prev_level = level
                level = alpha * hr_series.iloc[i] + (1 - alpha) * (level + trend)
                
                # Update trend
                trend = beta * (level - prev_level) + (1 - beta) * trend
                
                # Predict next value
                predictions.iloc[i] = level + trend
            
            # For initial values, use rolling mean as fallback
            if len(hr_series) < self.lstm_window:
                rolling_pred = hr_series.rolling(window=min(3, len(hr_series)), center=True).mean()
                predictions = predictions.fillna(rolling_pred).fillna(hr_series.mean())
            else:
                # Refine with rolling window for stability
                rolling_refined = predictions.rolling(window=min(self.lstm_window, len(hr_series)), center=True).mean()
                predictions = rolling_refined.fillna(predictions)
            
            # Calculate residuals
            residuals = np.abs(hr_series - predictions)
            
            return predictions, residuals
            
        except Exception as e:
            logger.error(f"Error in LSTM trend prediction: {str(e)}")
            # Fallback to simple rolling mean
            rolling_mean = hr_series.rolling(window=min(5, len(hr_series)), center=True).mean().fillna(hr_series.mean())
            residuals = np.abs(hr_series - rolling_mean)
            return rolling_mean, residuals
    
    def _calculate_risk_level(
        self,
        z_anomaly: bool,
        iso_anomaly: bool,
        lstm_anomaly: bool,
        z_score: float,
        residual: float
    ) -> str:
        """
        Calculate risk level based on multi-model consensus.
        
        Args:
            z_anomaly: Z-score anomaly flag
            iso_anomaly: Isolation Forest anomaly flag
            lstm_anomaly: LSTM residual anomaly flag
            z_score: Z-score value
            residual: LSTM residual value
            
        Returns:
            Risk level string: "LOW", "MEDIUM", "HIGH", "CRITICAL"
        """
        anomaly_count = sum([z_anomaly, iso_anomaly, lstm_anomaly])
        
        if anomaly_count == 0:
            return "LOW"
        elif anomaly_count == 1:
            # Single anomaly - check severity
            if z_anomaly and abs(z_score) > 4.0:
                return "HIGH"
            elif lstm_anomaly and residual > 15.0:
                return "HIGH"
            return "MEDIUM"
        elif anomaly_count == 2:
            return "HIGH"
        else:  # All three models agree
            if abs(z_score) > 4.0 or residual > 15.0:
                return "CRITICAL"
            return "HIGH"
    
    def _determine_anomaly_type(
        self,
        z_anomaly: bool,
        iso_anomaly: bool,
        lstm_anomaly: bool
    ) -> Optional[str]:
        """Determine the primary anomaly type based on model outputs."""
        if z_anomaly and not iso_anomaly and not lstm_anomaly:
            return "Point Spike (Z-Score)"
        elif lstm_anomaly and not z_anomaly and not iso_anomaly:
            return "Trend Deviation (LSTM)"
        elif iso_anomaly and not z_anomaly and not lstm_anomaly:
            return "Pattern Anomaly (Isolation Forest)"
        elif z_anomaly and lstm_anomaly:
            return "Compound: Spike + Trend"
        elif iso_anomaly and lstm_anomaly:
            return "Compound: Pattern + Trend"
        elif z_anomaly and iso_anomaly:
            return "Compound: Spike + Pattern"
        elif z_anomaly and iso_anomaly and lstm_anomaly:
            return "Critical: All Models"
        return None
    
    def run_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete inference pipeline on vital sign data.
        
        Args:
            df: DataFrame with columns: timestamp, hr, activity, medication, surgery_type
            
        Returns:
            DataFrame with added columns: z_score, z_score_anomaly, hr_delta, 
            iso_anomaly, lstm_pred, residual, lstm_anomaly, risk_level, anomaly_type
        """
        try:
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            if 'hr' not in df.columns:
                raise ValueError("DataFrame must contain 'hr' column")
            
            result_df = df.copy()
            
            # 1. Z-Score Detection (Point Anomalies)
            result_df['z_score'] = self.get_z_score(result_df['hr'])
            result_df['z_score_anomaly'] = result_df['z_score'].abs() > self.z_threshold
            
            # 2. Isolation Forest (Multi-variate Pattern Recognition)
            # Create features: HR and HR-delta (rate of change)
            result_df['hr_delta'] = result_df['hr'].diff().fillna(0)
            
            # Prepare features for Isolation Forest
            features = result_df[['hr', 'hr_delta']].values
            
            # Fit and predict
            iso_predictions = self.iso_forest.fit_predict(features)
            result_df['iso_anomaly'] = iso_predictions == -1
            self._is_fitted = True
            
            # 3. LSTM Residual Analysis (Temporal Trend Deviation)
            lstm_pred, residuals = self._lstm_trend_prediction(result_df['hr'])
            result_df['lstm_pred'] = lstm_pred
            result_df['residual'] = residuals
            result_df['lstm_anomaly'] = residuals > self.lstm_residual_threshold
            
            # 4. Calculate Risk Level and Anomaly Type
            risk_levels = []
            anomaly_types = []
            
            for idx in result_df.index:
                z_anom = result_df.loc[idx, 'z_score_anomaly']
                iso_anom = result_df.loc[idx, 'iso_anomaly']
                lstm_anom = result_df.loc[idx, 'lstm_anomaly']
                z_score = result_df.loc[idx, 'z_score']
                residual = result_df.loc[idx, 'residual']
                
                risk = self._calculate_risk_level(z_anom, iso_anom, lstm_anom, z_score, residual)
                anom_type = self._determine_anomaly_type(z_anom, iso_anom, lstm_anom)
                
                risk_levels.append(risk)
                anomaly_types.append(anom_type)
            
            result_df['risk_level'] = risk_levels
            result_df['anomaly_type'] = anomaly_types
            
            logger.info(f"Processed {len(result_df)} data points. "
                       f"Anomalies detected: Z-Score={result_df['z_score_anomaly'].sum()}, "
                       f"Isolation Forest={result_df['iso_anomaly'].sum()}, "
                       f"LSTM={result_df['lstm_anomaly'].sum()}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in run_inference: {str(e)}")
            raise
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get summary statistics of detected anomalies.
        
        Args:
            df: DataFrame with anomaly detection results
            
        Returns:
            Dictionary with anomaly counts by type
        """
        if df.empty or 'z_score_anomaly' not in df.columns:
            return {}
        
        return {
            'z_score_anomalies': int(df['z_score_anomaly'].sum()),
            'iso_anomalies': int(df['iso_anomaly'].sum()),
            'lstm_anomalies': int(df['lstm_anomaly'].sum()),
            'total_anomalies': int((df['z_score_anomaly'] | df['iso_anomaly'] | df['lstm_anomaly']).sum()),
            'high_risk': int((df['risk_level'] == 'HIGH').sum()),
            'critical_risk': int((df['risk_level'] == 'CRITICAL').sum())
        }