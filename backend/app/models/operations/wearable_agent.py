"""
Wearable Device Integration Agent for IoMT Data Processing
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random
import logging

from app.config.settings import settings

logger = logging.getLogger(__name__)

class WearableDataAgent:
    """
    AI agent for processing and analyzing wearable device data (IoMT)
    """

    def __init__(self):
        self.device_types = self._define_device_types()
        self.data_processors = self._initialize_data_processors()
        self.alert_thresholds = self._define_alert_thresholds()
        self.analytics_models = self._load_analytics_models()
        logger.info("✅ Wearable Data Agent initialized")

    def _define_device_types(self) -> Dict[str, Any]:
        """Define supported wearable device types"""
        return {
            "fitness_trackers": {
                "fitbit": {
                    "metrics": ["steps", "heart_rate", "sleep", "calories", "distance"],
                    "sampling_rate": "1_minute",
                    "battery_life": "7_days",
                    "data_format": "json"
                },
                "apple_watch": {
                    "metrics": ["steps", "heart_rate", "ecg", "blood_oxygen", "activity"],
                    "sampling_rate": "continuous",
                    "battery_life": "18_hours",
                    "data_format": "healthkit"
                },
                "garmin": {
                    "metrics": ["steps", "heart_rate", "gps", "stress", "body_battery"],
                    "sampling_rate": "1_second",
                    "battery_life": "14_days",
                    "data_format": "fit"
                }
            },
            "medical_devices": {
                "continuous_glucose_monitor": {
                    "metrics": ["glucose_level", "glucose_trend"],
                    "sampling_rate": "1_minute",
                    "battery_life": "14_days",
                    "data_format": "proprietary"
                },
                "blood_pressure_monitor": {
                    "metrics": ["systolic_bp", "diastolic_bp", "heart_rate"],
                    "sampling_rate": "on_demand",
                    "battery_life": "1_year",
                    "data_format": "bluetooth"
                },
                "pulse_oximeter": {
                    "metrics": ["oxygen_saturation", "heart_rate"],
                    "sampling_rate": "continuous",
                    "battery_life": "24_hours",
                    "data_format": "bluetooth"
                }
            },
            "smart_clothing": {
                "smart_shirt": {
                    "metrics": ["heart_rate", "breathing_rate", "body_temperature"],
                    "sampling_rate": "continuous",
                    "battery_life": "24_hours",
                    "data_format": "textile_sensors"
                },
                "smart_socks": {
                    "metrics": ["gait_analysis", "pressure_distribution", "balance"],
                    "sampling_rate": "continuous",
                    "battery_life": "12_hours",
                    "data_format": "pressure_sensors"
                }
            }
        }

    def _initialize_data_processors(self) -> Dict[str, Any]:
        """Initialize data processing pipelines"""
        return {
            "preprocessing": {
                "noise_reduction": "moving_average",
                "outlier_detection": "iqr_method",
                "missing_data_handling": "interpolation",
                "data_validation": "range_checking"
            },
            "feature_extraction": {
                "time_domain": ["mean", "std", "min", "max", "range"],
                "frequency_domain": ["fft", "power_spectral_density"],
                "statistical": ["skewness", "kurtosis", "percentiles"],
                "temporal": ["trends", "patterns", "anomalies"]
            },
            "aggregation": {
                "temporal_windows": ["1_minute", "5_minutes", "1_hour", "1_day"],
                "statistical_measures": ["mean", "median", "std", "percentiles"],
                "trend_analysis": ["linear_regression", "seasonal_decomposition"]
            }
        }

    def _define_alert_thresholds(self) -> Dict[str, Any]:
        """Define alert thresholds for different metrics"""
        return {
            "heart_rate": {
                "resting": {"low": 50, "high": 100},
                "active": {"low": 60, "high": 180},
                "critical": {"low": 40, "high": 200}
            },
            "blood_pressure": {
                "normal": {"systolic": [90, 120], "diastolic": [60, 80]},
                "elevated": {"systolic": [120, 129], "diastolic": [60, 80]},
                "high": {"systolic": [130, 180], "diastolic": [80, 120]},
                "crisis": {"systolic": 180, "diastolic": 120}
            },
            "glucose": {
                "normal": [70, 140],
                "prediabetic": [140, 200],
                "diabetic": [200, 400],
                "critical_low": 50,
                "critical_high": 400
            },
            "oxygen_saturation": {
                "normal": [95, 100],
                "concerning": [90, 95],
                "critical": 90
            },
            "activity": {
                "sedentary": {"steps_per_day": 5000},
                "active": {"steps_per_day": 10000},
                "very_active": {"steps_per_day": 15000}
            }
        }

    def _load_analytics_models(self) -> Dict[str, Any]:
        """Load analytics models for wearable data"""
        return {
            "anomaly_detection": {
                "method": "isolation_forest",
                "sensitivity": 0.1,
                "window_size": 24  # hours
            },
            "trend_analysis": {
                "method": "seasonal_decompose",
                "period": 7,  # days
                "trend_threshold": 0.1
            },
            "pattern_recognition": {
                "sleep_patterns": "hmm_model",
                "activity_patterns": "clustering",
                "heart_rate_variability": "frequency_analysis"
            },
            "predictive_models": {
                "health_deterioration": "random_forest",
                "medication_adherence": "logistic_regression",
                "emergency_prediction": "gradient_boosting"
            }
        }

    async def process_wearable_data(
            self,
            patient_id: str,
            device_data: Dict[str, Any],
            analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Process and analyze wearable device data
        """
        try:
            # Validate and preprocess data
            processed_data = self._preprocess_data(device_data)

            # Extract features
            features = self._extract_features(processed_data)

            # Detect anomalies
            anomalies = self._detect_anomalies(processed_data, features)

            # Analyze trends
            trends = self._analyze_trends(processed_data)

            # Generate alerts
            alerts = self._generate_alerts(processed_data, anomalies)

            # Create health insights
            insights = self._generate_health_insights(
                processed_data, features, trends, anomalies
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                insights, alerts, patient_id
            )

            # Calculate health scores
            health_scores = self._calculate_health_scores(processed_data, features)

            return {
                "patient_id": patient_id,
                "analysis_date": datetime.utcnow().isoformat(),
                "data_summary": self._create_data_summary(processed_data),
                "processed_metrics": processed_data,
                "extracted_features": features,
                "anomaly_detection": anomalies,
                "trend_analysis": trends,
                "alerts": alerts,
                "health_insights": insights,
                "recommendations": recommendations,
                "health_scores": health_scores,
                "data_quality": self._assess_data_quality(device_data),
                "next_analysis": self._schedule_next_analysis(alerts)
            }

        except Exception as e:
            logger.error(f"Wearable data processing error: {e}")
            return {"error": str(e), "patient_id": patient_id}

    def _preprocess_data(self, device_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess raw wearable device data"""

        processed = {}

        for metric, data_points in device_data.items():
            if not data_points:
                continue

            # Convert to pandas DataFrame for easier processing
            df = pd.DataFrame(data_points)

            # Handle missing timestamps
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.date_range(
                    start=datetime.utcnow() - timedelta(hours=len(data_points)),
                    periods=len(data_points),
                    freq='1min'
                )
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Sort by timestamp
            df = df.sort_values('timestamp')

            # Remove outliers using IQR method
            if 'value' in df.columns:
                Q1 = df['value'].quantile(0.25)
                Q3 = df['value'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

                # Apply smoothing (moving average)
                df['smoothed_value'] = df['value'].rolling(window=5, center=True).mean()
                df['smoothed_value'].fillna(df['value'], inplace=True)

            processed[metric] = {
                "raw_data": data_points,
                "processed_data": df.to_dict('records'),
                "data_points": len(df),
                "time_range": {
                    "start": df['timestamp'].min().isoformat(),
                    "end": df['timestamp'].max().isoformat(),
                    "duration_hours": (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
                }
            }

        return processed

    def _extract_features(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from processed data"""

        features = {}

        for metric, data in processed_data.items():
            df = pd.DataFrame(data["processed_data"])

            if 'value' in df.columns or 'smoothed_value' in df.columns:
                values = df['smoothed_value'] if 'smoothed_value' in df.columns else df['value']

                # Time domain features
                time_features = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "range": float(values.max() - values.min()),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75))
                }

                # Temporal features
                temporal_features = {
                    "trend": self._calculate_trend(values),
                    "variability": float(values.std() / values.mean()) if values.mean() != 0 else 0,
                    "stability": self._calculate_stability(values)
                }

                # Pattern features
                pattern_features = {
                    "daily_pattern": self._extract_daily_pattern(df),
                    "weekly_pattern": self._extract_weekly_pattern(df) if len(df) > 7*24*60 else None
                }

                features[metric] = {
                    "time_domain": time_features,
                    "temporal": temporal_features,
                    "patterns": pattern_features
                }

        return features

    def _calculate_trend(self, values: pd.Series) -> Dict[str, Any]:
        """Calculate trend in time series data"""
        if len(values) < 2:
            return {"direction": "insufficient_data", "strength": 0.0}

        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)

        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"

        return {
            "direction": direction,
            "slope": float(slope),
            "strength": float(abs(slope)),
            "r_squared": float(r_squared),
            "significance": "significant" if r_squared > 0.5 else "weak"
        }

    def _calculate_stability(self, values: pd.Series) -> float:
        """Calculate stability score (inverse of coefficient of variation)"""
        if values.mean() == 0:
            return 0.0
        cv = values.std() / values.mean()
        return float(1 / (1 + cv))  # Higher score = more stable

    def _extract_daily_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract daily patterns from data"""
        if 'timestamp' not in df.columns or len(df) < 24:
            return {"pattern": "insufficient_data"}

        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_avg = df.groupby('hour')['value'].mean() if 'value' in df.columns else None

        if hourly_avg is not None:
            peak_hour = int(hourly_avg.idxmax())
            low_hour = int(hourly_avg.idxmin())

            return {
                "peak_hour": peak_hour,
                "low_hour": low_hour,
                "peak_value": float(hourly_avg.max()),
                "low_value": float(hourly_avg.min()),
                "daily_range": float(hourly_avg.max() - hourly_avg.min()),
                "hourly_averages": hourly_avg.to_dict()
            }

        return {"pattern": "no_clear_pattern"}

    def _extract_weekly_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract weekly patterns from data"""
        if 'timestamp' not in df.columns or len(df) < 7*24:
            return {"pattern": "insufficient_data"}

        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.day_name()
        daily_avg = df.groupby('day_of_week')['value'].mean() if 'value' in df.columns else None

        if daily_avg is not None:
            return {
                "daily_averages": daily_avg.to_dict(),
                "most_active_day": daily_avg.idxmax(),
                "least_active_day": daily_avg.idxmin(),
                "weekly_variation": float(daily_avg.std())
            }

        return {"pattern": "no_clear_pattern"}

    def _detect_anomalies(self, processed_data: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in wearable data"""

        anomalies = {}

        for metric, data in processed_data.items():
            df = pd.DataFrame(data["processed_data"])

            if 'value' in df.columns:
                values = df['value']

                # Statistical anomaly detection (Z-score method)
                z_scores = np.abs((values - values.mean()) / values.std())
                statistical_anomalies = df[z_scores > 3].to_dict('records')

                # Threshold-based anomalies
                threshold_anomalies = self._detect_threshold_anomalies(metric, df)

                # Trend anomalies
                trend_anomalies = self._detect_trend_anomalies(metric, features.get(metric, {}))

                anomalies[metric] = {
                    "statistical_anomalies": statistical_anomalies,
                    "threshold_anomalies": threshold_anomalies,
                    "trend_anomalies": trend_anomalies,
                    "total_anomalies": len(statistical_anomalies) + len(threshold_anomalies),
                    "anomaly_rate": (len(statistical_anomalies) + len(threshold_anomalies)) / len(df) if len(df) > 0 else 0
                }

        return anomalies

    def _detect_threshold_anomalies(self, metric: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect threshold-based anomalies"""

        anomalies = []
        thresholds = self.alert_thresholds.get(metric, {})

        if not thresholds or 'value' not in df.columns:
            return anomalies

        for _, row in df.iterrows():
            value = row['value']

            # Check different threshold levels
            if metric == "heart_rate":
                if value < thresholds.get("critical", {}).get("low", 0) or value > thresholds.get("critical", {}).get("high", 1000):
                    anomalies.append({
                        "timestamp": row['timestamp'],
                        "value": value,
                        "type": "critical_threshold",
                        "severity": "critical"
                    })
                elif value < thresholds.get("resting", {}).get("low", 0) or value > thresholds.get("resting", {}).get("high", 1000):
                    anomalies.append({
                        "timestamp": row['timestamp'],
                        "value": value,
                        "type": "abnormal_threshold",
                        "severity": "warning"
                    })

            elif metric == "glucose":
                if value < thresholds.get("critical_low", 0) or value > thresholds.get("critical_high", 1000):
                    anomalies.append({
                        "timestamp": row['timestamp'],
                        "value": value,
                        "type": "critical_glucose",
                        "severity": "critical"
                    })

        return anomalies

    def _detect_trend_anomalies(self, metric: str, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect trend-based anomalies"""

        anomalies = []
        temporal_features = features.get("temporal", {})
        trend = temporal_features.get("trend", {})

        # Detect concerning trends
        if trend.get("direction") == "decreasing" and metric in ["oxygen_saturation", "activity"]:
            if trend.get("strength", 0) > 0.1 and trend.get("significance") == "significant":
                anomalies.append({
                    "type": "declining_trend",
                    "metric": metric,
                    "severity": "warning",
                    "description": f"Significant declining trend in {metric}"
                })

        elif trend.get("direction") == "increasing" and metric in ["heart_rate", "glucose"]:
            if trend.get("strength", 0) > 0.1 and trend.get("significance") == "significant":
                anomalies.append({
                    "type": "increasing_trend",
                    "metric": metric,
                    "severity": "warning",
                    "description": f"Significant increasing trend in {metric}"
                })

        # Detect high variability
        variability = temporal_features.get("variability", 0)
        if variability > 0.3:  # High coefficient of variation
            anomalies.append({
                "type": "high_variability",
                "metric": metric,
                "severity": "info",
                "description": f"High variability detected in {metric}",
                "variability_score": variability
            })

        return anomalies

    def _analyze_trends(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in wearable data"""

        trends = {}

        for metric, data in processed_data.items():
            df = pd.DataFrame(data["processed_data"])

            if 'value' in df.columns and len(df) > 1:
                values = df['value']

                # Short-term trend (last 24 hours)
                recent_data = df.tail(min(24*60, len(df)))  # Last 24 hours or all data
                short_term_trend = self._calculate_trend(recent_data['value'])

                # Long-term trend (all data)
                long_term_trend = self._calculate_trend(values)

                # Seasonal patterns (if enough data)
                seasonal_analysis = self._analyze_seasonal_patterns(df) if len(df) > 7*24*60 else None

                trends[metric] = {
                    "short_term": short_term_trend,
                    "long_term": long_term_trend,
                    "seasonal": seasonal_analysis,
                    "trend_consistency": self._assess_trend_consistency(short_term_trend, long_term_trend)
                }

        return trends

    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in data"""

        # Group by day of week and hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

        # Calculate average by day of week
        daily_pattern = df.groupby('day_of_week')['value'].mean()

        # Calculate average by hour
        hourly_pattern = df.groupby('hour')['value'].mean()

        return {
            "daily_pattern": daily_pattern.to_dict(),
            "hourly_pattern": hourly_pattern.to_dict(),
            "weekend_vs_weekday": {
                "weekday_avg": float(daily_pattern[0:5].mean()),
                "weekend_avg": float(daily_pattern[5:7].mean()),
                "difference": float(daily_pattern[5:7].mean() - daily_pattern[0:5].mean())
            }
        }

    def _assess_trend_consistency(self, short_term: Dict[str, Any], long_term: Dict[str, Any]) -> str:
        """Assess consistency between short-term and long-term trends"""

        short_direction = short_term.get("direction", "stable")
        long_direction = long_term.get("direction", "stable")

        if short_direction == long_direction:
            return "consistent"
        elif short_direction == "stable" or long_direction == "stable":
            return "partially_consistent"
        else:
            return "inconsistent"

    def _generate_alerts(self, processed_data: Dict[str, Any], anomalies: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on data analysis"""

        alerts = []

        for metric, anomaly_data in anomalies.items():
            # Critical threshold alerts
            critical_anomalies = [a for a in anomaly_data.get("threshold_anomalies", []) if a.get("severity") == "critical"]

            for anomaly in critical_anomalies:
                alerts.append({
                    "alert_id": f"alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{metric}",
                    "type": "critical_threshold",
                    "metric": metric,
                    "severity": "critical",
                    "message": f"Critical {metric} value detected: {anomaly['value']}",
                    "timestamp": anomaly["timestamp"],
                    "action_required": "immediate_medical_attention",
                    "escalation": "emergency_contact"
                })

            # High anomaly rate alerts
            anomaly_rate = anomaly_data.get("anomaly_rate", 0)
            if anomaly_rate > 0.1:  # More than 10% anomalies
                alerts.append({
                    "alert_id": f"alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{metric}_rate",
                    "type": "high_anomaly_rate",
                    "metric": metric,
                    "severity": "warning",
                    "message": f"High anomaly rate detected in {metric}: {anomaly_rate:.1%}",
                    "action_required": "review_device_and_data",
                    "escalation": "healthcare_provider"
                })

            # Trend alerts
            trend_anomalies = anomaly_data.get("trend_anomalies", [])
            for trend_anomaly in trend_anomalies:
                if trend_anomaly.get("severity") == "warning":
                    alerts.append({
                        "alert_id": f"alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{metric}_trend",
                        "type": "concerning_trend",
                        "metric": metric,
                        "severity": "warning",
                        "message": trend_anomaly["description"],
                        "action_required": "monitor_closely",
                        "escalation": "routine_follow_up"
                    })

        # Sort alerts by severity
        severity_order = {"critical": 3, "warning": 2, "info": 1}
        alerts.sort(key=lambda x: severity_order.get(x["severity"], 0), reverse=True)

        return alerts

    def _generate_health_insights(
            self,
            processed_data: Dict[str, Any],
            features: Dict[str, Any],
            trends: Dict[str, Any],
            anomalies: Dict[str, Any]
    ) -> List[str]:
        """Generate health insights from wearable data analysis"""

        insights = []

        # Activity insights
        if "steps" in features:
            steps_features = features["steps"]["time_domain"]
            avg_steps = steps_features.get("mean", 0)

            if avg_steps < 5000:
                insights.append("Low daily activity detected - consider increasing physical activity")
            elif avg_steps > 10000:
                insights.append("Excellent activity levels maintained - keep up the good work!")

            # Activity trend insights
            if "steps" in trends:
                steps_trend = trends["steps"]["long_term"]
                if steps_trend.get("direction") == "increasing":
                    insights.append("Positive trend in daily activity levels")
                elif steps_trend.get("direction") == "decreasing":
                    insights.append("Declining activity trend - may need intervention")

        # Heart rate insights
        if "heart_rate" in features:
            hr_features = features["heart_rate"]["time_domain"]
            avg_hr = hr_features.get("mean", 0)
            hr_variability = features["heart_rate"]["temporal"].get("variability", 0)

            if avg_hr > 100:
                insights.append("Elevated resting heart rate detected - consider medical evaluation")
            elif avg_hr < 50:
                insights.append("Low resting heart rate - may indicate excellent fitness or medical condition")

            if hr_variability > 0.2:
                insights.append("High heart rate variability detected - may indicate stress or irregular rhythm")

        # Sleep insights (if available)
        if "sleep" in features:
            sleep_patterns = features["sleep"]["patterns"]
            if sleep_patterns.get("daily_pattern", {}).get("pattern") != "insufficient_data":
                insights.append("Sleep pattern analysis available - review for optimization opportunities")

        # Data quality insights
        total_anomalies = sum(data.get("total_anomalies", 0) for data in anomalies.values())
        if total_anomalies > 10:
            insights.append("Multiple data anomalies detected - check device placement and functionality")

        # Trend consistency insights
        inconsistent_trends = [
            metric for metric, trend_data in trends.items()
            if trend_data.get("trend_consistency") == "inconsistent"
        ]

        if inconsistent_trends:
            insights.append(f"Inconsistent trends detected in {', '.join(inconsistent_trends)} - monitor for pattern changes")

        return insights

    def _generate_recommendations(
            self,
            insights: List[str],
            alerts: List[Dict[str, Any]],
            patient_id: str
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""

        recommendations = []

        # Critical alert recommendations
        critical_alerts = [alert for alert in alerts if alert["severity"] == "critical"]
        if critical_alerts:
            recommendations.append({
                "type": "immediate_action",
                "priority": "critical",
                "recommendation": "Seek immediate medical attention",
                "reason": f"{len(critical_alerts)} critical alerts detected",
                "timeframe": "immediate"
            })

        # Activity recommendations
        activity_insights = [insight for insight in insights if "activity" in insight.lower()]
        if activity_insights:
            recommendations.append({
                "type": "lifestyle_modification",
                "priority": "medium",
                "recommendation": "Increase daily physical activity",
                "reason": "Low activity levels detected",
                "timeframe": "within_week",
                "specific_actions": [
                    "Aim for 10,000 steps per day",
                    "Take stairs instead of elevators",
                    "Schedule regular walking breaks"
                ]
            })

        # Device maintenance recommendations
        high_anomaly_alerts = [alert for alert in alerts if alert["type"] == "high_anomaly_rate"]
        if high_anomaly_alerts:
            recommendations.append({
                "type": "device_maintenance",
                "priority": "medium",
                "recommendation": "Check wearable device functionality",
                "reason": "High rate of data anomalies detected",
                "timeframe": "within_day",
                "specific_actions": [
                    "Ensure proper device placement",
                    "Check battery level",
                    "Clean device sensors",
                    "Restart device if necessary"
                ]
            })

        # Monitoring recommendations
        warning_alerts = [alert for alert in alerts if alert["severity"] == "warning"]
        if warning_alerts:
            recommendations.append({
                "type": "enhanced_monitoring",
                "priority": "medium",
                "recommendation": "Increase monitoring frequency",
                "reason": f"{len(warning_alerts)} warning alerts detected",
                "timeframe": "within_day",
                "specific_actions": [
                    "Monitor symptoms closely",
                    "Keep detailed health diary",
                    "Schedule follow-up with healthcare provider"
                ]
            })

        return recommendations

    def _calculate_health_scores(self, processed_data: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall health scores from wearable data"""

        scores = {}

        # Activity score
        if "steps" in features:
            avg_steps = features["steps"]["time_domain"].get("mean", 0)
            activity_score = min(100, (avg_steps / 10000) * 100)
            scores["activity"] = {
                "score": activity_score,
                "grade": self._get_grade(activity_score),
                "description": "Based on daily step count"
            }

        # Heart health score
        if "heart_rate" in features:
            hr_features = features["heart_rate"]["time_domain"]
            avg_hr = hr_features.get("mean", 70)
            hr_variability = features["heart_rate"]["temporal"].get("variability", 0.1)

            # Score based on resting heart rate (lower is better, within reason)
            if 60 <= avg_hr <= 80:
                hr_score = 100
            elif 50 <= avg_hr < 60 or 80 < avg_hr <= 90:
                hr_score = 80
            elif 90 < avg_hr <= 100:
                hr_score = 60
            else:
                hr_score = 40

            # Adjust for variability (moderate variability is healthy)
            if 0.1 <= hr_variability <= 0.2:
                hr_score += 10
            elif hr_variability > 0.3:
                hr_score -= 20

            hr_score = max(0, min(100, hr_score))

            scores["heart_health"] = {
                "score": hr_score,
                "grade": self._get_grade(hr_score),
                "description": "Based on heart rate patterns"
            }

        # Overall health score
        if scores:
            overall_score = sum(score_data["score"] for score_data in scores.values()) / len(scores)
            scores["overall"] = {
                "score": overall_score,
                "grade": self._get_grade(overall_score),
                "description": "Composite health score from all metrics"
            }

        return scores

    def _get_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _create_data_summary(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of processed data"""

        summary = {
            "metrics_analyzed": list(processed_data.keys()),
            "total_data_points": sum(data["data_points"] for data in processed_data.values()),
            "time_coverage": {},
            "data_completeness": {}
        }

        # Calculate time coverage
        all_start_times = []
        all_end_times = []

        for metric, data in processed_data.items():
            time_range = data["time_range"]
            all_start_times.append(pd.to_datetime(time_range["start"]))
            all_end_times.append(pd.to_datetime(time_range["end"]))

            summary["data_completeness"][metric] = {
                "data_points": data["data_points"],
                "duration_hours": time_range["duration_hours"],
                "coverage": "good" if time_range["duration_hours"] > 12 else "limited"
            }

        if all_start_times and all_end_times:
            summary["time_coverage"] = {
                "earliest_data": min(all_start_times).isoformat(),
                "latest_data": max(all_end_times).isoformat(),
                "total_duration_hours": (max(all_end_times) - min(all_start_times)).total_seconds() / 3600
            }

        return summary

    def _assess_data_quality(self, device_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of incoming device data"""

        quality_metrics = {
            "completeness": 0.0,
            "consistency": 0.0,
            "accuracy": 0.0,
            "timeliness": 0.0,
            "overall_quality": 0.0
        }

        total_metrics = len(device_data)
        if total_metrics == 0:
            return quality_metrics

        # Assess completeness
        non_empty_metrics = sum(1 for data in device_data.values() if data)
        quality_metrics["completeness"] = non_empty_metrics / total_metrics

        # Assess consistency (simplified)
        consistent_metrics = 0
        for metric, data_points in device_data.items():
            if data_points and len(data_points) > 1:
                # Check for consistent data structure
                first_keys = set(data_points[0].keys()) if data_points else set()
                all_consistent = all(set(point.keys()) == first_keys for point in data_points)
                if all_consistent:
                    consistent_metrics += 1

        quality_metrics["consistency"] = consistent_metrics / total_metrics if total_metrics > 0 else 0

        # Assess timeliness (data recency)
        most_recent_time = None
        for data_points in device_data.values():
            if data_points:
                for point in data_points:
                    if 'timestamp' in point:
                        timestamp = pd.to_datetime(point['timestamp'])
                        if most_recent_time is None or timestamp > most_recent_time:
                            most_recent_time = timestamp

        if most_recent_time:
            hours_since_last_data = (datetime.utcnow() - most_recent_time.replace(tzinfo=None)).total_seconds() / 3600
            quality_metrics["timeliness"] = max(0, 1 - (hours_since_last_data / 24))  # Decreases over 24 hours

        # Accuracy assessment (simplified - based on reasonable value ranges)
        quality_metrics["accuracy"] = 0.9  # Assume good accuracy for demo

        # Overall quality
        quality_metrics["overall_quality"] = (
                quality_metrics["completeness"] * 0.3 +
                quality_metrics["consistency"] * 0.2 +
                quality_metrics["accuracy"] * 0.3 +
                quality_metrics["timeliness"] * 0.2
        )

        return quality_metrics

    def _schedule_next_analysis(self, alerts: List[Dict[str, Any]]) -> str:
        """Schedule next analysis based on alert severity"""

        critical_alerts = [alert for alert in alerts if alert["severity"] == "critical"]
        warning_alerts = [alert for alert in alerts if alert["severity"] == "warning"]

        if critical_alerts:
            # Continuous monitoring for critical alerts
            next_analysis = datetime.utcnow() + timedelta(minutes=15)
        elif warning_alerts:
            # Hourly monitoring for warnings
            next_analysis = datetime.utcnow() + timedelta(hours=1)
        else:
            # Regular monitoring
            next_analysis = datetime.utcnow() + timedelta(hours=4)

        return next_analysis.isoformat()
