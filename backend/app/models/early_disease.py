"""
Early Disease Detection using Longitudinal EHR and Multimodal AI
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import json

from app.config.settings import settings

logger = logging.getLogger(__name__)

class EarlyDiseasePredictor:
    """
    Advanced early disease detection using longitudinal data and multimodal AI
    """

    def __init__(self):
        self.disease_models = self._initialize_disease_models()
        self.anomaly_detectors = self._initialize_anomaly_detectors()
        self.risk_calculators = self._initialize_risk_calculators()
        self.biomarker_patterns = self._load_biomarker_patterns()
        self.temporal_models = self._initialize_temporal_models()
        logger.info("✅ Early Disease Predictor initialized")

    def _initialize_disease_models(self) -> Dict[str, Any]:
        """Initialize disease-specific prediction models"""
        return {
            "alzheimers": {
                "model": xgb.XGBClassifier(n_estimators=100, random_state=42),
                "features": ["age", "cognitive_scores", "brain_imaging", "genetic_markers", "biomarkers"],
                "risk_factors": ["apoe4", "family_history", "education_level", "cardiovascular_risk"],
                "early_indicators": ["memory_decline", "language_difficulties", "executive_dysfunction"],
                "biomarkers": ["amyloid_beta", "tau_protein", "neurofilament_light"]
            },
            "parkinsons": {
                "model": xgb.XGBClassifier(n_estimators=100, random_state=42),
                "features": ["motor_symptoms", "non_motor_symptoms", "datscan", "genetic_testing"],
                "risk_factors": ["age", "male_gender", "pesticide_exposure", "head_trauma"],
                "early_indicators": ["tremor", "bradykinesia", "rigidity", "postural_instability"],
                "biomarkers": ["alpha_synuclein", "dopamine_transporter"]
            },
            "cancer_lung": {
                "model": xgb.XGBClassifier(n_estimators=150, random_state=42),
                "features": ["smoking_history", "ct_imaging", "biomarkers", "genetic_mutations"],
                "risk_factors": ["smoking", "radon_exposure", "asbestos", "family_history"],
                "early_indicators": ["persistent_cough", "chest_pain", "weight_loss", "fatigue"],
                "biomarkers": ["cea", "cyfra21_1", "scc_antigen", "ctdna"]
            },
            "diabetes_type2": {
                "model": xgb.XGBClassifier(n_estimators=100, random_state=42),
                "features": ["glucose_trends", "insulin_resistance", "bmi_trajectory", "genetic_risk"],
                "risk_factors": ["obesity", "family_history", "sedentary_lifestyle", "metabolic_syndrome"],
                "early_indicators": ["prediabetes", "insulin_resistance", "metabolic_dysfunction"],
                "biomarkers": ["hba1c", "fasting_glucose", "insulin", "c_peptide"]
            },
            "cardiovascular": {
                "model": xgb.XGBClassifier(n_estimators=120, random_state=42),
                "features": ["lipid_profile", "blood_pressure", "inflammation_markers", "imaging"],
                "risk_factors": ["hypertension", "dyslipidemia", "smoking", "diabetes"],
                "early_indicators": ["endothelial_dysfunction", "arterial_stiffness", "subclinical_atherosclerosis"],
                "biomarkers": ["troponin", "bnp", "crp", "ldl_cholesterol"]
            }
        }

    def _initialize_anomaly_detectors(self) -> Dict[str, Any]:
        """Initialize anomaly detection models for each disease"""
        detectors = {}

        for disease in self.disease_models.keys():
            detectors[disease] = {
                "isolation_forest": IsolationForest(contamination=0.1, random_state=42),
                "scaler": StandardScaler(),
                "threshold": -0.5
            }

        return detectors

    def _initialize_risk_calculators(self) -> Dict[str, Any]:
        """Initialize risk calculation algorithms"""
        return {
            "framingham_risk": {
                "factors": ["age", "gender", "cholesterol", "hdl", "blood_pressure", "smoking", "diabetes"],
                "weights": [0.2, 0.1, 0.15, 0.1, 0.2, 0.15, 0.1]
            },
            "ascvd_risk": {
                "factors": ["age", "gender", "race", "cholesterol", "hdl", "blood_pressure", "smoking", "diabetes"],
                "weights": [0.25, 0.1, 0.05, 0.15, 0.1, 0.2, 0.1, 0.05]
            },
            "diabetes_risk": {
                "factors": ["age", "bmi", "family_history", "hypertension", "physical_activity"],
                "weights": [0.2, 0.3, 0.2, 0.15, 0.15]
            }
        }

    def _load_biomarker_patterns(self) -> Dict[str, Any]:
        """Load biomarker patterns for early detection"""
        return {
            "alzheimers": {
                "amyloid_beta_42": {"normal": ">500 pg/mL", "concerning": "<500 pg/mL", "abnormal": "<200 pg/mL"},
                "tau_protein": {"normal": "<300 pg/mL", "concerning": "300-400 pg/mL", "abnormal": ">400 pg/mL"},
                "ptau_181": {"normal": "<20 pg/mL", "concerning": "20-30 pg/mL", "abnormal": ">30 pg/mL"}
            },
            "cardiovascular": {
                "troponin_i": {"normal": "<0.04 ng/mL", "concerning": "0.04-0.1 ng/mL", "abnormal": ">0.1 ng/mL"},
                "bnp": {"normal": "<100 pg/mL", "concerning": "100-300 pg/mL", "abnormal": ">300 pg/mL"},
                "crp": {"normal": "<1 mg/L", "concerning": "1-3 mg/L", "abnormal": ">3 mg/L"}
            },
            "cancer": {
                "cea": {"normal": "<3 ng/mL", "concerning": "3-5 ng/mL", "abnormal": ">5 ng/mL"},
                "ca_125": {"normal": "<35 U/mL", "concerning": "35-50 U/mL", "abnormal": ">50 U/mL"},
                "psa": {"normal": "<4 ng/mL", "concerning": "4-10 ng/mL", "abnormal": ">10 ng/mL"}
            }
        }

    def _initialize_temporal_models(self) -> Dict[str, Any]:
        """Initialize temporal pattern recognition models"""
        return {
            "trend_analysis": {
                "window_size": 12,  # months
                "min_data_points": 3,
                "significance_threshold": 0.05
            },
            "seasonality_detection": {
                "periods": [3, 6, 12],  # months
                "min_cycles": 2
            },
            "change_point_detection": {
                "sensitivity": 0.8,
                "min_segment_length": 2
            }
        }

    async def predict_early_disease(
            self,
            patient_id: str,
            longitudinal_data: Optional[Dict[str, Any]] = None,
            target_diseases: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Predict early disease onset using longitudinal data
        """
        try:
            # Get patient's longitudinal data
            if not longitudinal_data:
                longitudinal_data = await self._get_longitudinal_data(patient_id)

            # Determine diseases to screen for
            if not target_diseases:
                target_diseases = list(self.disease_models.keys())

            results = {
                "patient_id": patient_id,
                "screening_date": datetime.utcnow().isoformat(),
                "diseases_screened": target_diseases,
                "predictions": {},
                "risk_summary": {},
                "recommendations": [],
                "follow_up_plan": {}
            }

            # Analyze each target disease
            for disease in target_diseases:
                if disease in self.disease_models:
                    prediction = await self._predict_single_disease(
                        disease, longitudinal_data, patient_id
                    )
                    results["predictions"][disease] = prediction

            # Generate risk summary
            results["risk_summary"] = self._generate_risk_summary(results["predictions"])

            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(results["predictions"])

            # Create follow-up plan
            results["follow_up_plan"] = self._create_follow_up_plan(results["predictions"])

            # Add temporal analysis
            results["temporal_analysis"] = await self._analyze_temporal_patterns(longitudinal_data)

            return results

        except Exception as e:
            logger.error(f"Early disease prediction error: {e}")
            return {"error": str(e), "patient_id": patient_id}

    async def _get_longitudinal_data(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's longitudinal data"""
        # In production, this would query the database for historical data
        # For demo, generate synthetic longitudinal data

        import random
        from datetime import datetime, timedelta

        # Generate 2 years of monthly data
        data_points = []
        base_date = datetime.utcnow() - timedelta(days=730)

        for i in range(24):  # 24 months
            date = base_date + timedelta(days=30*i)

            # Simulate gradual changes over time
            trend_factor = i / 24.0  # 0 to 1 over 2 years

            data_point = {
                "date": date.isoformat(),
                "age": 65 + (i / 12),  # Age increases
                "bmi": 28 + random.uniform(-2, 2) + (trend_factor * 2),  # Gradual weight gain
                "blood_pressure_systolic": 130 + random.uniform(-10, 10) + (trend_factor * 15),
                "blood_pressure_diastolic": 80 + random.uniform(-5, 5) + (trend_factor * 8),
                "cholesterol_total": 200 + random.uniform(-20, 20) + (trend_factor * 30),
                "cholesterol_ldl": 120 + random.uniform(-15, 15) + (trend_factor * 25),
                "cholesterol_hdl": 50 + random.uniform(-5, 5) - (trend_factor * 5),
                "glucose_fasting": 95 + random.uniform(-10, 10) + (trend_factor * 20),
                "hba1c": 5.5 + random.uniform(-0.2, 0.2) + (trend_factor * 0.8),
                "creatinine": 1.0 + random.uniform(-0.1, 0.1) + (trend_factor * 0.3),
                "gfr": 90 - random.uniform(0, 5) - (trend_factor * 15),
                "cognitive_score": 30 - random.uniform(0, 2) - (trend_factor * 3),  # Declining
                "exercise_frequency": max(0, 5 - random.uniform(0, 1) - (trend_factor * 2)),
                "smoking_status": random.choice([0, 1]) if i < 12 else 0,  # Quit smoking
                "medication_adherence": max(0.6, 1.0 - random.uniform(0, 0.1) - (trend_factor * 0.2))
            }

            data_points.append(data_point)

        return {
            "patient_id": patient_id,
            "data_points": data_points,
            "timespan_months": 24,
            "data_frequency": "monthly"
        }

    async def _predict_single_disease(
            self,
            disease: str,
            longitudinal_data: Dict[str, Any],
            patient_id: str
    ) -> Dict[str, Any]:
        """Predict single disease risk"""

        disease_config = self.disease_models[disease]
        data_points = longitudinal_data["data_points"]

        # Extract features for this disease
        features = self._extract_disease_features(disease, data_points)

        # Analyze temporal patterns
        temporal_analysis = self._analyze_disease_temporal_patterns(disease, data_points)

        # Calculate risk using multiple approaches
        ml_risk = self._calculate_ml_risk(disease, features)
        biomarker_risk = self._calculate_biomarker_risk(disease, data_points)
        temporal_risk = self._calculate_temporal_risk(disease, temporal_analysis)

        # Combine risk scores
        combined_risk = self._combine_risk_scores(ml_risk, biomarker_risk, temporal_risk)

        # Detect anomalies
        anomaly_score = self._detect_anomalies(disease, features)

        # Generate disease-specific insights
        insights = self._generate_disease_insights(disease, data_points, temporal_analysis)

        return {
            "disease": disease,
            "risk_scores": {
                "ml_risk": ml_risk,
                "biomarker_risk": biomarker_risk,
                "temporal_risk": temporal_risk,
                "combined_risk": combined_risk,
                "anomaly_score": anomaly_score
            },
            "risk_level": self._categorize_risk(combined_risk),
            "confidence": self._calculate_confidence(ml_risk, biomarker_risk, temporal_risk),
            "temporal_analysis": temporal_analysis,
            "insights": insights,
            "early_indicators": self._identify_early_indicators(disease, data_points),
            "recommended_tests": self._recommend_tests(disease, combined_risk),
            "intervention_opportunities": self._identify_interventions(disease, data_points)
        }

    def _extract_disease_features(self, disease: str, data_points: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features relevant to specific disease"""

        if not data_points:
            return np.array([]).reshape(0, -1)

        # Get latest data point
        latest = data_points[-1]

        # Disease-specific feature extraction
        if disease == "alzheimers":
            features = [
                latest.get("age", 65),
                latest.get("cognitive_score", 30),
                latest.get("education_years", 12),
                1 if latest.get("apoe4_positive") else 0,
                1 if latest.get("family_history_dementia") else 0,
                latest.get("brain_volume", 1000),
                latest.get("amyloid_pet", 0.5),
                latest.get("tau_pet", 0.3)
            ]
        elif disease == "diabetes_type2":
            features = [
                latest.get("age", 65),
                latest.get("bmi", 28),
                latest.get("glucose_fasting", 95),
                latest.get("hba1c", 5.5),
                1 if latest.get("family_history_diabetes") else 0,
                latest.get("exercise_frequency", 3),
                latest.get("waist_circumference", 90),
                latest.get("triglycerides", 150)
            ]
        elif disease == "cardiovascular":
            features = [
                latest.get("age", 65),
                latest.get("blood_pressure_systolic", 130),
                latest.get("cholesterol_total", 200),
                latest.get("cholesterol_hdl", 50),
                latest.get("smoking_status", 0),
                1 if latest.get("diabetes") else 0,
                latest.get("crp", 1.0),
                latest.get("calcium_score", 0)
            ]
        else:
            # Generic features
            features = [
                latest.get("age", 65),
                latest.get("bmi", 28),
                latest.get("blood_pressure_systolic", 130),
                latest.get("cholesterol_total", 200),
                latest.get("glucose_fasting", 95)
            ]

        return np.array(features).reshape(1, -1)

    def _analyze_disease_temporal_patterns(self, disease: str, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns for specific disease"""

        if len(data_points) < 3:
            return {"insufficient_data": True}

        # Extract time series for key biomarkers
        dates = [point["date"] for point in data_points]

        patterns = {
            "trends": {},
            "volatility": {},
            "change_points": {},
            "acceleration": {}
        }

        # Disease-specific biomarker analysis
        if disease == "diabetes_type2":
            biomarkers = ["glucose_fasting", "hba1c", "bmi"]
        elif disease == "cardiovascular":
            biomarkers = ["blood_pressure_systolic", "cholesterol_ldl", "crp"]
        elif disease == "alzheimers":
            biomarkers = ["cognitive_score", "brain_volume"]
        else:
            biomarkers = ["glucose_fasting", "blood_pressure_systolic", "cholesterol_total"]

        for biomarker in biomarkers:
            values = [point.get(biomarker, 0) for point in data_points]

            if len(values) >= 3:
                # Calculate trend
                patterns["trends"][biomarker] = self._calculate_trend(values)

                # Calculate volatility
                patterns["volatility"][biomarker] = np.std(values)

                # Detect change points
                patterns["change_points"][biomarker] = self._detect_change_points(values)

                # Calculate acceleration (second derivative)
                if len(values) >= 4:
                    patterns["acceleration"][biomarker] = self._calculate_acceleration(values)

        return patterns

    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend in time series"""
        if len(values) < 2:
            return {"slope": 0.0, "r_squared": 0.0}

        x = np.arange(len(values))
        y = np.array(values)

        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)

        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            "slope": float(slope),
            "r_squared": float(r_squared),
            "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        }

    def _detect_change_points(self, values: List[float]) -> List[int]:
        """Detect change points in time series"""
        if len(values) < 4:
            return []

        change_points = []
        threshold = np.std(values) * 1.5  # Threshold for significant change

        for i in range(2, len(values) - 1):
            # Compare before and after segments
            before = np.mean(values[:i])
            after = np.mean(values[i:])

            if abs(after - before) > threshold:
                change_points.append(i)

        return change_points

    def _calculate_acceleration(self, values: List[float]) -> float:
        """Calculate acceleration (second derivative)"""
        if len(values) < 3:
            return 0.0

        # Calculate first differences
        first_diff = np.diff(values)

        # Calculate second differences (acceleration)
        second_diff = np.diff(first_diff)

        return float(np.mean(second_diff))

    def _calculate_ml_risk(self, disease: str, features: np.ndarray) -> float:
        """Calculate ML-based risk score"""
        if features.size == 0:
            return 0.5  # Default moderate risk

        # For demo, use a simple scoring function
        # In production, this would use trained models

        model_config = self.disease_models[disease]

        # Normalize features (simplified)
        normalized_features = features[0] / (features[0].max() + 1e-8)

        # Calculate weighted score
        weights = np.random.random(len(normalized_features))
        weights = weights / weights.sum()

        risk_score = np.dot(normalized_features, weights)

        return float(np.clip(risk_score, 0.0, 1.0))

    def _calculate_biomarker_risk(self, disease: str, data_points: List[Dict[str, Any]]) -> float:
        """Calculate biomarker-based risk score"""
        if not data_points:
            return 0.5

        latest = data_points[-1]
        risk_score = 0.0

        # Disease-specific biomarker analysis
        if disease == "diabetes_type2":
            hba1c = latest.get("hba1c", 5.5)
            glucose = latest.get("glucose_fasting", 95)

            # Risk increases with higher values
            risk_score += min(1.0, (hba1c - 5.0) / 2.0) * 0.6  # HbA1c contribution
            risk_score += min(1.0, (glucose - 90) / 40.0) * 0.4  # Glucose contribution

        elif disease == "cardiovascular":
            bp_sys = latest.get("blood_pressure_systolic", 130)
            cholesterol = latest.get("cholesterol_ldl", 120)

            risk_score += min(1.0, (bp_sys - 120) / 40.0) * 0.5
            risk_score += min(1.0, (cholesterol - 100) / 60.0) * 0.5

        elif disease == "alzheimers":
            cognitive = latest.get("cognitive_score", 30)

            # Risk increases with lower cognitive scores
            risk_score += max(0.0, (28 - cognitive) / 10.0)

        return float(np.clip(risk_score, 0.0, 1.0))

    def _calculate_temporal_risk(self, disease: str, temporal_analysis: Dict[str, Any]) -> float:
        """Calculate temporal pattern-based risk"""
        if temporal_analysis.get("insufficient_data"):
            return 0.5

        risk_score = 0.0
        trends = temporal_analysis.get("trends", {})

        # Analyze concerning trends
        for biomarker, trend_data in
