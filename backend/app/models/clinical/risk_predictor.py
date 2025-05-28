"""
Advanced Risk Prediction with Multiple ML Models
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import openai
import json
import pickle
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from app.config.settings import settings

logger = logging.getLogger(__name__)

class RiskPredictor:
    """
    Advanced risk prediction using ensemble of ML models
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.openai_client = None
        self._initialize_models()
        logger.info("✅ Risk Predictor initialized")

    def _initialize_models(self):
        """Initialize ML models and OpenAI client"""
        try:
            # Initialize OpenAI client
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

            # Initialize ML models
            self._setup_readmission_model()
            self._setup_mortality_model()
            self._setup_complication_model()

            logger.info("✅ All risk prediction models initialized")

        except Exception as e:
            logger.error(f"❌ Risk predictor initialization error: {e}")

    def _setup_readmission_model(self):
        """Setup 30-day readmission prediction model"""
        # XGBoost model for structured data
        self.models['readmission_xgb'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )

        # Random Forest for comparison
        self.models['readmission_rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        # Scaler for features
        self.scalers['readmission'] = StandardScaler()

        # Train with synthetic data (in production, use real data)
        self._train_readmission_model()

    def _setup_mortality_model(self):
        """Setup mortality risk prediction model"""
        self.models['mortality_xgb'] = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.05,
            random_state=42
        )

        self.scalers['mortality'] = StandardScaler()
        self._train_mortality_model()

    def _setup_complication_model(self):
        """Setup complication risk prediction model"""
        self.models['complication_xgb'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )

        self.scalers['complication'] = StandardScaler()
        self._train_complication_model()

    def _train_readmission_model(self):
        """Train readmission model with synthetic data"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 10000

        # Features: age, length_of_stay, num_diagnoses, num_medications,
        # emergency_admission, discharge_disposition, comorbidity_score
        X = np.random.randn(n_samples, 15)

        # Create realistic correlations
        X[:, 0] = np.random.uniform(18, 95, n_samples)  # age
        X[:, 1] = np.random.exponential(3, n_samples)   # length_of_stay
        X[:, 2] = np.random.poisson(2, n_samples)       # num_diagnoses
        X[:, 3] = np.random.poisson(5, n_samples)       # num_medications
        X[:, 4] = np.random.binomial(1, 0.3, n_samples) # emergency_admission

        # Generate target with realistic relationships
        risk_score = (
                0.02 * X[:, 0] +  # age effect
                0.1 * X[:, 1] +   # length of stay effect
                0.3 * X[:, 2] +   # diagnoses effect
                0.1 * X[:, 3] +   # medications effect
                0.5 * X[:, 4] +   # emergency admission effect
                np.random.normal(0, 0.5, n_samples)  # noise
        )

        y = (risk_score > np.percentile(risk_score, 70)).astype(int)

        # Scale features
        X_scaled = self.scalers['readmission'].fit_transform(X)

        # Train models
        self.models['readmission_xgb'].fit(X_scaled, y)
        self.models['readmission_rf'].fit(X_scaled, y)

        # Store feature importance
        self.feature_importance['readmission'] = {
            'xgb': self.models['readmission_xgb'].feature_importances_,
            'rf': self.models['readmission_rf'].feature_importances_
        }

    def _train_mortality_model(self):
        """Train mortality model with synthetic data"""
        np.random.seed(43)
        n_samples = 8000

        X = np.random.randn(n_samples, 12)
        X[:, 0] = np.random.uniform(18, 95, n_samples)  # age
        X[:, 1] = np.random.exponential(5, n_samples)   # severity_score

        # Higher mortality risk with age and severity
        risk_score = (
                0.05 * X[:, 0] +
                0.8 * X[:, 1] +
                np.random.normal(0, 1, n_samples)
        )

        y = (risk_score > np.percentile(risk_score, 85)).astype(int)

        X_scaled = self.scalers['mortality'].fit_transform(X)
        self.models['mortality_xgb'].fit(X_scaled, y)

        self.feature_importance['mortality'] = {
            'xgb': self.models['mortality_xgb'].feature_importances_
        }

    def _train_complication_model(self):
        """Train complication model with synthetic data"""
        np.random.seed(44)
        n_samples = 12000

        X = np.random.randn(n_samples, 10)
        X[:, 0] = np.random.uniform(18, 95, n_samples)  # age
        X[:, 1] = np.random.poisson(3, n_samples)       # procedures

        risk_score = (
                0.03 * X[:, 0] +
                0.4 * X[:, 1] +
                np.random.normal(0, 0.8, n_samples)
        )

        y = (risk_score > np.percentile(risk_score, 75)).astype(int)

        X_scaled = self.scalers['complication'].fit_transform(X)
        self.models['complication_xgb'].fit(X_scaled, y)

        self.feature_importance['complication'] = {
            'xgb': self.models['complication_xgb'].feature_importances_
        }

    async def predict_risk(
            self,
            patient_data: Dict[str, Any],
            risk_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Predict multiple types of risk for a patient
        """
        try:
            if risk_types is None:
                risk_types = ['readmission', 'mortality', 'complication']

            results = {
                'patient_id': patient_data.get('patient_id'),
                'predictions': {},
                'overall_risk': {},
                'recommendations': [],
                'timestamp': datetime.utcnow().isoformat()
            }

            # Extract features
            features = self._extract_features(patient_data)

            # Predict each risk type
            for risk_type in risk_types:
                if risk_type in self.models:
                    prediction = await self._predict_single_risk(risk_type, features, patient_data)
                    results['predictions'][risk_type] = prediction

            # Calculate overall risk
            results['overall_risk'] = self._calculate_overall_risk(results['predictions'])

            # Generate recommendations
            results['recommendations'] = await self._generate_recommendations(
                results['predictions'], patient_data
            )

            # Add explainability
            results['explanations'] = await self._generate_explanations(
                results['predictions'], features, patient_data
            )

            return results

        except Exception as e:
            logger.error(f"Risk prediction error: {e}")
            return {'error': str(e), 'patient_id': patient_data.get('patient_id')}

    def _extract_features(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from patient data"""
        # Standard feature extraction
        features = [
            patient_data.get('age', 65),
            patient_data.get('length_of_stay', 3),
            len(patient_data.get('conditions', [])),
            len(patient_data.get('medications', [])),
            1 if patient_data.get('emergency_admission') else 0,
            patient_data.get('discharge_disposition', 1),
            patient_data.get('comorbidity_score', 0),
            patient_data.get('lab_abnormalities', 0),
            patient_data.get('vital_instability', 0),
            patient_data.get('functional_status', 1),
            1 if patient_data.get('social_support_limited') else 0,
            patient_data.get('prior_admissions', 0),
            patient_data.get('medication_adherence_score', 0.8),
            patient_data.get('cognitive_impairment', 0),
            patient_data.get('polypharmacy_risk', 0)
        ]

        return np.array(features).reshape(1, -1)

    async def _predict_single_risk(
            self,
            risk_type: str,
            features: np.ndarray,
            patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict single risk type"""

        # Adjust features for specific risk type
        if risk_type == 'readmission':
            risk_features = features[:, :15]  # Use all 15 features
        elif risk_type == 'mortality':
            risk_features = features[:, :12]  # Use first 12 features
        elif risk_type == 'complication':
            risk_features = features[:, :10]  # Use first 10 features
        else:
            risk_features = features

        # Scale features
        scaled_features = self.scalers[risk_type].transform(risk_features)

        # Get predictions from ensemble
        model_key = f'{risk_type}_xgb'
        if model_key in self.models:
            # XGBoost prediction
            xgb_prob = self.models[model_key].predict_proba(scaled_features)[0][1]

            # Random Forest prediction (if available)
            rf_key = f'{risk_type}_rf'
            if rf_key in self.models:
                rf_prob = self.models[rf_key].predict_proba(scaled_features)[0][1]
                ensemble_prob = (xgb_prob + rf_prob) / 2
            else:
                ensemble_prob = xgb_prob

            # Add clinical context with LLM
            clinical_adjustment = await self._get_clinical_adjustment(
                risk_type, patient_data, ensemble_prob
            )

            final_prob = min(1.0, max(0.0, ensemble_prob + clinical_adjustment))

            return {
                'risk_score': final_prob,
                'risk_level': self._categorize_risk(final_prob),
                'model_scores': {
                    'xgb': xgb_prob,
                    'rf': rf_prob if rf_key in self.models else None,
                    'ensemble': ensemble_prob,
                    'clinical_adjustment': clinical_adjustment
                },
                'confidence': self._calculate_prediction_confidence(final_prob, ensemble_prob)
            }

        return {'error': f'Model not found for {risk_type}'}

    async def _get_clinical_adjustment(
            self,
            risk_type: str,
            patient_data: Dict[str, Any],
            base_risk: float
    ) -> float:
        """Get clinical adjustment using LLM"""

        if not self.openai_client:
            return 0.0

        try:
            prompt = f"""
            As a clinical expert, adjust the {risk_type} risk prediction based on clinical context:
            
            Base ML Risk Score: {base_risk:.3f}
            Patient Data: {json.dumps(patient_data, indent=2)}
            
            Consider:
            1. Clinical factors not captured by ML model
            2. Social determinants of health
            3. Patient-specific circumstances
            4. Recent clinical guidelines
            
            Provide adjustment between -0.2 and +0.2 as JSON:
            {{"adjustment": 0.05, "reasoning": "explanation"}}
            """

            response = await self.openai_client.chat.completions.create(
                model=settings.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a clinical expert providing risk adjustments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            result = json.loads(response.choices[0].message.content)
            return float(result.get('adjustment', 0.0))

        except Exception as e:
            logger.error(f"Clinical adjustment error: {e}")
            return 0.0

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk level"""
        if risk_score < 0.3:
            return 'Low'
        elif risk_score < 0.6:
            return 'Moderate'
        elif risk_score < 0.8:
            return 'High'
        else:
            return 'Very High'

    def _calculate_prediction_confidence(self, final_prob: float, ensemble_prob: float) -> float:
        """Calculate prediction confidence"""
        # Higher confidence when models agree and probability is not near 0.5
        model_agreement = 1.0 - abs(final_prob - ensemble_prob)
        probability_certainty = 2 * abs(final_prob - 0.5)

        return (model_agreement + probability_certainty) / 2

    def _calculate_overall_risk(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall patient risk"""
        risk_scores = []
        risk_levels = []

        for risk_type, prediction in predictions.items():
            if 'risk_score' in prediction:
                risk_scores.append(prediction['risk_score'])
                risk_levels.append(prediction['risk_level'])

        if not risk_scores:
            return {'overall_score': 0.0, 'overall_level': 'Unknown'}

        # Weighted average (readmission gets higher weight)
        weights = {'readmission': 0.4, 'mortality': 0.4, 'complication': 0.2}
        weighted_score = 0.0
        total_weight = 0.0

        for risk_type, prediction in predictions.items():
            if 'risk_score' in prediction:
                weight = weights.get(risk_type, 0.33)
                weighted_score += prediction['risk_score'] * weight
                total_weight += weight

        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0

        return {
            'overall_score': overall_score,
            'overall_level': self._categorize_risk(overall_score),
            'component_risks': len(risk_scores),
            'highest_risk': max(risk_scores),
            'risk_distribution': {level: risk_levels.count(level) for level in set(risk_levels)}
        }

    async def _generate_recommendations(
            self,
            predictions: Dict[str, Any],
            patient_data: Dict[str, Any]
    ) -> List[str]:
        """Generate risk-based recommendations"""

        recommendations = []

        # Risk-specific recommendations
        for risk_type, prediction in predictions.items():
            if 'risk_level' in prediction:
                risk_level = prediction['risk_level']

                if risk_type == 'readmission':
                    if risk_level in ['High', 'Very High']:
                        recommendations.extend([
                            'Schedule follow-up within 48-72 hours',
                            'Ensure medication reconciliation',
                            'Arrange home health services',
                            'Provide 24/7 contact information'
                        ])
                    elif risk_level == 'Moderate':
                        recommendations.extend([
                            'Schedule follow-up within 7 days',
                            'Review discharge instructions',
                            'Confirm medication understanding'
                        ])

                elif risk_type == 'mortality':
                    if risk_level in ['High', 'Very High']:
                        recommendations.extend([
                            'Consider palliative care consultation',
                            'Advanced directive discussion',
                            'Intensive monitoring',
                            'Family meeting recommended'
                        ])

                elif risk_type == 'complication':
                    if risk_level in ['High', 'Very High']:
                        recommendations.extend([
                            'Enhanced monitoring protocols',
                            'Prophylactic interventions',
                            'Specialist consultation',
                            'Patient/family education on warning signs'
                        ])

        # Remove duplicates
        return list(set(recommendations))

    async def _generate_explanations(
            self,
            predictions: Dict[str, Any],
            features: np.ndarray,
            patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate explanations for predictions"""

        explanations = {}

        for risk_type, prediction in predictions.items():
            if 'risk_score' in prediction:
                # Feature importance explanation
                if risk_type in self.feature_importance:
                    feature_names = [
                        'age', 'length_of_stay', 'num_conditions', 'num_medications',
                        'emergency_admission', 'discharge_disposition', 'comorbidity_score',
                        'lab_abnormalities', 'vital_instability', 'functional_status',
                        'social_support_limited', 'prior_admissions', 'medication_adherence',
                        'cognitive_impairment', 'polypharmacy_risk'
                    ]

                    importance = self.feature_importance[risk_type]['xgb']
                    top_features = sorted(
                        zip(feature_names[:len(importance)], importance, features[0][:len(importance)]),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]

                    explanations[risk_type] = {
                        'top_risk_factors': [
                            {
                                'feature': feat,
                                'importance': float(imp),
                                'value': float(val),
                                'impact': 'High' if imp > 0.1 else 'Moderate' if imp > 0.05 else 'Low'
                            }
                            for feat, imp, val in top_features
                        ],
                        'risk_score': prediction['risk_score'],
                        'interpretation': self._interpret_risk_factors(top_features, risk_type)
                    }

        return explanations

    def _interpret_risk_factors(self, top_features: List, risk_type: str) -> str:
        """Interpret top risk factors"""
        interpretations = []

        for feature, importance, value in top_features[:3]:
            if feature == 'age' and value > 65:
                interpretations.append(f"Advanced age ({value:.0f} years) increases {risk_type} risk")
            elif feature == 'length_of_stay' and value > 5:
                interpretations.append(f"Extended hospital stay ({value:.1f} days) indicates complexity")
            elif feature == 'emergency_admission' and value > 0:
                interpretations.append("Emergency admission suggests acute condition")
            elif feature == 'num_conditions' and value > 2:
                interpretations.append(f"Multiple conditions ({value:.0f}) increase risk")

        return "; ".join(interpretations) if interpretations else "Standard risk factors present"
