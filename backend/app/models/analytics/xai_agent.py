"""
Explainable AI Agent for Transparent Healthcare Predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json

from app.config.settings import settings

logger = logging.getLogger(__name__)

class ExplainableAIAgent:
    """
    AI agent for providing explainable AI predictions and transparent decision support
    """

    def __init__(self):
        self.explanation_methods = self._initialize_explanation_methods()
        self.feature_importance_cache = {}
        self.explanation_templates = self._load_explanation_templates()
        logger.info("✅ Explainable AI Agent initialized")

    def _initialize_explanation_methods(self) -> Dict[str, Any]:
        """Initialize explanation methods and tools"""
        return {
            "feature_importance": {
                "method": "permutation_importance",
                "supported_models": ["xgboost", "random_forest", "linear"]
            },
            "local_explanations": {
                "method": "lime_like",
                "neighborhood_size": 1000,
                "feature_selection": "auto"
            },
            "global_explanations": {
                "method": "partial_dependence",
                "feature_interactions": True
            },
            "counterfactual": {
                "method": "nearest_neighbor",
                "max_changes": 3,
                "feasibility_check": True
            }
        }

    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load explanation templates for different prediction types"""
        return {
            "risk_prediction": """
## Risk Prediction Explanation

**Patient Risk Level**: {risk_level} ({risk_score:.1%})

### Key Contributing Factors:
{contributing_factors}

### Model Confidence: {confidence:.1%}

### Clinical Reasoning:
{clinical_reasoning}

### What This Means:
{interpretation}

### Recommended Actions:
{recommendations}
""",
            "diagnosis_support": """
## Diagnostic Support Explanation

**Suggested Diagnosis**: {diagnosis}
**Confidence**: {confidence:.1%}

### Supporting Evidence:
{supporting_evidence}

### Alternative Considerations:
{alternatives}

### Recommended Next Steps:
{next_steps}
""",
            "treatment_recommendation": """
## Treatment Recommendation Explanation

**Recommended Treatment**: {treatment}
**Expected Outcome**: {expected_outcome}

### Why This Treatment:
{rationale}

### Contraindications Checked:
{contraindications}

### Monitoring Requirements:
{monitoring}
"""
        }

    async def explain_prediction(
            self,
            prediction_result: Dict[str, Any],
            model_info: Dict[str, Any],
            patient_data: Dict[str, Any],
            explanation_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for AI prediction
        """
        try:
            # Determine prediction type
            prediction_type = self._identify_prediction_type(prediction_result)

            # Generate feature importance explanation
            feature_explanation = self._explain_feature_importance(
                prediction_result, model_info, patient_data
            )

            # Generate local explanation (instance-specific)
            local_explanation = self._generate_local_explanation(
                prediction_result, model_info, patient_data
            )

            # Generate counterfactual explanations
            counterfactual_explanation = self._generate_counterfactual_explanation(
                prediction_result, patient_data
            )

            # Generate clinical reasoning
            clinical_reasoning = await self._generate_clinical_reasoning(
                prediction_result, feature_explanation, patient_data
            )

            # Create human-readable explanation
            human_explanation = self._create_human_readable_explanation(
                prediction_type, prediction_result, feature_explanation, clinical_reasoning
            )

            # Calculate explanation quality metrics
            explanation_quality = self._assess_explanation_quality(
                feature_explanation, local_explanation, clinical_reasoning
            )

            return {
                "explanation_id": f"xai_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "prediction_type": prediction_type,
                "explanation_type": explanation_type,
                "feature_importance": feature_explanation,
                "local_explanation": local_explanation,
                "counterfactual_explanation": counterfactual_explanation,
                "clinical_reasoning": clinical_reasoning,
                "human_readable": human_explanation,
                "explanation_quality": explanation_quality,
                "model_transparency": self._assess_model_transparency(model_info),
                "generated_at": datetime.utcnow().isoformat(),
                "disclaimer": "This explanation is for educational purposes and should not replace clinical judgment"
            }

        except Exception as e:
            logger.error(f"XAI explanation error: {e}")
            return {"error": str(e), "prediction_result": prediction_result}

    def _identify_prediction_type(self, prediction_result: Dict[str, Any]) -> str:
        """Identify the type of prediction to explain"""
        if "risk_score" in prediction_result:
            return "risk_prediction"
        elif "diagnosis" in prediction_result:
            return "diagnosis_support"
        elif "treatment" in prediction_result:
            return "treatment_recommendation"
        elif "readmission" in str(prediction_result).lower():
            return "readmission_risk"
        else:
            return "general_prediction"

    def _explain_feature_importance(
            self,
            prediction_result: Dict[str, Any],
            model_info: Dict[str, Any],
            patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Explain feature importance for the prediction"""

        # Extract features used in prediction
        features = self._extract_prediction_features(patient_data)

        # Calculate feature importance (simplified SHAP-like approach)
        feature_importance = self._calculate_feature_importance(features, prediction_result)

        # Rank features by importance
        ranked_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Generate explanations for top features
        top_features_explained = []
        for feature, importance in ranked_features[:10]:
            explanation = self._explain_single_feature(
                feature, importance, patient_data.get(feature)
            )
            top_features_explained.append(explanation)

        return {
            "method": "feature_importance_analysis",
            "top_features": top_features_explained,
            "feature_contributions": dict(ranked_features),
            "total_features_analyzed": len(features),
            "explanation_coverage": min(1.0, len(ranked_features) / len(features))
        }

    def _extract_prediction_features(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features used in prediction"""
        # Standard healthcare prediction features
        features = {
            "age": patient_data.get("age", 65),
            "gender": 1 if patient_data.get("gender") == "male" else 0,
            "length_of_stay": patient_data.get("length_of_stay", 3),
            "num_conditions": len(patient_data.get("conditions", [])),
            "num_medications": len(patient_data.get("medications", [])),
            "emergency_admission": 1 if patient_data.get("emergency_admission") else 0,
            "prior_admissions": patient_data.get("prior_admissions", 0),
            "comorbidity_score": patient_data.get("comorbidity_score", 0),
            "functional_status": patient_data.get("functional_status", 1),
            "social_support": 1 if patient_data.get("social_support") else 0
        }

        # Add condition-specific features
        conditions = patient_data.get("conditions", [])
        for condition in ["diabetes", "hypertension", "heart_failure", "copd"]:
            features[f"has_{condition}"] = 1 if condition in conditions else 0

        return features

    def _calculate_feature_importance(
            self,
            features: Dict[str, Any],
            prediction_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate feature importance scores"""

        # Simplified feature importance calculation
        # In production, this would use SHAP, LIME, or model-specific methods

        base_weights = {
            "age": 0.15,
            "length_of_stay": 0.12,
            "num_conditions": 0.10,
            "emergency_admission": 0.15,
            "prior_admissions": 0.10,
            "comorbidity_score": 0.08,
            "has_diabetes": 0.06,
            "has_heart_failure": 0.08,
            "has_hypertension": 0.05,
            "functional_status": 0.06,
            "social_support": 0.05
        }

        # Adjust weights based on feature values
        importance_scores = {}
        for feature, value in features.items():
            base_weight = base_weights.get(feature, 0.02)

            # Adjust importance based on feature value
            if isinstance(value, (int, float)):
                if feature == "age" and value > 75:
                    importance_scores[feature] = base_weight * 1.5
                elif feature == "length_of_stay" and value > 7:
                    importance_scores[feature] = base_weight * 1.3
                elif feature == "num_conditions" and value > 3:
                    importance_scores[feature] = base_weight * 1.4
                else:
                    importance_scores[feature] = base_weight
            else:
                importance_scores[feature] = base_weight if value else base_weight * 0.5

        # Normalize scores
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            importance_scores = {
                k: v / total_importance for k, v in importance_scores.items()
            }

        return importance_scores

    def _explain_single_feature(self, feature: str, importance: float, value: Any) -> Dict[str, Any]:
        """Explain a single feature's contribution"""

        feature_explanations = {
            "age": "Patient age affects risk due to physiological changes and comorbidity accumulation",
            "length_of_stay": "Longer hospital stays often indicate more complex medical conditions",
            "num_conditions": "Multiple medical conditions increase care complexity and risk",
            "emergency_admission": "Emergency admissions suggest acute, potentially unstable conditions",
            "prior_admissions": "Previous hospitalizations indicate ongoing health challenges",
            "comorbidity_score": "Higher comorbidity burden increases overall medical complexity",
            "has_diabetes": "Diabetes affects multiple organ systems and requires careful management",
            "has_heart_failure": "Heart failure significantly impacts prognosis and care requirements",
            "has_hypertension": "High blood pressure increases cardiovascular and stroke risk",
            "functional_status": "Functional limitations affect recovery and independence",
            "social_support": "Social support influences adherence and recovery outcomes"
        }

        # Determine impact direction
        impact_direction = "increases" if importance > 0 else "decreases"
        impact_magnitude = "significantly" if abs(importance) > 0.1 else "moderately" if abs(importance) > 0.05 else "slightly"

        return {
            "feature": feature.replace("_", " ").title(),
            "value": value,
            "importance_score": importance,
            "impact_direction": impact_direction,
            "impact_magnitude": impact_magnitude,
            "explanation": feature_explanations.get(feature, "This factor contributes to the overall risk assessment"),
            "clinical_significance": self._assess_clinical_significance(feature, value, importance)
        }

    def _assess_clinical_significance(self, feature: str, value: Any, importance: float) -> str:
        """Assess clinical significance of feature"""

        if abs(importance) > 0.15:
            significance = "High"
        elif abs(importance) > 0.08:
            significance = "Moderate"
        else:
            significance = "Low"

        # Feature-specific adjustments
        if feature == "age" and value > 80:
            significance = "High"
        elif feature == "emergency_admission" and value == 1:
            significance = "High"
        elif feature == "num_conditions" and value > 4:
            significance = "High"

        return significance

    def _generate_local_explanation(
            self,
            prediction_result: Dict[str, Any],
            model_info: Dict[str, Any],
            patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate local (instance-specific) explanation"""

        # Identify key factors for this specific patient
        patient_specific_factors = []

        age = patient_data.get("age", 65)
        if age > 75:
            patient_specific_factors.append({
                "factor": "Advanced Age",
                "value": f"{age} years",
                "impact": "Increases risk due to age-related physiological changes",
                "modifiable": False
            })

        conditions = patient_data.get("conditions", [])
        if len(conditions) > 2:
            patient_specific_factors.append({
                "factor": "Multiple Comorbidities",
                "value": f"{len(conditions)} conditions",
                "impact": "Increases complexity of care and risk of complications",
                "modifiable": True
            })

        if patient_data.get("emergency_admission"):
            patient_specific_factors.append({
                "factor": "Emergency Admission",
                "value": "Yes",
                "impact": "Suggests acute condition requiring careful monitoring",
                "modifiable": False
            })

        # Generate patient-specific recommendations
        recommendations = self._generate_patient_specific_recommendations(
            patient_specific_factors, prediction_result
        )

        return {
            "patient_specific_factors": patient_specific_factors,
            "risk_modifiers": self._identify_risk_modifiers(patient_data),
            "recommendations": recommendations,
            "confidence_factors": self._identify_confidence_factors(patient_data, prediction_result)
        }

    def _identify_risk_modifiers(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify factors that could modify risk"""
        modifiers = []

        # Modifiable risk factors
        if not patient_data.get("social_support"):
            modifiers.append({
                "factor": "Limited Social Support",
                "type": "modifiable",
                "intervention": "Social work consultation, family engagement",
                "potential_impact": "moderate"
            })

        if len(patient_data.get("medications", [])) > 10:
            modifiers.append({
                "factor": "Polypharmacy",
                "type": "modifiable",
                "intervention": "Medication review and reconciliation",
                "potential_impact": "moderate"
            })

        # Non-modifiable risk factors
        if patient_data.get("age", 0) > 80:
            modifiers.append({
                "factor": "Advanced Age",
                "type": "non_modifiable",
                "intervention": "Enhanced monitoring and support",
                "potential_impact": "high"
            })

        return modifiers

    def _identify_confidence_factors(self, patient_data: Dict[str, Any], prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Identify factors affecting prediction confidence"""

        confidence_factors = {
            "data_completeness": self._assess_data_completeness(patient_data),
            "model_certainty": prediction_result.get("confidence", 0.8),
            "edge_cases": self._detect_edge_cases(patient_data),
            "uncertainty_sources": []
        }

        # Identify uncertainty sources
        if patient_data.get("age", 65) > 90:
            confidence_factors["uncertainty_sources"].append("Very advanced age - limited training data")

        if len(patient_data.get("conditions", [])) > 8:
            confidence_factors["uncertainty_sources"].append("Unusually high number of comorbidities")

        if not confidence_factors["uncertainty_sources"]:
            confidence_factors["uncertainty_sources"].append("Standard case - high model confidence")

        return confidence_factors

    def _assess_data_completeness(self, patient_data: Dict[str, Any]) -> float:
        """Assess completeness of patient data"""
        required_fields = ["age", "gender", "conditions", "medications", "length_of_stay"]
        complete_fields = sum(1 for field in required_fields if patient_data.get(field) is not None)
        return complete_fields / len(required_fields)

    def _detect_edge_cases(self, patient_data: Dict[str, Any]) -> List[str]:
        """Detect edge cases that might affect prediction reliability"""
        edge_cases = []

        age = patient_data.get("age", 65)
        if age < 18 or age > 95:
            edge_cases.append(f"Unusual age: {age}")

        num_conditions = len(patient_data.get("conditions", []))
        if num_conditions > 10:
            edge_cases.append(f"Very high number of conditions: {num_conditions}")

        los = patient_data.get("length_of_stay", 3)
        if los > 30:
            edge_cases.append(f"Very long stay: {los} days")

        return edge_cases

    def _generate_counterfactual_explanation(
            self,
            prediction_result: Dict[str, Any],
            patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate counterfactual explanations (what-if scenarios)"""

        current_risk = prediction_result.get("risk_score", 0.5)
        counterfactuals = []

        # Age scenario (non-modifiable)
        age = patient_data.get("age", 65)
        if age > 65:
            younger_risk = max(0.1, current_risk - 0.15)
            counterfactuals.append({
                "scenario": f"If patient were 10 years younger ({age-10})",
                "risk_change": younger_risk - current_risk,
                "new_risk": younger_risk,
                "feasibility": "not_modifiable",
                "explanation": "Age is a non-modifiable risk factor"
            })

        # Social support scenario (modifiable)
        if not patient_data.get("social_support"):
            better_support_risk = max(0.1, current_risk - 0.08)
            counterfactuals.append({
                "scenario": "If patient had strong social support",
                "risk_change": better_support_risk - current_risk,
                "new_risk": better_support_risk,
                "feasibility": "modifiable",
                "explanation": "Social support can be improved through interventions"
            })

        # Medication adherence scenario
        if len(patient_data.get("medications", [])) > 5:
            better_adherence_risk = max(0.1, current_risk - 0.06)
            counterfactuals.append({
                "scenario": "If medication regimen were simplified",
                "risk_change": better_adherence_risk - current_risk,
                "new_risk": better_adherence_risk,
                "feasibility": "modifiable",
                "explanation": "Medication simplification can improve adherence"
            })

        return {
            "counterfactuals": counterfactuals,
            "most_impactful": max(counterfactuals, key=lambda x: abs(x["risk_change"])) if counterfactuals else None,
            "modifiable_scenarios": [cf for cf in counterfactuals if cf["feasibility"] == "modifiable"]
        }

    async def _generate_clinical_reasoning(
            self,
            prediction_result: Dict[str, Any],
            feature_explanation: Dict[str, Any],
            patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate clinical reasoning for the prediction"""

        # Extract key clinical factors
        top_features = feature_explanation.get("top_features", [])[:5]
        risk_score = prediction_result.get("risk_score", 0.5)
        risk_level = prediction_result.get("risk_level", "Moderate")

        # Generate reasoning narrative
        reasoning_points = []

        # Risk level reasoning
        if risk_level == "High":
            reasoning_points.append("Multiple high-risk factors are present, indicating significant concern")
        elif risk_level == "Moderate":
            reasoning_points.append("Some risk factors are present, warranting careful monitoring")
        else:
            reasoning_points.append("Risk factors are minimal, suggesting lower likelihood of adverse outcomes")

        # Feature-specific reasoning
        for feature in top_features[:3]:
            if feature["clinical_significance"] == "High":
                reasoning_points.append(
                    f"{feature['feature']} ({feature['value']}) {feature['impact_direction']} risk {feature['impact_magnitude']}"
                )

        # Clinical context
        clinical_context = self._generate_clinical_context(patient_data, risk_level)

        # Evidence strength
        evidence_strength = self._assess_evidence_strength(feature_explanation, prediction_result)

        return {
            "reasoning_narrative": reasoning_points,
            "clinical_context": clinical_context,
            "evidence_strength": evidence_strength,
            "clinical_guidelines": self._reference_clinical_guidelines(patient_data),
            "differential_considerations": self._generate_differential_considerations(patient_data)
        }

    def _generate_clinical_context(self, patient_data: Dict[str, Any], risk_level: str) -> str:
        """Generate clinical context for the prediction"""

        age = patient_data.get("age", 65)
        conditions = patient_data.get("conditions", [])

        context_parts = []

        # Age context
        if age > 75:
            context_parts.append("advanced age increases vulnerability")
        elif age < 50:
            context_parts.append("younger age is generally protective")

        # Comorbidity context
        if len(conditions) > 3:
            context_parts.append("multiple comorbidities complicate management")
        elif len(conditions) == 0:
            context_parts.append("absence of major comorbidities is favorable")

        # Risk level context
        risk_context = {
            "High": "requires intensive monitoring and intervention",
            "Moderate": "warrants enhanced care coordination",
            "Low": "allows for standard care protocols"
        }

        context_parts.append(risk_context.get(risk_level, "requires clinical judgment"))

        return ". ".join(context_parts).capitalize() + "."

    def _assess_evidence_strength(self, feature_explanation: Dict[str, Any], prediction_result: Dict[str, Any]) -> str:
        """Assess strength of evidence for prediction"""

        confidence = prediction_result.get("confidence", 0.8)
        coverage = feature_explanation.get("explanation_coverage", 0.8)

        # Calculate evidence strength
        evidence_score = (confidence + coverage) / 2

        if evidence_score > 0.8:
            return "Strong"
        elif evidence_score > 0.6:
            return "Moderate"
        else:
            return "Limited"

    def _reference_clinical_guidelines(self, patient_data: Dict[str, Any]) -> List[str]:
        """Reference relevant clinical guidelines"""
        guidelines = []
        conditions = patient_data.get("conditions", [])

        for condition in conditions:
            if "diabetes" in condition.lower():
                guidelines.append("ADA 2024 Standards of Care in Diabetes")
            elif "heart" in condition.lower():
                guidelines.append("AHA/ACC Heart Failure Guidelines")
            elif "hypertension" in condition.lower():
                guidelines.append("AHA/ACC Hypertension Guidelines")

        if not guidelines:
            guidelines.append("General medical care guidelines")

        return guidelines

    def _generate_differential_considerations(self, patient_data: Dict[str, Any]) -> List[str]:
        """Generate differential considerations"""
        considerations = []

        age = patient_data.get("age", 65)
        conditions = patient_data.get("conditions", [])

        if age > 80:
            considerations.append("Frailty assessment may be indicated")

        if len(conditions) > 5:
            considerations.append("Medication interactions should be carefully reviewed")

        if patient_data.get("emergency_admission"):
            considerations.append("Acute condition may affect baseline risk assessment")

        return considerations

    def _generate_patient_specific_recommendations(
            self,
            patient_factors: List[Dict[str, Any]],
            prediction_result: Dict[str, Any]
    ) -> List[str]:
        """Generate patient-specific recommendations"""
        recommendations = []

        risk_level = prediction_result.get("risk_level", "Moderate")

        # Risk-level specific recommendations
        if risk_level == "High":
            recommendations.extend([
                "Consider intensive case management",
                "Schedule early follow-up appointment",
                "Implement enhanced monitoring protocols"
            ])
        elif risk_level == "Moderate":
            recommendations.extend([
                "Ensure adequate discharge planning",
                "Schedule timely follow-up",
                "Review medication adherence"
            ])

        # Factor-specific recommendations
        for factor in patient_factors:
            if factor["modifiable"]:
                if "Comorbidities" in factor["factor"]:
                    recommendations.append("Optimize management of chronic conditions")
                elif "Social Support" in factor["factor"]:
                    recommendations.append("Engage social services for support assessment")

        return list(set(recommendations))  # Remove duplicates

    def _create_human_readable_explanation(
            self,
            prediction_type: str,
            prediction_result: Dict[str, Any],
            feature_explanation: Dict[str, Any],
            clinical_reasoning: Dict[str, Any]
    ) -> str:
        """Create human-readable explanation"""

        template = self.explanation_templates.get(prediction_type, self.explanation_templates["risk_prediction"])

        # Prepare template variables
        risk_score = prediction_result.get("risk_score", 0.5)
        risk_level = prediction_result.get("risk_level", "Moderate")
        confidence = prediction_result.get("confidence", 0.8)

        # Format contributing factors
        top_features = feature_explanation.get("top_features", [])[:5]
        contributing_factors = "\n".join([
            f"• {feature['feature']}: {feature['explanation']}"
            for feature in top_features
        ])

        # Format clinical reasoning
        reasoning_narrative = clinical_reasoning.get("reasoning_narrative", [])
        clinical_reasoning_text = "\n".join([f"• {point}" for point in reasoning_narrative])

        # Format interpretation
        interpretation = self._create_interpretation(risk_level, risk_score)

        # Format recommendations
        recommendations = prediction_result.get("recommendations", [])
        recommendations_text = "\n".join([f"• {rec}" for rec in recommendations[:5]])

        # Fill template
        try:
            formatted_explanation = template.format(
                risk_level=risk_level,
                risk_score=risk_score,
                contributing_factors=contributing_factors,
                confidence=confidence,
                clinical_reasoning=clinical_reasoning_text,
                interpretation=interpretation,
                recommendations=recommendations_text
            )
        except KeyError:
            # Fallback if template formatting fails
            formatted_explanation = f"""
## Prediction Explanation

**Risk Level**: {risk_level} ({risk_score:.1%})
**Confidence**: {confidence:.1%}

### Key Factors:
{contributing_factors}

### Clinical Reasoning:
{clinical_reasoning_text}

### Recommendations:
{recommendations_text}
"""

        return formatted_explanation

    def _create_interpretation(self, risk_level: str, risk_score: float) -> str:
        """Create interpretation of the prediction"""

        interpretations = {
            "High": f"The prediction indicates a high likelihood ({risk_score:.1%}) of the predicted outcome. This warrants immediate attention and intervention.",
            "Moderate": f"The prediction suggests a moderate likelihood ({risk_score:.1%}) of the predicted outcome. Careful monitoring and preventive measures are recommended.",
            "Low": f"The prediction indicates a low likelihood ({risk_score:.1%}) of the predicted outcome. Standard care protocols are appropriate."
        }

        return interpretations.get(risk_level, f"The prediction score is {risk_score:.1%}. Clinical judgment should guide next steps.")

    def _assess_explanation_quality(
            self,
            feature_explanation: Dict[str, Any],
            local_explanation: Dict[str, Any],
            clinical_reasoning: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess quality of the generated explanation"""

        # Completeness score
        completeness_factors = [
            feature_explanation.get("explanation_coverage", 0),
            1.0 if local_explanation.get("patient_specific_factors") else 0.5,
            1.0 if clinical_reasoning.get("reasoning_narrative") else 0.5
        ]
        completeness = sum(completeness_factors) / len(completeness_factors)

        # Clarity score (based on explanation structure)
        clarity_factors = [
            1.0 if len(feature_explanation.get("top_features", [])) >= 3 else 0.5,
            1.0 if clinical_reasoning.get("evidence_strength") == "Strong" else 0.7 if clinical_reasoning.get("evidence_strength") == "Moderate" else 0.5
        ]
        clarity = sum(clarity_factors) / len(clarity_factors)

        # Overall quality
        overall_quality = (completeness * 0.6 + clarity * 0.4)

        return {
            "completeness_score": completeness,
            "clarity_score": clarity,
            "overall_quality": overall_quality,
            "quality_grade": "Excellent" if overall_quality > 0.8 else "Good" if overall_quality > 0.6 else "Fair"
        }

    def _assess_model_transparency(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess transparency of the underlying model"""

        model_type = model_info.get("model_type", "unknown")

        transparency_scores = {
            "linear": 0.9,
            "logistic_regression": 0.9,
            "decision_tree": 0.8,
            "random_forest": 0.6,
            "xgboost": 0.6,
            "neural_network": 0.3,
            "deep_learning": 0.2,
            "unknown": 0.5
        }

        transparency_score = transparency_scores.get(model_type, 0.5)

        return {
            "model_type": model_type,
            "transparency_score": transparency_score,
            "interpretability": "High" if transparency_score > 0.7 else "Medium" if transparency_score > 0.4 else "Low",
            "explanation_reliability": "High" if transparency_score > 0.6 else "Medium"
        }
