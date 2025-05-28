"""
Chronic Disease Management Agent
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random
import logging

from app.config.settings import settings

logger = logging.getLogger(__name__)

class ChronicCareAgent:
    """
    AI agent for chronic disease monitoring and management
    """

    def __init__(self):
        self.chronic_conditions = self._define_chronic_conditions()
        self.monitoring_protocols = self._define_monitoring_protocols()
        self.alert_thresholds = self._define_alert_thresholds()
        self.intervention_strategies = self._load_intervention_strategies()
        logger.info("✅ Chronic Care Agent initialized")

    def _define_chronic_conditions(self) -> Dict[str, Any]:
        """Define chronic conditions and their management parameters"""
        return {
            "diabetes_type2": {
                "key_metrics": ["hba1c", "fasting_glucose", "blood_pressure", "weight"],
                "target_ranges": {
                    "hba1c": {"min": 0, "max": 7.0, "unit": "%"},
                    "fasting_glucose": {"min": 80, "max": 130, "unit": "mg/dL"},
                    "blood_pressure": {"min": 90, "max": 140, "unit": "mmHg systolic"},
                    "weight": {"min": -5, "max": 5, "unit": "% change from baseline"}
                },
                "monitoring_frequency": {
                    "hba1c": "quarterly",
                    "fasting_glucose": "daily",
                    "blood_pressure": "weekly",
                    "weight": "daily"
                },
                "complications": ["nephropathy", "retinopathy", "neuropathy", "cardiovascular"],
                "medications": ["metformin", "insulin", "sglt2_inhibitors", "glp1_agonists"]
            },
            "hypertension": {
                "key_metrics": ["systolic_bp", "diastolic_bp", "heart_rate", "weight"],
                "target_ranges": {
                    "systolic_bp": {"min": 90, "max": 130, "unit": "mmHg"},
                    "diastolic_bp": {"min": 60, "max": 80, "unit": "mmHg"},
                    "heart_rate": {"min": 60, "max": 100, "unit": "bpm"},
                    "weight": {"min": -5, "max": 5, "unit": "% change"}
                },
                "monitoring_frequency": {
                    "systolic_bp": "daily",
                    "diastolic_bp": "daily",
                    "heart_rate": "daily",
                    "weight": "weekly"
                },
                "complications": ["stroke", "heart_attack", "kidney_disease", "heart_failure"],
                "medications": ["ace_inhibitors", "beta_blockers", "diuretics", "calcium_blockers"]
            },
            "heart_failure": {
                "key_metrics": ["weight", "symptoms", "exercise_tolerance", "ejection_fraction"],
                "target_ranges": {
                    "weight": {"min": -2, "max": 2, "unit": "lbs from baseline"},
                    "symptoms": {"min": 0, "max": 2, "unit": "severity scale"},
                    "exercise_tolerance": {"min": 3, "max": 5, "unit": "NYHA class"},
                    "ejection_fraction": {"min": 40, "max": 100, "unit": "%"}
                },
                "monitoring_frequency": {
                    "weight": "daily",
                    "symptoms": "daily",
                    "exercise_tolerance": "weekly",
                    "ejection_fraction": "quarterly"
                },
                "complications": ["arrhythmias", "kidney_dysfunction", "sudden_death"],
                "medications": ["ace_inhibitors", "beta_blockers", "diuretics", "aldosterone_antagonists"]
            },
            "copd": {
                "key_metrics": ["peak_flow", "oxygen_saturation", "symptoms", "exacerbations"],
                "target_ranges": {
                    "peak_flow": {"min": 80, "max": 120, "unit": "% of personal best"},
                    "oxygen_saturation": {"min": 88, "max": 100, "unit": "%"},
                    "symptoms": {"min": 0, "max": 2, "unit": "severity scale"},
                    "exacerbations": {"min": 0, "max": 2, "unit": "per year"}
                },
                "monitoring_frequency": {
                    "peak_flow": "daily",
                    "oxygen_saturation": "daily",
                    "symptoms": "daily",
                    "exacerbations": "continuous"
                },
                "complications": ["respiratory_failure", "cor_pulmonale", "pneumonia"],
                "medications": ["bronchodilators", "corticosteroids", "oxygen_therapy"]
            }
        }

    def _define_monitoring_protocols(self) -> Dict[str, Any]:
        """Define monitoring protocols for chronic conditions"""
        return {
            "data_collection": {
                "patient_reported": ["symptoms", "medication_adherence", "lifestyle_factors"],
                "device_measured": ["weight", "blood_pressure", "glucose", "peak_flow"],
                "clinical_assessed": ["lab_results", "imaging", "physical_exam"]
            },
            "alert_triggers": {
                "immediate": ["severe_symptoms", "critical_values", "emergency_situations"],
                "urgent": ["trend_deterioration", "missed_medications", "target_violations"],
                "routine": ["scheduled_reviews", "medication_adjustments", "lifestyle_coaching"]
            },
            "escalation_pathways": {
                "level_1": "automated_intervention",
                "level_2": "care_team_notification",
                "level_3": "provider_alert",
                "level_4": "emergency_services"
            }
        }

    def _define_alert_thresholds(self) -> Dict[str, Any]:
        """Define alert thresholds for monitoring"""
        return {
            "critical": {
                "glucose": {"low": 50, "high": 400},
                "blood_pressure": {"systolic_high": 180, "diastolic_high": 110},
                "weight_gain": 5,  # pounds in 2 days
                "oxygen_saturation": 85
            },
            "warning": {
                "glucose": {"low": 70, "high": 250},
                "blood_pressure": {"systolic_high": 160, "diastolic_high": 100},
                "weight_gain": 3,  # pounds in 2 days
                "oxygen_saturation": 88
            },
            "trend": {
                "consecutive_high_readings": 3,
                "deterioration_period": 7,  # days
                "improvement_threshold": 0.1  # 10% improvement
            }
        }

    def _load_intervention_strategies(self) -> Dict[str, Any]:
        """Load intervention strategies for chronic conditions"""
        return {
            "medication_management": {
                "adherence_support": [
                    "Medication reminders",
                    "Pill organizers",
                    "Pharmacy coordination",
                    "Side effect management"
                ],
                "optimization": [
                    "Dose adjustments",
                    "Drug substitutions",
                    "Combination therapy",
                    "Deprescribing"
                ]
            },
            "lifestyle_interventions": {
                "diet": [
                    "Nutritionist referral",
                    "Meal planning",
                    "Carbohydrate counting",
                    "Sodium restriction"
                ],
                "exercise": [
                    "Physical therapy",
                    "Cardiac rehabilitation",
                    "Pulmonary rehabilitation",
                    "Home exercise programs"
                ],
                "behavioral": [
                    "Smoking cessation",
                    "Stress management",
                    "Sleep hygiene",
                    "Alcohol reduction"
                ]
            },
            "monitoring_intensification": {
                "frequency_increase": [
                    "Daily monitoring",
                    "Twice daily monitoring",
                    "Continuous monitoring",
                    "Real-time alerts"
                ],
                "additional_metrics": [
                    "New biomarkers",
                    "Symptom tracking",
                    "Activity monitoring",
                    "Sleep tracking"
                ]
            },
            "care_coordination": {
                "team_expansion": [
                    "Specialist referrals",
                    "Care manager assignment",
                    "Pharmacist consultation",
                    "Social worker involvement"
                ],
                "communication": [
                    "Care plan updates",
                    "Provider notifications",
                    "Family involvement",
                    "Patient education"
                ]
            }
        }

    async def monitor_patient(
            self,
            patient_id: str,
            monitoring_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive chronic disease monitoring for a patient
        """
        try:
            # Get patient's chronic conditions
            patient_conditions = await self._get_patient_conditions(patient_id)

            # Get recent monitoring data
            if not monitoring_data:
                monitoring_data = await self._get_monitoring_data(patient_id)

            # Analyze each condition
            condition_analyses = {}
            overall_status = "stable"
            alerts = []

            for condition in patient_conditions:
                if condition in self.chronic_conditions:
                    analysis = await self._analyze_condition(
                        condition, patient_id, monitoring_data
                    )
                    condition_analyses[condition] = analysis

                    # Update overall status
                    if analysis["status"] == "critical":
                        overall_status = "critical"
                    elif analysis["status"] == "deteriorating" and overall_status != "critical":
                        overall_status = "deteriorating"

                    # Collect alerts
                    alerts.extend(analysis.get("alerts", []))

            # Generate interventions
            interventions = await self._generate_interventions(
                condition_analyses, patient_id
            )

            # Calculate risk scores
            risk_assessment = self._calculate_risk_scores(condition_analyses)

            # Generate care recommendations
            care_recommendations = self._generate_care_recommendations(
                condition_analyses, risk_assessment
            )

            return {
                "patient_id": patient_id,
                "monitoring_date": datetime.utcnow().isoformat(),
                "overall_status": overall_status,
                "conditions_monitored": list(condition_analyses.keys()),
                "condition_analyses": condition_analyses,
                "alerts": alerts,
                "risk_assessment": risk_assessment,
                "recommended_interventions": interventions,
                "care_recommendations": care_recommendations,
                "next_monitoring": self._calculate_next_monitoring(condition_analyses),
                "care_plan_updates": self._suggest_care_plan_updates(condition_analyses)
            }

        except Exception as e:
            logger.error(f"Chronic monitoring error: {e}")
            return {"error": str(e), "patient_id": patient_id}

    async def _get_patient_conditions(self, patient_id: str) -> List[str]:
        """Get patient's chronic conditions"""
        # In production, query from database
        # For demo, return sample conditions
        conditions = ["diabetes_type2", "hypertension", "heart_failure"]
        return random.sample(conditions, k=random.randint(1, 2))

    async def _get_monitoring_data(self, patient_id: str) -> Dict[str, Any]:
        """Get recent monitoring data for patient"""
        # Generate sample monitoring data
        return {
            "hba1c": random.uniform(6.5, 9.0),
            "fasting_glucose": random.uniform(90, 180),
            "systolic_bp": random.uniform(120, 160),
            "diastolic_bp": random.uniform(70, 100),
            "weight": random.uniform(-3, 5),  # change from baseline
            "heart_rate": random.uniform(60, 90),
            "oxygen_saturation": random.uniform(92, 98),
            "peak_flow": random.uniform(70, 110),
            "symptoms": random.randint(0, 3),
            "medication_adherence": random.uniform(0.7, 1.0),
            "last_updated": datetime.utcnow() - timedelta(hours=random.randint(1, 24))
        }

    async def _analyze_condition(
            self,
            condition: str,
            patient_id: str,
            monitoring_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze specific chronic condition"""

        condition_config = self.chronic_conditions[condition]
        key_metrics = condition_config["key_metrics"]
        target_ranges = condition_config["target_ranges"]

        # Analyze each metric
        metric_analyses = {}
        alerts = []
        status_scores = []

        for metric in key_metrics:
            if metric in monitoring_data:
                value = monitoring_data[metric]
                target = target_ranges.get(metric, {})

                # Assess metric status
                metric_status = self._assess_metric_status(metric, value, target)
                metric_analyses[metric] = metric_status

                # Generate alerts if needed
                if metric_status["alert_level"] in ["warning", "critical"]:
                    alerts.append({
                        "metric": metric,
                        "value": value,
                        "alert_level": metric_status["alert_level"],
                        "message": metric_status["message"],
                        "action_required": metric_status["action_required"]
                    })

                status_scores.append(metric_status["score"])

        # Calculate overall condition status
        avg_score = sum(status_scores) / len(status_scores) if status_scores else 0.5

        if avg_score < 0.3:
            overall_status = "critical"
        elif avg_score < 0.6:
            overall_status = "deteriorating"
        elif avg_score < 0.8:
            overall_status = "suboptimal"
        else:
            overall_status = "stable"

        # Identify trends
        trends = self._analyze_trends(condition, patient_id, monitoring_data)

        # Generate condition-specific insights
        insights = self._generate_condition_insights(
            condition, metric_analyses, trends
        )

        return {
            "condition": condition,
            "status": overall_status,
            "status_score": avg_score,
            "metric_analyses": metric_analyses,
            "alerts": alerts,
            "trends": trends,
            "insights": insights,
            "last_assessment": datetime.utcnow().isoformat()
        }

    def _assess_metric_status(
            self,
            metric: str,
            value: float,
            target: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess status of individual metric"""

        min_val = target.get("min", 0)
        max_val = target.get("max", 100)
        unit = target.get("unit", "")

        # Determine if value is in range
        if min_val <= value <= max_val:
            alert_level = "normal"
            score = 1.0
            message = f"{metric} is within target range"
            action_required = "continue_current_plan"
        else:
            # Check severity of deviation
            if metric in self.alert_thresholds["critical"]:
                critical_thresholds = self.alert_thresholds["critical"][metric]
                if isinstance(critical_thresholds, dict):
                    if value < critical_thresholds.get("low", 0) or value > critical_thresholds.get("high", 1000):
                        alert_level = "critical"
                        score = 0.0
                        message = f"{metric} is at critical level: {value} {unit}"
                        action_required = "immediate_intervention"
                    else:
                        alert_level = "warning"
                        score = 0.3
                        message = f"{metric} is outside target range: {value} {unit}"
                        action_required = "adjust_treatment"
                else:
                    if value > critical_thresholds:
                        alert_level = "critical"
                        score = 0.0
                        message = f"{metric} exceeds critical threshold: {value} {unit}"
                        action_required = "immediate_intervention"
                    else:
                        alert_level = "warning"
                        score = 0.3
                        message = f"{metric} is elevated: {value} {unit}"
                        action_required = "monitor_closely"
            else:
                alert_level = "warning"
                score = 0.5
                message = f"{metric} is outside target range: {value} {unit}"
                action_required = "review_and_adjust"

        return {
            "metric": metric,
            "value": value,
            "target_range": f"{min_val}-{max_val} {unit}",
            "alert_level": alert_level,
            "score": score,
            "message": message,
            "action_required": action_required
        }

    def _analyze_trends(
            self,
            condition: str,
            patient_id: str,
            current_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze trends in monitoring data"""

        # Generate sample trend data (in production, query historical data)
        trends = {}
        condition_config = self.chronic_conditions[condition]

        for metric in condition_config["key_metrics"]:
            if metric in current_data:
                # Simulate trend analysis
                trend_direction = random.choice(["improving", "stable", "declining"])
                trend_strength = random.uniform(0.1, 0.8)

                trends[metric] = {
                    "direction": trend_direction,
                    "strength": trend_strength,
                    "significance": "significant" if trend_strength > 0.5 else "moderate",
                    "period": "30_days",
                    "confidence": random.uniform(0.7, 0.95)
                }

        return trends

    def _generate_condition_insights(
            self,
            condition: str,
            metric_analyses: Dict[str, Any],
            trends: Dict[str, Any]
    ) -> List[str]:
        """Generate insights for specific condition"""

        insights = []

        # Condition-specific insights
        if condition == "diabetes_type2":
            hba1c_analysis = metric_analyses.get("hba1c", {})
            if hba1c_analysis.get("alert_level") == "warning":
                insights.append("HbA1c above target suggests need for medication adjustment")

            glucose_trend = trends.get("fasting_glucose", {})
            if glucose_trend.get("direction") == "declining":
                insights.append("Improving glucose control indicates effective management")

        elif condition == "hypertension":
            bp_analysis = metric_analyses.get("systolic_bp", {})
            if bp_analysis.get("alert_level") == "critical":
                insights.append("Severe hypertension requires immediate intervention")

            weight_trend = trends.get("weight", {})
            if weight_trend.get("direction") == "improving":
                insights.append("Weight loss contributing to blood pressure improvement")

        elif condition == "heart_failure":
            weight_analysis = metric_analyses.get("weight", {})
            if weight_analysis.get("alert_level") == "warning":
                insights.append("Weight gain may indicate fluid retention")

            symptoms_trend = trends.get("symptoms", {})
            if symptoms_trend.get("direction") == "declining":
                insights.append("Symptom improvement suggests stable heart failure")

        # General insights
        improving_metrics = [
            metric for metric, trend in trends.items()
            if trend.get("direction") == "improving"
        ]

        if len(improving_metrics) > 1:
            insights.append(f"Multiple improving metrics: {', '.join(improving_metrics)}")

        declining_metrics = [
            metric for metric, trend in trends.items()
            if trend.get("direction") == "declining"
        ]

        if len(declining_metrics) > 1:
            insights.append(f"Concerning trends in: {', '.join(declining_metrics)}")

        return insights

    async def _generate_interventions(
            self,
            condition_analyses: Dict[str, Any],
            patient_id: str
    ) -> List[Dict[str, Any]]:
        """Generate targeted interventions"""

        interventions = []

        for condition, analysis in condition_analyses.items():
            condition_status = analysis["status"]
            alerts = analysis.get("alerts", [])

            # Generate interventions based on status
            if condition_status == "critical":
                interventions.extend(self._get_critical_interventions(condition, alerts))
            elif condition_status == "deteriorating":
                interventions.extend(self._get_urgent_interventions(condition, analysis))
            elif condition_status == "suboptimal":
                interventions.extend(self._get_optimization_interventions(condition, analysis))

            # Add condition-specific interventions
            condition_interventions = self._get_condition_specific_interventions(
                condition, analysis
            )
            interventions.extend(condition_interventions)

        # Prioritize and deduplicate interventions
        prioritized_interventions = self._prioritize_interventions(interventions)

        return prioritized_interventions[:10]  # Return top 10 interventions

    def _get_critical_interventions(
            self,
            condition: str,
            alerts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get interventions for critical status"""

        interventions = []

        for alert in alerts:
            if alert["alert_level"] == "critical":
                interventions.append({
                    "type": "immediate_action",
                    "condition": condition,
                    "intervention": f"Immediate intervention for {alert['metric']}",
                    "priority": "critical",
                    "timeframe": "immediate",
                    "action": alert["action_required"],
                    "description": f"Address critical {alert['metric']} level: {alert['value']}"
                })

        return interventions

    def _get_urgent_interventions(
            self,
            condition: str,
            analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get interventions for deteriorating status"""

        interventions = [
            {
                "type": "medication_review",
                "condition": condition,
                "intervention": "Comprehensive medication review",
                "priority": "high",
                "timeframe": "within_24_hours",
                "action": "review_and_adjust_medications",
                "description": f"Review {condition} medications due to deteriorating status"
            },
            {
                "type": "monitoring_intensification",
                "condition": condition,
                "intervention": "Increase monitoring frequency",
                "priority": "high",
                "timeframe": "immediate",
                "action": "intensify_monitoring",
                "description": f"Increase monitoring for {condition} management"
            }
        ]

        return interventions

    def _get_optimization_interventions(
            self,
            condition: str,
            analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get interventions for optimization"""

        interventions = [
            {
                "type": "lifestyle_modification",
                "condition": condition,
                "intervention": "Lifestyle counseling",
                "priority": "medium",
                "timeframe": "within_week",
                "action": "lifestyle_counseling",
                "description": f"Optimize lifestyle factors for {condition} management"
            },
            {
                "type": "education",
                "condition": condition,
                "intervention": "Patient education reinforcement",
                "priority": "medium",
                "timeframe": "within_week",
                "action": "patient_education",
                "description": f"Reinforce {condition} self-management education"
            }
        ]

        return interventions

    def _get_condition_specific_interventions(
            self,
            condition: str,
            analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get condition-specific interventions"""

        interventions = []

        if condition == "diabetes_type2":
            hba1c_analysis = analysis["metric_analyses"].get("hba1c", {})
            if hba1c_analysis.get("alert_level") in ["warning", "critical"]:
                interventions.append({
                    "type": "specialist_referral",
                    "condition": condition,
                    "intervention": "Endocrinology consultation",
                    "priority": "high",
                    "timeframe": "within_week",
                    "action": "specialist_referral",
                    "description": "Refer to endocrinologist for diabetes optimization"
                })

        elif condition == "heart_failure":
            weight_analysis = analysis["metric_analyses"].get("weight", {})
            if weight_analysis.get("alert_level") == "warning":
                interventions.append({
                    "type": "medication_adjustment",
                    "condition": condition,
                    "intervention": "Diuretic adjustment",
                    "priority": "high",
                    "timeframe": "within_24_hours",
                    "action": "adjust_diuretics",
                    "description": "Adjust diuretic therapy for fluid management"
                })

        return interventions

    def _prioritize_interventions(
            self,
            interventions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prioritize interventions by urgency and impact"""

        priority_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        # Sort by priority and remove duplicates
        unique_interventions = []
        seen_interventions = set()

        for intervention in interventions:
            key = f"{intervention['type']}_{intervention['condition']}"
            if key not in seen_interventions:
                seen_interventions.add(key)
                unique_interventions.append(intervention)

        # Sort by priority
        unique_interventions.sort(
            key=lambda x: priority_weights.get(x["priority"], 0),
            reverse=True
        )

        return unique_interventions

    def _calculate_risk_scores(
            self,
            condition_analyses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate risk scores for chronic conditions"""

        risk_scores = {}
        overall_risk = 0.0

        for condition, analysis in condition_analyses.items():
            status_score = analysis["status_score"]

            # Calculate condition-specific risk
            if condition == "diabetes_type2":
                # Higher risk for complications
                complication_risk = 1.0 - status_score
                risk_scores[condition] = {
                    "current_control": status_score,
                    "complication_risk": complication_risk,
                    "overall_risk": (status_score + complication_risk) / 2
                }
            else:
                risk_scores[condition] = {
                    "current_control": status_score,
                    "overall_risk": 1.0 - status_score
                }

            overall_risk += risk_scores[condition]["overall_risk"]

        # Calculate overall risk
        if condition_analyses:
            overall_risk = overall_risk / len(condition_analyses)

        return {
            "condition_risks": risk_scores,
            "overall_risk": overall_risk,
            "risk_level": self._categorize_risk_level(overall_risk),
            "risk_factors": self._identify_risk_factors(condition_analyses)
        }

    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize overall risk level"""
        if risk_score < 0.3:
            return "Low"
        elif risk_score < 0.6:
            return "Moderate"
        elif risk_score < 0.8:
            return "High"
        else:
            return "Critical"

    def _identify_risk_factors(
            self,
            condition_analyses: Dict[str, Any]
    ) -> List[str]:
        """Identify key risk factors"""

        risk_factors = []

        for condition, analysis in condition_analyses.items():
            if analysis["status"] in ["critical", "deteriorating"]:
                risk_factors.append(f"Poorly controlled {condition}")

            # Check for specific risk factors
            alerts = analysis.get("alerts", [])
            for alert in alerts:
                if alert["alert_level"] == "critical":
                    risk_factors.append(f"Critical {alert['metric']} in {condition}")

        return risk_factors

    def _generate_care_recommendations(
            self,
            condition_analyses: Dict[str, Any],
            risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate care recommendations"""

        recommendations = []
        overall_risk = risk_assessment["risk_level"]

        # Risk-based recommendations
        if overall_risk == "Critical":
            recommendations.extend([
                "Consider hospitalization or intensive outpatient management",
                "Daily monitoring and frequent provider contact",
                "Emergency action plan review with patient and family"
            ])
        elif overall_risk == "High":
            recommendations.extend([
                "Increase monitoring frequency and provider contact",
                "Consider care management or case management services",
                "Review and optimize all medications"
            ])
        elif overall_risk == "Moderate":
            recommendations.extend([
                "Enhance patient education and self-management support",
                "Regular monitoring with scheduled check-ins",
                "Lifestyle modification counseling"
            ])

        # Condition-specific recommendations
        for condition, analysis in condition_analyses.items():
            if condition == "diabetes_type2" and analysis["status"] != "stable":
                recommendations.append("Diabetes self-management education refresher")
            elif condition == "heart_failure" and analysis["status"] != "stable":
                recommendations.append("Heart failure education and daily weight monitoring")
            elif condition == "hypertension" and analysis["status"] != "stable":
                recommendations.append("Blood pressure monitoring and lifestyle counseling")

        return list(set(recommendations))  # Remove duplicates

    def _calculate_next_monitoring(
            self,
            condition_analyses: Dict[str, Any]
    ) -> Dict[str, str]:
        """Calculate next monitoring schedule"""

        next_monitoring = {}

        for condition, analysis in condition_analyses.items():
            status = analysis["status"]

            if status == "critical":
                next_monitoring[condition] = "immediate"
            elif status == "deteriorating":
                next_monitoring[condition] = "within_24_hours"
            elif status == "suboptimal":
                next_monitoring[condition] = "within_3_days"
            else:
                # Use standard monitoring frequency
                condition_config = self.chronic_conditions[condition]
                frequencies = condition_config["monitoring_frequency"]

                # Use most frequent monitoring requirement
                min_frequency = min(frequencies.values(), key=lambda x: {
                    "daily": 1, "weekly": 7, "monthly": 30, "quarterly": 90
                }.get(x, 30))

                next_monitoring[condition] = min_frequency

        return next_monitoring

    def _suggest_care_plan_updates(
            self,
            condition_analyses: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest care plan updates"""

        updates = []

        for condition, analysis in condition_analyses.items():
            if analysis["status"] in ["critical", "deteriorating"]:
                updates.append({
                    "condition": condition,
                    "update_type": "medication_adjustment",
                    "description": f"Review and adjust {condition} medications",
                    "priority": "high"
                })

                updates.append({
                    "condition": condition,
                    "update_type": "monitoring_frequency",
                    "description": f"Increase monitoring frequency for {condition}",
                    "priority": "high"
                })

            # Check for specific metric issues
            for metric, metric_analysis in analysis.get("metric_analyses", {}).items():
                if metric_analysis.get("alert_level") == "critical":
                    updates.append({
                        "condition": condition,
                        "update_type": "target_adjustment",
                        "description": f"Review {metric} targets for {condition}",
                        "priority": "medium"
                    })

        return updates
