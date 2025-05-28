"""
Population Health Analytics Agent
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import json

from app.config.settings import settings

logger = logging.getLogger(__name__)

class PopulationAnalyticsAgent:
    """
    AI agent for population health analytics and insights
    """

    def __init__(self):
        self.health_indicators = self._load_health_indicators()
        self.risk_models = self._initialize_risk_models()
        self.benchmark_data = self._load_benchmark_data()
        logger.info("✅ Population Analytics Agent initialized")

    def _load_health_indicators(self) -> Dict[str, Any]:
        """Load population health indicators"""
        return {
            "chronic_diseases": {
                "diabetes": {"prevalence_target": 0.11, "control_target": 0.70},
                "hypertension": {"prevalence_target": 0.45, "control_target": 0.80},
                "heart_disease": {"prevalence_target": 0.06, "control_target": 0.75},
                "copd": {"prevalence_target": 0.04, "control_target": 0.65}
            },
            "preventive_care": {
                "mammography": {"target_rate": 0.80, "age_range": [50, 74]},
                "colonoscopy": {"target_rate": 0.75, "age_range": [45, 75]},
                "flu_vaccination": {"target_rate": 0.70, "age_range": [65, 100]},
                "blood_pressure_screening": {"target_rate": 0.90, "age_range": [18, 100]}
            },
            "quality_measures": {
                "readmission_rate": {"target": 0.15, "direction": "lower"},
                "mortality_rate": {"target": 0.02, "direction": "lower"},
                "patient_satisfaction": {"target": 0.85, "direction": "higher"},
                "medication_adherence": {"target": 0.80, "direction": "higher"}
            },
            "social_determinants": {
                "food_insecurity": {"prevalence_threshold": 0.20},
                "housing_instability": {"prevalence_threshold": 0.15},
                "transportation_barriers": {"prevalence_threshold": 0.25},
                "social_isolation": {"prevalence_threshold": 0.30}
            }
        }

    def _initialize_risk_models(self) -> Dict[str, Any]:
        """Initialize population risk stratification models"""
        return {
            "high_risk_criteria": {
                "age_threshold": 65,
                "comorbidity_count": 3,
                "recent_admissions": 2,
                "medication_count": 10
            },
            "risk_weights": {
                "age": 0.25,
                "comorbidities": 0.30,
                "admissions": 0.25,
                "medications": 0.20
            },
            "intervention_thresholds": {
                "care_management": 0.70,
                "case_management": 0.85,
                "intensive_management": 0.95
            }
        }

    def _load_benchmark_data(self) -> Dict[str, Any]:
        """Load national and regional benchmark data"""
        return {
            "national_benchmarks": {
                "readmission_rate": 0.158,
                "mortality_rate": 0.024,
                "patient_satisfaction": 0.82,
                "diabetes_control": 0.68,
                "hypertension_control": 0.76
            },
            "regional_benchmarks": {
                "readmission_rate": 0.145,
                "mortality_rate": 0.021,
                "patient_satisfaction": 0.85,
                "diabetes_control": 0.72,
                "hypertension_control": 0.78
            },
            "top_decile": {
                "readmission_rate": 0.12,
                "mortality_rate": 0.015,
                "patient_satisfaction": 0.92,
                "diabetes_control": 0.85,
                "hypertension_control": 0.88
            }
        }

    async def analyze_population_health(
            self,
            population_filters: Optional[Dict[str, Any]] = None,
            analysis_period: Optional[str] = "12_months"
    ) -> Dict[str, Any]:
        """
        Comprehensive population health analysis
        """
        try:
            # Get population data
            population_data = await self._get_population_data(population_filters, analysis_period)

            # Analyze chronic disease management
            chronic_disease_analysis = self._analyze_chronic_diseases(population_data)

            # Analyze preventive care
            preventive_care_analysis = self._analyze_preventive_care(population_data)

            # Analyze quality measures
            quality_analysis = self._analyze_quality_measures(population_data)

            # Analyze health disparities
            disparity_analysis = self._analyze_health_disparities(population_data)

            # Risk stratification
            risk_stratification = self._perform_risk_stratification(population_data)

            # Social determinants analysis
            sdoh_analysis = self._analyze_social_determinants(population_data)

            # Generate insights and recommendations
            insights = self._generate_population_insights(
                chronic_disease_analysis, preventive_care_analysis,
                quality_analysis, disparity_analysis, risk_stratification, sdoh_analysis
            )

            # Calculate population health score
            population_health_score = self._calculate_population_health_score(
                chronic_disease_analysis, preventive_care_analysis, quality_analysis
            )

            return {
                "analysis_id": f"pop_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "analysis_period": analysis_period,
                "population_filters": population_filters or {},
                "population_summary": self._create_population_summary(population_data),
                "chronic_disease_analysis": chronic_disease_analysis,
                "preventive_care_analysis": preventive_care_analysis,
                "quality_analysis": quality_analysis,
                "disparity_analysis": disparity_analysis,
                "risk_stratification": risk_stratification,
                "sdoh_analysis": sdoh_analysis,
                "population_health_score": population_health_score,
                "insights_and_recommendations": insights,
                "benchmark_comparisons": self._compare_to_benchmarks(quality_analysis),
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Population analytics error: {e}")
            return {"error": str(e), "analysis_period": analysis_period}

    async def _get_population_data(self, filters: Optional[Dict[str, Any]], period: str) -> Dict[str, Any]:
        """Get population data for analysis"""
        # In production, this would query the database
        # For demo, generate synthetic population data

        import random
        from datetime import datetime, timedelta

        # Generate synthetic population
        population_size = 10000
        population = []

        for i in range(population_size):
            # Demographics
            age = random.randint(18, 95)
            gender = random.choice(["male", "female"])
            race = random.choice(["white", "black", "hispanic", "asian", "other"])

            # Health conditions
            conditions = []
            if age > 45 and random.random() < 0.3:
                conditions.append("diabetes")
            if age > 50 and random.random() < 0.4:
                conditions.append("hypertension")
            if age > 60 and random.random() < 0.15:
                conditions.append("heart_disease")
            if age > 55 and random.random() < 0.08:
                conditions.append("copd")

            # Healthcare utilization
            admissions_last_year = random.poisson(0.5) if conditions else random.poisson(0.1)
            er_visits_last_year = random.poisson(0.8) if conditions else random.poisson(0.3)

            # Quality measures
            last_admission_date = datetime.utcnow() - timedelta(days=random.randint(1, 365)) if admissions_last_year > 0 else None
            readmitted_30_days = random.random() < 0.15 if last_admission_date else False

            # Preventive care
            preventive_care = {}
            if gender == "female" and 50 <= age <= 74:
                preventive_care["mammography"] = random.random() < 0.75
            if age >= 45:
                preventive_care["colonoscopy"] = random.random() < 0.70
            if age >= 65:
                preventive_care["flu_vaccine"] = random.random() < 0.65

            # Social determinants
            income_level = random.choice(["low", "medium", "high"])
            insurance_type = random.choice(["medicare", "medicaid", "commercial", "uninsured"])

            sdoh_factors = {
                "food_insecurity": random.random() < 0.15 if income_level == "low" else random.random() < 0.05,
                "housing_instability": random.random() < 0.20 if income_level == "low" else random.random() < 0.08,
                "transportation_barriers": random.random() < 0.25 if income_level == "low" else random.random() < 0.10,
                "social_isolation": random.random() < 0.30 if age > 75 else random.random() < 0.15
            }

            patient = {
                "patient_id": f"patient_{i:05d}",
                "age": age,
                "gender": gender,
                "race": race,
                "conditions": conditions,
                "admissions_last_year": admissions_last_year,
                "er_visits_last_year": er_visits_last_year,
                "readmitted_30_days": readmitted_30_days,
                "preventive_care": preventive_care,
                "income_level": income_level,
                "insurance_type": insurance_type,
                "sdoh_factors": sdoh_factors,
                "medication_count": len(conditions) * 2 + random.randint(0, 3),
                "patient_satisfaction": random.uniform(0.6, 1.0)
            }

            population.append(patient)

        return {
            "population": population,
            "total_patients": len(population),
            "analysis_period": period,
            "data_generated": datetime.utcnow().isoformat()
        }

    def _analyze_chronic_diseases(self, population_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze chronic disease prevalence and management"""
        population = population_data["population"]
        total_patients = len(population)

        chronic_analysis = {}

        for disease in self.health_indicators["chronic_diseases"].keys():
            # Calculate prevalence
            patients_with_disease = [p for p in population if disease in p["conditions"]]
            prevalence = len(patients_with_disease) / total_patients

            # Calculate control rate (simplified)
            controlled_patients = sum(1 for p in patients_with_disease if random.random() < 0.7)
            control_rate = controlled_patients / len(patients_with_disease) if patients_with_disease else 0

            # Get targets
            targets = self.health_indicators["chronic_diseases"][disease]

            chronic_analysis[disease] = {
                "prevalence": prevalence,
                "prevalence_target": targets["prevalence_target"],
                "control_rate": control_rate,
                "control_target": targets["control_target"],
                "total_patients": len(patients_with_disease),
                "controlled_patients": controlled_patients,
                "meets_prevalence_target": prevalence <= targets["prevalence_target"],
                "meets_control_target": control_rate >= targets["control_target"]
            }

        return chronic_analysis

    def _analyze_preventive_care(self, population_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze preventive care measures"""
        population = population_data["population"]

        preventive_analysis = {}

        for measure, criteria in self.health_indicators["preventive_care"].items():
            min_age, max_age = criteria["age_range"]
            target_rate = criteria["target_rate"]

            # Find eligible population
            if measure == "mammography":
                eligible = [p for p in population if p["gender"] == "female" and min_age <= p["age"] <= max_age]
            else:
                eligible = [p for p in population if min_age <= p["age"] <= max_age]

            # Calculate completion rate
            if eligible:
                completed = sum(1 for p in eligible if p["preventive_care"].get(measure, False))
                completion_rate = completed / len(eligible)
            else:
                completion_rate = 0
                completed = 0

            preventive_analysis[measure] = {
                "eligible_population": len(eligible),
                "completed": completed,
                "completion_rate": completion_rate,
                "target_rate": target_rate,
                "meets_target": completion_rate >= target_rate,
                "gap": max(0, target_rate - completion_rate)
            }

        return preventive_analysis

    def _analyze_quality_measures(self, population_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quality measures"""
        population = population_data["population"]
        total_patients = len(population)

        # Readmission rate
        patients_with_admissions = [p for p in population if p["admissions_last_year"] > 0]
        readmitted_patients = sum(1 for p in patients_with_admissions if p["readmitted_30_days"])
        readmission_rate = readmitted_patients / len(patients_with_admissions) if patients_with_admissions else 0

        # Mortality rate (simplified)
        mortality_rate = random.uniform(0.015, 0.025)

        # Patient satisfaction
        satisfaction_scores = [p["patient_satisfaction"] for p in population]
        avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)

        # Medication adherence (simplified)
        adherence_rate = random.uniform(0.75, 0.85)

        quality_analysis = {
            "readmission_rate": {
                "rate": readmission_rate,
                "target": self.health_indicators["quality_measures"]["readmission_rate"]["target"],
                "meets_target": readmission_rate <= self.health_indicators["quality_measures"]["readmission_rate"]["target"],
                "total_admissions": len(patients_with_admissions),
                "readmissions": readmitted_patients
            },
            "mortality_rate": {
                "rate": mortality_rate,
                "target": self.health_indicators["quality_measures"]["mortality_rate"]["target"],
                "meets_target": mortality_rate <= self.health_indicators["quality_measures"]["mortality_rate"]["target"]
            },
            "patient_satisfaction": {
                "score": avg_satisfaction,
                "target": self.health_indicators["quality_measures"]["patient_satisfaction"]["target"],
                "meets_target": avg_satisfaction >= self.health_indicators["quality_measures"]["patient_satisfaction"]["target"]
            },
            "medication_adherence": {
                "rate": adherence_rate,
                "target": self.health_indicators["quality_measures"]["medication_adherence"]["target"],
                "meets_target": adherence_rate >= self.health_indicators["quality_measures"]["medication_adherence"]["target"]
            }
        }

        return quality_analysis

    def _analyze_health_disparities(self, population_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze health disparities by demographics"""
        population = population_data["population"]

        disparities = {
            "by_race": {},
            "by_gender": {},
            "by_income": {},
            "by_insurance": {}
        }

        # Analyze by race
        races = list(set(p["race"] for p in population))
        for race in races:
            race_population = [p for p in population if p["race"] == race]
            disparities["by_race"][race] = self._calculate_disparity_metrics(race_population)

        # Analyze by gender
        genders = list(set(p["gender"] for p in population))
        for gender in genders:
            gender_population = [p for p in population if p["gender"] == gender]
            disparities["by_gender"][gender] = self._calculate_disparity_metrics(gender_population)

        # Analyze by income
        income_levels = list(set(p["income_level"] for p in population))
        for income in income_levels:
            income_population = [p for p in population if p["income_level"] == income]
            disparities["by_income"][income] = self._calculate_disparity_metrics(income_population)

        # Analyze by insurance
        insurance_types = list(set(p["insurance_type"] for p in population))
        for insurance in insurance_types:
            insurance_population = [p for p in population if p["insurance_type"] == insurance]
            disparities["by_insurance"][insurance] = self._calculate_disparity_metrics(insurance_population)

        return disparities

    def _calculate_disparity_metrics(self, subpopulation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate key metrics for a subpopulation"""
        if not subpopulation:
            return {}

        # Chronic disease prevalence
        diabetes_prev = sum(1 for p in subpopulation if "diabetes" in p["conditions"]) / len(subpopulation)
        hypertension_prev = sum(1 for p in subpopulation if "hypertension" in p["conditions"]) / len(subpopulation)

        # Healthcare utilization
        avg_admissions = sum(p["admissions_last_year"] for p in subpopulation) / len(subpopulation)
        avg_er_visits = sum(p["er_visits_last_year"] for p in subpopulation) / len(subpopulation)

        # Readmission rate
        patients_with_admissions = [p for p in subpopulation if p["admissions_last_year"] > 0]
        readmission_rate = sum(1 for p in patients_with_admissions if p["readmitted_30_days"]) / len(patients_with_admissions) if patients_with_admissions else 0

        # Patient satisfaction
        avg_satisfaction = sum(p["patient_satisfaction"] for p in subpopulation) / len(subpopulation)

        return {
            "population_size": len(subpopulation),
            "diabetes_prevalence": diabetes_prev,
            "hypertension_prevalence": hypertension_prev,
            "avg_admissions_per_year": avg_admissions,
            "avg_er_visits_per_year": avg_er_visits,
            "readmission_rate": readmission_rate,
            "avg_patient_satisfaction": avg_satisfaction
        }

    def _perform_risk_stratification(self, population_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform population risk stratification"""
        population = population_data["population"]

        risk_stratification = {
            "low_risk": [],
            "medium_risk": [],
            "high_risk": [],
            "very_high_risk": []
        }

        for patient in population:
            risk_score = self._calculate_individual_risk_score(patient)

            if risk_score < 0.25:
                risk_stratification["low_risk"].append(patient["patient_id"])
            elif risk_score < 0.50:
                risk_stratification["medium_risk"].append(patient["patient_id"])
            elif risk_score < 0.75:
                risk_stratification["high_risk"].append(patient["patient_id"])
            else:
                risk_stratification["very_high_risk"].append(patient["patient_id"])

        # Calculate distribution
        total_patients = len(population)
        distribution = {
            "low_risk": len(risk_stratification["low_risk"]) / total_patients,
            "medium_risk": len(risk_stratification["medium_risk"]) / total_patients,
            "high_risk": len(risk_stratification["high_risk"]) / total_patients,
            "very_high_risk": len(risk_stratification["very_high_risk"]) / total_patients
        }

        return {
            "risk_distribution": distribution,
            "risk_counts": {k: len(v) for k, v in risk_stratification.items()},
            "high_risk_patients": len(risk_stratification["high_risk"]) + len(risk_stratification["very_high_risk"]),
            "care_management_eligible": len(risk_stratification["high_risk"]) + len(risk_stratification["very_high_risk"])
        }

    def _calculate_individual_risk_score(self, patient: Dict[str, Any]) -> float:
        """Calculate individual patient risk score"""
        risk_score = 0.0
        weights = self.risk_models["risk_weights"]

        # Age component
        age_risk = min(1.0, patient["age"] / 100)
        risk_score += age_risk * weights["age"]

        # Comorbidity component
        comorbidity_risk = min(1.0, len(patient["conditions"]) / 5)
        risk_score += comorbidity_risk * weights["comorbidities"]

        # Admission component
        admission_risk = min(1.0, patient["admissions_last_year"] / 3)
        risk_score += admission_risk * weights["admissions"]

        # Medication component
        medication_risk = min(1.0, patient["medication_count"] / 15)
        risk_score += medication_risk * weights["medications"]

        return min(1.0, risk_score)

    def _analyze_social_determinants(self, population_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social determinants of health"""
        population = population_data["population"]
        total_patients = len(population)

        sdoh_analysis = {}

        for factor, threshold_data in self.health_indicators["social_determinants"].items():
            affected_patients = sum(1 for p in population if p["sdoh_factors"].get(factor, False))
            prevalence = affected_patients / total_patients
            threshold = threshold_data["prevalence_threshold"]

            sdoh_analysis[factor] = {
                "prevalence": prevalence,
                "affected_patients": affected_patients,
                "threshold": threshold,
                "exceeds_threshold": prevalence > threshold,
                "impact_level": "high" if prevalence > threshold * 1.5 else "moderate" if prevalence > threshold else "low"
            }

        return sdoh_analysis

    def _generate_population_insights(self, *analyses) -> Dict[str, Any]:
        """Generate insights and recommendations"""
        chronic_analysis, preventive_analysis, quality_analysis, disparity_analysis, risk_analysis, sdoh_analysis = analyses

        insights = {
            "key_findings": [],
            "priority_areas": [],
            "recommendations": [],
            "intervention_opportunities": []
        }

        # Chronic disease insights
        for disease, data in chronic_analysis.items():
            if not data["meets_control_target"]:
                insights["key_findings"].append(f"{disease.title()} control rate ({data['control_rate']:.1%}) below target ({data['control_target']:.1%})")
                insights["priority_areas"].append(f"{disease.title()} management")

        # Preventive care insights
        for measure, data in preventive_analysis.items():
            if not data["meets_target"]:
                insights["key_findings"].append(f"{measure.replace('_', ' ').title()} completion rate ({data['completion_rate']:.1%}) below target ({data['target_rate']:.1%})")
                insights["priority_areas"].append(f"{measure.replace('_', ' ').title()} screening")

        # Quality measure insights
        for measure, data in quality_analysis.items():
            if not data["meets_target"]:
                insights["key_findings"].append(f"{measure.replace('_', ' ').title()} needs improvement")
                insights["priority_areas"].append(f"{measure.replace('_', ' ').title()} improvement")

        # Generate recommendations
        if "diabetes management" in insights["priority_areas"]:
            insights["recommendations"].extend([
                "Implement diabetes self-management education programs",
                "Enhance medication adherence support",
                "Increase endocrinology referrals for uncontrolled patients"
            ])

        if "readmission_rate improvement" in insights["priority_areas"]:
            insights["recommendations"].extend([
                "Strengthen discharge planning processes",
                "Implement post-discharge follow-up calls",
                "Enhance care transitions coordination"
            ])

        # Risk stratification insights
        high_risk_percentage = risk_analysis["risk_distribution"]["high_risk"] + risk_analysis["risk_distribution"]["very_high_risk"]
        if high_risk_percentage > 0.20:
            insights["intervention_opportunities"].append("High percentage of high-risk patients - consider care management expansion")

        return insights

    def _calculate_population_health_score(self, chronic_analysis: Dict, preventive_analysis: Dict, quality_analysis: Dict) -> Dict[str, Any]:
        """Calculate overall population health score"""

        # Chronic disease score (0-100)
        chronic_scores = []
        for disease, data in chronic_analysis.items():
            control_score = data["control_rate"] * 100
            chronic_scores.append(control_score)

        chronic_score = sum(chronic_scores) / len(chronic_scores) if chronic_scores else 50

        # Preventive care score (0-100)
        preventive_scores = []
        for measure, data in preventive_analysis.items():
            completion_score = data["completion_rate"] * 100
            preventive_scores.append(completion_score)

        preventive_score = sum(preventive_scores) / len(preventive_scores) if preventive_scores else 50

        # Quality score (0-100)
        quality_scores = []
        for measure, data in quality_analysis.items():
            if measure in ["readmission_rate", "mortality_rate"]:
                # Lower is better
                score = max(0, 100 - (data["rate"] * 1000))
            else:
                # Higher is better
                score = data.get("score", data.get("rate", 0.5)) * 100
            quality_scores.append(score)

        quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 50

        # Overall score (weighted average)
        overall_score = (chronic_score * 0.4 + preventive_score * 0.3 + quality_score * 0.3)

        return {
            "overall_score": overall_score,
            "chronic_disease_score": chronic_score,
            "preventive_care_score": preventive_score,
            "quality_score": quality_score,
            "grade": self._get_health_grade(overall_score),
            "percentile": self._get_percentile_rank(overall_score)
        }

    def _get_health_grade(self, score: float) -> str:
        """Convert score to letter grade"""
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

    def _get_percentile_rank(self, score: float) -> int:
        """Get percentile rank (simplified)"""
        return min(99, max(1, int(score)))

    def _compare_to_benchmarks(self, quality_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare quality measures to benchmarks"""
        comparisons = {}

        for measure, data in quality_analysis.items():
            if measure in self.benchmark_data["national_benchmarks"]:
                national = self.benchmark_data["national_benchmarks"][measure]
                regional = self.benchmark_data["regional_benchmarks"][measure]
                top_decile = self.benchmark_data["top_decile"][measure]

                current_value = data.get("rate", data.get("score", 0))

                comparisons[measure] = {
                    "current": current_value,
                    "national_benchmark": national,
                    "regional_benchmark": regional,
                    "top_decile": top_decile,
                    "vs_national": "above" if current_value > national else "below",
                    "vs_regional": "above" if current_value > regional else "below",
                    "vs_top_decile": "above" if current_value > top_decile else "below"
                }

        return comparisons

    def _create_population_summary(self, population_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create population summary statistics"""
        population = population_data["population"]

        # Demographics
        avg_age = sum(p["age"] for p in population) / len(population)
        gender_dist = {}
        race_dist = {}

        for p in population:
            gender_dist[p["gender"]] = gender_dist.get(p["gender"], 0) + 1
            race_dist[p["race"]] = race_dist.get(p["race"], 0) + 1

        # Convert to percentages
        total = len(population)
        gender_dist = {k: v/total for k, v in gender_dist.items()}
        race_dist = {k: v/total for k, v in race_dist.items()}

        return {
            "total_patients": total,
            "average_age": avg_age,
            "gender_distribution": gender_dist,
            "race_distribution": race_dist,
            "analysis_period": population_data["analysis_period"]
        }
