"""
Preventive Care Agent for Personalized Prevention Recommendations
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random
import logging

from app.config.settings import settings

logger = logging.getLogger(__name__)

class PreventionAgent:
    """
    AI agent for generating personalized preventive care recommendations
    """

    def __init__(self):
        self.prevention_guidelines = self._load_prevention_guidelines()
        self.risk_calculators = self._load_risk_calculators()
        self.screening_protocols = self._load_screening_protocols()
        self.vaccination_schedules = self._load_vaccination_schedules()
        logger.info("✅ Prevention Agent initialized")

    def _load_prevention_guidelines(self) -> Dict[str, Any]:
        """Load evidence-based prevention guidelines"""
        return {
            "cardiovascular": {
                "risk_factors": ["age", "smoking", "diabetes", "hypertension", "cholesterol", "family_history"],
                "interventions": {
                    "lifestyle": ["diet_modification", "exercise", "smoking_cessation", "weight_management"],
                    "medications": ["statins", "aspirin", "ace_inhibitors"],
                    "monitoring": ["blood_pressure", "cholesterol", "glucose"]
                },
                "targets": {
                    "ldl_cholesterol": {"high_risk": 70, "moderate_risk": 100, "low_risk": 130},
                    "blood_pressure": {"target": 130, "unit": "mmHg systolic"},
                    "hba1c": {"target": 7.0, "unit": "%"}
                }
            },
            "cancer": {
                "screening_types": {
                    "breast": {"method": "mammography", "age_start": 40, "frequency": "annual"},
                    "cervical": {"method": "pap_smear", "age_start": 21, "frequency": "3_years"},
                    "colorectal": {"method": "colonoscopy", "age_start": 45, "frequency": "10_years"},
                    "lung": {"method": "ldct", "age_start": 50, "frequency": "annual", "criteria": "smoking_history"},
                    "prostate": {"method": "psa", "age_start": 50, "frequency": "annual", "gender": "male"}
                },
                "risk_factors": {
                    "breast": ["age", "family_history", "brca_mutation", "hormone_exposure"],
                    "colorectal": ["age", "family_history", "inflammatory_bowel_disease", "polyps"],
                    "lung": ["smoking", "age", "occupational_exposure", "family_history"]
                }
            },
            "infectious_disease": {
                "vaccinations": {
                    "influenza": {"frequency": "annual", "age_groups": "all", "high_risk": True},
                    "covid19": {"frequency": "annual", "age_groups": "all", "high_risk": True},
                    "pneumonia": {"frequency": "once", "age_start": 65, "high_risk": True},
                    "shingles": {"frequency": "once", "age_start": 50},
                    "tdap": {"frequency": "10_years", "age_groups": "all"}
                },
                "prevention_measures": ["hand_hygiene", "mask_wearing", "social_distancing", "isolation_when_sick"]
            }
        }

    def _load_risk_calculators(self) -> Dict[str, Any]:
        """Load risk calculation algorithms"""
        return {
            "framingham_cvd": {
                "factors": ["age", "gender", "smoking", "diabetes", "systolic_bp", "cholesterol", "hdl"],
                "weights": {"age": 0.2, "gender": 0.1, "smoking": 0.15, "diabetes": 0.15, "systolic_bp": 0.15, "cholesterol": 0.15, "hdl": 0.1}
            },
            "ascvd_pooled_cohort": {
                "factors": ["age", "gender", "race", "smoking", "diabetes", "systolic_bp", "cholesterol", "hdl", "bp_treatment"],
                "weights": {"age": 0.25, "gender": 0.1, "race": 0.05, "smoking": 0.15, "diabetes": 0.1, "systolic_bp": 0.15, "cholesterol": 0.1, "hdl": 0.05, "bp_treatment": 0.05}
            },
            "breast_cancer_gail": {
                "factors": ["age", "age_menarche", "age_first_birth", "relatives_breast_cancer", "biopsies", "atypical_hyperplasia"],
                "weights": {"age": 0.3, "age_menarche": 0.1, "age_first_birth": 0.15, "relatives_breast_cancer": 0.25, "biopsies": 0.1, "atypical_hyperplasia": 0.1}
            }
        }

    def _load_screening_protocols(self) -> Dict[str, Any]:
        """Load screening protocols and schedules"""
        return {
            "age_based_screening": {
                "18-39": ["blood_pressure", "cholesterol_if_risk", "depression_screening"],
                "40-49": ["blood_pressure", "cholesterol", "diabetes_screening", "mammography_if_risk"],
                "50-64": ["blood_pressure", "cholesterol", "diabetes", "mammography", "colonoscopy", "depression"],
                "65+": ["blood_pressure", "cholesterol", "diabetes", "mammography", "colonoscopy", "bone_density", "depression", "cognitive_assessment"]
            },
            "gender_specific": {
                "female": ["mammography", "pap_smear", "bone_density", "pregnancy_planning"],
                "male": ["prostate_screening", "abdominal_aortic_aneurysm"]
            },
            "risk_based": {
                "family_history_cancer": ["genetic_counseling", "enhanced_screening"],
                "smoking_history": ["lung_cancer_screening", "copd_screening"],
                "diabetes_risk": ["glucose_tolerance_test", "hba1c"]
            }
        }

    def _load_vaccination_schedules(self) -> Dict[str, Any]:
        """Load vaccination schedules"""
        return {
            "routine_adult": {
                "influenza": {"frequency": "annual", "months": [9, 10, 11]},
                "covid19": {"frequency": "annual", "months": [9, 10, 11]},
                "tdap": {"frequency": "10_years", "booster": True}
            },
            "age_specific": {
                "50+": ["shingles"],
                "65+": ["pneumonia", "high_dose_influenza"]
            },
            "condition_specific": {
                "diabetes": ["influenza", "pneumonia", "covid19"],
                "heart_disease": ["influenza", "pneumonia", "covid19"],
                "copd": ["influenza", "pneumonia", "covid19"],
                "immunocompromised": ["all_recommended", "avoid_live_vaccines"]
            }
        }

    async def generate_prevention_plan(
            self,
            patient_id: str,
            patient_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive personalized prevention plan
        """
        try:
            # Get patient data if not provided
            if not patient_data:
                patient_data = await self._get_patient_data(patient_id)

            # Calculate risk scores
            risk_assessment = self._calculate_risk_scores(patient_data)

            # Generate screening recommendations
            screening_recommendations = self._generate_screening_recommendations(patient_data, risk_assessment)

            # Generate vaccination recommendations
            vaccination_recommendations = self._generate_vaccination_recommendations(patient_data)

            # Generate lifestyle recommendations
            lifestyle_recommendations = self._generate_lifestyle_recommendations(patient_data, risk_assessment)

            # Generate medication recommendations for prevention
            medication_recommendations = self._generate_prevention_medications(patient_data, risk_assessment)

            # Create prevention timeline
            prevention_timeline = self._create_prevention_timeline(
                screening_recommendations, vaccination_recommendations, patient_data
            )

            # Calculate prevention score
            prevention_score = self._calculate_prevention_score(
                screening_recommendations, vaccination_recommendations, lifestyle_recommendations
            )

            return {
                "patient_id": patient_id,
                "prevention_plan_id": f"prev_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "risk_assessment": risk_assessment,
                "screening_recommendations": screening_recommendations,
                "vaccination_recommendations": vaccination_recommendations,
                "lifestyle_recommendations": lifestyle_recommendations,
                "medication_recommendations": medication_recommendations,
                "prevention_timeline": prevention_timeline,
                "prevention_score": prevention_score,
                "next_review_date": self._calculate_next_review(patient_data),
                "generated_at": datetime.utcnow().isoformat(),
                "valid_until": (datetime.utcnow() + timedelta(days=365)).isoformat()
            }

        except Exception as e:
            logger.error(f"Prevention plan generation error: {e}")
            return {"error": str(e), "patient_id": patient_id}

    async def _get_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Get patient data for prevention planning"""
        # Generate sample patient data for demo
        return {
            "patient_id": patient_id,
            "age": random.randint(30, 80),
            "gender": random.choice(["male", "female"]),
            "race": random.choice(["white", "black", "hispanic", "asian"]),
            "smoking_status": random.choice(["never", "former", "current"]),
            "family_history": {
                "heart_disease": random.choice([True, False]),
                "cancer": random.choice([True, False]),
                "diabetes": random.choice([True, False])
            },
            "current_conditions": random.sample(["diabetes", "hypertension", "high_cholesterol"], k=random.randint(0, 2)),
            "medications": random.sample(["metformin", "lisinopril", "atorvastatin"], k=random.randint(0, 2)),
            "vital_signs": {
                "blood_pressure": random.randint(110, 160),
                "cholesterol": random.randint(150, 280),
                "hdl": random.randint(30, 80),
                "glucose": random.randint(80, 140)
            },
            "last_screenings": {
                "mammography": "2023-01-15" if random.choice([True, False]) else None,
                "colonoscopy": "2020-06-20" if random.choice([True, False]) else None,
                "pap_smear": "2023-08-10" if random.choice([True, False]) else None
            },
            "vaccinations": {
                "influenza": "2024-10-01" if random.choice([True, False]) else None,
                "covid19": "2024-09-15" if random.choice([True, False]) else None,
                "pneumonia": "2022-03-10" if random.choice([True, False]) else None
            }
        }

    def _calculate_risk_scores(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate various disease risk scores"""
        risk_scores = {}

        # Cardiovascular risk using Framingham-like calculation
        cvd_risk = self._calculate_cvd_risk(patient_data)
        risk_scores["cardiovascular"] = cvd_risk

        # Cancer risks
        if patient_data["gender"] == "female":
            breast_cancer_risk = self._calculate_breast_cancer_risk(patient_data)
            risk_scores["breast_cancer"] = breast_cancer_risk

        if patient_data["gender"] == "male" and patient_data["age"] > 50:
            prostate_cancer_risk = self._calculate_prostate_cancer_risk(patient_data)
            risk_scores["prostate_cancer"] = prostate_cancer_risk

        # Colorectal cancer risk
        colorectal_risk = self._calculate_colorectal_cancer_risk(patient_data)
        risk_scores["colorectal_cancer"] = colorectal_risk

        # Diabetes risk
        diabetes_risk = self._calculate_diabetes_risk(patient_data)
        risk_scores["diabetes"] = diabetes_risk

        # Osteoporosis risk
        if patient_data["age"] > 50:
            osteoporosis_risk = self._calculate_osteoporosis_risk(patient_data)
            risk_scores["osteoporosis"] = osteoporosis_risk

        return risk_scores

    def _calculate_cvd_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cardiovascular disease risk"""
        age = patient_data["age"]
        gender = patient_data["gender"]
        smoking = patient_data["smoking_status"] == "current"
        diabetes = "diabetes" in patient_data["current_conditions"]
        bp = patient_data["vital_signs"]["blood_pressure"]
        cholesterol = patient_data["vital_signs"]["cholesterol"]
        hdl = patient_data["vital_signs"]["hdl"]

        # Simplified Framingham risk calculation
        risk_score = 0.0

        # Age factor
        if gender == "male":
            risk_score += max(0, (age - 20) * 0.04)
        else:
            risk_score += max(0, (age - 20) * 0.03)

        # Risk factors
        if smoking:
            risk_score += 0.15
        if diabetes:
            risk_score += 0.12
        if bp > 140:
            risk_score += 0.10
        if cholesterol > 240:
            risk_score += 0.08
        if hdl < 40:
            risk_score += 0.06

        # Family history
        if patient_data["family_history"]["heart_disease"]:
            risk_score += 0.08

        # Convert to 10-year risk percentage
        ten_year_risk = min(50.0, risk_score * 100)

        return {
            "ten_year_risk": ten_year_risk,
            "risk_category": self._categorize_cvd_risk(ten_year_risk),
            "risk_factors": self._identify_cvd_risk_factors(patient_data),
            "modifiable_factors": self._identify_modifiable_cvd_factors(patient_data)
        }

    def _categorize_cvd_risk(self, risk_percentage: float) -> str:
        """Categorize CVD risk level"""
        if risk_percentage < 5:
            return "Low"
        elif risk_percentage < 10:
            return "Intermediate"
        elif risk_percentage < 20:
            return "High"
        else:
            return "Very High"

    def _identify_cvd_risk_factors(self, patient_data: Dict[str, Any]) -> List[str]:
        """Identify CVD risk factors"""
        risk_factors = []

        if patient_data["age"] > 65:
            risk_factors.append("Advanced age")
        if patient_data["smoking_status"] == "current":
            risk_factors.append("Current smoking")
        if "diabetes" in patient_data["current_conditions"]:
            risk_factors.append("Diabetes")
        if "hypertension" in patient_data["current_conditions"]:
            risk_factors.append("Hypertension")
        if patient_data["vital_signs"]["cholesterol"] > 240:
            risk_factors.append("High cholesterol")
        if patient_data["family_history"]["heart_disease"]:
            risk_factors.append("Family history of heart disease")

        return risk_factors

    def _identify_modifiable_cvd_factors(self, patient_data: Dict[str, Any]) -> List[str]:
        """Identify modifiable CVD risk factors"""
        modifiable = []

        if patient_data["smoking_status"] == "current":
            modifiable.append("Smoking cessation")
        if patient_data["vital_signs"]["blood_pressure"] > 140:
            modifiable.append("Blood pressure control")
        if patient_data["vital_signs"]["cholesterol"] > 200:
            modifiable.append("Cholesterol management")
        if "diabetes" in patient_data["current_conditions"]:
            modifiable.append("Diabetes control")

        modifiable.extend(["Diet improvement", "Regular exercise", "Weight management"])

        return modifiable

    def _calculate_breast_cancer_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate breast cancer risk for women"""
        age = patient_data["age"]
        family_history = patient_data["family_history"]["cancer"]

        # Simplified Gail model calculation
        risk_score = 0.0

        # Age factor
        if age >= 50:
            risk_score += 0.3
        elif age >= 40:
            risk_score += 0.2

        # Family history
        if family_history:
            risk_score += 0.4

        # Convert to lifetime risk
        lifetime_risk = min(25.0, risk_score * 50)

        return {
            "lifetime_risk": lifetime_risk,
            "risk_category": "High" if lifetime_risk > 20 else "Average",
            "screening_recommendation": "Enhanced" if lifetime_risk > 20 else "Standard"
        }

    def _calculate_prostate_cancer_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate prostate cancer risk for men"""
        age = patient_data["age"]
        race = patient_data["race"]
        family_history = patient_data["family_history"]["cancer"]

        risk_score = 0.0

        # Age factor
        if age >= 70:
            risk_score += 0.4
        elif age >= 60:
            risk_score += 0.3
        elif age >= 50:
            risk_score += 0.2

        # Race factor
        if race == "black":
            risk_score += 0.3

        # Family history
        if family_history:
            risk_score += 0.2

        lifetime_risk = min(20.0, risk_score * 40)

        return {
            "lifetime_risk": lifetime_risk,
            "risk_category": "High" if lifetime_risk > 15 else "Average",
            "screening_recommendation": "Discuss with provider" if lifetime_risk > 10 else "Standard"
        }

    def _calculate_colorectal_cancer_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate colorectal cancer risk"""
        age = patient_data["age"]
        family_history = patient_data["family_history"]["cancer"]

        risk_score = 0.0

        if age >= 50:
            risk_score += 0.3
        if family_history:
            risk_score += 0.2

        lifetime_risk = min(15.0, risk_score * 30)

        return {
            "lifetime_risk": lifetime_risk,
            "risk_category": "High" if lifetime_risk > 10 else "Average",
            "screening_age": 45 if family_history else 50
        }

    def _calculate_diabetes_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate type 2 diabetes risk"""
        age = patient_data["age"]
        family_history = patient_data["family_history"]["diabetes"]
        glucose = patient_data["vital_signs"]["glucose"]

        risk_score = 0.0

        if age >= 45:
            risk_score += 0.2
        if family_history:
            risk_score += 0.3
        if glucose >= 100:
            risk_score += 0.4

        ten_year_risk = min(50.0, risk_score * 80)

        return {
            "ten_year_risk": ten_year_risk,
            "risk_category": "High" if ten_year_risk > 20 else "Moderate" if ten_year_risk > 10 else "Low",
            "screening_recommendation": "Annual" if ten_year_risk > 20 else "Every 3 years"
        }

    def _calculate_osteoporosis_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate osteoporosis risk"""
        age = patient_data["age"]
        gender = patient_data["gender"]

        risk_score = 0.0

        if gender == "female":
            if age >= 65:
                risk_score += 0.5
            elif age >= 50:
                risk_score += 0.3
        else:  # male
            if age >= 70:
                risk_score += 0.4

        lifetime_risk = min(30.0, risk_score * 60)

        return {
            "lifetime_risk": lifetime_risk,
            "risk_category": "High" if lifetime_risk > 20 else "Moderate" if lifetime_risk > 10 else "Low",
            "screening_age": 65 if gender == "female" else 70
        }

    def _generate_screening_recommendations(
            self,
            patient_data: Dict[str, Any],
            risk_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate personalized screening recommendations"""
        recommendations = []
        age = patient_data["age"]
        gender = patient_data["gender"]
        last_screenings = patient_data["last_screenings"]

        # Mammography
        if gender == "female" and age >= 40:
            last_mammo = last_screenings.get("mammography")
            is_due = self._is_screening_due(last_mammo, "annual")

            if is_due:
                priority = "High" if risk_assessment.get("breast_cancer", {}).get("risk_category") == "High" else "Routine"
                recommendations.append({
                    "screening_type": "mammography",
                    "description": "Breast cancer screening",
                    "frequency": "Annual",
                    "priority": priority,
                    "due_date": "Now" if not last_mammo else "Overdue",
                    "risk_based": risk_assessment.get("breast_cancer", {}).get("risk_category") == "High"
                })

        # Colonoscopy
        if age >= 45:
            last_colonoscopy = last_screenings.get("colonoscopy")
            is_due = self._is_screening_due(last_colonoscopy, "10_years")

            if is_due:
                screening_age = risk_assessment.get("colorectal_cancer", {}).get("screening_age", 50)
                if age >= screening_age:
                    recommendations.append({
                        "screening_type": "colonoscopy",
                        "description": "Colorectal cancer screening",
                        "frequency": "Every 10 years",
                        "priority": "Routine",
                        "due_date": "Now" if not last_colonoscopy else "Overdue",
                        "alternatives": ["FIT test annually", "Cologuard every 3 years"]
                    })

        # Pap smear
        if gender == "female" and 21 <= age <= 65:
            last_pap = last_screenings.get("pap_smear")
            is_due = self._is_screening_due(last_pap, "3_years")

            if is_due:
                recommendations.append({
                    "screening_type": "pap_smear",
                    "description": "Cervical cancer screening",
                    "frequency": "Every 3 years",
                    "priority": "Routine",
                    "due_date": "Now" if not last_pap else "Overdue"
                })

        # Bone density
        if (gender == "female" and age >= 65) or (gender == "male" and age >= 70):
            osteoporosis_risk = risk_assessment.get("osteoporosis", {})
            if osteoporosis_risk.get("risk_category") in ["Moderate", "High"]:
                recommendations.append({
                    "screening_type": "bone_density",
                    "description": "Osteoporosis screening",
                    "frequency": "Every 2 years",
                    "priority": "Routine",
                    "due_date": "Schedule"
                })

        # Cardiovascular screening
        cvd_risk = risk_assessment.get("cardiovascular", {})
        if cvd_risk.get("risk_category") in ["High", "Very High"]:
            recommendations.extend([
                {
                    "screening_type": "lipid_panel",
                    "description": "Cholesterol screening",
                    "frequency": "Annual",
                    "priority": "High",
                    "due_date": "Schedule"
                },
                {
                    "screening_type": "blood_pressure",
                    "description": "Blood pressure monitoring",
                    "frequency": "Every 6 months",
                    "priority": "High",
                    "due_date": "Schedule"
                }
            ])

        return recommendations

    def _is_screening_due(self, last_screening_date: Optional[str], frequency: str) -> bool:
        """Check if screening is due based on frequency"""
        if not last_screening_date:
            return True

        try:
            last_date = datetime.strptime(last_screening_date, "%Y-%m-%d")
            current_date = datetime.utcnow()

            frequency_days = {
                "annual": 365,
                "3_years": 365 * 3,
                "10_years": 365 * 10,
                "6_months": 180
            }

            days_since = (current_date - last_date).days
            return days_since >= frequency_days.get(frequency, 365)

        except:
            return True

    def _generate_vaccination_recommendations(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate vaccination recommendations"""
        recommendations = []
        age = patient_data["age"]
        conditions = patient_data["current_conditions"]
        vaccinations = patient_data["vaccinations"]

        # Annual vaccines
        current_month = datetime.utcnow().month
        if current_month >= 9:  # Flu season
            if not vaccinations.get("influenza") or self._is_vaccination_due(vaccinations.get("influenza"), "annual"):
                recommendations.append({
                    "vaccine": "influenza",
                    "description": "Annual flu vaccine",
                    "priority": "High" if conditions else "Routine",
                    "due_date": "Now",
                    "reason": "Annual protection against influenza"
                })

            if not vaccinations.get("covid19") or self._is_vaccination_due(vaccinations.get("covid19"), "annual"):
                recommendations.append({
                    "vaccine": "covid19",
                    "description": "COVID-19 vaccine",
                    "priority": "High",
                    "due_date": "Now",
                    "reason": "Annual protection against COVID-19"
                })

        # Age-specific vaccines
        if age >= 50 and not vaccinations.get("shingles"):
            recommendations.append({
                "vaccine": "shingles",
                "description": "Shingles vaccine (Shingrix)",
                "priority": "Routine",
                "due_date": "Schedule",
                "reason": "Prevention of shingles and post-herpetic neuralgia"
            })

        if age >= 65 and not vaccinations.get("pneumonia"):
            recommendations.append({
                "vaccine": "pneumonia",
                "description": "Pneumococcal vaccine",
                "priority": "High",
                "due_date": "Schedule",
                "reason": "Prevention of pneumococcal disease"
            })

        # Condition-specific vaccines
        if conditions:
            high_risk_vaccines = ["influenza", "pneumonia", "covid19"]
            for vaccine in high_risk_vaccines:
                if vaccine not in [rec["vaccine"] for rec in recommendations]:
                    if not vaccinations.get(vaccine):
                        recommendations.append({
                            "vaccine": vaccine,
                            "description": f"{vaccine.title()} vaccine (high-risk)",
                            "priority": "High",
                            "due_date": "Schedule",
                            "reason": f"High-risk due to {', '.join(conditions)}"
                        })

        return recommendations

    def _is_vaccination_due(self, last_vaccination_date: str, frequency: str) -> bool:
        """Check if vaccination is due"""
        try:
            last_date = datetime.strptime(last_vaccination_date, "%Y-%m-%d")
            current_date = datetime.utcnow()

            if frequency == "annual":
                return (current_date - last_date).days >= 365
            elif frequency == "10_years":
                return (current_date - last_date).days >= 365 * 10

            return False
        except:
            return True

    def _generate_lifestyle_recommendations(
            self,
            patient_data: Dict[str, Any],
            risk_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate lifestyle modification recommendations"""
        recommendations = []

        # Smoking cessation
        if patient_data["smoking_status"] == "current":
            recommendations.append({
                "category": "smoking_cessation",
                "recommendation": "Quit smoking",
                "priority": "Critical",
                "interventions": ["Nicotine replacement therapy", "Counseling", "Prescription medications"],
                "expected_benefit": "Reduces cardiovascular and cancer risk significantly",
                "resources": ["Quitline: 1-800-QUIT-NOW", "Smoking cessation apps", "Healthcare provider support"]
            })

        # Diet and nutrition
        cvd_risk = risk_assessment.get("cardiovascular", {}).get("risk_category")
        if cvd_risk in ["High", "Very High"] or "diabetes" in patient_data["current_conditions"]:
            recommendations.append({
                "category": "nutrition",
                "recommendation": "Heart-healthy diet",
                "priority": "High",
                "interventions": ["Mediterranean diet", "DASH diet", "Reduce sodium", "Increase fiber"],
                "expected_benefit": "Reduces cardiovascular risk and improves diabetes control",
                "resources": ["Nutritionist referral", "Meal planning apps", "Cooking classes"]
            })

        # Physical activity
        recommendations.append({
            "category": "physical_activity",
            "recommendation": "Regular exercise",
            "priority": "High",
            "interventions": ["150 minutes moderate aerobic activity per week", "Strength training 2x/week"],
            "expected_benefit": "Reduces risk of chronic diseases and improves overall health",
            "resources": ["Fitness apps", "Community programs", "Physical therapy if needed"]
        })

        # Weight management
        recommendations.append({
            "category": "weight_management",
            "recommendation": "Maintain healthy weight",
            "priority": "Medium",
            "interventions": ["Calorie awareness", "Portion control", "Regular weigh-ins"],
            "expected_benefit": "Reduces risk of diabetes, cardiovascular disease, and cancer",
            "resources": ["Weight management programs", "Dietitian consultation", "Support groups"]
        })

        # Stress management
        recommendations.append({
            "category": "stress_management",
            "recommendation": "Stress reduction techniques",
            "priority": "Medium",
            "interventions": ["Meditation", "Yoga", "Deep breathing", "Regular sleep schedule"],
            "expected_benefit": "Improves mental health and reduces cardiovascular risk",
            "resources": ["Meditation apps", "Stress management classes", "Mental health counseling"]
        })

        return recommendations

    def _generate_prevention_medications(
            self,
            patient_data: Dict[str, Any],
            risk_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate medication recommendations for prevention"""
        recommendations = []

        # Aspirin for cardiovascular prevention
        cvd_risk = risk_assessment.get("cardiovascular", {})
        if cvd_risk.get("risk_category") in ["High", "Very High"] and patient_data["age"] >= 40:
            recommendations.append({
                "medication": "aspirin",
                "indication": "Cardiovascular disease prevention",
                "dosage": "81mg daily",
                "priority": "Consider",
                "benefits": "Reduces risk of heart attack and stroke",
                "risks": "Increased bleeding risk",
                "contraindications": ["History of bleeding", "Allergy to aspirin"],
                "monitoring": "Annual assessment of bleeding risk"
            })

        # Statins for cholesterol
        if cvd_risk.get("risk_category") in ["High", "Very High"] or patient_data["vital_signs"]["cholesterol"] > 240:
            recommendations.append({
                "medication": "statin",
                "indication": "Cholesterol management and cardiovascular prevention",
                "dosage": "Moderate to high intensity",
                "priority": "Recommended",
                "benefits": "Reduces cardiovascular events by 25-35%",
                "risks": "Muscle pain, liver enzyme elevation",
                "monitoring": "Lipid panel and liver enzymes in 6-8 weeks, then annually"
            })

        # Vitamin D and Calcium for bone health
        osteoporosis_risk = risk_assessment.get("osteoporosis", {})
        if osteoporosis_risk and osteoporosis_risk.get("risk_category") in ["Moderate", "High"]:
            recommendations.append({
                "medication": "calcium_vitamin_d",
                "indication": "Bone health and osteoporosis prevention",
                "dosage": "Calcium 1200mg + Vitamin D 800-1000 IU daily",
                "priority": "Recommended",
                "benefits": "Reduces fracture risk",
                "risks": "Kidney stones (rare), constipation",
                "monitoring": "Annual vitamin D level"
            })

        return recommendations

    def _create_prevention_timeline(
            self,
            screening_recs: List[Dict[str, Any]],
            vaccination_recs: List[Dict[str, Any]],
            patient_data: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Create prevention timeline"""
        timeline = {
            "immediate": [],
            "within_3_months": [],
            "within_6_months": [],
            "annual": []
        }

        # Categorize screenings by urgency
        for screening in screening_recs:
            if screening["due_date"] in ["Now", "Overdue"]:
                timeline["immediate"].append(screening)
            elif screening["priority"] == "High":
                timeline["within_3_months"].append(screening)
            else:
                timeline["within_6_months"].append(screening)

        # Categorize vaccinations
        for vaccination in vaccination_recs:
            if vaccination["due_date"] == "Now":
                timeline["immediate"].append(vaccination)
            else:
                timeline["within_3_months"].append(vaccination)

        # Annual items
        timeline["annual"].extend([
            {"item": "Annual physical exam", "type": "screening"},
            {"item": "Blood pressure check", "type": "screening"},
            {"item": "Influenza vaccination", "type": "vaccination"}
        ])

        return timeline

    def _calculate_prevention_score(
            self,
            screening_recs: List[Dict[str, Any]],
            vaccination_recs: List[Dict[str, Any]],
            lifestyle_recs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate prevention adherence score"""

        # Count completed vs recommended items
        total_recommendations = len(screening_recs) + len(vaccination_recs) + len(lifestyle_recs)

        # For demo, assume some compliance
        completed_screenings = len([s for s in screening_recs if s["due_date"] != "Now"])
        completed_vaccinations = len([v for v in vaccination_recs if v["due_date"] != "Now"])

        if total_recommendations > 0:
            compliance_score = (completed_screenings + completed_vaccinations) / total_recommendations
        else:
            compliance_score = 1.0

        # Convert to percentage
        prevention_score = compliance_score * 100

        return {
            "overall_score": prevention_score,
            "grade": self._get_prevention_grade(prevention_score),
            "screening_compliance": (completed_screenings / len(screening_recs)) * 100 if screening_recs else 100,
            "vaccination_compliance": (completed_vaccinations / len(vaccination_recs)) * 100 if vaccination_recs else 100,
            "improvement_opportunities": len(screening_recs) + len(vaccination_recs)
        }

    def _get_prevention_grade(self, score: float) -> str:
        """Convert prevention score to letter grade"""
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

    def _calculate_next_review(self, patient_data: Dict[str, Any]) -> str:
        """Calculate next prevention plan review date"""
        age = patient_data["age"]

        # Younger patients need less frequent reviews
        if age < 40:
            months = 24
        elif age < 65:
            months = 12
        else:
            months = 6

        next_review = datetime.utcnow() + timedelta(days=30 * months)
        return next_review.strftime("%Y-%m-%d")
