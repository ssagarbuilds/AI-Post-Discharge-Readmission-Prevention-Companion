"""
Medication and Treatment Adherence Monitoring Agent
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random
import logging

from app.config.settings import settings

logger = logging.getLogger(__name__)

class AdherenceAgent:
    """
    AI agent for monitoring and improving medication and treatment adherence
    """

    def __init__(self):
        self.adherence_metrics = self._define_adherence_metrics()
        self.barrier_categories = self._define_barrier_categories()
        self.intervention_strategies = self._load_intervention_strategies()
        self.adherence_thresholds = self._define_adherence_thresholds()
        logger.info("✅ Adherence Agent initialized")

    def _define_adherence_metrics(self) -> Dict[str, Any]:
        """Define adherence measurement methods"""
        return {
            "medication_adherence": {
                "pdc": "Proportion of Days Covered",
                "mpr": "Medication Possession Ratio",
                "self_report": "Patient Self-Report",
                "pill_count": "Pill Count Method",
                "electronic_monitoring": "Electronic Monitoring Devices"
            },
            "appointment_adherence": {
                "show_rate": "Appointment Show Rate",
                "cancellation_rate": "Last-Minute Cancellation Rate",
                "reschedule_rate": "Rescheduling Rate",
                "no_show_rate": "No-Show Rate"
            },
            "lifestyle_adherence": {
                "diet_compliance": "Dietary Recommendation Compliance",
                "exercise_compliance": "Exercise Program Adherence",
                "monitoring_compliance": "Self-Monitoring Adherence",
                "lifestyle_modification": "Lifestyle Change Adherence"
            },
            "treatment_plan_adherence": {
                "follow_up_compliance": "Follow-up Care Compliance",
                "test_completion": "Diagnostic Test Completion",
                "referral_completion": "Specialist Referral Follow-through",
                "care_plan_adherence": "Overall Care Plan Adherence"
            }
        }

    def _define_barrier_categories(self) -> Dict[str, Any]:
        """Define categories of adherence barriers"""
        return {
            "patient_factors": {
                "knowledge": ["lack_of_understanding", "health_literacy", "language_barriers"],
                "beliefs": ["medication_concerns", "side_effect_fears", "cultural_beliefs"],
                "psychological": ["depression", "anxiety", "cognitive_impairment", "motivation"],
                "physical": ["difficulty_swallowing", "dexterity_issues", "vision_problems"]
            },
            "medication_factors": {
                "complexity": ["multiple_medications", "frequent_dosing", "complex_regimen"],
                "side_effects": ["adverse_effects", "tolerability_issues", "drug_interactions"],
                "formulation": ["pill_size", "taste", "administration_method"]
            },
            "healthcare_system_factors": {
                "access": ["appointment_availability", "transportation", "clinic_hours"],
                "communication": ["provider_communication", "care_coordination", "follow_up"],
                "cost": ["medication_cost", "copayments", "insurance_coverage"]
            },
            "social_economic_factors": {
                "financial": ["medication_cost", "lost_wages", "insurance_issues"],
                "social": ["family_support", "caregiver_availability", "social_isolation"],
                "environmental": ["pharmacy_access", "medication_storage", "reminders"]
            }
        }

    def _load_intervention_strategies(self) -> Dict[str, Any]:
        """Load evidence-based intervention strategies"""
        return {
            "educational_interventions": {
                "patient_education": [
                    "Disease education sessions",
                    "Medication counseling",
                    "Side effect management education",
                    "Importance of adherence education"
                ],
                "health_literacy": [
                    "Simplified instructions",
                    "Visual aids and diagrams",
                    "Teach-back method",
                    "Language-appropriate materials"
                ]
            },
            "behavioral_interventions": {
                "reminder_systems": [
                    "Medication reminder apps",
                    "Pill organizers",
                    "Alarm systems",
                    "Calendar reminders"
                ],
                "habit_formation": [
                    "Routine establishment",
                    "Cue-based reminders",
                    "Habit stacking",
                    "Environmental modifications"
                ],
                "motivational_support": [
                    "Motivational interviewing",
                    "Goal setting",
                    "Self-efficacy building",
                    "Peer support groups"
                ]
            },
            "system_interventions": {
                "medication_management": [
                    "Medication synchronization",
                    "Automatic refills",
                    "Blister packaging",
                    "Long-acting formulations"
                ],
                "care_coordination": [
                    "Care team communication",
                    "Pharmacist involvement",
                    "Case management",
                    "Integrated care plans"
                ],
                "technology_solutions": [
                    "Electronic monitoring",
                    "Smart pill bottles",
                    "Mobile health apps",
                    "Telemedicine follow-ups"
                ]
            },
            "financial_interventions": {
                "cost_reduction": [
                    "Generic substitution",
                    "Patient assistance programs",
                    "Insurance optimization",
                    "Pharmacy shopping"
                ],
                "payment_assistance": [
                    "Copay assistance programs",
                    "Manufacturer coupons",
                    "Charitable programs",
                    "Government assistance"
                ]
            }
        }

    def _define_adherence_thresholds(self) -> Dict[str, float]:
        """Define adherence thresholds for different categories"""
        return {
            "excellent": 0.95,
            "good": 0.80,
            "suboptimal": 0.60,
            "poor": 0.40
        }

    async def assess_adherence(
            self,
            patient_id: str,
            adherence_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive adherence assessment for a patient
        """
        try:
            # Get adherence data if not provided
            if not adherence_data:
                adherence_data = await self._get_adherence_data(patient_id)

            # Calculate adherence scores
            adherence_scores = self._calculate_adherence_scores(adherence_data)

            # Identify barriers
            barriers = await self._identify_barriers(patient_id, adherence_data)

            # Generate interventions
            interventions = self._generate_interventions(adherence_scores, barriers)

            # Create adherence improvement plan
            improvement_plan = self._create_improvement_plan(adherence_scores, barriers, interventions)

            # Calculate risk assessment
            risk_assessment = self._assess_adherence_risk(adherence_scores, barriers)

            # Generate monitoring plan
            monitoring_plan = self._create_monitoring_plan(adherence_scores, risk_assessment)

            return {
                "patient_id": patient_id,
                "assessment_date": datetime.utcnow().isoformat(),
                "adherence_scores": adherence_scores,
                "identified_barriers": barriers,
                "risk_assessment": risk_assessment,
                "recommended_interventions": interventions,
                "improvement_plan": improvement_plan,
                "monitoring_plan": monitoring_plan,
                "next_assessment": self._calculate_next_assessment(adherence_scores),
                "adherence_goals": self._set_adherence_goals(adherence_scores)
            }

        except Exception as e:
            logger.error(f"Adherence assessment error: {e}")
            return {"error": str(e), "patient_id": patient_id}

    async def _get_adherence_data(self, patient_id: str) -> Dict[str, Any]:
        """Get adherence data for patient"""
        # Generate sample adherence data for demo
        return {
            "medications": [
                {
                    "name": "Lisinopril",
                    "prescribed_days": 90,
                    "days_covered": random.randint(60, 90),
                    "refills_on_time": random.randint(2, 4),
                    "total_refills": 4,
                    "last_refill_date": "2024-11-15",
                    "side_effects_reported": random.choice([True, False])
                },
                {
                    "name": "Metformin",
                    "prescribed_days": 90,
                    "days_covered": random.randint(70, 90),
                    "refills_on_time": random.randint(3, 4),
                    "total_refills": 4,
                    "last_refill_date": "2024-11-20",
                    "side_effects_reported": random.choice([True, False])
                }
            ],
            "appointments": {
                "scheduled": 6,
                "attended": random.randint(4, 6),
                "cancelled": random.randint(0, 2),
                "no_shows": random.randint(0, 1),
                "rescheduled": random.randint(0, 2)
            },
            "lifestyle_adherence": {
                "diet_compliance": random.uniform(0.5, 0.9),
                "exercise_compliance": random.uniform(0.4, 0.8),
                "blood_pressure_monitoring": random.uniform(0.6, 0.95),
                "weight_monitoring": random.uniform(0.5, 0.9)
            },
            "self_reported": {
                "medication_adherence": random.uniform(0.7, 0.95),
                "missed_doses_last_week": random.randint(0, 5),
                "reasons_for_missing": random.sample(["forgot", "side_effects", "cost", "feeling_better"], k=random.randint(0, 2))
            },
            "barriers_reported": {
                "cost_concerns": random.choice([True, False]),
                "side_effects": random.choice([True, False]),
                "forgetfulness": random.choice([True, False]),
                "complex_regimen": random.choice([True, False]),
                "transportation_issues": random.choice([True, False])
            }
        }

    def _calculate_adherence_scores(self, adherence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate adherence scores across different domains"""

        # Medication adherence (PDC - Proportion of Days Covered)
        medication_scores = []
        medication_details = []

        for med in adherence_data["medications"]:
            pdc = med["days_covered"] / med["prescribed_days"]
            refill_adherence = med["refills_on_time"] / med["total_refills"]

            medication_scores.append(pdc)
            medication_details.append({
                "medication": med["name"],
                "pdc": pdc,
                "refill_adherence": refill_adherence,
                "adherence_level": self._categorize_adherence(pdc),
                "last_refill": med["last_refill_date"],
                "side_effects": med["side_effects_reported"]
            })

        avg_medication_adherence = sum(medication_scores) / len(medication_scores) if medication_scores else 0

        # Appointment adherence
        appointments = adherence_data["appointments"]
        appointment_adherence = appointments["attended"] / appointments["scheduled"] if appointments["scheduled"] > 0 else 1.0
        no_show_rate = appointments["no_shows"] / appointments["scheduled"] if appointments["scheduled"] > 0 else 0

        # Lifestyle adherence
        lifestyle = adherence_data["lifestyle_adherence"]
        lifestyle_adherence = sum(lifestyle.values()) / len(lifestyle.values()) if lifestyle else 0

        # Self-reported adherence
        self_reported_adherence = adherence_data["self_reported"]["medication_adherence"]

        # Overall adherence score (weighted average)
        overall_adherence = (
                avg_medication_adherence * 0.4 +
                appointment_adherence * 0.2 +
                lifestyle_adherence * 0.2 +
                self_reported_adherence * 0.2
        )

        return {
            "overall_adherence": overall_adherence,
            "overall_level": self._categorize_adherence(overall_adherence),
            "medication_adherence": {
                "average_score": avg_medication_adherence,
                "level": self._categorize_adherence(avg_medication_adherence),
                "details": medication_details
            },
            "appointment_adherence": {
                "score": appointment_adherence,
                "level": self._categorize_adherence(appointment_adherence),
                "show_rate": appointment_adherence,
                "no_show_rate": no_show_rate,
                "details": appointments
            },
            "lifestyle_adherence": {
                "score": lifestyle_adherence,
                "level": self._categorize_adherence(lifestyle_adherence),
                "components": lifestyle
            },
            "self_reported": {
                "score": self_reported_adherence,
                "level": self._categorize_adherence(self_reported_adherence),
                "missed_doses": adherence_data["self_reported"]["missed_doses_last_week"],
                "reasons": adherence_data["self_reported"]["reasons_for_missing"]
            }
        }

    def _categorize_adherence(self, score: float) -> str:
        """Categorize adherence level based on score"""
        if score >= self.adherence_thresholds["excellent"]:
            return "Excellent"
        elif score >= self.adherence_thresholds["good"]:
            return "Good"
        elif score >= self.adherence_thresholds["suboptimal"]:
            return "Suboptimal"
        else:
            return "Poor"

    async def _identify_barriers(self, patient_id: str, adherence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify barriers to adherence"""

        barriers = {
            "identified_barriers": [],
            "barrier_categories": {},
            "severity_assessment": {},
            "modifiable_barriers": [],
            "patient_reported": adherence_data["barriers_reported"]
        }

        # Analyze patient-reported barriers
        reported_barriers = adherence_data["barriers_reported"]
        for barrier, present in reported_barriers.items():
            if present:
                barriers["identified_barriers"].append(barrier)

                # Categorize barrier
                category = self._categorize_barrier(barrier)
                if category not in barriers["barrier_categories"]:
                    barriers["barrier_categories"][category] = []
                barriers["barrier_categories"][category].append(barrier)

                # Assess modifiability
                if self._is_modifiable_barrier(barrier):
                    barriers["modifiable_barriers"].append(barrier)

        # Infer barriers from adherence patterns
        inferred_barriers = self._infer_barriers_from_patterns(adherence_data)
        barriers["identified_barriers"].extend(inferred_barriers)

        # Assess barrier severity
        for barrier in barriers["identified_barriers"]:
            barriers["severity_assessment"][barrier] = self._assess_barrier_severity(barrier, adherence_data)

        return barriers

    def _categorize_barrier(self, barrier: str) -> str:
        """Categorize barrier into main categories"""
        barrier_mapping = {
            "cost_concerns": "financial",
            "side_effects": "medication_factors",
            "forgetfulness": "patient_factors",
            "complex_regimen": "medication_factors",
            "transportation_issues": "healthcare_system_factors"
        }
        return barrier_mapping.get(barrier, "other")

    def _is_modifiable_barrier(self, barrier: str) -> bool:
        """Determine if barrier is modifiable"""
        modifiable_barriers = [
            "cost_concerns", "forgetfulness", "complex_regimen",
            "side_effects", "transportation_issues"
        ]
        return barrier in modifiable_barriers

    def _infer_barriers_from_patterns(self, adherence_data: Dict[str, Any]) -> List[str]:
        """Infer barriers from adherence patterns"""
        inferred = []

        # Low refill adherence suggests access or cost issues
        for med in adherence_data["medications"]:
            refill_rate = med["refills_on_time"] / med["total_refills"]
            if refill_rate < 0.7:
                inferred.append("medication_access_issues")

        # High no-show rate suggests access or motivation issues
        appointments = adherence_data["appointments"]
        if appointments["scheduled"] > 0:
            no_show_rate = appointments["no_shows"] / appointments["scheduled"]
            if no_show_rate > 0.2:
                inferred.append("appointment_barriers")

        # Inconsistent self-monitoring suggests motivation or knowledge gaps
        lifestyle = adherence_data["lifestyle_adherence"]
        if any(score < 0.6 for score in lifestyle.values()):
            inferred.append("self_management_challenges")

        return inferred

    def _assess_barrier_severity(self, barrier: str, adherence_data: Dict[str, Any]) -> str:
        """Assess severity of identified barrier"""
        # Simplified severity assessment
        severity_map = {
            "cost_concerns": "high",
            "side_effects": "high",
            "forgetfulness": "medium",
            "complex_regimen": "medium",
            "transportation_issues": "high"
        }
        return severity_map.get(barrier, "medium")

    def _generate_interventions(
            self,
            adherence_scores: Dict[str, Any],
            barriers: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate targeted interventions based on adherence scores and barriers"""

        interventions = []
        overall_adherence = adherence_scores["overall_adherence"]
        identified_barriers = barriers["identified_barriers"]

        # Priority interventions based on adherence level
        if overall_adherence < self.adherence_thresholds["poor"]:
            interventions.extend(self._get_intensive_interventions())
        elif overall_adherence < self.adherence_thresholds["suboptimal"]:
            interventions.extend(self._get_moderate_interventions())
        else:
            interventions.extend(self._get_maintenance_interventions())

        # Barrier-specific interventions
        for barrier in identified_barriers:
            barrier_interventions = self._get_barrier_specific_interventions(barrier)
            interventions.extend(barrier_interventions)

        # Domain-specific interventions
        if adherence_scores["medication_adherence"]["level"] in ["Poor", "Suboptimal"]:
            interventions.extend(self._get_medication_specific_interventions(adherence_scores))

        if adherence_scores["appointment_adherence"]["level"] in ["Poor", "Suboptimal"]:
            interventions.extend(self._get_appointment_specific_interventions())

        # Prioritize and deduplicate
        prioritized_interventions = self._prioritize_interventions(interventions)

        return prioritized_interventions[:8]  # Return top 8 interventions

    def _get_intensive_interventions(self) -> List[Dict[str, Any]]:
        """Get intensive interventions for poor adherence"""
        return [
            {
                "type": "case_management",
                "intervention": "Assign dedicated case manager",
                "priority": "High",
                "timeframe": "Immediate",
                "description": "Intensive case management with weekly contact",
                "expected_outcome": "Improved adherence monitoring and support"
            },
            {
                "type": "medication_management",
                "intervention": "Medication synchronization and packaging",
                "priority": "High",
                "timeframe": "Within 1 week",
                "description": "Synchronize all medications and provide blister packaging",
                "expected_outcome": "Simplified medication regimen"
            },
            {
                "type": "education",
                "intervention": "Intensive patient education",
                "priority": "High",
                "timeframe": "Within 1 week",
                "description": "Comprehensive education on disease and medication importance",
                "expected_outcome": "Improved understanding and motivation"
            }
        ]

    def _get_moderate_interventions(self) -> List[Dict[str, Any]]:
        """Get moderate interventions for suboptimal adherence"""
        return [
            {
                "type": "reminder_system",
                "intervention": "Electronic reminder system",
                "priority": "Medium",
                "timeframe": "Within 2 weeks",
                "description": "Set up medication reminder app and pill organizer",
                "expected_outcome": "Reduced missed doses due to forgetfulness"
            },
            {
                "type": "pharmacist_consultation",
                "intervention": "Pharmacist medication review",
                "priority": "Medium",
                "timeframe": "Within 2 weeks",
                "description": "Comprehensive medication review with pharmacist",
                "expected_outcome": "Optimized medication regimen and education"
            }
        ]

    def _get_maintenance_interventions(self) -> List[Dict[str, Any]]:
        """Get maintenance interventions for good adherence"""
        return [
            {
                "type": "positive_reinforcement",
                "intervention": "Adherence recognition and encouragement",
                "priority": "Low",
                "timeframe": "Ongoing",
                "description": "Acknowledge good adherence and encourage continuation",
                "expected_outcome": "Maintained high adherence"
            },
            {
                "type": "monitoring",
                "intervention": "Regular adherence monitoring",
                "priority": "Low",
                "timeframe": "Monthly",
                "description": "Regular check-ins to maintain adherence",
                "expected_outcome": "Early detection of adherence decline"
            }
        ]

    def _get_barrier_specific_interventions(self, barrier: str) -> List[Dict[str, Any]]:
        """Get interventions specific to identified barriers"""

        interventions = []

        if barrier == "cost_concerns":
            interventions.append({
                "type": "financial_assistance",
                "intervention": "Patient assistance program enrollment",
                "priority": "High",
                "timeframe": "Within 1 week",
                "description": "Enroll in manufacturer or pharmacy assistance programs",
                "expected_outcome": "Reduced medication costs"
            })

        elif barrier == "side_effects":
            interventions.append({
                "type": "medication_adjustment",
                "intervention": "Side effect management",
                "priority": "High",
                "timeframe": "Within 3 days",
                "description": "Review and adjust medications to minimize side effects",
                "expected_outcome": "Improved medication tolerance"
            })

        elif barrier == "forgetfulness":
            interventions.append({
                "type": "reminder_system",
                "intervention": "Multi-modal reminder system",
                "priority": "Medium",
                "timeframe": "Within 1 week",
                "description": "Set up phone alarms, pill organizer, and family reminders",
                "expected_outcome": "Reduced missed doses"
            })

        elif barrier == "complex_regimen":
            interventions.append({
                "type": "regimen_simplification",
                "intervention": "Medication regimen simplification",
                "priority": "Medium",
                "timeframe": "
