"""
Cognitive Assessment Agent for Mental Health Evaluation
"""

import openai
import json
import random
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from app.config.settings import settings

logger = logging.getLogger(__name__)

class CognitiveAssessmentAgent:
    """
    AI agent for cognitive function evaluation and mental health screening
    """

    def __init__(self):
        self.openai_client = None
        self.assessment_tools = self._load_assessment_tools()
        self.cognitive_domains = self._define_cognitive_domains()
        self.screening_protocols = self._load_screening_protocols()
        self._initialize_client()
        logger.info("✅ Cognitive Assessment Agent initialized")

    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ Cognitive assessment OpenAI client initialized")
        except Exception as e:
            logger.error(f"❌ Cognitive assessment initialization error: {e}")

    def _load_assessment_tools(self) -> Dict[str, Any]:
        """Load standardized cognitive assessment tools"""
        return {
            "mmse": {
                "name": "Mini-Mental State Examination",
                "domains": ["orientation", "attention", "memory", "language", "visuospatial"],
                "max_score": 30,
                "cutoffs": {"normal": 24, "mild_impairment": 18, "severe_impairment": 0}
            },
            "moca": {
                "name": "Montreal Cognitive Assessment",
                "domains": ["visuospatial", "naming", "attention", "language", "abstraction", "memory", "orientation"],
                "max_score": 30,
                "cutoffs": {"normal": 26, "mild_impairment": 18, "severe_impairment": 0}
            },
            "phq9": {
                "name": "Patient Health Questionnaire-9",
                "domains": ["depression_symptoms"],
                "max_score": 27,
                "cutoffs": {"minimal": 4, "mild": 9, "moderate": 14, "severe": 19}
            },
            "gad7": {
                "name": "Generalized Anxiety Disorder-7",
                "domains": ["anxiety_symptoms"],
                "max_score": 21,
                "cutoffs": {"minimal": 4, "mild": 9, "moderate": 14, "severe": 21}
            }
        }

    def _define_cognitive_domains(self) -> Dict[str, Any]:
        """Define cognitive domains and their components"""
        return {
            "memory": {
                "short_term": ["immediate_recall", "working_memory"],
                "long_term": ["episodic_memory", "semantic_memory"],
                "tests": ["word_list_recall", "story_recall", "digit_span"]
            },
            "attention": {
                "sustained": ["continuous_performance"],
                "selective": ["focused_attention"],
                "divided": ["dual_task_performance"],
                "tests": ["digit_span", "trail_making_a", "continuous_performance"]
            },
            "executive_function": {
                "planning": ["problem_solving", "strategy_formation"],
                "inhibition": ["response_inhibition", "interference_control"],
                "flexibility": ["set_shifting", "cognitive_flexibility"],
                "tests": ["trail_making_b", "stroop_test", "tower_test"]
            },
            "language": {
                "comprehension": ["auditory_comprehension", "reading_comprehension"],
                "expression": ["verbal_fluency", "naming"],
                "tests": ["boston_naming", "verbal_fluency", "comprehension_tasks"]
            },
            "visuospatial": {
                "perception": ["visual_perception", "spatial_perception"],
                "construction": ["drawing", "copying"],
                "tests": ["clock_drawing", "cube_copying", "block_design"]
            }
        }

    def _load_screening_protocols(self) -> Dict[str, Any]:
        """Load mental health screening protocols"""
        return {
            "depression_screening": {
                "primary_tool": "phq9",
                "follow_up_tools": ["phq2", "beck_depression"],
                "risk_assessment": True,
                "suicide_screening": True
            },
            "anxiety_screening": {
                "primary_tool": "gad7",
                "follow_up_tools": ["beck_anxiety", "hamilton_anxiety"],
                "panic_assessment": True
            },
            "cognitive_screening": {
                "primary_tool": "moca",
                "follow_up_tools": ["mmse", "detailed_neuropsych"],
                "dementia_assessment": True
            },
            "substance_use_screening": {
                "primary_tool": "audit",
                "follow_up_tools": ["cage", "dast"],
                "intervention_brief": True
            }
        }

    async def conduct_cognitive_assessment(
            self,
            patient_id: str,
            assessment_type: str = "comprehensive",
            patient_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Conduct comprehensive cognitive assessment
        """
        try:
            # Determine assessment battery
            if assessment_type == "screening":
                assessments = ["moca", "phq9", "gad7"]
            elif assessment_type == "cognitive_only":
                assessments = ["moca", "mmse"]
            elif assessment_type == "mental_health":
                assessments = ["phq9", "gad7"]
            else:  # comprehensive
                assessments = ["moca", "mmse", "phq9", "gad7"]

            # Conduct assessments
            assessment_results = {}
            for assessment in assessments:
                result = await self._conduct_single_assessment(assessment, patient_data)
                assessment_results[assessment] = result

            # Analyze results
            cognitive_profile = self._analyze_cognitive_profile(assessment_results)

            # Generate recommendations
            recommendations = self._generate_recommendations(assessment_results, cognitive_profile)

            # Risk assessment
            risk_assessment = self._assess_cognitive_risks(assessment_results, patient_data)

            return {
                "assessment_id": f"cog_assess_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "patient_id": patient_id,
                "assessment_type": assessment_type,
                "assessment_results": assessment_results,
                "cognitive_profile": cognitive_profile,
                "risk_assessment": risk_assessment,
                "recommendations": recommendations,
                "follow_up_needed": self._determine_follow_up_needs(assessment_results),
                "assessment_date": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Cognitive assessment error: {e}")
            return {"error": str(e), "patient_id": patient_id}

    async def _conduct_single_assessment(
            self,
            assessment_name: str,
            patient_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Conduct a single cognitive assessment"""

        assessment_config = self.assessment_tools.get(assessment_name)
        if not assessment_config:
            return {"error": f"Unknown assessment: {assessment_name}"}

        # Simulate assessment (in production, this would be interactive)
        if assessment_name == "moca":
            return await self._simulate_moca_assessment(patient_data)
        elif assessment_name == "mmse":
            return await self._simulate_mmse_assessment(patient_data)
        elif assessment_name == "phq9":
            return await self._simulate_phq9_assessment(patient_data)
        elif assessment_name == "gad7":
            return await self._simulate_gad7_assessment(patient_data)
        else:
            return {"error": f"Assessment not implemented: {assessment_name}"}

    async def _simulate_moca_assessment(self, patient_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate MoCA assessment"""

        # Generate realistic scores based on patient factors
        age = patient_data.get("age", 65) if patient_data else 65
        education = patient_data.get("education_years", 12) if patient_data else 12

        # Age and education adjustments
        base_score = 28
        if age > 75:
            base_score -= random.randint(1, 4)
        if education < 12:
            base_score -= random.randint(1, 2)

        # Add some randomness
        total_score = max(0, min(30, base_score + random.randint(-3, 2)))

        # Domain scores
        domain_scores = {
            "visuospatial": min(5, max(0, 4 + random.randint(-1, 1))),
            "naming": min(3, max(0, 3 + random.randint(-1, 0))),
            "attention": min(6, max(0, 5 + random.randint(-2, 1))),
            "language": min(3, max(0, 2 + random.randint(-1, 1))),
            "abstraction": min(2, max(0, 2 + random.randint(-1, 0))),
            "delayed_recall": min(5, max(0, 3 + random.randint(-2, 2))),
            "orientation": min(6, max(0, 6 + random.randint(-1, 0)))
        }

        # Adjust total to match domain sum
        total_score = sum(domain_scores.values())

        # Interpretation
        if total_score >= 26:
            interpretation = "Normal cognitive function"
            impairment_level = "none"
        elif total_score >= 22:
            interpretation = "Mild cognitive impairment possible"
            impairment_level = "mild"
        elif total_score >= 18:
            interpretation = "Moderate cognitive impairment"
            impairment_level = "moderate"
        else:
            interpretation = "Severe cognitive impairment"
            impairment_level = "severe"

        return {
            "assessment": "moca",
            "total_score": total_score,
            "max_score": 30,
            "domain_scores": domain_scores,
            "interpretation": interpretation,
            "impairment_level": impairment_level,
            "percentile": self._score_to_percentile(total_score, 30),
            "recommendations": self._get_moca_recommendations(total_score)
        }

    async def _simulate_mmse_assessment(self, patient_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate MMSE assessment"""

        age = patient_data.get("age", 65) if patient_data else 65

        # Generate score based on age
        base_score = 29
        if age > 80:
            base_score -= random.randint(2, 5)
        elif age > 70:
            base_score -= random.randint(1, 3)

        total_score = max(0, min(30, base_score + random.randint(-2, 1)))

        # Domain scores
        domain_scores = {
            "orientation": min(10, max(0, 9 + random.randint(-1, 1))),
            "registration": min(3, max(0, 3)),
            "attention": min(5, max(0, 4 + random.randint(-2, 1))),
            "recall": min(3, max(0, 2 + random.randint(-2, 1))),
            "language": min(9, max(0, 8 + random.randint(-1, 1)))
        }

        total_score = sum(domain_scores.values())

        # Interpretation
        if total_score >= 24:
            interpretation = "Normal cognitive function"
            impairment_level = "none"
        elif total_score >= 18:
            interpretation = "Mild cognitive impairment"
            impairment_level = "mild"
        else:
            interpretation = "Significant cognitive impairment"
            impairment_level = "severe"

        return {
            "assessment": "mmse",
            "total_score": total_score,
            "max_score": 30,
            "domain_scores": domain_scores,
            "interpretation": interpretation,
            "impairment_level": impairment_level,
            "percentile": self._score_to_percentile(total_score, 30),
            "recommendations": self._get_mmse_recommendations(total_score)
        }

    async def _simulate_phq9_assessment(self, patient_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate PHQ-9 depression assessment"""

        # Generate realistic depression scores
        base_score = random.randint(2, 8)  # Most people have some symptoms

        # Adjust based on patient factors
        if patient_data:
            conditions = patient_data.get("conditions", [])
            if any("depression" in cond.lower() for cond in conditions):
                base_score += random.randint(5, 10)
            if any("chronic" in cond.lower() for cond in conditions):
                base_score += random.randint(2, 5)

        total_score = min(27, max(0, base_score))

        # Individual item scores (0-3 each)
        item_scores = []
        remaining_score = total_score
        for i in range(9):
            if i == 8:  # Last item gets remaining score
                item_scores.append(min(3, remaining_score))
            else:
                item_score = min(3, random.randint(0, min(3, remaining_score)))
                item_scores.append(item_score)
                remaining_score -= item_score

        # Interpretation
        if total_score <= 4:
            severity = "Minimal depression"
            risk_level = "low"
        elif total_score <= 9:
            severity = "Mild depression"
            risk_level = "low"
        elif total_score <= 14:
            severity = "Moderate depression"
            risk_level = "moderate"
        elif total_score <= 19:
            severity = "Moderately severe depression"
            risk_level = "high"
        else:
            severity = "Severe depression"
            risk_level = "high"

        # Suicide risk assessment (item 9)
        suicide_risk = item_scores[8] > 0

        return {
            "assessment": "phq9",
            "total_score": total_score,
            "max_score": 27,
            "item_scores": item_scores,
            "severity": severity,
            "risk_level": risk_level,
            "suicide_risk_indicated": suicide_risk,
            "suicide_risk_score": item_scores[8],
            "recommendations": self._get_phq9_recommendations(total_score, suicide_risk)
        }

    async def _simulate_gad7_assessment(self, patient_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate GAD-7 anxiety assessment"""

        # Generate realistic anxiety scores
        base_score = random.randint(1, 6)

        # Adjust based on patient factors
        if patient_data:
            conditions = patient_data.get("conditions", [])
            if any("anxiety" in cond.lower() for cond in conditions):
                base_score += random.randint(4, 8)
            age = patient_data.get("age", 50)
            if age < 40:  # Younger patients may have higher anxiety
                base_score += random.randint(1, 3)

        total_score = min(21, max(0, base_score))

        # Individual item scores (0-3 each)
        item_scores = []
        remaining_score = total_score
        for i in range(7):
            if i == 6:  # Last item gets remaining score
                item_scores.append(min(3, remaining_score))
            else:
                item_score = min(3, random.randint(0, min(3, remaining_score)))
                item_scores.append(item_score)
                remaining_score -= item_score

        # Interpretation
        if total_score <= 4:
            severity = "Minimal anxiety"
            risk_level = "low"
        elif total_score <= 9:
            severity = "Mild anxiety"
            risk_level = "low"
        elif total_score <= 14:
            severity = "Moderate anxiety"
            risk_level = "moderate"
        else:
            severity = "Severe anxiety"
            risk_level = "high"

        return {
            "assessment": "gad7",
            "total_score": total_score,
            "max_score": 21,
            "item_scores": item_scores,
            "severity": severity,
            "risk_level": risk_level,
            "recommendations": self._get_gad7_recommendations(total_score)
        }

    def _score_to_percentile(self, score: int, max_score: int) -> int:
        """Convert score to percentile"""
        return int((score / max_score) * 100)

    def _get_moca_recommendations(self, score: int) -> List[str]:
        """Get MoCA-specific recommendations"""
        if score >= 26:
            return ["Continue regular cognitive activities", "Annual cognitive screening"]
        elif score >= 22:
            return ["Cognitive training exercises", "Follow-up in 6 months", "Lifestyle modifications"]
        else:
            return ["Neuropsychological evaluation", "Medical workup for cognitive impairment", "Consider specialist referral"]

    def _get_mmse_recommendations(self, score: int) -> List[str]:
        """Get MMSE-specific recommendations"""
        if score >= 24:
            return ["Normal cognitive function", "Routine follow-up"]
        elif score >= 18:
            return ["Monitor cognitive function", "Consider detailed assessment", "Safety evaluation"]
        else:
            return ["Comprehensive evaluation needed", "Caregiver support", "Safety planning"]

    def _get_phq9_recommendations(self, score: int, suicide_risk: bool) -> List[str]:
        """Get PHQ-9 specific recommendations"""
        recommendations = []

        if suicide_risk:
            recommendations.extend([
                "IMMEDIATE: Suicide risk assessment required",
                "Safety planning",
                "Consider emergency evaluation"
            ])

        if score <= 4:
            recommendations.extend(["Minimal depression - routine monitoring", "Lifestyle counseling"])
        elif score <= 9:
            recommendations.extend(["Mild depression - counseling recommended", "Monitor symptoms"])
        elif score <= 14:
            recommendations.extend(["Moderate depression - therapy recommended", "Consider medication evaluation"])
