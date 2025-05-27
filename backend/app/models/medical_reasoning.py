"""
Medical Reasoning Engine with Clinical Decision Support
"""

import openai
import json
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from app.config.settings import settings

logger = logging.getLogger(__name__)

class MedicalReasoningEngine:
    """
    Advanced medical reasoning engine for clinical decision support
    """

    def __init__(self):
        self.openai_client = None
        self.medical_knowledge = self._load_medical_knowledge()
        self.reasoning_chains = self._initialize_reasoning_chains()
        self._initialize_client()
        logger.info("✅ Medical Reasoning Engine initialized")

    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ Medical reasoning client initialized")
            else:
                logger.warning("⚠️ Medical reasoning using fallback mode")
        except Exception as e:
            logger.error(f"❌ Medical reasoning initialization error: {e}")

    def _load_medical_knowledge(self) -> Dict[str, Any]:
        """Load medical knowledge base"""
        return {
            "clinical_guidelines": {
                "hypertension": {
                    "definition": "Blood pressure ≥130/80 mmHg",
                    "risk_factors": ["age", "obesity", "smoking", "diabetes", "family_history"],
                    "treatment": ["lifestyle_modification", "ACE_inhibitors", "diuretics"],
                    "monitoring": "BP checks every 3-6 months"
                },
                "diabetes": {
                    "definition": "HbA1c ≥6.5% or FPG ≥126 mg/dL",
                    "risk_factors": ["obesity", "family_history", "sedentary_lifestyle"],
                    "treatment": ["metformin", "insulin", "lifestyle_modification"],
                    "monitoring": "HbA1c every 3-6 months"
                },
                "heart_failure": {
                    "definition": "Inability of heart to pump adequate blood",
                    "risk_factors": ["CAD", "hypertension", "diabetes", "cardiomyopathy"],
                    "treatment": ["ACE_inhibitors", "beta_blockers", "diuretics"],
                    "monitoring": "Echo every 6-12 months"
                }
            },
            "drug_interactions": {
                "warfarin": ["aspirin", "NSAIDs", "antibiotics"],
                "digoxin": ["diuretics", "ACE_inhibitors"],
                "lithium": ["diuretics", "ACE_inhibitors", "NSAIDs"]
            },
            "contraindications": {
                "ACE_inhibitors": ["pregnancy", "hyperkalemia", "bilateral_renal_stenosis"],
                "beta_blockers": ["asthma", "COPD", "heart_block"],
                "metformin": ["kidney_disease", "liver_disease", "heart_failure"]
            }
        }

    def _initialize_reasoning_chains(self) -> Dict[str, List[str]]:
        """Initialize clinical reasoning chains"""
        return {
            "differential_diagnosis": [
                "symptom_analysis",
                "risk_factor_assessment",
                "physical_examination_review",
                "diagnostic_test_interpretation",
                "probability_ranking"
            ],
            "treatment_planning": [
                "diagnosis_confirmation",
                "contraindication_check",
                "drug_interaction_screening",
                "dosage_calculation",
                "monitoring_plan"
            ],
            "risk_assessment": [
                "baseline_risk_calculation",
                "modifiable_risk_factors",
                "protective_factors",
                "intervention_recommendations",
                "follow_up_planning"
            ]
        }

    async def clinical_reasoning(
            self,
            patient_data: Dict[str, Any],
            reasoning_type: str = "differential_diagnosis",
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform clinical reasoning based on patient data
        """
        try:
            reasoning_chain = self.reasoning_chains.get(reasoning_type, [])

            if not reasoning_chain:
                raise ValueError(f"Unknown reasoning type: {reasoning_type}")

            # Execute reasoning chain
            reasoning_results = {}
            for step in reasoning_chain:
                step_result = await self._execute_reasoning_step(
                    step, patient_data, reasoning_results, context or {}
                )
                reasoning_results[step] = step_result

            # Generate final reasoning
            final_reasoning = await self._generate_final_reasoning(
                reasoning_type, reasoning_results, patient_data
            )

            return {
                "reasoning_type": reasoning_type,
                "patient_id": patient_data.get("patient_id"),
                "reasoning_steps": reasoning_results,
                "final_reasoning": final_reasoning,
                "confidence": self._calculate_confidence(reasoning_results),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Clinical reasoning error: {e}")
            return {"error": str(e), "reasoning_type": reasoning_type}

    async def _execute_reasoning_step(
            self,
            step: str,
            patient_data: Dict[str, Any],
            previous_results: Dict[str, Any],
            context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute individual reasoning step"""

        if step == "symptom_analysis":
            return await self._analyze_symptoms(patient_data)
        elif step == "risk_factor_assessment":
            return await self._assess_risk_factors(patient_data)
        elif step == "physical_examination_review":
            return await self._review_physical_exam(patient_data)
        elif step == "diagnostic_test_interpretation":
            return await self._interpret_diagnostic_tests(patient_data)
        elif step == "probability_ranking":
            return await self._rank_probabilities(previous_results)
        elif step == "diagnosis_confirmation":
            return await self._confirm_diagnosis(patient_data, context)
        elif step == "contraindication_check":
            return await self._check_contraindications(patient_data)
        elif step == "drug_interaction_screening":
            return await self._screen_drug_interactions(patient_data)
        elif step == "dosage_calculation":
            return await self._calculate_dosage(patient_data)
        elif step == "monitoring_plan":
            return await self._create_monitoring_plan(patient_data)
        elif step == "baseline_risk_calculation":
            return await self._calculate_baseline_risk(patient_data)
        elif step == "modifiable_risk_factors":
            return await self._identify_modifiable_risks(patient_data)
        elif step == "protective_factors":
            return await self._identify_protective_factors(patient_data)
        elif step == "intervention_recommendations":
            return await self._recommend_interventions(patient_data, previous_results)
        elif step == "follow_up_planning":
            return await self._plan_follow_up(patient_data, previous_results)
        else:
            return {"error": f"Unknown reasoning step: {step}"}

    async def _analyze_symptoms(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patient symptoms"""
        symptoms = patient_data.get("symptoms", [])
        chief_complaint = patient_data.get("chief_complaint", "")

        if self.openai_client:
            prompt = f"""
            Analyze these symptoms for differential diagnosis:
            
            Chief Complaint: {chief_complaint}
            Symptoms: {', '.join(symptoms)}
            
            Provide:
            1. Primary symptom clusters
            2. Associated symptoms
            3. Red flag symptoms
            4. Possible organ systems involved
            
            Return as JSON.
            """

            response = await self.openai_client.chat.completions.create(
                model=settings.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a clinical reasoning expert. Analyze symptoms systematically."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            try:
                return json.loads(response.choices[0].message.content)
            except:
                return {"analysis": response.choices[0].message.content}

        # Fallback analysis
        return {
            "symptom_count": len(symptoms),
            "chief_complaint": chief_complaint,
            "requires_ai_analysis": True
        }

    async def _assess_risk_factors(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess patient risk factors"""
        age = patient_data.get("age", 0)
        gender = patient_data.get("gender", "")
        medical_history = patient_data.get("medical_history", [])
        family_history = patient_data.get("family_history", [])
        social_history = patient_data.get("social_history", {})

        risk_factors = {
            "demographic": {
                "age_risk": "high" if age > 65 else "moderate" if age > 45 else "low",
                "gender_specific_risks": self._get_gender_risks(gender, age)
            },
            "medical_history": {
                "chronic_conditions": medical_history,
                "risk_level": "high" if len(medical_history) > 2 else "moderate" if medical_history else "low"
            },
            "family_history": {
                "hereditary_risks": family_history,
                "genetic_predisposition": "high" if family_history else "low"
            },
            "lifestyle": {
                "smoking": social_history.get("smoking", False),
                "alcohol": social_history.get("alcohol_use", "none"),
                "exercise": social_history.get("exercise", "unknown")
            }
        }

        return risk_factors

    async def _check_contraindications(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for medication contraindications"""
        medications = patient_data.get("medications", [])
        conditions = patient_data.get("conditions", [])
        allergies = patient_data.get("allergies", [])

        contraindications = {}

        for medication in medications:
            med_lower = medication.lower()
            contraindicated = []

            # Check against known contraindications
            for med_class, contras in self.medical_knowledge["contraindications"].items():
                if med_class.lower() in med_lower:
                    for condition in conditions:
                        if any(contra.lower() in condition.lower() for contra in contras):
                            contraindicated.append(condition)

            # Check allergies
            for allergy in allergies:
                if allergy.lower() in med_lower:
                    contraindicated.append(f"Allergy: {allergy}")

            if contraindicated:
                contraindications[medication] = contraindicated

        return {
            "contraindications_found": contraindications,
            "safety_level": "high_risk" if contraindications else "safe",
            "recommendations": self._generate_safety_recommendations(contraindications)
        }

    async def _screen_drug_interactions(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Screen for drug interactions"""
        medications = patient_data.get("medications", [])

        interactions = {}

        for i, med1 in enumerate(medications):
            for j, med2 in enumerate(medications[i+1:], i+1):
                interaction = self._check_interaction(med1, med2)
                if interaction:
                    interactions[f"{med1} + {med2}"] = interaction

        return {
            "interactions_found": interactions,
            "interaction_count": len(interactions),
            "severity": self._assess_interaction_severity(interactions)
        }

    def _check_interaction(self, med1: str, med2: str) -> Optional[Dict[str, Any]]:
        """Check interaction between two medications"""
        med1_lower = med1.lower()
        med2_lower = med2.lower()

        for drug, interacting_drugs in self.medical_knowledge["drug_interactions"].items():
            if drug.lower() in med1_lower:
                for interacting in interacting_drugs:
                    if interacting.lower() in med2_lower:
                        return {
                            "severity": "moderate",
                            "mechanism": "Known interaction",
                            "recommendation": "Monitor closely"
                        }

        return None

    def _get_gender_risks(self, gender: str, age: int) -> List[str]:
        """Get gender-specific risk factors"""
        risks = []

        if gender.lower() == "female":
            if age > 50:
                risks.extend(["osteoporosis", "cardiovascular_disease_post_menopause"])
            if 15 <= age <= 45:
                risks.append("pregnancy_considerations")
        elif gender.lower() == "male":
            if age > 40:
                risks.extend(["prostate_issues", "cardiovascular_disease"])

        return risks

    def _generate_safety_recommendations(self, contraindications: Dict[str, List[str]]) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []

        if contraindications:
            recommendations.append("Review medication list with pharmacist")
            recommendations.append("Consider alternative medications")
            recommendations.append("Monitor for adverse effects")
        else:
            recommendations.append("Current medications appear safe")

        return recommendations

    def _assess_interaction_severity(self, interactions: Dict[str, Any]) -> str:
        """Assess overall interaction severity"""
        if not interactions:
            return "none"

        # Simple severity assessment
        return "moderate" if len(interactions) > 2 else "mild"

    async def _generate_final_reasoning(
            self,
            reasoning_type: str,
            reasoning_results: Dict[str, Any],
            patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final clinical reasoning summary"""

        if self.openai_client:
            summary_prompt = f"""
            Based on the clinical reasoning analysis for {reasoning_type}:
            
            Patient Data: {json.dumps(patient_data, indent=2)}
            Reasoning Results: {json.dumps(reasoning_results, indent=2)}
            
            Provide a comprehensive clinical summary with:
            1. Key findings
            2. Clinical recommendations
            3. Risk assessment
            4. Next steps
            
            Format as JSON.
            """

            try:
                response = await self.openai_client.chat.completions.create(
                    model=settings.DEFAULT_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert clinician providing comprehensive reasoning summaries."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1500
                )

                return json.loads(response.choices[0].message.content)

            except Exception as e:
                logger.error(f"Final reasoning generation error: {e}")
                return {"summary": "Clinical reasoning completed", "details": reasoning_results}

        return {
            "summary": f"Clinical reasoning for {reasoning_type} completed",
            "key_findings": list(reasoning_results.keys()),
            "recommendations": ["Review with clinical team"]
        }

    def _calculate_confidence(self, reasoning_results: Dict[str, Any]) -> float:
        """Calculate confidence in reasoning"""
        completed_steps = len([r for r in reasoning_results.values() if not r.get("error")])
        total_steps = len(reasoning_results)

        if total_steps == 0:
            return 0.0

        return completed_steps / total_steps
