"""
Intelligent Symptom Triage Agent with Emergency Detection
"""

import openai
import json
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from app.config.settings import settings

logger = logging.getLogger(__name__)

class SymptomTriageAgent:
    """
    AI agent for intelligent symptom triage and emergency detection
    """

    def __init__(self):
        self.openai_client = None
        self.triage_protocols = self._load_triage_protocols()
        self.emergency_indicators = self._load_emergency_indicators()
        self.symptom_mappings = self._load_symptom_mappings()
        self._initialize_client()
        logger.info("✅ Symptom Triage Agent initialized")

    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ Symptom triage OpenAI client initialized")
        except Exception as e:
            logger.error(f"❌ Symptom triage initialization error: {e}")

    def _load_triage_protocols(self) -> Dict[str, Any]:
        """Load evidence-based triage protocols"""
        return {
            "emergency_protocol": {
                "response_time": "immediate",
                "action": "call_911",
                "escalation": "emergency_services"
            },
            "urgent_protocol": {
                "response_time": "within_1_hour",
                "action": "emergency_department",
                "escalation": "urgent_care"
            },
            "semi_urgent_protocol": {
                "response_time": "within_24_hours",
                "action": "primary_care",
                "escalation": "same_day_appointment"
            },
            "routine_protocol": {
                "response_time": "within_week",
                "action": "schedule_appointment",
                "escalation": "routine_care"
            }
        }

    def _load_emergency_indicators(self) -> Dict[str, List[str]]:
        """Load emergency symptom indicators"""
        return {
            "cardiovascular": [
                "chest pain", "severe chest pressure", "crushing chest pain",
                "shortness of breath", "difficulty breathing", "heart palpitations",
                "fainting", "loss of consciousness", "severe dizziness"
            ],
            "neurological": [
                "severe headache", "sudden confusion", "slurred speech",
                "weakness on one side", "facial drooping", "seizure",
                "loss of vision", "severe neck stiffness"
            ],
            "respiratory": [
                "severe difficulty breathing", "cannot speak in full sentences",
                "blue lips", "blue fingernails", "wheezing severely"
            ],
            "gastrointestinal": [
                "severe abdominal pain", "vomiting blood", "blood in stool",
                "severe dehydration", "unable to keep fluids down"
            ],
            "trauma": [
                "severe bleeding", "deep cuts", "broken bones",
                "head injury", "loss of consciousness", "severe burns"
            ]
        }

    def _load_symptom_mappings(self) -> Dict[str, Any]:
        """Load symptom to condition mappings"""
        return {
            "chest_pain": {
                "conditions": ["heart_attack", "angina", "pulmonary_embolism", "pneumonia"],
                "urgency": "emergency",
                "questions": [
                    "Is the pain crushing or pressure-like?",
                    "Does it radiate to arm, jaw, or back?",
                    "Are you short of breath?",
                    "Do you feel nauseous?"
                ]
            },
            "headache": {
                "conditions": ["migraine", "tension_headache", "cluster_headache", "stroke"],
                "urgency": "varies",
                "questions": [
                    "Is this the worst headache of your life?",
                    "Do you have fever?",
                    "Any vision changes?",
                    "Neck stiffness?"
                ]
            },
            "fever": {
                "conditions": ["infection", "flu", "covid19", "sepsis"],
                "urgency": "varies",
                "questions": [
                    "What is your temperature?",
                    "How long have you had fever?",
                    "Any other symptoms?",
                    "Recent travel or exposures?"
                ]
            }
        }

    async def triage_symptoms(
            self,
            symptoms: List[str],
            patient_data: Optional[Dict[str, Any]] = None,
            additional_info: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive symptom triage
        """
        try:
            # Emergency detection
            emergency_assessment = self._detect_emergency_symptoms(symptoms)

            # Symptom analysis
            symptom_analysis = await self._analyze_symptoms(symptoms, patient_data)

            # Risk stratification
            risk_assessment = self._assess_risk_level(symptoms, symptom_analysis, patient_data)

            # Generate triage recommendation
            triage_recommendation = self._generate_triage_recommendation(
                emergency_assessment, risk_assessment, symptom_analysis
            )

            # Additional questions
            follow_up_questions = self._generate_follow_up_questions(symptoms, symptom_analysis)

            # Care instructions
            care_instructions = self._generate_care_instructions(triage_recommendation, symptoms)

            return {
                "triage_id": f"triage_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "symptoms_reported": symptoms,
                "emergency_assessment": emergency_assessment,
                "symptom_analysis": symptom_analysis,
                "risk_assessment": risk_assessment,
                "triage_recommendation": triage_recommendation,
                "follow_up_questions": follow_up_questions,
                "care_instructions": care_instructions,
                "timestamp": datetime.utcnow().isoformat(),
                "requires_immediate_attention": emergency_assessment["is_emergency"]
            }

        except Exception as e:
            logger.error(f"Symptom triage error: {e}")
            return {"error": str(e), "symptoms": symptoms}

    def _detect_emergency_symptoms(self, symptoms: List[str]) -> Dict[str, Any]:
        """Detect emergency symptoms"""
        emergency_detected = False
        emergency_categories = []
        emergency_symptoms = []

        symptoms_text = " ".join(symptoms).lower()

        for category, indicators in self.emergency_indicators.items():
            for indicator in indicators:
                if indicator.lower() in symptoms_text:
                    emergency_detected = True
                    emergency_categories.append(category)
                    emergency_symptoms.append(indicator)

        return {
            "is_emergency": emergency_detected,
            "emergency_categories": list(set(emergency_categories)),
            "emergency_symptoms": list(set(emergency_symptoms)),
            "confidence": 0.9 if emergency_detected else 0.1,
            "action_required": "immediate_911" if emergency_detected else "continue_assessment"
        }

    async def _analyze_symptoms(
            self,
            symptoms: List[str],
            patient_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze symptoms using AI"""

        if not self.openai_client:
            return self._analyze_symptoms_rules(symptoms)

        try:
            # Prepare context
            context = ""
            if patient_data:
                age = patient_data.get("age", "unknown")
                gender = patient_data.get("gender", "unknown")
                conditions = patient_data.get("conditions", [])
                context = f"Patient: {age} year old {gender}, Medical history: {', '.join(conditions)}"

            prompt = f"""
            Analyze these symptoms for medical triage:
            
            Symptoms: {', '.join(symptoms)}
            {context}
            
            Provide analysis in JSON format:
            {{
                "primary_symptoms": ["list of main symptoms"],
                "associated_symptoms": ["list of related symptoms"],
                "possible_conditions": [
                    {{"condition": "name", "probability": 0.8, "urgency": "high"}}
                ],
                "red_flags": ["list of concerning symptoms"],
                "system_involvement": ["cardiovascular", "respiratory", etc.],
                "severity_assessment": "mild/moderate/severe",
                "confidence": 0.8
            }}
            """

            response = await self.openai_client.chat.completions.create(
                model=settings.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a medical triage expert. Analyze symptoms for urgency and possible conditions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"AI symptom analysis error: {e}")
            return self._analyze_symptoms_rules(symptoms)

    def _analyze_symptoms_rules(self, symptoms: List[str]) -> Dict[str, Any]:
        """Rule-based symptom analysis"""
        symptoms_text = " ".join(symptoms).lower()

        # Basic analysis
        analysis = {
            "primary_symptoms": symptoms[:3],  # First 3 as primary
            "associated_symptoms": symptoms[3:] if len(symptoms) > 3 else [],
            "possible_conditions": [],
            "red_flags": [],
            "system_involvement": [],
            "severity_assessment": "moderate",
            "confidence": 0.6
        }

        # Check for red flags
        red_flag_keywords = ["severe", "sudden", "worst", "crushing", "cannot", "unable"]
        for keyword in red_flag_keywords:
            if keyword in symptoms_text:
                analysis["red_flags"].append(f"Contains '{keyword}' - concerning")

        # Determine system involvement
        if any(word in symptoms_text for word in ["chest", "heart", "pressure"]):
            analysis["system_involvement"].append("cardiovascular")
        if any(word in symptoms_text for word in ["breath", "cough", "wheeze"]):
            analysis["system_involvement"].append("respiratory")
        if any(word in symptoms_text for word in ["head", "dizzy", "confusion"]):
            analysis["system_involvement"].append("neurological")

        return analysis

    def _assess_risk_level(
            self,
            symptoms: List[str],
            symptom_analysis: Dict[str, Any],
            patient_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess overall risk level"""

        risk_score = 0.0
        risk_factors = []

        # Symptom-based risk
        severity = symptom_analysis.get("severity_assessment", "moderate")
        if severity == "severe":
            risk_score += 0.4
            risk_factors.append("Severe symptoms reported")
        elif severity == "moderate":
            risk_score += 0.2

        # Red flags
        red_flags = symptom_analysis.get("red_flags", [])
        if red_flags:
            risk_score += len(red_flags) * 0.1
            risk_factors.append(f"{len(red_flags)} red flag symptoms")

        # Patient factors
        if patient_data:
            age = patient_data.get("age", 0)
            if age > 65:
                risk_score += 0.1
                risk_factors.append("Advanced age")

            conditions = patient_data.get("conditions", [])
            high_risk_conditions = ["diabetes", "heart disease", "copd", "cancer"]
            for condition in conditions:
                if any(hrc in condition.lower() for hrc in high_risk_conditions):
                    risk_score += 0.1
                    risk_factors.append(f"High-risk condition: {condition}")

        # System involvement
        systems = symptom_analysis.get("system_involvement", [])
        critical_systems = ["cardiovascular", "neurological", "respiratory"]
        for system in systems:
            if system in critical_systems:
                risk_score += 0.1
                risk_factors.append(f"Critical system involvement: {system}")

        # Categorize risk
        if risk_score >= 0.7:
            risk_level = "Critical"
        elif risk_score >= 0.5:
            risk_level = "High"
        elif risk_score >= 0.3:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        return {
            "risk_score": min(1.0, risk_score),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "confidence": 0.8
        }

    def _generate_triage_recommendation(
            self,
            emergency_assessment: Dict[str, Any],
            risk_assessment: Dict[str, Any],
            symptom_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate triage recommendation"""

        # Emergency override
        if emergency_assessment["is_emergency"]:
            protocol = self.triage_protocols["emergency_protocol"]
            return {
                "urgency": "Emergency",
                "protocol": protocol,
                "recommendation": "Call 911 immediately",
                "reasoning": f"Emergency symptoms detected: {', '.join(emergency_assessment['emergency_symptoms'])}",
                "time_frame": "Immediate"
            }

        # Risk-based triage
        risk_level = risk_assessment["risk_level"]

        if risk_level == "Critical":
            protocol = self.triage_protocols["urgent_protocol"]
            recommendation = "Go to emergency department immediately"
            time_frame = "Within 1 hour"
        elif risk_level == "High":
            protocol = self.triage_protocols["semi_urgent_protocol"]
            recommendation = "Seek medical attention today"
            time_frame = "Within 24 hours"
        elif risk_level == "Moderate":
            protocol = self.triage_protocols["routine_protocol"]
            recommendation = "Schedule appointment with primary care"
            time_frame = "Within 1 week"
        else:
            protocol = self.triage_protocols["routine_protocol"]
            recommendation = "Monitor symptoms and schedule routine care if needed"
            time_frame = "As needed"

        return {
            "urgency": risk_level,
            "protocol": protocol,
            "recommendation": recommendation,
            "reasoning": f"Risk assessment based on: {', '.join(risk_assessment['risk_factors'])}",
            "time_frame": time_frame
        }

    def _generate_follow_up_questions(
            self,
            symptoms: List[str],
            symptom_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate follow-up questions"""

        questions = []
        symptoms_text = " ".join(symptoms).lower()

        # Symptom-specific questions
        for symptom_key, mapping in self.symptom_mappings.items():
            if symptom_key.replace("_", " ") in symptoms_text:
                questions.extend(mapping["questions"])

        # General assessment questions
        general_questions = [
            "When did these symptoms start?",
            "Have the symptoms gotten worse?",
            "What makes the symptoms better or worse?",
            "Have you taken any medications for this?",
            "Do you have any other symptoms not mentioned?"
        ]

        # Add general questions if no specific ones found
        if not questions:
            questions = general_questions[:3]

        # Remove duplicates and limit
        return list(set(questions))[:5]

    def _generate_care_instructions(
            self,
            triage_recommendation: Dict[str, Any],
            symptoms: List[str]
    ) -> Dict[str, Any]:
        """Generate care instructions"""

        urgency = triage_recommendation["urgency"]

        if urgency == "Emergency":
            return {
                "immediate_actions": [
                    "Call 911 immediately",
                    "Do not drive yourself",
                    "Stay calm and follow dispatcher instructions"
                ],
                "what_to_bring": [
                    "List of current medications",
                    "Insurance cards",
                    "Emergency contact information"
                ],
                "warning_signs": [
                    "If symptoms worsen, call 911 again",
                    "If you lose consciousness, have someone call 911"
                ]
            }

        elif urgency in ["Critical", "High"]:
            return {
                "immediate_actions": [
                    "Go to emergency department or urgent care",
                    "Have someone drive you if possible",
                    "Bring medication list and insurance cards"
                ],
                "monitoring": [
                    "Monitor symptoms closely",
                    "Note any changes or worsening",
                    "Keep track of when symptoms occur"
                ],
                "when_to_call_911": [
                    "If symptoms suddenly worsen",
                    "If you develop new concerning symptoms",
                    "If you feel unsafe or very unwell"
                ]
            }

        else:
            return {
                "self_care": [
                    "Rest and monitor symptoms",
                    "Stay hydrated",
                    "Take over-the-counter medications as appropriate"
                ],
                "when_to_seek_care": [
                    "If symptoms worsen or persist",
                    "If new symptoms develop",
                    "If you become concerned about your condition"
                ],
                "follow_up": [
                    "Schedule appointment with primary care provider",
                    "Keep symptom diary",
                    "Follow up as recommended"
                ]
            }
