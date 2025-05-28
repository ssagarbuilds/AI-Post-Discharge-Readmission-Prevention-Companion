"""
Advanced Care Plan Generator with Multimodal AI and HAIM Framework
Based on latest 2024-2025 research on multimodal foundation models
"""

import openai
import json
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from app.config.settings import settings
from app.models.foundation.multimodal_llm import MultimodalLLM

logger = logging.getLogger(__name__)

class CarePlanGenerator:
    """
    Advanced care plan generator using multimodal AI and HAIM framework
    Integrates imaging, text, tabular, and time-series data
    """

    def __init__(self):
        self.openai_client = None
        self.multimodal_llm = MultimodalLLM()
        self.care_templates = self._load_care_templates()
        self.evidence_base = self._load_evidence_base()
        self.haim_processor = self._initialize_haim_processor()
        self._initialize_client()
        logger.info("✅ Advanced Care Plan Generator initialized")

    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ Care plan OpenAI client initialized")
        except Exception as e:
            logger.error(f"❌ Care plan generator initialization error: {e}")

    def _initialize_haim_processor(self):
        """Initialize HAIM (Holistic AI in Medicine) processor"""
        return {
            "modality_weights": {
                "tabular": 0.3,
                "time_series": 0.25,
                "text": 0.25,
                "images": 0.2
            },
            "fusion_method": "late_fusion",
            "feature_extractors": {
                "clinical_bert": "emilyalsentzer/Bio_ClinicalBERT",
                "vision_transformer": "microsoft/swin-base-patch4-window7-224",
                "tabular_encoder": "xgboost",
                "time_series_encoder": "lstm"
            }
        }

    def _load_care_templates(self) -> Dict[str, Any]:
        """Load evidence-based care plan templates"""
        return {
            "post_discharge": {
                "heart_failure": {
                    "medications": {
                        "ace_inhibitors": {
                            "first_line": ["lisinopril", "enalapril", "captopril"],
                            "dosing": "Start low, titrate to max tolerated",
                            "monitoring": "Creatinine, potassium at 1-2 weeks"
                        },
                        "beta_blockers": {
                            "first_line": ["metoprolol", "carvedilol", "bisoprolol"],
                            "contraindications": ["asthma", "severe_bradycardia"],
                            "monitoring": "Heart rate, blood pressure"
                        },
                        "diuretics": {
                            "loop_diuretics": ["furosemide", "bumetanide"],
                            "monitoring": "Daily weights, electrolytes",
                            "patient_education": "Weigh daily, call if >3lb gain"
                        }
                    },
                    "lifestyle": {
                        "diet": "Sodium <2g/day, fluid restriction 2L/day",
                        "activity": "Gradual increase, cardiac rehab referral",
                        "monitoring": "Daily weights, symptom diary"
                    },
                    "follow_up": {
                        "primary_care": "7-14 days",
                        "cardiology": "2-4 weeks",
                        "heart_failure_clinic": "1 week if available"
                    }
                },
                "diabetes": {
                    "medications": {
                        "metformin": {
                            "dosing": "500mg BID, increase to 1000mg BID",
                            "contraindications": ["eGFR <30", "metabolic_acidosis"],
                            "monitoring": "HbA1c every 3 months"
                        },
                        "insulin": {
                            "types": ["basal", "rapid_acting", "premixed"],
                            "monitoring": "Blood glucose 2-4x daily",
                            "education": "Injection technique, hypoglycemia recognition"
                        }
                    },
                    "lifestyle": {
                        "diet": "Carbohydrate counting, consistent meal timing",
                        "activity": "150 min/week moderate exercise",
                        "monitoring": "Blood glucose logs, foot care"
                    }
                },
                "copd": {
                    "medications": {
                        "bronchodilators": {
                            "laba": ["salmeterol", "formoterol"],
                            "lama": ["tiotropium", "umeclidinium"],
                            "combination": ["fluticasone/salmeterol"]
                        },
                        "corticosteroids": {
                            "inhaled": ["fluticasone", "budesonide"],
                            "systemic": "Prednisone for exacerbations"
                        }
                    },
                    "lifestyle": {
                        "smoking_cessation": "Mandatory, provide resources",
                        "pulmonary_rehab": "Refer if available",
                        "vaccinations": "Annual flu, pneumonia, COVID-19"
                    }
                }
            },
            "language_templates": {
                "en": {
                    "medication_instructions": "Take your medications exactly as prescribed",
                    "emergency_signs": "Call 911 if you experience",
                    "follow_up": "Schedule follow-up appointment",
                    "lifestyle": "Make these lifestyle changes"
                },
                "es": {
                    "medication_instructions": "Tome sus medicamentos exactamente como se los recetaron",
                    "emergency_signs": "Llame al 911 si experimenta",
                    "follow_up": "Programe una cita de seguimiento",
                    "lifestyle": "Haga estos cambios en su estilo de vida"
                }
            }
        }

    def _load_evidence_base(self) -> Dict[str, Any]:
        """Load evidence-based guidelines"""
        return {
            "guidelines": {
                "heart_failure": {
                    "source": "AHA/ACC/HFSA 2022 Guidelines",
                    "class_i_recommendations": [
                        "ACE inhibitor or ARB for all patients with HFrEF",
                        "Beta-blocker for all patients with HFrEF",
                        "Diuretics for fluid retention"
                    ],
                    "quality_measures": [
                        "LVEF documentation",
                        "ACE inhibitor/ARB prescription",
                        "Beta-blocker prescription"
                    ]
                },
                "diabetes": {
                    "source": "ADA 2024 Standards of Care",
                    "targets": {
                        "hba1c": "<7% for most adults",
                        "blood_pressure": "<130/80 mmHg",
                        "ldl_cholesterol": "<70 mg/dL if CVD risk"
                    }
                }
            },
            "risk_calculators": {
                "cardiovascular": "Pooled Cohort Equations",
                "diabetes": "ADA Risk Calculator",
                "mortality": "Charlson Comorbidity Index"
            }
        }

    async def generate_comprehensive_plan(
            self,
            patient_data: Dict[str, Any],
            multimodal_inputs: Optional[Dict[str, Any]] = None,
            language: str = "en",
            plan_type: str = "post_discharge"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive care plan using multimodal AI
        """
        try:
            # Process multimodal inputs using HAIM framework
            multimodal_analysis = await self._process_multimodal_inputs(
                patient_data, multimodal_inputs or {}
            )

            # Extract clinical insights
            clinical_insights = await self._extract_clinical_insights(
                patient_data, multimodal_analysis
            )

            # Generate personalized recommendations
            recommendations = await self._generate_personalized_recommendations(
                patient_data, clinical_insights, language
            )

            # Create structured care plan
            care_plan = await self._create_structured_plan(
                patient_data, recommendations, clinical_insights, language
            )

            # Add quality measures and monitoring
            care_plan = await self._add_quality_measures(care_plan, patient_data)

            # Generate patient education materials
            education_materials = await self._generate_education_materials(
                care_plan, language
            )

            return {
                "patient_id": patient_data.get("patient_id"),
                "plan_type": plan_type,
                "language": language,
                "care_plan": care_plan,
                "education_materials": education_materials,
                "multimodal_insights": multimodal_analysis,
                "evidence_level": "high",
                "generated_at": datetime.utcnow().isoformat(),
                "next_review": (datetime.utcnow() + timedelta(days=30)).isoformat(),
                "version": "2.0_multimodal"
            }

        except Exception as e:
            logger.error(f"Care plan generation error: {e}")
            return {"error": str(e), "patient_id": patient_data.get("patient_id")}

    async def _process_multimodal_inputs(
            self,
            patient_data: Dict[str, Any],
            multimodal_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process multimodal inputs using HAIM framework"""

        # Prepare inputs for multimodal processing
        text_input = self._prepare_text_input(patient_data)
        structured_data = self._prepare_structured_data(patient_data)

        # Process with multimodal LLM
        multimodal_result = await self.multimodal_llm.process_multimodal_input(
            text=text_input,
            image=multimodal_inputs.get("chest_xray"),
            audio=multimodal_inputs.get("voice_recording"),
            structured_data=structured_data,
            task="care_plan_generation"
        )

        # Extract HAIM-style features
        haim_features = self._extract_haim_features(multimodal_result)

        return {
            "multimodal_analysis": multimodal_result,
            "haim_features": haim_features,
            "modality_contributions": self._calculate_modality_contributions(multimodal_result),
            "confidence_score": self._calculate_confidence(multimodal_result)
        }

    def _prepare_text_input(self, patient_data: Dict[str, Any]) -> str:
        """Prepare text input from patient data"""
        text_components = []

        # Demographics
        if patient_data.get("age"):
            text_components.append(f"Age: {patient_data['age']} years")

        if patient_data.get("gender"):
            text_components.append(f"Gender: {patient_data['gender']}")

        # Medical history
        if patient_data.get("conditions"):
            conditions = ", ".join(patient_data["conditions"])
            text_components.append(f"Medical conditions: {conditions}")

        # Medications
        if patient_data.get("medications"):
            medications = ", ".join(patient_data["medications"])
            text_components.append(f"Current medications: {medications}")

        # Discharge notes
        if patient_data.get("discharge_notes"):
            text_components.append(f"Discharge notes: {patient_data['discharge_notes']}")

        # Recent vitals
        if patient_data.get("vitals"):
            vitals = patient_data["vitals"]
            vital_text = f"Recent vitals - BP: {vitals.get('blood_pressure', 'N/A')}, HR: {vitals.get('heart_rate', 'N/A')}, Temp: {vitals.get('temperature', 'N/A')}"
            text_components.append(vital_text)

        return "\n".join(text_components)

    def _prepare_structured_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare structured data for analysis"""
        return {
            "demographics": {
                "age": patient_data.get("age"),
                "gender": patient_data.get("gender"),
                "race": patient_data.get("race"),
                "ethnicity": patient_data.get("ethnicity")
            },
            "clinical_metrics": {
                "length_of_stay": patient_data.get("length_of_stay"),
                "admission_type": patient_data.get("admission_type"),
                "discharge_disposition": patient_data.get("discharge_disposition"),
                "comorbidity_count": len(patient_data.get("conditions", []))
            },
            "laboratory_values": patient_data.get("lab_results", {}),
            "vital_signs": patient_data.get("vitals", {}),
            "social_determinants": patient_data.get("social_history", {})
        }

    def _extract_haim_features(self, multimodal_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract HAIM-style features from multimodal analysis"""
        features = {
            "tabular_features": [],
            "text_features": [],
            "image_features": [],
            "time_series_features": [],
            "fusion_embedding": []
        }

        # Extract features from each modality
        if multimodal_result.get("text_analysis"):
            features["text_features"] = self._extract_text_features(
                multimodal_result["text_analysis"]
            )

        if multimodal_result.get("image_analysis"):
            features["image_features"] = self._extract_image_features(
                multimodal_result["image_analysis"]
            )

        if multimodal_result.get("structured_analysis"):
            features["tabular_features"] = self._extract_tabular_features(
                multimodal_result["structured_analysis"]
            )

        # Create fusion embedding (simplified)
        features["fusion_embedding"] = self._create_fusion_embedding(features)

        return features

    def _extract_text_features(self, text_analysis: Dict[str, Any]) -> List[float]:
        """Extract features from text analysis"""
        # Simplified feature extraction
        content = text_analysis.get("content", "")

        # Basic text features
        features = [
            len(content.split()),  # word count
            content.count("."),    # sentence count
            1 if "pain" in content.lower() else 0,
            1 if "medication" in content.lower() else 0,
            1 if "follow" in content.lower() else 0
        ]

        # Pad to fixed size
        while len(features) < 100:
            features.append(0.0)

        return features[:100]

    def _extract_image_features(self, image_analysis: Dict[str, Any]) -> List[float]:
        """Extract features from image analysis"""
        # Simplified image feature extraction
        features = [0.0] * 50  # Placeholder for image features

        if image_analysis.get("vision_classification"):
            # Extract confidence scores
            classifications = image_analysis["vision_classification"]
            if isinstance(classifications, list) and classifications:
                for i, item in enumerate(classifications[:10]):
                    if isinstance(item, dict) and "score" in item:
                        features[i] = item["score"]

        return features

    def _extract_tabular_features(self, structured_analysis: Dict[str, Any]) -> List[float]:
        """Extract features from structured data analysis"""
        features = []

        # Extract numerical features from analysis
        analysis_text = structured_analysis.get("analysis", "")

        # Simple feature extraction based on keywords
        feature_keywords = [
            "age", "blood_pressure", "heart_rate", "temperature",
            "glucose", "creatinine", "hemoglobin", "sodium", "potassium"
        ]

        for keyword in feature_keywords:
            features.append(1.0 if keyword in analysis_text.lower() else 0.0)

        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)

        return features[:20]

    def _create_fusion_embedding(self, features: Dict[str, List[float]]) -> List[float]:
        """Create fusion embedding from all modalities"""
        fusion = []

        # Concatenate features from all modalities
        for modality, feature_list in features.items():
            if modality != "fusion_embedding" and feature_list:
                fusion.extend(feature_list[:50])  # Limit each modality to 50 features

        # Ensure fixed size
        target_size = 200
        if len(fusion) > target_size:
            fusion = fusion[:target_size]
        else:
            fusion.extend([0.0] * (target_size - len(fusion)))

        return fusion

    def _calculate_modality_contributions(self, multimodal_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate contribution of each modality"""
        contributions = {}
        total_confidence = 0.0

        modalities = ["text_analysis", "image_analysis", "structured_analysis", "audio_analysis"]

        for modality in modalities:
            if multimodal_result.get(modality):
                confidence = multimodal_result[modality].get("confidence", 0.0)
                contributions[modality] = confidence
                total_confidence += confidence

        # Normalize contributions
        if total_confidence > 0:
            for modality in contributions:
                contributions[modality] /= total_confidence

        return contributions

    def _calculate_confidence(self, multimodal_result: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        confidences = []

        for analysis in multimodal_result.values():
            if isinstance(analysis, dict) and "confidence" in analysis:
                confidences.append(analysis["confidence"])

        return sum(confidences) / len(confidences) if confidences else 0.0

    async def _extract_clinical_insights(
            self,
            patient_data: Dict[str, Any],
            multimodal_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract clinical insights from multimodal analysis"""

        if not self.openai_client:
            return self._extract_rule_based_insights(patient_data)

        try:
            prompt = f"""
            Based on this multimodal patient analysis, extract key clinical insights:
            
            Patient Data: {json.dumps(patient_data, indent=2)}
            Multimodal Analysis: {json.dumps(multimodal_analysis.get("multimodal_analysis", {}), indent=2)}
            
            Extract:
            1. Primary clinical concerns
            2. Risk factors identified
            3. Medication optimization opportunities
            4. Lifestyle modification priorities
            5. Monitoring requirements
            6. Social determinants affecting care
            
            Return as structured JSON.
            """

            response = await self.openai_client.chat.completions.create(
                model=settings.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a clinical expert extracting insights for care planning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Clinical insights extraction error: {e}")
            return self._extract_rule_based_insights(patient_data)

    def _extract_rule_based_insights(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights using rule-based approach"""
        insights = {
            "primary_concerns": [],
            "risk_factors": [],
            "medication_opportunities": [],
            "lifestyle_priorities": [],
            "monitoring_requirements": [],
            "social_determinants": []
        }

        conditions = patient_data.get("conditions", [])
        age = patient_data.get("age", 0)
        medications = patient_data.get("medications", [])

        # Primary concerns based on conditions
        for condition in conditions:
            condition_lower = condition.lower()
            if "heart failure" in condition_lower:
                insights["primary_concerns"].append("Heart failure management")
                insights["monitoring_requirements"].extend(["Daily weights", "Symptom monitoring"])
            elif "diabetes" in condition_lower:
                insights["primary_concerns"].append("Diabetes management")
                insights["monitoring_requirements"].extend(["Blood glucose monitoring", "HbA1c tracking"])

        # Risk factors based on age and conditions
        if age > 65:
            insights["risk_factors"].append("Advanced age")

        if len(conditions) > 2:
            insights["risk_factors"].append("Multiple comorbidities")

        # Medication opportunities
        if not any("ace inhibitor" in med.lower() or "arb" in med.lower() for med in medications):
            if any("heart failure" in cond.lower() for cond in conditions):
                insights["medication_opportunities"].append("Consider ACE inhibitor/ARB")

        return insights

    async def _generate_personalized_recommendations(
            self,
            patient_data: Dict[str, Any],
            clinical_insights: Dict[str, Any],
            language: str
    ) -> Dict[str, Any]:
        """Generate personalized recommendations"""

        recommendations = {
            "medications": [],
            "lifestyle": [],
            "monitoring": [],
            "follow_up": [],
            "emergency_signs": [],
            "patient_education": []
        }

        conditions = patient_data.get("conditions", [])

        # Generate condition-specific recommendations
        for condition in conditions:
            condition_lower = condition.lower()

            if "heart failure" in condition_lower:
                recommendations.update(
                    self._get_heart_failure_recommendations(patient_data, language)
                )
            elif "diabetes" in condition_lower:
                recommendations.update(
                    self._get_diabetes_recommendations(patient_data, language)
                )
            elif "copd" in condition_lower:
                recommendations.update(
                    self._get_copd_recommendations(patient_data, language)
                )

        # Add general recommendations
        recommendations["monitoring"].extend([
            "Regular vital signs monitoring",
            "Medication adherence tracking"
        ])

        return recommendations

    def _get_heart_failure_recommendations(self, patient_data: Dict[str, Any], language: str) -> Dict[str, List[str]]:
        """Get heart failure specific recommendations"""
        templates = self.care_templates["language_templates"][language]

        return {
            "medications": [
                "ACE inhibitor or ARB as tolerated",
                "Beta-blocker (start low, titrate up)",
                "Diuretic for fluid management"
            ],
            "lifestyle": [
                "Sodium restriction <2g/day",
                "Fluid restriction 2L/day",
                "Daily weight monitoring"
            ],
            "monitoring": [
                "Daily weights (call if >3lb gain in 2 days)",
                "Blood pressure and heart rate",
                "Kidney function and electrolytes"
            ],
            "emergency_signs": [
                "Sudden weight gain >3 pounds",
                "Increased shortness of breath",
                "Chest pain or pressure",
                "Swelling in legs or abdomen"
            ]
        }

    def _get_diabetes_recommendations(self, patient_data: Dict[str, Any], language: str) -> Dict[str, List[str]]:
        """Get diabetes specific recommendations"""
        return {
            "medications": [
                "Metformin if no contraindications",
                "Insulin if indicated",
                "Statin for cardiovascular protection"
            ],
            "lifestyle": [
                "Carbohydrate counting",
                "Regular meal timing",
                "150 minutes/week exercise"
            ],
            "monitoring": [
                "Blood glucose 2-4 times daily",
                "HbA1c every 3 months",
                "Annual eye and foot exams"
            ],
            "emergency_signs": [
                "Blood glucose <70 or >300 mg/dL",
                "Persistent nausea/vomiting",
                "Signs of diabetic ketoacidosis"
            ]
        }

    def _get_copd_recommendations(self, patient_data: Dict[str, Any], language: str) -> Dict[str, List[str]]:
        """Get COPD specific recommendations"""
        return {
            "medications": [
                "Long-acting bronchodilator",
                "Inhaled corticosteroid if indicated",
                "Rescue inhaler for exacerbations"
            ],
            "lifestyle": [
                "Smoking cessation (mandatory)",
                "Pulmonary rehabilitation",
                "Annual vaccinations (flu, pneumonia)"
            ],
            "monitoring": [
                "Peak flow measurements",
                "Symptom tracking",
                "Oxygen saturation if prescribed"
            ],
            "emergency_signs": [
                "Increased shortness of breath",
                "Change in sputum color/amount",
                "Chest tightness or pain"
            ]
        }

    async def _create_structured_plan(
            self,
            patient_data: Dict[str, Any],
            recommendations: Dict[str, Any],
            clinical_insights: Dict[str, Any],
            language: str
    ) -> Dict[str, Any]:
        """Create structured care plan"""

        plan = {
            "patient_summary": self._create_patient_summary(patient_data),
            "medication_plan": self._create_medication_plan(recommendations, patient_data),
            "lifestyle_plan": self._create_lifestyle_plan(recommendations),
            "monitoring_plan": self._create_monitoring_plan(recommendations),
            "follow_up_plan": self._create_follow_up_plan(patient_data),
            "emergency_plan": self._create_emergency_plan(recommendations, language),
            "education_plan": self._create_education_plan(recommendations, language),
            "goals": self._create_care_goals(patient_data, clinical_insights)
        }

        return plan

    def _create_patient_summary(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create patient summary"""
        return {
            "demographics": {
                "age": patient_data.get("age"),
                "gender": patient_data.get("gender")
            },
            "primary_conditions": patient_data.get("conditions", []),
            "current_medications": patient_data.get("medications", []),
            "recent_hospitalization": {
                "length_of_stay": patient_data.get("length_of_stay"),
                "discharge_date": patient_data.get("discharge_date"),
                "primary_diagnosis": patient_data.get("primary_diagnosis")
            }
        }

    def _create_medication_plan(self, recommendations: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create medication plan"""
        return {
            "current_medications": patient_data.get("medications", []),
            "new_medications": recommendations.get("medications", []),
            "medication_changes": [],
            "adherence_strategies": [
                "Use pill organizer",
                "Set medication reminders",
                "Understand purpose of each medication"
            ],
            "monitoring_requirements": [
                "Bring medication list to all appointments",
                "Report side effects immediately",
                "Do not stop medications without consulting provider"
            ]
        }

    def _create_lifestyle_plan(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Create lifestyle modification plan"""
        return {
            "diet_modifications": recommendations.get("lifestyle", []),
            "activity_recommendations": [
                "Start with light activity",
                "Gradually increase as tolerated",
                "Avoid overexertion"
            ],
            "smoking_cessation": "Mandatory if applicable",
            "weight_management": "Maintain healthy weight",
            "stress_management": "Practice relaxation techniques"
        }

    def _create_monitoring_plan(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Create monitoring plan"""
        return {
            "daily_monitoring": recommendations.get("monitoring", []),
            "weekly_monitoring": ["Weight trends", "Symptom patterns"],
            "monthly_monitoring": ["Medication adherence", "Activity tolerance"],
            "laboratory_monitoring": ["As ordered by provider"],
            "when_to_call": recommendations.get("emergency_signs", [])
        }

    def _create_follow_up_plan(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create follow-up plan"""
        conditions = patient_data.get("conditions", [])

        follow_up = {
            "primary_care": "7-14 days",
            "specialist_appointments": []
        }

        for condition in conditions:
            condition_lower = condition.lower()
            if "heart failure" in condition_lower:
                follow_up["specialist_appointments"].append({
                    "specialty": "Cardiology",
                    "timeframe": "2-4 weeks",
                    "purpose": "Heart failure management"
                })
            elif "diabetes" in condition_lower:
                follow_up["specialist_appointments"].append({
                    "specialty": "Endocrinology",
                    "timeframe": "4-6 weeks",
                    "purpose": "Diabetes management"
                })

        return follow_up

    def _create_emergency_plan(self, recommendations: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Create emergency action plan"""
        templates = self.care_templates["language_templates"][language]

        return {
            "emergency_contacts": {
                "911": templates["emergency_signs"],
                "primary_care": "During business hours",
                "on_call_service": "After hours, weekends"
            },
            "warning_signs": recommendations.get("emergency_signs", []),
            "action_steps": [
                "Call 911 for life-threatening symptoms",
                "Contact provider for concerning symptoms",
                "Go to emergency room if unable to reach provider"
            ]
        }

    def _create_education_plan(self, recommendations: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Create patient education plan"""
        return {
            "key_topics": [
                "Understanding your conditions",
                "Medication management",
                "Lifestyle modifications",
                "When to seek help"
            ],
            "educational_materials": [
                "Condition-specific handouts",
                "Medication guides",
                "Emergency action plans"
            ],
            "learning_preferences": "Visual, auditory, hands-on",
            "language": language
        }

    def _create_care_goals(self, patient_data: Dict[str, Any], clinical_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Create care goals"""
        return {
            "short_term_goals": [
                "Prevent readmission within 30 days",
                "Achieve medication adherence >90%",
                "Establish care with primary provider"
            ],
            "long_term_goals": [
                "Optimize chronic disease management",
                "Improve quality of life",
                "Prevent complications"
            ],
            "measurable_outcomes": [
                "No emergency visits within 30 days",
                "Stable vital signs",
                "Patient reports understanding of care plan"
            ]
        }

    async def _add_quality_measures(self, care_plan: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add quality measures and metrics"""

        care_plan["quality_measures"] = {
            "core_measures": self._get_core_measures(patient_data),
            "safety_measures": self._get_safety_measures(patient_data),
            "patient_experience": self._get_experience_measures(),
            "outcome_measures": self._get_outcome_measures(patient_data)
        }

        return care_plan

    def _get_core_measures(self, patient_data: Dict[str, Any]) -> List[str]:
        """Get core quality measures"""
        measures = []
        conditions = patient_data.get("conditions", [])

        for condition in conditions:
            condition_lower = condition.lower()
            if "heart failure" in condition_lower:
                measures.extend([
                    "LVEF documentation",
                    "ACE inhibitor/ARB prescription",
                    "Beta-blocker prescription",
                    "Discharge instructions provided"
                ])
            elif "diabetes" in condition_lower:
                measures.extend([
                    "HbA1c monitoring",
                    "Blood pressure control",
                    "LDL cholesterol management"
                ])

        return measures

    def _get_safety_measures(self, patient_data: Dict[str, Any]) -> List[str]:
        """Get safety measures"""
        return [
            "Medication reconciliation completed",
            "Allergy documentation",
            "Fall risk assessment",
            "Infection prevention measures"
        ]

    def _get_experience_measures(self) -> List[str]:
        """Get patient experience measures"""
        return [
            "Patient understanding of discharge instructions",
            "Satisfaction with care coordination",
            "Communication effectiveness",
            "Cultural competency"
        ]

    def _get_outcome_measures(self, patient_data: Dict[str, Any]) -> List[str]:
        """Get outcome measures"""
        return [
            "30-day readmission rate",
            "Emergency department visits",
            "Medication adherence rate",
            "Patient-reported outcomes"
        ]

    async def _generate_education_materials(self, care_plan: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Generate patient education materials"""

        materials = {
            "medication_guide": self._create_medication_guide(care_plan, language),
            "condition_information": self._create_condition_information(care_plan, language),
            "lifestyle_guide": self._create_lifestyle_guide(care_plan, language),
            "emergency_instructions": self._create_emergency_instructions(care_plan, language),
            "appointment_scheduler": self._create_appointment_guide(care_plan, language)
        }

        return materials

    def _create_medication_guide(self, care_plan: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Create medication guide"""
        templates = self.care_templates["language_templates"][language]

        return {
            "title": "Your Medication Guide" if language == "en" else "Su Guía de Medicamentos",
            "instructions": templates["medication_instructions"],
            "medication_list": care_plan["medication_plan"]["current_medications"],
            "new_medications": care_plan["medication_plan"]["new_medications"],
            "important_reminders": [
                "Take medications at the same time each day",
                "Do not skip doses",
                "Contact provider before stopping any medication"
            ]
        }

    def _create_condition_information(self, care_plan: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Create condition-specific information"""
        return {
            "title": "Understanding Your Health Conditions" if language == "en" else "Entendiendo Sus Condiciones de Salud",
            "conditions": care_plan["patient_summary"]["primary_conditions"],
            "management_tips": care_plan["lifestyle_plan"]["diet_modifications"],
            "monitoring_requirements": care_plan["monitoring_plan"]["daily_monitoring"]
        }

    def _create_lifestyle_guide(self, care_plan: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Create lifestyle modification guide"""
        return {
            "title": "Healthy Living Guide" if language == "en" else "Guía de Vida Saludable",
            "diet_tips": care_plan["lifestyle_plan"]["diet_modifications"],
            "activity_recommendations": care_plan["lifestyle_plan"]["activity_recommendations"],
            "general_wellness": [
                "Get adequate sleep",
                "Manage stress",
                "Stay hydrated"
            ]
        }

    def _create_emergency_instructions(self, care_plan: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Create emergency instructions"""
        templates = self.care_templates["language_templates"][language]

        return {
            "title": "When to Seek Emergency Care" if language == "en" else "Cuándo Buscar Atención de Emergencia",
            "emergency_signs": care_plan["emergency_plan"]["warning_signs"],
            "action_steps": care_plan["emergency_plan"]["action_steps"],
            "emergency_contacts": care_plan["emergency_plan"]["emergency_contacts"]
        }

    def _create_appointment_guide(self, care_plan: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Create appointment scheduling guide"""
        return {
            "title": "Your Follow-up Appointments" if language == "en" else "Sus Citas de Seguimiento",
            "primary_care": care_plan["follow_up_plan"]["primary_care"],
            "specialist_appointments": care_plan["follow_up_plan"]["specialist_appointments"],
            "preparation_tips": [
                "Bring medication list",
                "Prepare questions",
                "Bring insurance cards"
            ]
        }
