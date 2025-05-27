"""
Bilingual Healthcare Chatbot with Medical Reasoning
"""

import openai
import asyncio
from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime

from app.config.settings import settings

logger = logging.getLogger(__name__)

class HealthcareAssistant:
    """
    Advanced bilingual healthcare chatbot with medical reasoning
    """

    def __init__(self):
        self.openai_client = None
        self.conversation_history = {}
        self.medical_knowledge = self._load_medical_knowledge()
        self.language_models = self._initialize_language_models()
        self._initialize_client()
        logger.info("✅ Healthcare Assistant initialized")

    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ Healthcare chatbot OpenAI client initialized")
        except Exception as e:
            logger.error(f"❌ Healthcare chatbot initialization error: {e}")

    def _initialize_language_models(self) -> Dict[str, Any]:
        """Initialize language-specific models"""
        return {
            "constellation_models": {
                "emergency_triage": {
                    "model": "gpt-4o",
                    "temperature": 0.0,
                    "role": "Emergency medical triage specialist"
                },
                "medication_advisor": {
                    "model": "gpt-4o",
                    "temperature": 0.1,
                    "role": "Medication safety and adherence specialist"
                },
                "symptom_checker": {
                    "model": "gpt-4o",
                    "temperature": 0.2,
                    "role": "Symptom assessment and guidance specialist"
                },
                "chronic_care": {
                    "model": "gpt-4o",
                    "temperature": 0.1,
                    "role": "Chronic disease management specialist"
                },
                "mental_health": {
                    "model": "gpt-4o",
                    "temperature": 0.3,
                    "role": "Mental health support specialist"
                }
            },
            "supported_languages": {
                "en": "English",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "zh": "Chinese",
                "ja": "Japanese",
                "ko": "Korean",
                "hi": "Hindi",
                "ar": "Arabic",
                "ru": "Russian",
                "nl": "Dutch",
                "sv": "Swedish"
            }
        }

    def _load_medical_knowledge(self) -> Dict[str, Any]:
        """Load medical knowledge base"""
        return {
            "emergency_keywords": {
                "en": [
                    "chest pain", "difficulty breathing", "severe bleeding", "unconscious",
                    "heart attack", "stroke", "severe allergic reaction", "poisoning",
                    "severe burns", "broken bones", "head injury", "suicide"
                ],
                "es": [
                    "dolor de pecho", "dificultad para respirar", "sangrado severo", "inconsciente",
                    "ataque cardíaco", "derrame cerebral", "reacción alérgica severa", "envenenamiento",
                    "quemaduras graves", "huesos rotos", "lesión en la cabeza", "suicidio"
                ],
                "fr": [
                    "douleur thoracique", "difficulté à respirer", "saignement sévère", "inconscient",
                    "crise cardiaque", "accident vasculaire cérébral", "réaction allergique sévère"
                ]
            },
            "medication_interactions": {
                "high_risk": [
                    {"drug1": "warfarin", "drug2": "aspirin", "risk": "bleeding"},
                    {"drug1": "digoxin", "drug2": "furosemide", "risk": "toxicity"},
                    {"drug1": "lithium", "drug2": "lisinopril", "risk": "toxicity"}
                ],
                "moderate_risk": [
                    {"drug1": "metformin", "drug2": "contrast", "risk": "kidney_damage"},
                    {"drug1": "statins", "drug2": "grapefruit", "risk": "muscle_damage"}
                ]
            },
            "symptom_patterns": {
                "cardiac": {
                    "symptoms": ["chest pain", "shortness of breath", "arm pain", "jaw pain"],
                    "urgency": "high",
                    "action": "seek_immediate_care"
                },
                "respiratory": {
                    "symptoms": ["cough", "wheezing", "chest tightness", "fever"],
                    "urgency": "moderate",
                    "action": "monitor_and_follow_up"
                },
                "neurological": {
                    "symptoms": ["headache", "dizziness", "confusion", "weakness"],
                    "urgency": "variable",
                    "action": "assess_severity"
                }
            },
            "chronic_conditions": {
                "diabetes": {
                    "monitoring": ["blood_glucose", "hba1c", "foot_care"],
                    "medications": ["metformin", "insulin", "sglt2_inhibitors"],
                    "lifestyle": ["diet", "exercise", "weight_management"]
                },
                "hypertension": {
                    "monitoring": ["blood_pressure", "kidney_function"],
                    "medications": ["ace_inhibitors", "beta_blockers", "diuretics"],
                    "lifestyle": ["low_sodium_diet", "exercise", "stress_management"]
                },
                "heart_failure": {
                    "monitoring": ["daily_weights", "symptoms", "fluid_intake"],
                    "medications": ["ace_inhibitors", "beta_blockers", "diuretics"],
                    "lifestyle": ["fluid_restriction", "sodium_restriction", "activity_modification"]
                }
            }
        }

    async def chat(
            self,
            message: str,
            patient_id: Optional[str] = None,
            language: str = "en",
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main chat interface with medical reasoning
        """
        try:
            # Detect language if auto
            if language == "auto":
                language = self._detect_language(message)

            # Validate language support
            if language not in self.language_models["supported_languages"]:
                language = "en"

            # Check for emergency situations
            if self._is_emergency(message, language):
                return await self._handle_emergency(message, language, patient_id)

            # Determine appropriate specialist model
            specialist = self._select_specialist(message, context or {})

            # Get conversation history
            history = self._get_conversation_history(patient_id or "anonymous")

            # Generate response
            if self.openai_client:
                response = await self._generate_ai_response(
                    message, specialist, language, history, context or {}
                )
            else:
                response = self._generate_fallback_response(message, language)

            # Update conversation history
            self._update_conversation_history(patient_id or "anonymous", message, response)

            # Add safety checks
            response = self._add_safety_checks(response, message, language)

            return {
                "message": response["content"],
                "language": language,
                "specialist_used": specialist["role"],
                "confidence": response.get("confidence", 0.8),
                "suggestions": response.get("suggestions", []),
                "safety_flags": response.get("safety_flags", []),
                "follow_up_questions": response.get("follow_up_questions", []),
                "medical_disclaimer": self._get_medical_disclaimer(language),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Healthcare chat error: {e}")
            return self._error_response(str(e), language)

    def _detect_language(self, message: str) -> str:
        """Detect language from message content"""
        message_lower = message.lower()

        # Language indicators
        language_indicators = {
            "es": ["hola", "como", "que", "dolor", "medicina", "ayuda", "gracias", "donde", "cuando", "por favor"],
            "fr": ["bonjour", "comment", "que", "douleur", "médicament", "aide", "merci", "où", "quand"],
            "de": ["hallo", "wie", "was", "schmerz", "medikament", "hilfe", "danke", "wo", "wann"],
            "it": ["ciao", "come", "che", "dolore", "medicina", "aiuto", "grazie", "dove", "quando"],
            "pt": ["olá", "como", "que", "dor", "medicamento", "ajuda", "obrigado", "onde", "quando"],
            "zh": ["你好", "怎么", "什么", "疼痛", "药物", "帮助", "谢谢"],
            "ja": ["こんにちは", "どう", "何", "痛み", "薬", "助け", "ありがとう"],
            "ko": ["안녕", "어떻게", "무엇", "통증", "약물", "도움", "감사"],
            "hi": ["नमस्ते", "कैसे", "क्या", "दर्द", "दवा", "मदद", "धन्यवाद"],
            "ar": ["مرحبا", "كيف", "ماذا", "ألم", "دواء", "مساعدة", "شكرا"],
            "ru": ["привет", "как", "что", "боль", "лекарство", "помощь", "спасибо"],
            "nl": ["hallo", "hoe", "wat", "pijn", "medicijn", "hulp", "dank"],
            "sv": ["hej", "hur", "vad", "smärta", "medicin", "hjälp", "tack"]
        }

        # Count language indicators
        language_scores = {}
        for lang, indicators in language_indicators.items():
            score = sum(1 for indicator in indicators if indicator in message_lower)
            if score > 0:
                language_scores[lang] = score

        # Return language with highest score, default to English
        if language_scores:
            return max(language_scores, key=language_scores.get)

        return "en"

    def _is_emergency(self, message: str, language: str) -> bool:
        """Check if message indicates emergency"""
        message_lower = message.lower()
        emergency_keywords = self.medical_knowledge["emergency_keywords"].get(language, [])

        return any(keyword in message_lower for keyword in emergency_keywords)

    async def _handle_emergency(self, message: str, language: str, patient_id: Optional[str]) -> Dict[str, Any]:
        """Handle emergency situations"""
        emergency_responses = {
            "en": "🚨 EMERGENCY DETECTED: Call 911 immediately or go to the nearest emergency room. Do not delay seeking immediate medical attention.",
            "es": "🚨 EMERGENCIA DETECTADA: Llame al 911 inmediatamente o vaya a la sala de emergencias más cercana. No demore en buscar atención médica inmediata.",
            "fr": "🚨 URGENCE DÉTECTÉE: Appelez le 911 immédiatement ou rendez-vous aux urgences les plus proches. Ne tardez pas à chercher une attention médicale immédiate.",
            "de": "🚨 NOTFALL ERKANNT: Rufen Sie sofort 911 an oder gehen Sie zur nächsten Notaufnahme. Zögern Sie nicht, sofortige medizinische Hilfe zu suchen.",
            "it": "🚨 EMERGENZA RILEVATA: Chiama il 911 immediatamente o vai al pronto soccorso più vicino. Non ritardare nel cercare assistenza medica immediata."
        }

        response_text = emergency_responses.get(language, emergency_responses["en"])

        # Log emergency event
        logger.warning(f"Emergency detected for patient {patient_id}: {message}")

        return {
            "message": response_text,
            "language": language,
            "is_emergency": True,
            "priority": "critical",
            "action_required": "immediate_medical_attention",
            "emergency_contacts": {
                "emergency_services": "911",
                "poison_control": "1-800-222-1222",
                "suicide_prevention": "988"
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    def _select_specialist(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate specialist model"""
        message_lower = message.lower()

        # Emergency triage
        if any(word in message_lower for word in ["emergency", "urgent", "severe", "critical"]):
            return self.language_models["constellation_models"]["emergency_triage"]

        # Medication-related
        if any(word in message_lower for word in ["medication", "drug", "pill", "prescription", "side effect"]):
            return self.language_models["constellation_models"]["medication_advisor"]

        # Symptom checking
        if any(word in message_lower for word in ["symptom", "pain", "hurt", "feel", "sick"]):
            return self.language_models["constellation_models"]["symptom_checker"]

        # Chronic disease management
        if any(word in message_lower for word in ["diabetes", "hypertension", "heart failure", "chronic"]):
            return self.language_models["constellation_models"]["chronic_care"]

        # Mental health
        if any(word in message_lower for word in ["anxiety", "depression", "stress", "mental", "mood"]):
            return self.language_models["constellation_models"]["mental_health"]

        # Default to symptom checker
        return self.language_models["constellation_models"]["symptom_checker"]

    def _get_conversation_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for patient"""
        return self.conversation_history.get(patient_id, [])[-10:]  # Last 10 exchanges

    async def _generate_ai_response(
            self,
            message: str,
            specialist: Dict[str, Any],
            language: str,
            history: List[Dict[str, Any]],
            context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate AI response using specialist model"""

        # Build system prompt
        system_prompt = self._build_system_prompt(specialist, language, context)

        # Build conversation context
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for exchange in history[-5:]:  # Last 5 exchanges
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})

        # Add current message
        messages.append({"role": "user", "content": message})

        try:
            response = await self.openai_client.chat.completions.create(
                model=specialist["model"],
                messages=messages,
                temperature=specialist["temperature"],
                max_tokens=settings.MAX_TOKENS
            )

            content = response.choices[0].message.content

            # Generate follow-up questions
            follow_up_questions = self._generate_follow_up_questions(message, language, specialist)

            # Generate suggestions
            suggestions = self._generate_suggestions(message, language, specialist)

            return {
                "content": content,
                "confidence": 0.9,
                "follow_up_questions": follow_up_questions,
                "suggestions": suggestions,
                "model_used": specialist["model"]
            }

        except Exception as e:
            logger.error(f"AI response generation error: {e}")
            return self._generate_fallback_response(message, language)

    def _build_system_prompt(self, specialist: Dict[str, Any], language: str, context: Dict[str, Any]) -> str:
        """Build system prompt for specialist"""

        language_name = self.language_models["supported_languages"].get(language, "English")

        base_prompt = f"""
        You are a {specialist['role']} providing healthcare guidance in {language_name}.
        
        Core Principles:
        1. Patient safety is the highest priority
        2. Provide accurate, evidence-based information
        3. Never diagnose or prescribe medications
        4. Always recommend consulting healthcare providers for medical decisions
        5. Be empathetic, supportive, and culturally sensitive
        6. Use clear, simple language (6th grade reading level)
        7. Acknowledge uncertainty when appropriate
        
        Medical Guidelines:
        - Follow current clinical practice guidelines
        - Use evidence-based medicine principles
        - Maintain patient privacy and confidentiality
        - Provide culturally competent care
        
        Communication Style:
        - Be warm, empathetic, and professional
        - Use active listening techniques
        - Provide actionable information
        - Encourage patient empowerment
        - Respect patient autonomy
        """

        # Add specialist-specific instructions
        if "emergency" in specialist["role"].lower():
            base_prompt += "\n\nEMERGENCY PROTOCOL: If emergency symptoms are detected, immediately advise calling emergency services (911)."

        elif "medication" in specialist["role"].lower():
            base_prompt += "\n\nMEDICATION SAFETY: Always check for drug interactions, allergies, and contraindications. Never recommend specific dosages."

        elif "symptom" in specialist["role"].lower():
            base_prompt += "\n\nSYMPTOM ASSESSMENT: Provide guidance on symptom severity and when to seek care. Never provide diagnoses."

        elif "chronic" in specialist["role"].lower():
            base_prompt += "\n\nCHRONIC CARE: Focus on self-management, adherence, and lifestyle modifications. Emphasize regular monitoring."

        elif "mental" in specialist["role"].lower():
            base_prompt += "\n\nMENTAL HEALTH: Be especially empathetic. Screen for suicidal ideation. Provide crisis resources when appropriate."

        # Add patient context if available
        if context.get("patient_conditions"):
            base_prompt += f"\n\nPatient has these conditions: {', '.join(context['patient_conditions'])}"

        if context.get("current_medications"):
            base_prompt += f"\n\nCurrent medications: {', '.join(context['current_medications'])}"

        return base_prompt

    def _generate_follow_up_questions(self, message: str, language: str, specialist: Dict[str, Any]) -> List[str]:
        """Generate follow-up questions"""

        questions = {
            "en": {
                "emergency_triage": [
                    "Are you experiencing chest pain or difficulty breathing?",
                    "Have you lost consciousness or are you confused?",
                    "Is there severe bleeding that won't stop?"
                ],
                "medication_advisor": [
                    "Are you taking your medications as prescribed?",
                    "Have you experienced any side effects?",
                    "Do you have questions about drug interactions?"
                ],
                "symptom_checker": [
                    "How long have you been experiencing these symptoms?",
                    "On a scale of 1-10, how severe is your discomfort?",
                    "Have you tried any treatments or remedies?"
                ],
                "chronic_care": [
                    "How are you managing your daily monitoring?",
                    "Have you noticed any changes in your symptoms?",
                    "Are you following your care plan recommendations?"
                ],
                "mental_health": [
                    "How has your mood been lately?",
                    "Are you getting enough sleep and support?",
                    "Have you been able to engage in activities you enjoy?"
                ]
            },
            "es": {
                "emergency_triage": [
                    "¿Está experimentando dolor en el pecho o dificultad para respirar?",
                    "¿Ha perdido el conocimiento o está confundido?",
                    "¿Hay sangrado severo que no se detiene?"
                ],
                "medication_advisor": [
                    "¿Está tomando sus medicamentos según lo prescrito?",
                    "¿Ha experimentado algún efecto secundario?",
                    "¿Tiene preguntas sobre interacciones de medicamentos?"
                ],
                "symptom_checker": [
                    "¿Cuánto tiempo ha estado experimentando estos síntomas?",
                    "En una escala del 1 al 10, ¿qué tan severo es su malestar?",
                    "¿Ha probado algún tratamiento o remedio?"
                ]
            }
        }

        specialist_key = specialist["role"].split()[0].lower() + "_" + specialist["role"].split()[-1].lower()
        language_questions = questions.get(language, questions["en"])

        return language_questions.get(specialist_key, language_questions.get("symptom_checker", []))

    def _generate_suggestions(self, message: str, language: str, specialist: Dict[str, Any]) -> List[str]:
        """Generate helpful suggestions"""

        suggestions = {
            "en": [
                "Tell me more about your symptoms",
                "Ask about medication interactions",
                "Get information about your condition",
                "Learn about preventive care",
                "Find local healthcare resources"
            ],
            "es": [
                "Cuénteme más sobre sus síntomas",
                "Pregunte sobre interacciones de medicamentos",
                "Obtenga información sobre su condición",
                "Aprenda sobre cuidados preventivos",
                "Encuentre recursos de salud locales"
            ]
        }

        return suggestions.get(language, suggestions["en"])

    def _generate_fallback_response(self, message: str, language: str) -> Dict[str, Any]:
        """Generate fallback response when AI is unavailable"""

        fallback_responses = {
            "en": "I'm currently experiencing technical difficulties. For medical questions, please consult your healthcare provider or call your doctor's office.",
            "es": "Actualmente estoy experimentando dificultades técnicas. Para preguntas médicas, consulte a su proveedor de atención médica o llame al consultorio de su médico.",
            "fr": "Je rencontre actuellement des difficultés techniques. Pour les questions médicales, veuillez consulter votre professionnel de la santé.",
            "de": "Ich habe derzeit technische Schwierigkeiten. Bei medizinischen Fragen wenden Sie sich bitte an Ihren Arzt.",
            "it": "Sto attualmente riscontrando difficoltà tecniche. Per domande mediche, consulta il tuo medico."
        }

        return {
            "content": fallback_responses.get(language, fallback_responses["en"]),
            "confidence": 0.5,
            "follow_up_questions": [],
            "suggestions": [],
            "model_used": "fallback"
        }

    def _update_conversation_history(self, patient_id: str, user_message: str, response: Dict[str, Any]):
        """Update conversation history"""
        if patient_id not in self.conversation_history:
            self.conversation_history[patient_id] = []

        self.conversation_history[patient_id].append({
            "user": user_message,
            "assistant": response.get("content", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "specialist": response.get("model_used", "unknown")
        })

        # Keep only last 50 exchanges per patient
        if len(self.conversation_history[patient_id]) > 50:
            self.conversation_history[patient_id] = self.conversation_history[patient_id][-50:]

    def _add_safety_checks(self, response: Dict[str, Any], original_message: str, language: str) -> Dict[str, Any]:
        """Add safety checks and warnings"""

        safety_flags = []
        content = response.get("content", "")

        # Check for emergency keywords in response
        if any(word in content.lower() for word in ["emergency", "911", "urgent", "immediate"]):
            safety_flags.append("emergency_mentioned")

        # Check for medication mentions
        if any(word in content.lower() for word in ["take", "dose", "medication", "drug", "prescription"]):
            safety_flags.append("medication_mentioned")

        # Check for diagnostic language
        if any(word in content.lower() for word in ["diagnose", "you have", "condition is"]):
            safety_flags.append("diagnostic_language")

        response["safety_flags"] = safety_flags

        return response

    def _get_medical_disclaimer(self, language: str) -> str:
        """Get medical disclaimer in appropriate language"""

        disclaimers = {
            "en": "This information is for educational purposes only and is not a substitute for professional medical advice. Always consult your healthcare provider for medical decisions.",
            "es": "Esta información es solo para fines educativos y no sustituye el consejo médico profesional. Siempre consulte a su proveedor de atención médica para decisiones médicas.",
            "fr": "Ces informations sont à des fins éducatives uniquement et ne remplacent pas les conseils médicaux professionnels. Consultez toujours votre professionnel de la santé.",
            "de": "Diese Informationen dienen nur Bildungszwecken und ersetzen keine professionelle medizinische Beratung. Konsultieren Sie immer Ihren Arzt.",
            "it": "Queste informazioni sono solo a scopo educativo e non sostituiscono la consulenza medica professionale. Consulta sempre il tuo medico."
        }

        return disclaimers.get(language, disclaimers["en"])

    def _error_response(self, error: str, language: str) -> Dict[str, Any]:
        """Generate error response"""

        error_messages = {
            "en": "I apologize, but I'm having trouble processing your request. Please try again or contact your healthcare provider.",
            "es": "Me disculpo, pero tengo problemas para procesar su solicitud. Inténtelo de nuevo o comuníquese con su proveedor de atención médica.",
            "fr": "Je m'excuse, mais j'ai des difficultés à traiter votre demande. Veuillez réessayer ou contacter votre professionnel de la santé.",
            "de": "Entschuldigung, aber ich habe Probleme bei der Bearbeitung Ihrer Anfrage. Versuchen Sie es erneut oder kontaktieren Sie Ihren Arzt.",
            "it": "Mi scuso, ma ho difficoltà a elaborare la tua richiesta. Riprova o contatta il tuo medico."
        }

        return {
            "message": error_messages.get(language, error_messages["en"]),
            "language": language,
            "error": True,
            "timestamp": datetime.utcnow().isoformat()
        }
