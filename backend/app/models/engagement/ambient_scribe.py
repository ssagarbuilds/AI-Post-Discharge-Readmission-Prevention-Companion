"""
Ambient AI Scribe for Clinical Documentation
"""

import openai
import whisper
import json
import tempfile
import os
from typing import Dict, Any, Optional, List
from fastapi import UploadFile
import base64
import io
import logging
from datetime import datetime

from app.config.settings import settings

logger = logging.getLogger(__name__)

class AmbientScribe:
    """
    Ambient AI scribe for automated clinical documentation
    """

    def __init__(self):
        self.openai_client = None
        self.whisper_model = None
        self.note_templates = self._load_note_templates()
        self.clinical_vocabularies = self._load_clinical_vocabularies()
        self._initialize_models()
        logger.info("✅ Ambient Scribe initialized")

    def _initialize_models(self):
        """Initialize AI models"""
        try:
            # OpenAI client
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ Ambient scribe OpenAI client initialized")

            # Whisper model for audio transcription
            try:
                self.whisper_model = whisper.load_model("base")
                logger.info("✅ Whisper model loaded for ambient scribe")
            except Exception as e:
                logger.warning(f"⚠️ Whisper model not loaded: {e}")

        except Exception as e:
            logger.error(f"❌ Ambient scribe initialization error: {e}")

    def _load_note_templates(self) -> Dict[str, str]:
        """Load clinical note templates"""
        return {
            "soap_note": """
SOAP NOTE

Date: {date}
Patient: {patient_name}
Provider: {provider_name}

SUBJECTIVE:
Chief Complaint: {chief_complaint}
History of Present Illness: {hpi}
Review of Systems: {ros}
Past Medical History: {pmh}
Medications: {medications}
Allergies: {allergies}
Social History: {social_history}
Family History: {family_history}

OBJECTIVE:
Vital Signs: {vital_signs}
Physical Examination: {physical_exam}
Laboratory Results: {lab_results}
Imaging: {imaging}

ASSESSMENT:
{assessment}

PLAN:
{plan}

Provider Signature: {provider_signature}
Date: {signature_date}
""",
            "discharge_summary": """
DISCHARGE SUMMARY

Patient: {patient_name}
DOB: {dob}
MRN: {mrn}
Admission Date: {admission_date}
Discharge Date: {discharge_date}
Length of Stay: {length_of_stay}
Attending Physician: {attending_physician}

ADMISSION DIAGNOSIS:
{admission_diagnosis}

DISCHARGE DIAGNOSIS:
{discharge_diagnosis}

HOSPITAL COURSE:
{hospital_course}

PROCEDURES PERFORMED:
{procedures}

DISCHARGE MEDICATIONS:
{discharge_medications}

DISCHARGE INSTRUCTIONS:
{discharge_instructions}

FOLLOW-UP:
{follow_up}

DISCHARGE CONDITION: {discharge_condition}

Provider: {provider_name}
Date: {discharge_date}
""",
            "progress_note": """
PROGRESS NOTE

Date: {date}
Time: {time}
Patient: {patient_name}
Service: {service}

INTERVAL HISTORY:
{interval_history}

PHYSICAL EXAMINATION:
{physical_exam}

LABORATORY/STUDIES:
{lab_studies}

ASSESSMENT AND PLAN:
{assessment_plan}

Provider: {provider_name}
""",
            "consultation_note": """
CONSULTATION NOTE

Date: {date}
Consulting Service: {consulting_service}
Requesting Service: {requesting_service}
Patient: {patient_name}

REASON FOR CONSULTATION:
{reason_for_consultation}

HISTORY:
{history}

PHYSICAL EXAMINATION:
{physical_exam}

REVIEW OF RECORDS:
{record_review}

IMPRESSION:
{impression}

RECOMMENDATIONS:
{recommendations}

Thank you for this consultation.

{consultant_name}, {credentials}
{consulting_service}
"""
        }

    def _load_clinical_vocabularies(self) -> Dict[str, List[str]]:
        """Load clinical vocabularies for better recognition"""
        return {
            "vital_signs": [
                "blood pressure", "heart rate", "respiratory rate", "temperature",
                "oxygen saturation", "pulse", "BP", "HR", "RR", "temp", "O2 sat"
            ],
            "physical_exam": [
                "inspection", "palpation", "percussion", "auscultation",
                "heart sounds", "lung sounds", "bowel sounds", "murmur",
                "rales", "rhonchi", "wheeze", "clear to auscultation"
            ],
            "medications": [
                "milligrams", "mg", "grams", "g", "units", "twice daily", "BID",
                "three times daily", "TID", "four times daily", "QID",
                "once daily", "daily", "as needed", "PRN"
            ],
            "procedures": [
                "intubation", "central line", "arterial line", "chest tube",
                "lumbar puncture", "paracentesis", "thoracentesis", "biopsy"
            ],
            "laboratory": [
                "complete blood count", "CBC", "comprehensive metabolic panel", "CMP",
                "liver function tests", "LFTs", "arterial blood gas", "ABG",
                "urinalysis", "UA", "blood culture", "chest x-ray", "CXR"
            ]
        }

    async def transcribe_and_document(
            self,
            audio_file: Optional[UploadFile] = None,
            transcript: Optional[str] = None,
            note_type: str = "soap_note",
            patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main function to transcribe audio and generate clinical documentation
        """
        try:
            # Get transcript
            if audio_file:
                transcript_result = await self._transcribe_audio(audio_file)
                transcript_text = transcript_result["transcript"]
                audio_quality = transcript_result["quality"]
            elif transcript:
                transcript_text = transcript
                audio_quality = {"confidence": 1.0, "clarity": "text_input"}
            else:
                raise ValueError("Either audio file or transcript must be provided")

            # Process transcript for clinical content
            clinical_content = await self._extract_clinical_content(transcript_text, note_type)

            # Generate structured note
            structured_note = await self._generate_structured_note(
                clinical_content, note_type, patient_context or {}
            )

            # Add quality metrics
            quality_metrics = self._calculate_quality_metrics(
                transcript_text, clinical_content, structured_note
            )

            # Generate suggestions for improvement
            suggestions = self._generate_improvement_suggestions(
                clinical_content, quality_metrics
            )

            return {
                "original_transcript": transcript_text,
                "clinical_content": clinical_content,
                "structured_note": structured_note,
                "note_type": note_type,
                "audio_quality": audio_quality,
                "quality_metrics": quality_metrics,
                "suggestions": suggestions,
                "generated_at": datetime.utcnow().isoformat(),
                "model_version": "ambient_scribe_v2.0"
            }

        except Exception as e:
            logger.error(f"Ambient scribe error: {e}")
            return {"error": str(e), "note_type": note_type}

    async def _transcribe_audio(self, audio_file: UploadFile) -> Dict[str, Any]:
        """Transcribe audio file using Whisper"""
        try:
            # Read audio content
            audio_content = await audio_file.read()

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_content)
                temp_path = temp_file.name

            try:
                if self.whisper_model:
                    # Transcribe with Whisper
                    result = self.whisper_model.transcribe(
                        temp_path,
                        language="en",  # Can be made configurable
                        task="transcribe"
                    )

                    transcript = result["text"]
                    language = result.get("language", "en")

                    # Calculate quality metrics
                    segments = result.get("segments", [])
                    avg_confidence = sum(seg.get("avg_logprob", 0) for seg in segments) / len(segments) if segments else 0

                    return {
                        "transcript": transcript,
                        "language": language,
                        "quality": {
                            "confidence": max(0, min(1, (avg_confidence + 5) / 5)),  # Normalize to 0-1
                            "clarity": "good" if avg_confidence > -0.5 else "fair" if avg_confidence > -1.0 else "poor",
                            "segments": len(segments),
                            "duration": result.get("duration", 0)
                        }
                    }
                else:
                    # Fallback when Whisper not available
                    return {
                        "transcript": "Audio transcription not available - Whisper model not loaded",
                        "language": "en",
                        "quality": {"confidence": 0.0, "clarity": "unavailable"}
                    }

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"Audio transcription error: {e}")
            return {
                "transcript": f"Transcription error: {str(e)}",
                "language": "en",
                "quality": {"confidence": 0.0, "clarity": "error"}
            }

    async def _extract_clinical_content(self, transcript: str, note_type: str) -> Dict[str, Any]:
        """Extract clinical content from transcript using AI"""

        if not self.openai_client:
            return self._extract_clinical_content_rules(transcript, note_type)

        try:
            # Create extraction prompt based on note type
            extraction_prompt = self._create_extraction_prompt(transcript, note_type)

            response = await self.openai_client.chat.completions.create(
                model=settings.DEFAULT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medical scribe. Extract clinical information from conversation transcripts and organize it into structured medical documentation."
                    },
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            # Parse the response
            try:
                clinical_content = json.loads(response.choices[0].message.content)
                return clinical_content
            except json.JSONDecodeError:
                # If JSON parsing fails, return the content as text
                return {"extracted_content": response.choices[0].message.content}

        except Exception as e:
            logger.error(f"Clinical content extraction error: {e}")
            return self._extract_clinical_content_rules(transcript, note_type)

    def _create_extraction_prompt(self, transcript: str, note_type: str) -> str:
        """Create extraction prompt based on note type"""

        base_prompt = f"""
        Extract clinical information from this medical conversation transcript for a {note_type}:
        
        TRANSCRIPT:
        {transcript}
        
        """

        if note_type == "soap_note":
            base_prompt += """
            Extract and organize the following information in JSON format:
            {
                "chief_complaint": "main reason for visit",
                "hpi": "history of present illness",
                "ros": "review of systems findings",
                "pmh": "past medical history",
                "medications": "current medications",
                "allergies": "known allergies",
                "social_history": "social history",
                "family_history": "family history",
                "vital_signs": "vital signs mentioned",
                "physical_exam": "physical examination findings",
                "lab_results": "laboratory results discussed",
                "imaging": "imaging results",
                "assessment": "clinical assessment/diagnosis",
                "plan": "treatment plan and follow-up"
            }
            """
        elif note_type == "discharge_summary":
            base_prompt += """
            Extract and organize the following information in JSON format:
            {
                "admission_diagnosis": "reason for admission",
                "discharge_diagnosis": "final diagnosis at discharge",
                "hospital_course": "summary of hospital stay",
                "procedures": "procedures performed",
                "discharge_medications": "medications at discharge",
                "discharge_instructions": "patient instructions",
                "follow_up": "follow-up appointments and care",
                "discharge_condition": "patient condition at discharge"
            }
            """
        elif note_type == "progress_note":
            base_prompt += """
            Extract and organize the following information in JSON format:
            {
                "interval_history": "changes since last visit",
                "physical_exam": "current physical examination",
                "lab_studies": "recent lab results or studies",
                "assessment_plan": "current assessment and plan"
            }
            """

        base_prompt += """
        
        Guidelines:
        - Use medical terminology appropriately
        - Be concise but complete
        - If information is not mentioned, use "Not documented" or leave empty
        - Maintain patient confidentiality
        - Use standard medical abbreviations when appropriate
        """

        return base_prompt

    def _extract_clinical_content_rules(self, transcript: str, note_type: str) -> Dict[str, Any]:
        """Rule-based clinical content extraction as fallback"""

        content = {
            "extraction_method": "rule_based",
            "confidence": 0.6
        }

        transcript_lower = transcript.lower()

        # Extract vital signs
        vital_signs = []
        for vital in self.clinical_vocabularies["vital_signs"]:
            if vital.lower() in transcript_lower:
                # Try to find the value after the vital sign
                import re
                pattern = rf"{vital.lower()}\s*:?\s*(\d+(?:\.\d+)?(?:/\d+)?)"
                matches = re.findall(pattern, transcript_lower)
                if matches:
                    vital_signs.append(f"{vital}: {matches[0]}")
                else:
                    vital_signs.append(vital)

        content["vital_signs"] = "; ".join(vital_signs) if vital_signs else "Not documented"

        # Extract medications mentioned
        medications = []
        for med_term in self.clinical_vocabularies["medications"]:
            if med_term.lower() in transcript_lower:
                medications.append(med_term)

        content["medications"] = "; ".join(medications) if medications else "Not documented"

        # Extract procedures
        procedures = []
        for proc in self.clinical_vocabularies["procedures"]:
            if proc.lower() in transcript_lower:
                procedures.append(proc)

        content["procedures"] = "; ".join(procedures) if procedures else "Not documented"

        # Basic content extraction based on common phrases
        if "chief complaint" in transcript_lower or "presenting with" in transcript_lower:
            content["chief_complaint"] = "Documented in transcript"
        else:
            content["chief_complaint"] = "Not clearly documented"

        if "physical exam" in transcript_lower or "examination" in transcript_lower:
            content["physical_exam"] = "Documented in transcript"
        else:
            content["physical_exam"] = "Not documented"

        if "plan" in transcript_lower or "treatment" in transcript_lower:
            content["plan"] = "Documented in transcript"
        else:
            content["plan"] = "Not documented"

        return content

    async def _generate_structured_note(
            self,
            clinical_content: Dict[str, Any],
            note_type: str,
            patient_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate structured clinical note"""

        # Get template
        template = self.note_templates.get(note_type, self.note_templates["soap_note"])

        # Prepare template variables
        template_vars = self._prepare_template_variables(clinical_content, patient_context)

        # Fill template
        try:
            formatted_note = template.format(**template_vars)
        except KeyError as e:
            # Handle missing template variables
            logger.warning(f"Missing template variable: {e}")
            formatted_note = template
            for key, value in template_vars.items():
                formatted_note = formatted_note.replace(f"{{{key}}}", str(value))

        # Generate additional metadata
        metadata = {
            "note_type": note_type,
            "generated_by": "ambient_scribe",
            "generation_time": datetime.utcnow().isoformat(),
            "template_version": "v2.0",
            "completeness_score": self._calculate_completeness(clinical_content, note_type)
        }

        return {
            "formatted_note": formatted_note,
            "metadata": metadata,
            "template_variables": template_vars
        }

    def _prepare_template_variables(self, clinical_content: Dict[str, Any], patient_context: Dict[str, Any]) -> Dict[str, str]:
        """Prepare variables for template formatting"""

        # Default values
        template_vars = {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "time": datetime.utcnow().strftime("%H:%M"),
            "patient_name": patient_context.get("patient_name", "[Patient Name]"),
            "provider_name": patient_context.get("provider_name", "[Provider Name]"),
            "dob": patient_context.get("dob", "[DOB]"),
            "mrn": patient_context.get("mrn", "[MRN]"),
            "service": patient_context.get("service", "[Service]"),
            "signature_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            "provider_signature": "[Electronic Signature]"
        }

        # Add clinical content
        for key, value in clinical_content.items():
            if isinstance(value, str):
                template_vars[key] = value
            elif isinstance(value, list):
                template_vars[key] = "; ".join(str(item) for item in value)
            else:
                template_vars[key] = str(value)

        # Ensure all values are strings and handle missing values
        for key, value in template_vars.items():
            if not value or value == "":
                template_vars[key] = "[Not documented]"
            else:
                template_vars[key] = str(value)

        return template_vars

    def _calculate_quality_metrics(
            self,
            transcript: str,
            clinical_content: Dict[str, Any],
            structured_note: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate quality metrics for the generated documentation"""

        # Completeness score
        completeness = self._calculate_completeness(clinical_content, "soap_note")

        # Accuracy indicators
        accuracy_indicators = {
            "medical_terms_detected": len([term for vocab in self.clinical_vocabularies.values()
                                           for term in vocab if term.lower() in transcript.lower()]),
            "structured_sections": len([k for k, v in clinical_content.items()
                                        if v and v != "Not documented"]),
            "transcript_length": len(transcript.split()),
            "note_length": len(structured_note.get("formatted_note", "").split())
        }

        # Overall quality score
        quality_score = (
                completeness * 0.4 +
                min(1.0, accuracy_indicators["medical_terms_detected"] / 10) * 0.3 +
                min(1.0, accuracy_indicators["structured_sections"] / 8) * 0.3
        )

        return {
            "completeness_score": completeness,
            "accuracy_indicators": accuracy_indicators,
            "overall_quality": quality_score,
            "quality_level": "excellent" if quality_score > 0.8 else
            "good" if quality_score > 0.6 else
            "fair" if quality_score > 0.4 else "poor"
        }

    def _calculate_completeness(self, clinical_content: Dict[str, Any], note_type: str) -> float:
        """Calculate completeness score based on note type"""

        required_fields = {
            "soap_note": [
                "chief_complaint", "hpi", "physical_exam", "assessment", "plan"
            ],
            "discharge_summary": [
                "admission_diagnosis", "discharge_diagnosis", "hospital_course",
                "discharge_medications", "discharge_instructions"
            ],
            "progress_note": [
                "interval_history", "physical_exam", "assessment_plan"
            ]
        }

        fields = required_fields.get(note_type, required_fields["soap_note"])
        completed_fields = 0

        for field in fields:
            value = clinical_content.get(field, "")
            if value and value != "Not documented" and value != "[Not documented]":
                completed_fields += 1

        return completed_fields / len(fields) if fields else 0.0

    def _generate_improvement_suggestions(
            self,
            clinical_content: Dict[str, Any],
            quality_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate suggestions for improving documentation"""

        suggestions = []

        # Completeness suggestions
        if quality_metrics["completeness_score"] < 0.7:
            suggestions.append("Consider documenting missing clinical sections for completeness")

        # Quality suggestions
        if quality_metrics["overall_quality"] < 0.6:
            suggestions.append("Review transcript for additional clinical details")

        # Specific content suggestions
        missing_sections = []
        for key, value in clinical_content.items():
            if not value or value == "Not documented":
                missing_sections.append(key.replace("_", " ").title())

        if missing_sections:
            suggestions.append(f"Consider adding information for: {', '.join(missing_sections[:3])}")

        # Audio quality suggestions
        if quality_metrics.get("accuracy_indicators", {}).get("medical_terms_detected", 0) < 3:
            suggestions.append("Ensure clear pronunciation of medical terms for better recognition")

        if not suggestions:
            suggestions.append("Documentation appears complete and well-structured")

        return suggestions
