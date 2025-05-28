"""
Social Determinants of Health (SDOH) Extractor
"""

import openai
import json
import re
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from app.config.settings import settings

logger = logging.getLogger(__name__)

class SDOHExtractor:
    """
    AI agent for extracting and analyzing social determinants of health
    """

    def __init__(self):
        self.openai_client = None
        self.sdoh_categories = self._load_sdoh_categories()
        self.intervention_mapping = self._load_intervention_mapping()
        self._initialize_client()
        logger.info("✅ SDOH Extractor initialized")

    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ SDOH extractor OpenAI client initialized")
        except Exception as e:
            logger.error(f"❌ SDOH extractor initialization error: {e}")

    def _load_sdoh_categories(self) -> Dict[str, Any]:
        """Load SDOH categories and indicators"""
        return {
            "economic_stability": {
                "keywords": [
                    "unemployed", "job loss", "financial hardship", "poverty", "low income",
                    "can't afford", "insurance problems", "medicaid", "food stamps", "welfare"
                ],
                "indicators": [
                    "employment_status", "income_level", "insurance_coverage",
                    "financial_strain", "government_assistance"
                ]
            },
            "education": {
                "keywords": [
                    "high school", "college", "education", "literacy", "reading problems",
                    "language barrier", "interpreter needed", "doesn't understand"
                ],
                "indicators": [
                    "education_level", "health_literacy", "language_proficiency",
                    "learning_disabilities"
                ]
            },
            "social_community": {
                "keywords": [
                    "lives alone", "no family", "isolated", "no support", "caregiver",
                    "social worker", "community resources", "church", "neighbors"
                ],
                "indicators": [
                    "social_support", "family_structure", "community_connections",
                    "caregiver_availability", "social_isolation"
                ]
            },
            "healthcare_access": {
                "keywords": [
                    "no doctor", "can't get appointment", "transportation", "distance",
                    "clinic closed", "waiting list", "specialist referral", "medication access"
                ],
                "indicators": [
                    "provider_access", "transportation_barriers", "appointment_availability",
                    "medication_access", "specialist_access"
                ]
            },
            "neighborhood_environment": {
                "keywords": [
                    "unsafe neighborhood", "crime", "pollution", "noise", "housing quality",
                    "homeless", "shelter", "eviction", "mold", "lead paint"
                ],
                "indicators": [
                    "housing_stability", "housing_quality", "neighborhood_safety",
                    "environmental_hazards", "walkability"
                ]
            },
            "food_security": {
                "keywords": [
                    "hungry", "food insecurity", "food bank", "skipping meals",
                    "can't afford food", "food stamps", "WIC", "nutrition"
                ],
                "indicators": [
                    "food_access", "nutrition_quality", "food_affordability",
                    "cooking_facilities", "food_assistance"
                ]
            }
        }

    def _load_intervention_mapping(self) -> Dict[str, List[str]]:
        """Load interventions for each SDOH category"""
        return {
            "economic_stability": [
                "Financial counseling referral",
                "Insurance enrollment assistance",
                "Employment services referral",
                "Government benefits application support",
                "Prescription assistance programs"
            ],
            "education": [
                "Health literacy materials in appropriate language",
                "Interpreter services",
                "Simplified discharge instructions",
                "Educational support programs",
                "Adult education referrals"
            ],
            "social_community": [
                "Social work consultation",
                "Community resource referrals",
                "Support group connections",
                "Caregiver support services",
                "Volunteer visitor programs"
            ],
            "healthcare_access": [
                "Transportation assistance",
                "Care coordination services",
                "Telehealth options",
                "Community health center referral",
                "Mobile clinic information"
            ],
            "neighborhood_environment": [
                "Housing assistance referral",
                "Environmental health assessment",
                "Safety planning",
                "Housing quality improvement resources",
                "Relocation assistance if needed"
            ],
            "food_security": [
                "Food bank referrals",
                "SNAP benefits enrollment",
                "WIC program referral",
                "Nutrition counseling",
                "Community garden programs"
            ]
        }

    async def extract_sdoh(
            self,
            text_data: str,
            patient_data: Optional[Dict[str, Any]] = None,
            data_source: str = "clinical_notes"
    ) -> Dict[str, Any]:
        """
        Extract SDOH factors from text data
        """
        try:
            # Rule-based extraction
            rule_based_results = self._rule_based_extraction(text_data)

            # AI-enhanced extraction if available
            if self.openai_client:
                ai_results = await self._ai_enhanced_extraction(text_data, data_source)
                combined_results = self._combine_extraction_results(rule_based_results, ai_results)
            else:
                combined_results = rule_based_results

            # Analyze severity and impact
            severity_analysis = self._analyze_severity(combined_results, patient_data or {})

            # Generate interventions
            interventions = self._generate_interventions(combined_results, severity_analysis)

            # Calculate SDOH risk score
            risk_score = self._calculate_sdoh_risk_score(combined_results, severity_analysis)

            # Generate insights
            insights = self._generate_sdoh_insights(combined_results, severity_analysis)

            return {
                "extraction_id": f"sdoh_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "data_source": data_source,
                "sdoh_factors": combined_results,
                "severity_analysis": severity_analysis,
                "risk_score": risk_score,
                "risk_level": self._categorize_risk_level(risk_score),
                "recommended_interventions": interventions,
                "insights": insights,
                "priority_areas": self._identify_priority_areas(combined_results, severity_analysis),
                "extracted_at": datetime.utcnow().isoformat(),
                "extraction_method": "ai_enhanced" if self.openai_client else "rule_based"
            }

        except Exception as e:
            logger.error(f"SDOH extraction error: {e}")
            return {"error": str(e), "text_data": text_data[:100] + "..."}

    def _rule_based_extraction(self, text: str) -> Dict[str, Any]:
        """Rule-based SDOH extraction using keywords"""
        text_lower = text.lower()
        extracted_factors = {}

        for category, data in self.sdoh_categories.items():
            category_findings = {
                "keywords_found": [],
                "indicators": [],
                "evidence_text": [],
                "confidence": 0.0
            }

            # Check for keywords
            for keyword in data["keywords"]:
                if keyword in text_lower:
                    category_findings["keywords_found"].append(keyword)

                    # Extract surrounding context
                    pattern = rf'.{{0,50}}{re.escape(keyword)}.{{0,50}}'
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    category_findings["evidence_text"].extend(matches)

            # Calculate confidence based on keyword matches
            if category_findings["keywords_found"]:
                category_findings["confidence"] = min(1.0, len(category_findings["keywords_found"]) / 3)
                category_findings["indicators"] = data["indicators"]
                extracted_factors[category] = category_findings

        return extracted_factors

    async def _ai_enhanced_extraction(self, text: str, data_source: str) -> Dict[str, Any]:
        """AI-enhanced SDOH extraction using GPT-4o"""

        prompt = f"""
        Analyze this {data_source} text for social determinants of health (SDOH) factors:
        
        TEXT: {text}
        
        Extract information about these SDOH categories:
        1. Economic Stability (employment, income, insurance, financial hardship)
        2. Education (education level, health literacy, language barriers)
        3. Social/Community Context (social support, family, isolation)
        4. Healthcare Access (provider access, transportation, appointments)
        5. Neighborhood/Environment (housing, safety, environmental factors)
        6. Food Security (food access, nutrition, food assistance)
        
        For each category found, provide:
        - severity: "mild", "moderate", "severe"
        - evidence: specific text that indicates the issue
        - confidence: 0.0 to 1.0
        - specific_factors: list of specific SDOH factors identified
        
        Return JSON format:
        {{
            "category_name": {{
                "severity": "moderate",
                "evidence": "specific text from input",
                "confidence": 0.8,
                "specific_factors": ["factor1", "factor2"]
            }}
        }}
        
        Only include categories where you find clear evidence.
        """

        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a social determinants of health expert. Extract SDOH factors from clinical and social text data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            logger.error(f"AI SDOH extraction error: {e}")
            return {}

    def _combine_extraction_results(self, rule_based: Dict[str, Any], ai_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine rule-based and AI extraction results"""
        combined = {}

        # Start with AI results (more structured)
        for category, ai_data in ai_results.items():
            # Map AI category names to our standard categories
            standard_category = self._map_to_standard_category(category)
            if standard_category:
                combined[standard_category] = {
                    "severity": ai_data.get("severity", "moderate"),
                    "evidence": ai_data.get("evidence", ""),
                    "confidence": ai_data.get("confidence", 0.5),
                    "specific_factors": ai_data.get("specific_factors", []),
                    "source": "ai_analysis"
                }

        # Enhance with rule-based findings
        for category, rule_data in rule_based.items():
            if category in combined:
                # Enhance existing finding
                combined[category]["keywords_found"] = rule_data["keywords_found"]
                combined[category]["confidence"] = max(
                    combined[category]["confidence"],
                    rule_data["confidence"]
                )
            else:
                # Add new finding from rules
                combined[category] = {
                    "severity": self._infer_severity_from_keywords(rule_data["keywords_found"]),
                    "evidence": "; ".join(rule_data["evidence_text"][:3]),
                    "confidence": rule_data["confidence"],
                    "specific_factors": rule_data["indicators"],
                    "keywords_found": rule_data["keywords_found"],
                    "source": "keyword_analysis"
                }

        return combined

    def _map_to_standard_category(self, ai_category: str) -> Optional[str]:
        """Map AI category names to standard SDOH categories"""
        category_mapping = {
            "economic_stability": "economic_stability",
            "economic": "economic_stability",
            "financial": "economic_stability",
            "education": "education",
            "educational": "education",
            "social_community": "social_community",
            "social": "social_community",
            "community": "social_community",
            "healthcare_access": "healthcare_access",
            "healthcare": "healthcare_access",
            "access": "healthcare_access",
            "neighborhood_environment": "neighborhood_environment",
            "neighborhood": "neighborhood_environment",
            "environment": "neighborhood_environment",
            "housing": "neighborhood_environment",
            "food_security": "food_security",
            "food": "food_security",
            "nutrition": "food_security"
        }

        return category_mapping.get(ai_category.lower())

    def _infer_severity_from_keywords(self, keywords: List[str]) -> str:
        """Infer severity from keyword matches"""
        high_severity_keywords = ["homeless", "eviction", "unemployed", "can't afford", "hungry"]

        if any(keyword in " ".join(keywords) for keyword in high_severity_keywords):
            return "severe"
        elif len(keywords) > 2:
            return "moderate"
        else:
            return "mild"

    def _analyze_severity(self, sdoh_factors: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze severity and impact of SDOH factors"""

        severity_analysis = {
            "overall_severity": "mild",
            "category_severities": {},
            "impact_assessment": {},
            "vulnerability_factors": []
        }

        # Analyze each category
        severity_scores = []
        for category, data in sdoh_factors.items():
            severity = data.get("severity", "mild")
            confidence = data.get("confidence", 0.5
