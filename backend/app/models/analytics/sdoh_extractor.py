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
            confidence = data.get("confidence", 0.5 )

            severity_scores.append(self._severity_to_score(severity))
            severity_analysis["category_severities"][category] = {
                "severity": severity,
                "confidence": confidence,
                "impact_factors": self._assess_category_impact(category, data, patient_data)
            }

        # Calculate overall severity
        if severity_scores:
            avg_severity_score = sum(severity_scores) / len(severity_scores)
            if avg_severity_score >= 2.5:
                severity_analysis["overall_severity"] = "severe"
            elif avg_severity_score >= 1.5:
                severity_analysis["overall_severity"] = "moderate"
            else:
                severity_analysis["overall_severity"] = "mild"

        # Identify vulnerability factors
        vulnerability_factors = []
        if patient_data.get("age", 0) > 65:
            vulnerability_factors.append("advanced_age")
        if len(patient_data.get("conditions", [])) > 2:
            vulnerability_factors.append("multiple_comorbidities")
        if patient_data.get("insurance_type") in ["medicaid", "uninsured"]:
            vulnerability_factors.append("insurance_barriers")

        severity_analysis["vulnerability_factors"] = vulnerability_factors

        return severity_analysis

    def _severity_to_score(self, severity: str) -> float:
        """Convert severity to numerical score"""
        severity_map = {"mild": 1.0, "moderate": 2.0, "severe": 3.0}
        return severity_map.get(severity, 1.0)

    def _assess_category_impact(self, category: str, data: Dict[str, Any], patient_data: Dict[str, Any]) -> List[str]:
        """Assess impact factors for specific SDOH category"""
        impact_factors = []

        if category == "economic_stability":
            if "unemployed" in data.get("keywords_found", []):
                impact_factors.append("income_loss")
            if "insurance" in " ".join(data.get("keywords_found", [])):
                impact_factors.append("healthcare_access_risk")

        elif category == "healthcare_access":
            if "transportation" in " ".join(data.get("keywords_found", [])):
                impact_factors.append("appointment_barriers")
            if "distance" in " ".join(data.get("keywords_found", [])):
                impact_factors.append("geographic_barriers")

        elif category == "food_security":
            if "hungry" in data.get("keywords_found", []):
                impact_factors.append("nutritional_deficiency_risk")
            if patient_data.get("conditions") and "diabetes" in patient_data["conditions"]:
                impact_factors.append("diabetes_management_risk")

        return impact_factors

    def _generate_interventions(self, sdoh_factors: Dict[str, Any], severity_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate targeted interventions for identified SDOH factors"""
        interventions = []

        for category, data in sdoh_factors.items():
            severity = data.get("severity", "mild")
            confidence = data.get("confidence", 0.5)

            # Get category-specific interventions
            category_interventions = self.intervention_mapping.get(category, [])

            for intervention in category_interventions:
                priority = self._calculate_intervention_priority(severity, confidence, category)

                interventions.append({
                    "category": category,
                    "intervention": intervention,
                    "priority": priority,
                    "urgency": "immediate" if severity == "severe" else "within_week" if severity == "moderate" else "routine",
                    "confidence": confidence,
                    "expected_impact": self._estimate_intervention_impact(category, intervention, severity)
                })

        # Sort by priority
        interventions.sort(key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["priority"]], reverse=True)

        return interventions[:10]  # Return top 10 interventions

    def _calculate_intervention_priority(self, severity: str, confidence: float, category: str) -> str:
        """Calculate intervention priority"""
        base_priority = {"severe": 3, "moderate": 2, "mild": 1}[severity]
        confidence_boost = 1 if confidence > 0.7 else 0

        # Category-specific priority adjustments
        critical_categories = ["economic_stability", "healthcare_access", "food_security"]
        category_boost = 1 if category in critical_categories else 0

        total_score = base_priority + confidence_boost + category_boost

        if total_score >= 4:
            return "high"
        elif total_score >= 2:
            return "medium"
        else:
            return "low"

    def _estimate_intervention_impact(self, category: str, intervention: str, severity: str) -> str:
        """Estimate expected impact of intervention"""
        high_impact_interventions = [
            "Financial counseling referral",
            "Transportation assistance",
            "Food bank referrals",
            "Housing assistance referral"
        ]

        if intervention in high_impact_interventions and severity in ["moderate", "severe"]:
            return "high"
        elif severity == "severe":
            return "medium"
        else:
            return "low"

    def _calculate_sdoh_risk_score(self, sdoh_factors: Dict[str, Any], severity_analysis: Dict[str, Any]) -> float:
        """Calculate overall SDOH risk score"""
        if not sdoh_factors:
            return 0.0

        # Base score from number of factors
        factor_count_score = min(1.0, len(sdoh_factors) / 6) * 0.4

        # Severity score
        severity_scores = []
        for category, data in sdoh_factors.items():
            severity = data.get("severity", "mild")
            confidence = data.get("confidence", 0.5)
            severity_score = self._severity_to_score(severity) / 3.0  # Normalize to 0-1
            weighted_score = severity_score * confidence
            severity_scores.append(weighted_score)

        avg_severity_score = sum(severity_scores) / len(severity_scores) * 0.6

        # Vulnerability adjustment
        vulnerability_count = len(severity_analysis.get("vulnerability_factors", []))
        vulnerability_score = min(0.2, vulnerability_count * 0.05)

        total_score = factor_count_score + avg_severity_score + vulnerability_score
        return min(1.0, total_score)

    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize SDOH risk level"""
        if risk_score >= 0.7:
            return "High"
        elif risk_score >= 0.4:
            return "Moderate"
        else:
            return "Low"

    def _generate_sdoh_insights(self, sdoh_factors: Dict[str, Any], severity_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from SDOH analysis"""
        insights = []

        # Overall insights
        factor_count = len(sdoh_factors)
        if factor_count >= 3:
            insights.append(f"Multiple SDOH factors identified ({factor_count}) - comprehensive intervention needed")
        elif factor_count >= 1:
            insights.append(f"SDOH factors present - targeted interventions recommended")

        # Category-specific insights
        for category, data in sdoh_factors.items():
            severity = data.get("severity", "mild")
            if severity == "severe":
                category_name = category.replace("_", " ").title()
                insights.append(f"Severe {category_name} issues require immediate attention")

        # Vulnerability insights
        vulnerability_factors = severity_analysis.get("vulnerability_factors", [])
        if vulnerability_factors:
            insights.append(f"Patient has additional vulnerability factors: {', '.join(vulnerability_factors)}")

        # Impact insights
        high_impact_categories = []
        for category, analysis in severity_analysis.get("category_severities", {}).items():
            if analysis.get("impact_factors"):
                high_impact_categories.append(category.replace("_", " ").title())

        if high_impact_categories:
            insights.append(f"High-impact areas for intervention: {', '.join(high_impact_categories)}")

        return insights

    def _identify_priority_areas(self, sdoh_factors: Dict[str, Any], severity_analysis: Dict[str, Any]) -> List[str]:
        """Identify priority areas for intervention"""
        priority_areas = []

        # Sort categories by severity and confidence
        category_priorities = []
        for category, data in sdoh_factors.items():
            severity_score = self._severity_to_score(data.get("severity", "mild"))
            confidence = data.get("confidence", 0.5)
            priority_score = severity_score * confidence
            category_priorities.append((category, priority_score))

        # Sort by priority score
        category_priorities.sort(key=lambda x: x[1], reverse=True)

        # Return top 3 priority areas
        for category, score in category_priorities[:3]:
            priority_areas.append(category.replace("_", " ").title())

        return priority_areas
