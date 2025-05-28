"""
Feedback Analysis Agent for Patient Experience
"""

import openai
import json
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import numpy as np
from textblob import TextBlob

from app.config.settings import settings

logger = logging.getLogger(__name__)

class FeedbackAnalysisAgent:
    """
    AI agent for analyzing patient feedback and experience data
    """

    def __init__(self):
        self.openai_client = None
        self.sentiment_analyzer = None
        self.feedback_categories = self._load_feedback_categories()
        self._initialize_models()
        logger.info("✅ Feedback Analysis Agent initialized")

    def _initialize_models(self):
        """Initialize AI models"""
        try:
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ Feedback agent OpenAI client initialized")
        except Exception as e:
            logger.error(f"❌ Feedback agent initialization error: {e}")

    def _load_feedback_categories(self) -> Dict[str, Any]:
        """Load feedback analysis categories"""
        return {
            "satisfaction_domains": {
                "communication": ["doctor_communication", "nurse_communication", "staff_responsiveness"],
                "care_quality": ["pain_management", "medication_explanation", "treatment_effectiveness"],
                "environment": ["room_cleanliness", "noise_level", "comfort"],
                "discharge": ["discharge_information", "follow_up_instructions", "medication_instructions"]
            },
            "sentiment_keywords": {
                "positive": ["excellent", "great", "satisfied", "helpful", "caring", "professional"],
                "negative": ["poor", "terrible", "rude", "unhelpful", "painful", "confused"],
                "neutral": ["okay", "average", "fine", "normal", "standard"]
            },
            "priority_issues": {
                "safety_concerns": ["medication_error", "fall", "infection", "wrong_treatment"],
                "communication_failures": ["not_informed", "conflicting_information", "language_barrier"],
                "access_issues": ["long_wait", "appointment_difficulty", "insurance_problems"]
            }
        }

    async def analyze_feedback(
            self,
            feedback_text: str,
            feedback_type: str = "general",
            patient_id: Optional[str] = None,
            additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze patient feedback comprehensively
        """
        try:
            # Basic sentiment analysis
            sentiment_analysis = self._analyze_sentiment(feedback_text)

            # Category classification
            category_analysis = await self._classify_feedback_categories(feedback_text)

            # Priority assessment
            priority_assessment = self._assess_priority(feedback_text, category_analysis)

            # Extract actionable insights
            actionable_insights = await self._extract_actionable_insights(
                feedback_text, category_analysis, priority_assessment
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                sentiment_analysis, category_analysis, priority_assessment
            )

            # Calculate satisfaction scores
            satisfaction_scores = self._calculate_satisfaction_scores(
                sentiment_analysis, category_analysis
            )

            return {
                "feedback_id": f"fb_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "patient_id": patient_id,
                "feedback_type": feedback_type,
                "original_feedback": feedback_text,
                "sentiment_analysis": sentiment_analysis,
                "category_analysis": category_analysis,
                "priority_assessment": priority_assessment,
                "actionable_insights": actionable_insights,
                "recommendations": recommendations,
                "satisfaction_scores": satisfaction_scores,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "requires_follow_up": priority_assessment.get("requires_follow_up", False)
            }

        except Exception as e:
            logger.error(f"Feedback analysis error: {e}")
            return {"error": str(e), "feedback_text": feedback_text}

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using TextBlob and keyword analysis"""
        try:
            # TextBlob sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1

            # Keyword-based sentiment enhancement
            text_lower = text.lower()
            positive_count = sum(1 for word in self.feedback_categories["sentiment_keywords"]["positive"]
                                 if word in text_lower)
            negative_count = sum(1 for word in self.feedback_categories["sentiment_keywords"]["negative"]
                                 if word in text_lower)

            # Combine scores
            keyword_polarity = (positive_count - negative_count) / max(1, positive_count + negative_count)
            combined_polarity = (polarity + keyword_polarity) / 2

            # Categorize sentiment
            if combined_polarity > 0.1:
                sentiment_label = "positive"
            elif combined_polarity < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            return {
                "polarity": combined_polarity,
                "subjectivity": subjectivity,
                "sentiment_label": sentiment_label,
                "confidence": abs(combined_polarity),
                "positive_keywords": positive_count,
                "negative_keywords": negative_count,
                "emotion_indicators": self._detect_emotions(text_lower)
            }

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"sentiment_label": "neutral", "polarity": 0.0, "confidence": 0.0}

    def _detect_emotions(self, text: str) -> List[str]:
        """Detect emotional indicators in text"""
        emotion_keywords = {
            "anger": ["angry", "furious", "mad", "outraged", "frustrated"],
            "fear": ["scared", "afraid", "worried", "anxious", "nervous"],
            "sadness": ["sad", "depressed", "disappointed", "upset", "hurt"],
            "joy": ["happy", "pleased", "delighted", "satisfied", "grateful"],
            "surprise": ["surprised", "shocked", "amazed", "unexpected"],
            "trust": ["trust", "confident", "reliable", "dependable"]
        }

        detected_emotions = []
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text for keyword in keywords):
                detected_emotions.append(emotion)

        return detected_emotions

    async def _classify_feedback_categories(self, text: str) -> Dict[str, Any]:
        """Classify feedback into categories using AI"""
        if not self.openai_client:
            return self._classify_categories_rules(text)

        try:
            prompt = f"""
            Classify this patient feedback into relevant healthcare categories:
            
            Feedback: "{text}"
            
            Analyze and return JSON with:
            {{
                "primary_category": "main category (communication/care_quality/environment/discharge)",
                "subcategories": ["list of specific subcategories"],
                "mentioned_staff": ["types of staff mentioned"],
                "specific_issues": ["specific problems or compliments"],
                "improvement_areas": ["areas needing improvement"],
                "positive_aspects": ["things done well"],
                "confidence": 0.8
            }}
            
            Categories:
            - Communication: doctor/nurse communication, staff responsiveness
            - Care Quality: pain management, medication explanation, treatment effectiveness
            - Environment: room cleanliness, noise level, comfort
            - Discharge: discharge information, follow-up instructions
            """

            response = await self.openai_client.chat.completions.create(
                model=settings.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a healthcare quality analyst specializing in patient feedback classification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"AI category classification error: {e}")
            return self._classify_categories_rules(text)

    def _classify_categories_rules(self, text: str) -> Dict[str, Any]:
        """Rule-based category classification"""
        text_lower = text.lower()
        categories = []
        subcategories = []

        # Check each domain
        for domain, subcats in self.feedback_categories["satisfaction_domains"].items():
            domain_score = 0
            found_subcats = []

            for subcat in subcats:
                subcat_keywords = subcat.replace("_", " ").split()
                if any(keyword in text_lower for keyword in subcat_keywords):
                    domain_score += 1
                    found_subcats.append(subcat)

            if domain_score > 0:
                categories.append(domain)
                subcategories.extend(found_subcats)

        # Determine primary category
        primary_category = categories[0] if categories else "general"

        return {
            "primary_category": primary_category,
            "subcategories": subcategories,
            "mentioned_staff": self._extract_staff_mentions(text_lower),
            "specific_issues": [],
            "improvement_areas": [],
            "positive_aspects": [],
            "confidence": 0.6
        }

    def _extract_staff_mentions(self, text: str) -> List[str]:
        """Extract mentions of healthcare staff"""
        staff_types = ["doctor", "nurse", "physician", "therapist", "technician", "staff", "provider"]
        mentioned_staff = []

        for staff_type in staff_types:
            if staff_type in text:
                mentioned_staff.append(staff_type)

        return mentioned_staff

    def _assess_priority(self, text: str, category_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess priority level of feedback"""
        text_lower = text.lower()
        priority_score = 0
        priority_factors = []

        # Check for safety concerns
        safety_issues = []
        for issue in self.feedback_categories["priority_issues"]["safety_concerns"]:
            issue_keywords = issue.replace("_", " ").split()
            if any(keyword in text_lower for keyword in issue_keywords):
                safety_issues.append(issue)
                priority_score += 3

        # Check for communication failures
        comm_issues = []
        for issue in self.feedback_categories["priority_issues"]["communication_failures"]:
            issue_keywords = issue.replace("_", " ").split()
            if any(keyword in text_lower for keyword in issue_keywords):
                comm_issues.append(issue)
                priority_score += 2

        # Check for access issues
        access_issues = []
        for issue in self.feedback_categories["priority_issues"]["access_issues"]:
            issue_keywords = issue.replace("_", " ").split()
            if any(keyword in text_lower for keyword in issue_keywords):
                access_issues.append(issue)
                priority_score += 1

        # Sentiment-based priority adjustment
        sentiment = category_analysis.get("sentiment_label", "neutral")
        if sentiment == "negative":
            priority_score += 1

        # Determine priority level
        if priority_score >= 4:
            priority_level = "critical"
        elif priority_score >= 2:
            priority_level = "high"
        elif priority_score >= 1:
            priority_level = "medium"
        else:
            priority_level = "low"

        return {
            "priority_level": priority_level,
            "priority_score": priority_score,
            "safety_issues": safety_issues,
            "communication_issues": comm_issues,
            "access_issues": access_issues,
            "requires_follow_up": priority_level in ["critical", "high"],
            "response_timeframe": self._get_response_timeframe(priority_level)
        }

    def _get_response_timeframe(self, priority_level: str) -> str:
        """Get recommended response timeframe"""
        timeframes = {
            "critical": "Within 24 hours",
            "high": "Within 3 days",
            "medium": "Within 1 week",
            "low": "Within 2 weeks"
        }
        return timeframes.get(priority_level, "Within 2 weeks")

    async def _extract_actionable_insights(
            self,
            text: str,
            category_analysis: Dict[str, Any],
            priority_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract actionable insights from feedback"""

        insights = {
            "immediate_actions": [],
            "process_improvements": [],
            "training_needs": [],
            "policy_reviews": [],
            "positive_reinforcements": []
        }

        # Based on priority issues
        if priority_assessment["safety_issues"]:
            insights["immediate_actions"].extend([
                "Review safety protocols",
                "Investigate incident",
                "Implement corrective measures"
            ])

        if priority_assessment["communication_issues"]:
            insights["training_needs"].extend([
                "Communication skills training",
                "Cultural competency training",
                "Patient education improvement"
            ])

        # Based on categories
        primary_category = category_analysis.get("primary_category", "")

        if primary_category == "communication":
            insights["process_improvements"].append("Enhance communication protocols")
            insights["training_needs"].append("Communication effectiveness training")

        elif primary_category == "care_quality":
            insights["process_improvements"].append("Review care delivery processes")
            insights["policy_reviews"].append("Clinical quality standards review")

        elif primary_category == "environment":
            insights["immediate_actions"].append("Facility maintenance review")
            insights["process_improvements"].append("Environmental services improvement")

        elif primary_category == "discharge":
            insights["process_improvements"].append("Discharge planning enhancement")
            insights["training_needs"].append("Discharge education training")

        return insights

    def _generate_recommendations(
            self,
            sentiment_analysis: Dict[str, Any],
            category_analysis: Dict[str, Any],
            priority_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate specific recommendations"""

        recommendations = []

        # Priority-based recommendations
        if priority_assessment["priority_level"] == "critical":
            recommendations.extend([
                "Immediate management review required",
                "Direct patient contact within 24 hours",
                "Root cause analysis initiation"
            ])

        elif priority_assessment["priority_level"] == "high":
            recommendations.extend([
                "Department head review",
                "Patient follow-up call",
                "Process improvement assessment"
            ])

        # Sentiment-based recommendations
        if sentiment_analysis["sentiment_label"] == "negative":
            recommendations.extend([
                "Service recovery intervention",
                "Staff coaching opportunity",
                "Patient satisfaction follow-up"
            ])

        elif sentiment_analysis["sentiment_label"] == "positive":
            recommendations.extend([
                "Share positive feedback with team",
                "Recognize staff excellence",
                "Document best practices"
            ])

        # Category-specific recommendations
        primary_category = category_analysis.get("primary_category", "")

        if primary_category == "communication":
            recommendations.append("Communication skills assessment")
        elif primary_category == "care_quality":
            recommendations.append("Clinical quality review")
        elif primary_category == "environment":
            recommendations.append("Facility improvement plan")
        elif primary_category == "discharge":
            recommendations.append("Discharge process optimization")

        return list(set(recommendations))  # Remove duplicates

    def _calculate_satisfaction_scores(
            self,
            sentiment_analysis: Dict[str, Any],
            category_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate satisfaction scores by domain"""

        base_score = 50  # Neutral baseline
        sentiment_adjustment = sentiment_analysis["polarity"] * 30  # -30 to +30

        # Overall satisfaction score
        overall_score = max(0, min(100, base_score + sentiment_adjustment))

        # Domain-specific scores
        domain_scores = {}
        for domain in self.feedback_categories["satisfaction_domains"].keys():
            if domain == category_analysis.get("primary_category"):
                # Primary category gets full sentiment impact
                domain_scores[domain] = overall_score
            else:
                # Other domains get neutral score
                domain_scores[domain] = base_score

        return {
            "overall_satisfaction": overall_score,
            "domain_scores": domain_scores,
            "confidence_level": sentiment_analysis.get("confidence", 0.5)
        }
