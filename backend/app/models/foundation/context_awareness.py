"""
Context Awareness Agent for Patient State Tracking
"""

import asyncio
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import json

from app.config.settings import settings

logger = logging.getLogger(__name__)

class ContextAwarenessAgent:
    """
    AI agent for maintaining and updating patient context across interactions
    """

    def __init__(self):
        self.context_store = {}
        self.session_history = {}
        self.context_templates = self._load_context_templates()
        self.memory_management = self._initialize_memory_management()
        logger.info("✅ Context Awareness Agent initialized")

    def _load_context_templates(self) -> Dict[str, Any]:
        """Load context templates for different interaction types"""
        return {
            "patient_session": {
                "patient_id": None,
                "session_start": None,
                "current_symptoms": [],
                "mentioned_conditions": [],
                "medications_discussed": [],
                "concerns_raised": [],
                "care_goals": [],
                "interaction_history": [],
                "emotional_state": "neutral",
                "urgency_level": "routine"
            },
            "clinical_encounter": {
                "encounter_id": None,
                "encounter_type": "consultation",
                "chief_complaint": None,
                "assessment_findings": [],
                "treatment_plans": [],
                "follow_up_needed": False,
                "provider_notes": [],
                "patient_questions": []
            },
            "care_coordination": {
                "care_team": [],
                "active_interventions": [],
                "pending_referrals": [],
                "scheduled_appointments": [],
                "care_transitions": [],
                "communication_preferences": {}
            }
        }

    def _initialize_memory_management(self) -> Dict[str, Any]:
        """Initialize memory management settings.py"""
        return {
            "short_term_memory": {
                "duration_hours": 24,
                "max_items": 50
            },
            "long_term_memory": {
                "duration_days": 365,
                "max_items": 1000,
                "importance_threshold": 0.7
            },
            "context_decay": {
                "enabled": True,
                "decay_rate": 0.1,
                "refresh_threshold": 0.3
            }
        }

    async def initialize_patient_context(
            self,
            patient_id: str,
            initial_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Initialize context for a new patient interaction
        """
        try:
            # Create new context from template
            context = self.context_templates["patient_session"].copy()
            context["patient_id"] = patient_id
            context["session_start"] = datetime.utcnow().isoformat()

            # Add any initial data
            if initial_data:
                context.update(initial_data)

            # Store context
            self.context_store[patient_id] = context

            # Initialize session history
            if patient_id not in self.session_history:
                self.session_history[patient_id] = []

            logger.info(f"✅ Context initialized for patient {patient_id}")

            return {
                "status": "initialized",
                "patient_id": patient_id,
                "context": context,
                "session_id": context["session_start"]
            }

        except Exception as e:
            logger.error(f"Error initializing context for patient {patient_id}: {e}")
            return {"status": "error", "error": str(e)}

    async def update_context(
            self,
            patient_id: str,
            update_data: Dict[str, Any],
            update_type: str = "incremental"
    ) -> Dict[str, Any]:
        """
        Update patient context with new information
        """
        try:
            # Get current context
            current_context = self.context_store.get(patient_id, {})

            if not current_context:
                # Initialize if doesn't exist
                await self.initialize_patient_context(patient_id, update_data)
                return self.context_store[patient_id]

            # Apply updates based on type
            if update_type == "incremental":
                updated_context = self._apply_incremental_update(current_context, update_data)
            elif update_type == "replace":
                updated_context = self._apply_replacement_update(current_context, update_data)
            elif update_type == "merge":
                updated_context = self._apply_merge_update(current_context, update_data)
            else:
                updated_context = {**current_context, **update_data}

            # Add timestamp
            updated_context["last_updated"] = datetime.utcnow().isoformat()

            # Store updated context
            self.context_store[patient_id] = updated_context

            # Add to session history
            self._add_to_session_history(patient_id, update_data, update_type)

            # Apply memory management
            await self._apply_memory_management(patient_id)

            logger.debug(f"Context updated for patient {patient_id}")

            return {
                "status": "updated",
                "patient_id": patient_id,
                "update_type": update_type,
                "context": updated_context
            }

        except Exception as e:
            logger.error(f"Error updating context for patient {patient_id}: {e}")
            return {"status": "error", "error": str(e)}

    def _apply_incremental_update(self, current: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Apply incremental updates to context"""
        updated = current.copy()

        for key, value in update.items():
            if key in updated:
                if isinstance(updated[key], list) and isinstance(value, list):
                    # Append to lists
                    updated[key].extend(value)
                    # Remove duplicates while preserving order
                    updated[key] = list(dict.fromkeys(updated[key]))
                elif isinstance(updated[key], dict) and isinstance(value, dict):
                    # Merge dictionaries
                    updated[key].update(value)
                else:
                    # Replace value
                    updated[key] = value
            else:
                updated[key] = value

        return updated

    def _apply_replacement_update(self, current: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Apply replacement updates to context"""
        updated = current.copy()
        updated.update(update)
        return updated

    def _apply_merge_update(self, current: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Apply deep merge updates to context"""
        def deep_merge(dict1, dict2):
            result = dict1.copy()
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        return deep_merge(current, update)

    def _add_to_session_history(self, patient_id: str, update_data: Dict[str, Any], update_type: str):
        """Add update to session history"""
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "update_type": update_type,
            "update_data": update_data,
            "importance": self._calculate_importance(update_data)
        }

        self.session_history[patient_id].append(history_entry)

        # Limit history size
        max_history = self.memory_management["long_term_memory"]["max_items"]
        if len(self.session_history[patient_id]) > max_history:
            self.session_history[patient_id] = self.session_history[patient_id][-max_history:]

    def _calculate_importance(self, update_data: Dict[str, Any]) -> float:
        """Calculate importance score for context update"""
        importance = 0.5  # Base importance

        # High importance keywords
        high_importance_keys = [
            "emergency", "urgent", "critical", "severe", "pain", "symptoms",
            "medication", "allergy", "diagnosis", "treatment"
        ]

        # Check for high importance content
        update_str = json.dumps(update_data, default=str).lower()
        for keyword in high_importance_keys:
            if keyword in update_str:
                importance += 0.1

        # Cap at 1.0
        return min(1.0, importance)

    async def _apply_memory_management(self, patient_id: str):
        """Apply memory management policies"""
        if not self.memory_management["context_decay"]["enabled"]:
            return

        context = self.context_store.get(patient_id, {})
        if not context:
            return

        # Apply decay to interaction history
        if "interaction_history" in context:
            decay_rate = self.memory_management["context_decay"]["decay_rate"]
            refresh_threshold = self.memory_management["context_decay"]["refresh_threshold"]

            # Remove old, low-importance interactions
            current_time = datetime.utcnow()
            filtered_history = []

            for interaction in context["interaction_history"]:
                if "timestamp" in interaction:
                    interaction_time = datetime.fromisoformat(interaction["timestamp"])
                    age_hours = (current_time - interaction_time).total_seconds() / 3600

                    # Calculate decay factor
                    decay_factor = max(0, 1 - (age_hours * decay_rate / 24))
                    importance = interaction.get("importance", 0.5)

                    # Keep if above threshold
                    if decay_factor * importance > refresh_threshold:
                        filtered_history.append(interaction)

            context["interaction_history"] = filtered_history
            self.context_store[patient_id] = context

    async def get_context(self, patient_id: str, context_type: str = "full") -> Dict[str, Any]:
        """
        Retrieve current context for a patient
        """
        try:
            context = self.context_store.get(patient_id, {})

            if context_type == "full":
                return context
            elif context_type == "summary":
                return self._create_context_summary(context)
            elif context_type == "recent":
                return self._get_recent_context(context)
            else:
                return context

        except Exception as e:
            logger.error(f"Error retrieving context for patient {patient_id}: {e}")
            return {}

    def _create_context_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the context"""
        summary = {
            "patient_id": context.get("patient_id"),
            "session_duration": self._calculate_session_duration(context),
            "key_symptoms": context.get("current_symptoms", [])[:5],
            "main_concerns": context.get("concerns_raised", [])[:3],
            "urgency_level": context.get("urgency_level", "routine"),
            "emotional_state": context.get("emotional_state", "neutral"),
            "interaction_count": len(context.get("interaction_history", []))
        }
        return summary

    def _get_recent_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get recent context items"""
        recent_hours = self.memory_management["short_term_memory"]["duration_hours"]
        cutoff_time = datetime.utcnow() - timedelta(hours=recent_hours)

        recent_context = {}

        # Filter recent interactions
        if "interaction_history" in context:
            recent_interactions = []
            for interaction in context["interaction_history"]:
                if "timestamp" in interaction:
                    interaction_time = datetime.fromisoformat(interaction["timestamp"])
                    if interaction_time > cutoff_time:
                        recent_interactions.append(interaction)

            recent_context["recent_interactions"] = recent_interactions

        # Add current session info
        recent_context.update({
            "patient_id": context.get("patient_id"),
            "current_symptoms": context.get("current_symptoms", []),
            "urgency_level": context.get("urgency_level", "routine"),
            "emotional_state": context.get("emotional_state", "neutral")
        })

        return recent_context

    def _calculate_session_duration(self, context: Dict[str, Any]) -> Optional[str]:
        """Calculate session duration"""
        start_time_str = context.get("session_start")
        if not start_time_str:
            return None

        try:
            start_time = datetime.fromisoformat(start_time_str)
            duration = datetime.utcnow() - start_time

            hours = int(duration.total_seconds() // 3600)
            minutes = int((duration.total_seconds() % 3600) // 60)

            if hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"

        except Exception:
            return None

    async def clear_context(self, patient_id: str, clear_type: str = "session") -> Dict[str, Any]:
        """
        Clear context for a patient
        """
        try:
            if clear_type == "session":
                # Clear current session but keep history
                if patient_id in self.context_store:
                    context = self.context_store[patient_id]

                    # Save important session data to history
                    session_summary = self._create_context_summary(context)
                    self._add_to_session_history(patient_id, session_summary, "session_end")

                    # Clear session context
                    del self.context_store[patient_id]

                    logger.info(f"Session context cleared for patient {patient_id}")
                    return {"status": "session_cleared", "patient_id": patient_id}

            elif clear_type == "all":
                # Clear everything
                if patient_id in self.context_store:
                    del self.context_store[patient_id]
                if patient_id in self.session_history:
                    del self.session_history[patient_id]

                logger.info(f"All context cleared for patient {patient_id}")
                return {"status": "all_cleared", "patient_id": patient_id}

            return {"status": "not_found", "patient_id": patient_id}

        except Exception as e:
            logger.error(f"Error clearing context for patient {patient_id}: {e}")
            return {"status": "error", "error": str(e)}

    async def get_context_insights(self, patient_id: str) -> Dict[str, Any]:
        """
        Generate insights from patient context
        """
        try:
            context = self.context_store.get(patient_id, {})
            history = self.session_history.get(patient_id, [])

            if not context and not history:
                return {"insights": [], "patient_id": patient_id}

            insights = []

            # Analyze current symptoms
            symptoms = context.get("current_symptoms", [])
            if len(symptoms) > 3:
                insights.append({
                    "type": "symptom_complexity",
                    "message": f"Patient has reported {len(symptoms)} symptoms",
                    "priority": "medium"
                })

            # Analyze emotional state
            emotional_state = context.get("emotional_state", "neutral")
            if emotional_state in ["anxious", "distressed", "angry"]:
                insights.append({
                    "type": "emotional_concern",
                    "message": f"Patient appears {emotional_state}",
                    "priority": "high"
                })

            # Analyze urgency
            urgency = context.get("urgency_level", "routine")
            if urgency in ["urgent", "emergency"]:
                insights.append({
                    "type": "urgency_alert",
                    "message": f"High urgency level detected: {urgency}",
                    "priority": "critical"
                })

            # Analyze interaction patterns
            if len(history) > 10:
                recent_interactions = [h for h in history if self._is_recent(h.get("timestamp"))]
                if len(recent_interactions) > 5:
                    insights.append({
                        "type": "high_interaction_frequency",
                        "message": "High frequency of recent interactions",
                        "priority": "medium"
                    })

            return {
                "insights": insights,
                "patient_id": patient_id,
                "context_score": self._calculate_context_score(context, history),
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating context insights for patient {patient_id}: {e}")
            return {"error": str(e), "patient_id": patient_id}

    def _is_recent(self, timestamp_str: Optional[str], hours: int = 24) -> bool:
        """Check if timestamp is recent"""
        if not timestamp_str:
            return False

        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            return timestamp > cutoff
        except:
            return False

    def _calculate_context_score(self, context: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
        """Calculate context richness score"""
        score = 0.0

        # Context completeness
        key_fields = ["current_symptoms", "mentioned_conditions", "concerns_raised"]
        for field in key_fields:
            if context.get(field):
                score += 0.2

        # History richness
        if len(history) > 5:
            score += 0.2

        # Recent activity
        recent_history = [h for h in history if self._is_recent(h.get("timestamp"))]
        if recent_history:
            score += 0.2

        return min(1.0, score)

    async def export_context(self, patient_id: str, format: str = "json") -> Dict[str, Any]:
        """
        Export patient context for external use
        """
        try:
            context = self.context_store.get(patient_id, {})
            history = self.session_history.get(patient_id, [])

            export_data = {
                "patient_id": patient_id,
                "current_context": context,
                "session_history": history,
                "export_timestamp": datetime.utcnow().isoformat(),
                "format": format
            }

            if format == "json":
                return {
                    "status": "exported",
                    "data": export_data,
                    "size": len(json.dumps(export_data, default=str))
                }
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported format: {format}"
                }

        except Exception as e:
            logger.error(f"Error exporting context for patient {patient_id}: {e}")
            return {"status": "error", "error": str(e)}
