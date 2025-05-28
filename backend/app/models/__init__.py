"""
AI Models Package - 25+ Specialized Healthcare Agents
"""

from app.models.foundation.multimodal_llm import MultimodalLLM
from app.models.foundation.medical_reasoning import MedicalReasoningEngine
from app.models.foundation.context_awareness import ContextAwarenessAgent

from app.models.clinical.risk_predictor import RiskPredictor
from app.models.clinical.care_plan_generator import CarePlanGenerator
from app.models.clinical.early_disease import EarlyDiseasePredictor
from app.models.clinical.symptom_triage import SymptomTriageAgent
from app.models.clinical.cognitive_agent import CognitiveAssessmentAgent
from app.models.clinical.imaging_agent import ImagingAnalysisAgent

from app.models.engagement.chatbot import HealthcareAssistant
from app.models.engagement.ambient_scribe import AmbientScribe
from app.models.engagement.feedback_agent import FeedbackAnalysisAgent

from app.models.analytics.population_analytics import PopulationAnalyticsAgent
from app.models.analytics.sdoh_extractor import SDOHExtractor
from app.models.analytics.xai_agent import ExplainableAIAgent

from app.models.operations.admin_agent import AdminAutomationAgent
from app.models.operations.chronic_agent import ChronicCareAgent
from app.models.operations.prevention_agent import PreventionAgent
from app.models.operations.adherence_agent import AdherenceAgent
from app.models.operations.wearable_agent import WearableDataAgent

from app.models.infrastructure.rag_agent import RAGAgent
from app.models.infrastructure.blockchain_logger import BlockchainLogger
from app.models.infrastructure.synthetic_data_agent import SyntheticDataAgent
from app.models.infrastructure.federated_learning import FederatedLearningCoordinator
from app.models.infrastructure.multiagent_orchestrator import MultiAgentOrchestrator

from app.models.emerging.quantum_ml import QuantumMLAgent
from app.models.emerging.neuromorphic_ai import NeuromorphicAgent
from app.models.emerging.swarm_intelligence import SwarmIntelligenceAgent
from app.models.emerging.edge_ai_optimizer import EdgeAIOptimizer

__all__ = [
    # Foundation
    "MultimodalLLM", "MedicalReasoningEngine", "ContextAwarenessAgent",

    # Clinical
    "RiskPredictor", "CarePlanGenerator", "EarlyDiseasePredictor",
    "SymptomTriageAgent", "CognitiveAssessmentAgent", "ImagingAnalysisAgent",

    # Engagement
    "HealthcareAssistant", "AmbientScribe", "FeedbackAnalysisAgent",

    # Analytics
    "PopulationAnalyticsAgent", "SDOHExtractor", "ExplainableAIAgent",

    # Operations
    "AdminAutomationAgent", "ChronicCareAgent", "PreventionAgent",
    "AdherenceAgent", "WearableDataAgent",

    # Infrastructure
    "RAGAgent", "BlockchainLogger", "SyntheticDataAgent",
    "FederatedLearningCoordinator", "MultiAgentOrchestrator",

    # Emerging
    "QuantumMLAgent", "NeuromorphicAgent", "SwarmIntelligenceAgent", "EdgeAIOptimizer"
]
