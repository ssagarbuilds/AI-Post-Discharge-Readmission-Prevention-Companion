"""
AI Healthcare Platform v4.0
Advanced post-discharge prevention with 25+ AI agents
"""

__version__ = "4.0.0"
__author__ = "AI Healthcare Platform Team"
__description__ = "Comprehensive AI-powered healthcare platform for post-discharge care"

# Import core modules for easy access
from app.config.settings import settings
from app.database.db_manager import get_database
from app.models.infrastructure.multiagent_orchestrator import MultiAgentOrchestrator

__all__ = [
    "settings",
    "get_database",
    "MultiAgentOrchestrator"
]
