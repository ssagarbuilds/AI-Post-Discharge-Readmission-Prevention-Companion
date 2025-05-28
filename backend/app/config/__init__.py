"""
Configuration Package
Application configuration and settings.py management
"""

from app.config.settings import Settings, settings

__all__ = [
    "Settings",
    "settings.py"
]

# Configuration metadata
__version__ = "1.0.0"
__description__ = "Healthcare AI Platform Configuration"

# Export commonly used settings.py for convenience
DATABASE_URL = settings.DATABASE_URL
REDIS_URL = settings.REDIS_URL
OPENAI_API_KEY = settings.OPENAI_API_KEY
ANTHROPIC_API_KEY = settings.ANTHROPIC_API_KEY

# Feature flags
ENABLE_MULTIMODAL = settings.ENABLE_MULTIMODAL
ENABLE_FEDERATED_LEARNING = settings.ENABLE_FEDERATED_LEARNING
ENABLE_QUANTUM_ML = settings.ENABLE_QUANTUM_ML
ENABLE_BLOCKCHAIN_LOGGING = settings.ENABLE_BLOCKCHAIN_LOGGING

# Environment info
ENVIRONMENT = settings.ENVIRONMENT
DEBUG = settings.DEBUG

