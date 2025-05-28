"""
Comprehensive application settings with latest AI configurations
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    """Application settings with AI and infrastructure configuration"""

    # Application
    APP_NAME: str = "AI Healthcare Platform"
    APP_VERSION: str = "4.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "your-jwt-secret-key-here")
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", "your-32-character-encryption-key")
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "0.0.0.0", "*.healthcare-ai.com"]
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://app.healthcare-ai.com"
    ]

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./healthcare.db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Vector Databases
    CHROMADB_HOST: str = os.getenv("CHROMADB_HOST", "localhost")
    CHROMADB_PORT: int = int(os.getenv("CHROMADB_PORT", "8001"))
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

    # AI Services
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")

    # Model Configuration
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))

    # Feature Flags
    ENABLE_BLOCKCHAIN: bool = os.getenv("ENABLE_BLOCKCHAIN", "false").lower() == "true"
    ENABLE_FEDERATED_LEARNING: bool = os.getenv("ENABLE_FEDERATED_LEARNING", "false").lower() == "true"
    ENABLE_SYNTHETIC_DATA: bool = os.getenv("ENABLE_SYNTHETIC_DATA", "true").lower() == "true"
    ENABLE_MULTIMODAL: bool = os.getenv("ENABLE_MULTIMODAL", "true").lower() == "true"
    ENABLE_QUANTUM_ML: bool = os.getenv("ENABLE_QUANTUM_ML", "false").lower() == "true"
    ENABLE_EDGE_AI: bool = os.getenv("ENABLE_EDGE_AI", "true").lower() == "true"

    # Performance
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
    RATE_LIMIT_PER_HOUR: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))

    # Monitoring
    PROMETHEUS_ENABLED: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
    GRAFANA_ENABLED: bool = os.getenv("GRAFANA_ENABLED", "true").lower() == "true"
    SENTRY_DSN: str = os.getenv("SENTRY_DSN", "")

    # External APIs
    FHIR_SERVER_URL: str = os.getenv("FHIR_SERVER_URL", "https://hapi.fhir.org/baseR4")
    HL7_ENABLED: bool = os.getenv("HL7_ENABLED", "false").lower() == "true"

    # Healthcare Specific
    SUPPORTED_LANGUAGES: List[str] = [
        "en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "hi", "ar", "ru", "nl", "sv"
    ]

    # AI Agent Configuration
    AGENT_TIMEOUT: int = int(os.getenv("AGENT_TIMEOUT", "30"))
    MAX_CONCURRENT_AGENTS: int = int(os.getenv("MAX_CONCURRENT_AGENTS", "10"))
    AGENT_RETRY_ATTEMPTS: int = int(os.getenv("AGENT_RETRY_ATTEMPTS", "3"))

    @validator("OPENAI_API_KEY")
    def validate_openai_key(cls, v):
        if not v or v == "your_openai_api_key_here":
            print("⚠️ Warning: OPENAI_API_KEY not configured. AI features will be limited.")
        return v

    @validator("ENCRYPTION_KEY")
    def validate_encryption_key(cls, v):
        if len(v) < 32:
            print("⚠️ Warning: ENCRYPTION_KEY should be at least 32 characters.")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()
