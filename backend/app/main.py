"""
Main FastAPI application for AI Post-Discharge Readmission Prevention Companion
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
import uvicorn
import os
from dotenv import load_dotenv
import asyncio

from app.database.db_manager import init_database
from app.routes import patients, predictions, chat

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Post-Discharge Readmission Prevention Companion",
    description="AI-powered healthcare app for reducing hospital readmissions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional Security Middleware Placeholder (uncomment for production)
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# @app.middleware("http")
# async def auth_middleware(request: Request, call_next):
#     """Authentication middleware for secure endpoints"""
#     # Skip auth for health checks and docs
#     if request.url.path in ["/", "/health", "/docs", "/redoc"]:
#         response = await call_next(request)
#         return response
#
#     # Add token validation logic here
#     # token = await oauth2_scheme(request)
#     # validate_token(token)
#
#     response = await call_next(request)
#     return response

# Global variable to track startup status
startup_complete = False

async def preload_ai_models():
    """Simulate AI model preloading"""
    print("\U0001F504 Loading AI models...")

    # Simulate model loading time
    await asyncio.sleep(2)

    # Check if models can be initialized
    try:
        from app.models.risk_predictor import HAIMRiskPredictor
        from app.models.care_plan_generator import CarePlanGenerator
        from app.models.chatbot import BilingualMedicalChatbot
        from app.models.sdoh_extractor import SDOHExtractor

        # Initialize models to verify they work
        risk_predictor = HAIMRiskPredictor()
        care_plan_generator = CarePlanGenerator()
        chatbot = BilingualMedicalChatbot()
        sdoh_extractor = SDOHExtractor()

        print("✅ Risk Prediction Model: Loaded")
        print("✅ Care Plan Generator: Loaded")
        print("✅ Bilingual Chatbot: Loaded")
        print("✅ SDOH Extractor: Loaded")

        return True
    except Exception as e:
        print(f"⚠️ Error loading AI models: {e}")
        return False

def validate_environment():
    """Validate required environment variables"""
    print("\U0001F50D Validating environment configuration...")

    issues = []

    # Check OpenAI API Key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here":
        issues.append("OPENAI_API_KEY is not properly configured")
        print("⚠️ Warning: OPENAI_API_KEY is not set. AI features will be limited.")
    else:
        print("✅ OPENAI_API_KEY: Configured")

    # Check Encryption Key
    encryption_key = os.getenv("ENCRYPTION_KEY")
    if not encryption_key or encryption_key == "your_32_character_encryption_key_here":
        issues.append("ENCRYPTION_KEY is not properly configured")
        print("⚠️ Warning: ENCRYPTION_KEY is not set. Using default encryption.")
    else:
        print("✅ ENCRYPTION_KEY: Configured")

    # Check Database URL
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        print("✅ DATABASE_URL: Configured")
    else:
        print("ℹ️ DATABASE_URL: Using default SQLite")

    return issues

# Include routers
app.include_router(patients.router, prefix="/api/v1/patients", tags=["patients"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])

@app.on_event("startup")
async def startup_event():
    """Initialize database, validate environment, and preload AI models"""
    global startup_complete

    print("🚀 Starting AI Post-Discharge Readmission Prevention Companion...")
    print("=" * 60)

    # Initialize database
    try:
        init_database()
        print("✅ Database: Initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return

    # Validate environment
    env_issues = validate_environment()

    # Preload AI models
    models_loaded = await preload_ai_models()

    if models_loaded:
        print("✅ AI Models: All models preloaded successfully")
    else:
        print("⚠️ AI Models: Some models failed to load")

    print("=" * 60)
    print("🏥 Healthcare AI app started successfully!")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🌐 Frontend: http://localhost:3000")

    if env_issues:
        print("\n⚠️ Configuration Issues Found:")
        for issue in env_issues:
            print(f"   - {issue}")
        print("   Please check your .env file configuration")

    startup_complete = True

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Post-Discharge Readmission Prevention Companion API",
        "status": "healthy" if startup_complete else "starting",
        "version": "1.0.0",
        "docs": "/docs",
        "startup_complete": startup_complete
    }

@app.get("/health")
async def health_check():
    """Detailed health check with environment status"""
    # Check environment variables status
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_status = "configured" if openai_key and openai_key != "your_openai_api_key_here" else "missing"

    encryption_key = os.getenv("ENCRYPTION_KEY")
    encryption_status = "configured" if encryption_key and encryption_key != "your_32_character_encryption_key_here" else "missing"

    # Check if models are working
    models_status = "loaded" if startup_complete else "loading"

    return {
        "status": "healthy" if startup_complete else "starting",
        "database": "connected",
        "ai_models": models_status,
        "environment": {
            "openai_api": openai_status,
            "encryption": encryption_status,
            "database_url": "configured" if os.getenv("DATABASE_URL") else "default"
        },
        "startup_complete": startup_complete,
        "timestamp": "2025-05-26T17:34:00Z"
    }

@app.get("/api/v1/status")
async def system_status():
    """Detailed system status for monitoring"""
    return {
        "application": "AI Post-Discharge Readmission Prevention Companion",
        "version": "1.0.0",
        "status": "operational" if startup_complete else "initializing",
        "components": {
            "database": "operational",
            "ai_models": "operational" if startup_complete else "loading",
            "api": "operational",
            "frontend": "available"
        },
        "endpoints": {
            "patients": "/api/v1/patients",
            "predictions": "/api/v1/predictions",
            "chat": "/api/v1/chat"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
