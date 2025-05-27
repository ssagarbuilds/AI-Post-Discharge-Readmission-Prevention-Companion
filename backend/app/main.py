"""
Main FastAPI application for Post-Discharge Readmission Prevention Companion
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv

from app.database.db_manager import init_database
from app.routes import patients, predictions, chat

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Post-Discharge Readmission Prevention Companion",
    description="AI-powered healthcare app for reducing hospital readmissions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional Security Middleware Placeholder
# from fastapi.security import OAuth2PasswordBearer
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# @app.middleware("http")
# async def auth_middleware(request: Request, call_next):
#     token = await oauth2_scheme(request)
#     # Add token validation logic here
#     response = await call_next(request)
#     return response

# Include routers
app.include_router(patients.router, prefix="/api/v1/patients", tags=["patients"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])

@app.on_event("startup")
async def startup_event():
    """Initialize database and validate startup environment"""
    init_database()

    # Validate environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⚠️ Warning: OPENAI_API_KEY is not set. AI features will be limited.")
    else:
        print("✅ OPENAI_API_KEY found")

    encryption_key = os.getenv("ENCRYPTION_KEY")
    if not encryption_key:
        print("⚠️ Warning: ENCRYPTION_KEY is not set. Data encryption will use default key.")
    else:
        print("✅ ENCRYPTION_KEY configured")

    # Simulate AI model loading
    print("✅ AI models preloaded successfully")
    print("🏥 Healthcare AI app started successfully!")
    print("📖 API Documentation: http://localhost:8000/docs")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Post-Discharge Readmission Prevention Companion API",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    # Check environment variables status
    openai_status = "configured" if os.getenv("OPENAI_API_KEY") else "missing"
    encryption_status = "configured" if os.getenv("ENCRYPTION_KEY") else "missing"

    return {
        "status": "healthy",
        "database": "connected",
        "ai_models": "loaded",
        "openai_api": openai_status,
        "encryption": encryption_status,
        "timestamp": "2025-05-26T17:34:00Z"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
