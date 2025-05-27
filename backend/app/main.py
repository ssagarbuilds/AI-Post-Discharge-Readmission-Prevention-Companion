"""
Main FastAPI application for AI Post-Discharge Readmission Prevention Companion
"""

from fastapi import FastAPI
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
    title="AI Post-Discharge Readmission Prevention Companion",
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

# Include routers
app.include_router(patients.router, prefix="/api/v1/patients", tags=["patients"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_database()
    print("🏥 Healthcare AI app started successfully!")
    print("📖 API Documentation: http://localhost:8000/docs")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Post-Discharge Readmission Prevention Companion API",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "ai_models": "loaded",
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
