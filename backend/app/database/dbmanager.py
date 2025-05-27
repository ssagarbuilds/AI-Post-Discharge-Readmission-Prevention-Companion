"""
Advanced database manager with vector database integration
"""

import sqlite3
import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from contextlib import asynccontextmanager

import chromadb
from qdrant_client import QdrantClient
from qdrant_client.http import models
import redis

from app.config.settings import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Unified database manager for SQL, NoSQL, and Vector databases
    """

    def __init__(self):
        self.sql_db_path = "healthcare.db"
        self.redis_client = None
        self.chroma_client = None
        self.qdrant_client = None
        self._initialize_connections()

    def _initialize_connections(self):
        """Initialize all database connections"""
        try:
            # Redis connection
            self.redis_client = redis.from_url(settings.REDIS_URL)
            logger.info("✅ Redis connected")

            # ChromaDB connection
            self.chroma_client = chromadb.HttpClient(
                host=settings.CHROMADB_HOST,
                port=settings.CHROMADB_PORT
            )
            logger.info("✅ ChromaDB connected")

            # Qdrant connection
            self.qdrant_client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT
            )
            logger.info("✅ Qdrant connected")

        except Exception as e:
            logger.error(f"❌ Database connection error: {e}")

    async def init_database(self):
        """Initialize all database schemas and collections"""
        await self._init_sql_schema()
        await self._init_vector_collections()
        logger.info("✅ All databases initialized")

    async def _init_sql_schema(self):
        """Initialize SQLite schema"""
        conn = sqlite3.connect(self.sql_db_path)
        cursor = conn.cursor()

        # Patients table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS patients (
                                                               id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                               patient_id TEXT UNIQUE NOT NULL,
                                                               name TEXT,
                                                               age INTEGER,
                                                               gender TEXT,
                                                               conditions TEXT,
                                                               medications TEXT,
                                                               discharge_notes TEXT,
                                                               risk_score REAL,
                                                               risk_level TEXT,
                                                               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                                               updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                       )
                       ''')

        # Risk assessments table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS risk_assessments (
                                                                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                       patient_id TEXT,
                                                                       assessment_type TEXT,
                                                                       risk_score REAL,
                                                                       risk_factors TEXT,
                                                                       recommendations TEXT,
                                                                       model_version TEXT,
                                                                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                                                       FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                           )
                       ''')

        # Care plans table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS care_plans (
                                                                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                 patient_id TEXT,
                                                                 plan_type TEXT,
                                                                 content TEXT,
                                                                 language TEXT,
                                                                 status TEXT DEFAULT 'active',
                                                                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                                                 FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                           )
                       ''')

        # Chat history table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS chat_history (
                                                                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                   patient_id TEXT,
                                                                   message TEXT,
                                                                   response TEXT,
                                                                   language TEXT,
                                                                   agent_used TEXT,
                                                                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                       )
                       ''')

        # Audit log table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS audit_log (
                                                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                event_type TEXT,
                                                                user_id TEXT,
                                                                patient_id TEXT,
                                                                details TEXT,
                                                                ip_address TEXT,
                                                                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                       )
                       ''')

        # Wearable data table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS wearable_data (
                                                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                    patient_id TEXT,
                                                                    device_type TEXT,
                                                                    data_type TEXT,
                                                                    value REAL,
                                                                    unit TEXT,
                                                                    timestamp TIMESTAMP,
                                                                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                       )
                       ''')

        # AI model performance table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS model_performance (
                                                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                        model_name TEXT,
                                                                        version TEXT,
                                                                        accuracy REAL,
                                                                        precision_score REAL,
                                                                        recall REAL,
                                                                        f1_score REAL,
                                                                        evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                       )
                       ''')

        conn.commit()
        conn.close()

    async def _init_vector_collections(self):
        """Initialize vector database collections"""
        try:
            # ChromaDB collections
            collections = [
                "patient_embeddings",
                "medical_knowledge",
                "care_plans",
                "clinical_notes"
            ]

            for collection_name in collections:
                try:
                    self.chroma_client.get_collection(collection_name)
                except:
                    self.chroma_client.create_collection(
                        name=collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )

            # Qdrant collections
            qdrant_collections = [
                {
                    "name": "patient_vectors",
                    "vector_size": 384,
                    "distance": models.Distance.COSINE
                },
                {
                    "name": "medical_embeddings",
                    "vector_size": 1536,
                    "distance": models.Distance.COSINE
                }
            ]

            for collection in qdrant_collections:
                try:
                    self.qdrant_client.get_collection(collection["name"])
                except:
                    self.qdrant_client.create_collection(
                        collection_name=collection["name"],
                        vectors_config=models.VectorParams(
                            size=collection["vector_size"],
                            distance=collection["distance"]
                        )
                    )

            logger.info("✅ Vector collections initialized")

        except Exception as e:
            logger.error(f"❌ Vector database initialization error: {e}")

    # Patient operations
    async def add_patient(self, patient_data: Dict[str, Any]) -> str:
        """Add new patient to database"""
        conn = sqlite3.connect(self.sql_db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       INSERT INTO patients (patient_id, name, age, gender, conditions, medications, discharge_notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ''', (
                           patient_data.get('patient_id'),
                           patient_data.get('name'),
                           patient_data.get('age'),
                           patient_data.get('gender'),
                           json.dumps(patient_data.get('conditions', [])),
                           json.dumps(patient_data.get('medications', [])),
                           patient_data.get('discharge_notes', '')
                       ))

        patient_id = patient_data.get('patient_id')
        conn.commit()
        conn.close()

        # Cache in Redis
        await self._cache_patient(patient_id, patient_data)

        return patient_id

    async def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient by ID with caching"""
        # Try cache first
        cached = await self._get_cached_patient(patient_id)
        if cached:
            return cached

        # Query database
        conn = sqlite3.connect(self.sql_db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            patient_data = {
                'id': row[0],
                'patient_id': row[1],
                'name': row[2],
                'age': row[3],
                'gender': row[4],
                'conditions': json.loads(row[5]) if row[5] else [],
                'medications': json.loads(row[6]) if row[6] else [],
                'discharge_notes': row[7],
                'risk_score': row[8],
                'risk_level': row[9],
                'created_at': row[10],
                'updated_at': row[11]
            }

            # Cache for future use
            await self._cache_patient(patient_id, patient_data)
            return patient_data

        return None

    async def _cache_patient(self, patient_id: str, patient_data: Dict[str, Any]):
        """Cache patient data in Redis"""
        try:
            self.redis_client.setex(
                f"patient:{patient_id}",
                3600,  # 1 hour TTL
                json.dumps(patient_data, default=str)
            )
        except Exception as e:
            logger.error(f"Cache error: {e}")

    async def _get_cached_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get cached patient data"""
        try:
            cached = self.redis_client.get(f"patient:{patient_id}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        return None

    # Vector operations
    async def add_patient_embedding(self, patient_id: str, embedding: List[float], metadata: Dict[str, Any]):
        """Add patient embedding to vector database"""
        try:
            # Add to ChromaDB
            self.chroma_client.get_collection("patient_embeddings").add(
                embeddings=[embedding],
                documents=[json.dumps(metadata)],
                ids=[patient_id]
            )

            # Add to Qdrant
            self.qdrant_client.upsert(
                collection_name="patient_vectors",
                points=[
                    models.PointStruct(
                        id=patient_id,
                        vector=embedding,
                        payload=metadata
                    )
                ]
            )

        except Exception as e:
            logger.error(f"Vector embedding error: {e}")

    async def search_similar_patients(self, embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar patients using vector similarity"""
        try:
            results = self.qdrant_client.search(
                collection_name="patient_vectors",
                query_vector=embedding,
                limit=limit
            )

            return [
                {
                    "patient_id": result.id,
                    "score": result.score,
                    "metadata": result.payload
                }
                for result in results
            ]

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    # Analytics operations
    async def get_population_stats(self) -> Dict[str, Any]:
        """Get population health statistics"""
        conn = sqlite3.connect(self.sql_db_path)
        cursor = conn.cursor()

        # Basic stats
        cursor.execute('SELECT COUNT(*) FROM patients')
        total_patients = cursor.fetchone()[0]

        cursor.execute('SELECT AVG(age) FROM patients WHERE age IS NOT NULL')
        avg_age = cursor.fetchone()[0] or 0

        cursor.execute('SELECT risk_level, COUNT(*) FROM patients WHERE risk_level IS NOT NULL GROUP BY risk_level')
        risk_distribution = dict(cursor.fetchall())

        # Recent assessments
        cursor.execute('SELECT COUNT(*) FROM risk_assessments WHERE created_at > datetime("now", "-7 days")')
        recent_assessments = cursor.fetchone()[0]

        conn.close()

        return {
            "total_patients": total_patients,
            "average_age": round(avg_age, 1),
            "risk_distribution": risk_distribution,
            "recent_assessments": recent_assessments,
            "last_updated": datetime.utcnow().isoformat()
        }

    async def log_audit_event(self, event_type: str, user_id: str = None, patient_id: str = None,
                              details: Dict[str, Any] = None, ip_address: str = None):
        """Log audit event"""
        conn = sqlite3.connect(self.sql_db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       INSERT INTO audit_log (event_type, user_id, patient_id, details, ip_address)
                       VALUES (?, ?, ?, ?, ?)
                       ''', (
                           event_type,
                           user_id,
                           patient_id,
                           json.dumps(details or {}),
                           ip_address
                       ))

        conn.commit()
        conn.close()

# Global database manager instance
db_manager = DatabaseManager()

async def init_database():
    """Initialize database"""
    await db_manager.init_database()

def get_database():
    """Get database manager instance"""
    return db_manager
