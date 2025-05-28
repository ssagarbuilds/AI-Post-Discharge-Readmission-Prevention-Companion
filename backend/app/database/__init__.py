"""
Database Package
Database connections, models, and data management
"""

from app.database.db_manager import DatabaseManager, get_db_manager

__all__ = [
    "DatabaseManager",
    "get_db_manager"
]

# Database metadata
__version__ = "1.0.0"
__description__ = "Healthcare AI Platform Database Layer"

# Initialize global database manager instance
db_manager = None

def initialize_database():
    """Initialize the global database manager"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

def get_database():
    """Get the global database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = initialize_database()
    return db_manager

# Export for convenience
get_db = get_database
