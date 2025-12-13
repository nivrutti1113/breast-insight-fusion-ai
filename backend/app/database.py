import os
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from dotenv import load_dotenv
import logging

from .models.prediction import PredictionHistory

load_dotenv()
logger = logging.getLogger(__name__)

class Database:
    client: AsyncIOMotorClient = None

database = Database()

async def connect_to_mongo():
    """Create database connection"""
    try:
        # Get MongoDB connection string from environment or use default
        mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        database_name = os.getenv("DATABASE_NAME", "breast_cancer_detection")
        
        logger.info(f"Connecting to MongoDB at {mongodb_url}")
        
        # Create Motor client
        database.client = AsyncIOMotorClient(mongodb_url)
        
        # Initialize Beanie with the Product document class
        await init_beanie(
            database=database.client[database_name],
            document_models=[PredictionHistory]
        )
        
        logger.info("Successfully connected to MongoDB")
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

async def close_mongo_connection():
    """Close database connection"""
    if database.client:
        database.client.close()
        logger.info("Disconnected from MongoDB")