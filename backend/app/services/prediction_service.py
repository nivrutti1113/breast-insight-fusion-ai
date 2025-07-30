from typing import List, Optional
from datetime import datetime, timedelta
from bson import ObjectId

from ..models.prediction import (
    PredictionHistory, 
    PredictionHistoryCreate, 
    PredictionHistoryResponse,
    PredictionResult,
    ImageMetadata,
    ModelInfo
)

class PredictionService:
    """Service class for handling prediction history operations"""
    
    @staticmethod
    async def create_prediction_history(
        prediction_data: PredictionHistoryCreate
    ) -> PredictionHistory:
        """Create a new prediction history record"""
        
        prediction_history = PredictionHistory(
            prediction=prediction_data.prediction,
            image_metadata=prediction_data.image_metadata,
            model_info=prediction_data.model_info or ModelInfo(),
            notes=prediction_data.notes,
            patient_id=prediction_data.patient_id,
            gradcam_path=prediction_data.gradcam_path
        )
        
        await prediction_history.insert()
        return prediction_history
    
    @staticmethod
    async def get_prediction_history(
        limit: int = 50,
        offset: int = 0,
        classification_filter: Optional[str] = None,
        patient_id_filter: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[PredictionHistory]:
        """Get prediction history with optional filters"""
        
        query = {}
        
        # Add classification filter
        if classification_filter:
            query["prediction.classification"] = classification_filter
        
        # Add patient ID filter
        if patient_id_filter:
            query["patient_id"] = patient_id_filter
        
        # Add date range filter
        if date_from or date_to:
            date_query = {}
            if date_from:
                date_query["$gte"] = date_from
            if date_to:
                date_query["$lte"] = date_to
            query["created_at"] = date_query
        
        # Execute query with pagination
        predictions = await PredictionHistory.find(query)\
            .sort(-PredictionHistory.created_at)\
            .skip(offset)\
            .limit(limit)\
            .to_list()
        
        return predictions
    
    @staticmethod
    async def get_prediction_by_id(prediction_id: str) -> Optional[PredictionHistory]:
        """Get a specific prediction by ID"""
        try:
            return await PredictionHistory.get(ObjectId(prediction_id))
        except:
            return None
    
    @staticmethod
    async def update_prediction_notes(
        prediction_id: str, 
        notes: str
    ) -> Optional[PredictionHistory]:
        """Update notes for a prediction"""
        try:
            prediction = await PredictionHistory.get(ObjectId(prediction_id))
            if prediction:
                prediction.notes = notes
                prediction.updated_at = datetime.utcnow()
                await prediction.save()
                return prediction
            return None
        except:
            return None
    
    @staticmethod
    async def delete_prediction(prediction_id: str) -> bool:
        """Delete a prediction history record"""
        try:
            prediction = await PredictionHistory.get(ObjectId(prediction_id))
            if prediction:
                await prediction.delete()
                return True
            return False
        except:
            return False
    
    @staticmethod
    async def get_statistics() -> dict:
        """Get prediction statistics"""
        try:
            # Total predictions
            total = await PredictionHistory.count()
            
            # Predictions by classification
            benign_count = await PredictionHistory.find(
                {"prediction.classification": "Benign"}
            ).count()
            malignant_count = await PredictionHistory.find(
                {"prediction.classification": "Malignant"}
            ).count()
            
            # Recent predictions (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_count = await PredictionHistory.find(
                {"created_at": {"$gte": thirty_days_ago}}
            ).count()
            
            # Average confidence
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "avg_confidence": {"$avg": "$prediction.confidence"},
                        "avg_probability": {"$avg": "$prediction.probability"}
                    }
                }
            ]
            
            aggregation_result = await PredictionHistory.aggregate(pipeline).to_list()
            avg_confidence = 0.0
            avg_probability = 0.0
            
            if aggregation_result:
                avg_confidence = aggregation_result[0].get("avg_confidence", 0.0)
                avg_probability = aggregation_result[0].get("avg_probability", 0.0)
            
            return {
                "total_predictions": total,
                "benign_count": benign_count,
                "malignant_count": malignant_count,
                "recent_predictions": recent_count,
                "average_confidence": round(avg_confidence, 3),
                "average_probability": round(avg_probability, 3),
                "benign_percentage": round((benign_count / total * 100) if total > 0 else 0, 1),
                "malignant_percentage": round((malignant_count / total * 100) if total > 0 else 0, 1)
            }
        except Exception as e:
            return {
                "total_predictions": 0,
                "benign_count": 0,
                "malignant_count": 0,
                "recent_predictions": 0,
                "average_confidence": 0.0,
                "average_probability": 0.0,
                "benign_percentage": 0.0,
                "malignant_percentage": 0.0,
                "error": str(e)
            }