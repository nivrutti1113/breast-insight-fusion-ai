from beanie import Document
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any
from bson import ObjectId

class PredictionResult(BaseModel):
    """Prediction result data structure"""
    probability: float = Field(..., description="Prediction probability (0-1)")
    classification: str = Field(..., description="Classification result (Benign/Malignant)")
    confidence: float = Field(..., description="Confidence score (0-1)")

class ImageMetadata(BaseModel):
    """Image metadata structure"""
    filename: str = Field(..., description="Original filename")
    size: str = Field(..., description="Image dimensions")
    content_type: str = Field(..., description="MIME type")
    file_size: int = Field(..., description="File size in bytes")

class ModelInfo(BaseModel):
    """Model information structure"""
    version: str = Field(default="1.0.0", description="Model version")
    architecture: str = Field(default="CNN with Transfer Learning", description="Model architecture")
    training_data: str = Field(default="Mammography dataset", description="Training dataset info")

class PredictionHistory(Document):
    """Document model for storing prediction history in MongoDB"""
    
    # Prediction details
    prediction: PredictionResult = Field(..., description="Prediction results")
    
    # Image information
    image_metadata: ImageMetadata = Field(..., description="Image metadata")
    
    # Model information
    model_info: ModelInfo = Field(default_factory=ModelInfo, description="Model information")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    # Optional fields
    notes: Optional[str] = Field(None, description="Optional notes")
    patient_id: Optional[str] = Field(None, description="Optional patient identifier")
    
    # Grad-CAM heatmap path (stored as relative path)
    gradcam_path: Optional[str] = Field(None, description="Path to Grad-CAM heatmap image")
    
    class Settings:
        name = "prediction_history"
        indexes = [
            "created_at",
            "prediction.classification",
            "patient_id",
        ]

    def dict(self, **kwargs) -> Dict[str, Any]:
        """Override dict method to include string representation of ObjectId"""
        data = super().dict(**kwargs)
        if hasattr(self, 'id') and self.id:
            data['id'] = str(self.id)
        return data

class PredictionHistoryResponse(BaseModel):
    """Response model for prediction history API"""
    id: str
    prediction: PredictionResult
    image_metadata: ImageMetadata
    model_info: ModelInfo
    created_at: datetime
    updated_at: datetime
    notes: Optional[str] = None
    patient_id: Optional[str] = None
    gradcam_path: Optional[str] = None

class PredictionHistoryCreate(BaseModel):
    """Request model for creating prediction history"""
    prediction: PredictionResult
    image_metadata: ImageMetadata
    model_info: Optional[ModelInfo] = None
    notes: Optional[str] = None
    patient_id: Optional[str] = None
    gradcam_path: Optional[str] = None