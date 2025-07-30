from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import json
import logging
import base64

from .models.breast_cancer_model import BreastCancerModel
from .utils.gradcam import GradCAM
from .utils.pdf_generator import PDFReportGenerator
from .utils.image_processor import ImageProcessor
from .database import connect_to_mongo, close_mongo_connection
from .services.prediction_service import PredictionService
from .models.prediction import (
    PredictionHistoryCreate,
    PredictionHistoryResponse,
    PredictionResult,
    ImageMetadata,
    ModelInfo
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Deep learning-based breast cancer prediction from mammogram images",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
grad_cam = None
pdf_generator = None
image_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and utilities on startup"""
    global model, grad_cam, pdf_generator, image_processor
    
    logger.info("Connecting to MongoDB...")
    await connect_to_mongo()
    
    logger.info("Initializing breast cancer prediction model...")
    model = BreastCancerModel()
    await model.load_model()
    
    logger.info("Initializing Grad-CAM visualization...")
    grad_cam = GradCAM(model.model)
    
    logger.info("Initializing PDF report generator...")
    pdf_generator = PDFReportGenerator()
    
    logger.info("Initializing image processor...")
    image_processor = ImageProcessor()
    
    logger.info("Startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Closing MongoDB connection...")
    await close_mongo_connection()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Breast Cancer Prediction API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict_breast_cancer(file: UploadFile = File(...)):
    """
    Predict breast cancer from mammogram image and save to history
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Process image for prediction
        processed_image = image_processor.preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        
        # Generate Grad-CAM heatmap
        heatmap = grad_cam.generate_heatmap(processed_image)
        
        # Save Grad-CAM heatmap
        gradcam_filename = f"gradcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        gradcam_path = f"/tmp/{gradcam_filename}"
        plt.imsave(gradcam_path, heatmap, cmap='jet')
        
        # Convert heatmap to base64 for response
        plt.figure(figsize=(8, 8))
        plt.imshow(heatmap, cmap='jet', alpha=0.6)
        plt.imshow(processed_image.squeeze(), cmap='gray', alpha=0.4)
        plt.axis('off')
        plt.tight_layout()
        
        # Save overlay image
        overlay_path = f"/tmp/overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
        # Convert overlay to base64
        with open(overlay_path, "rb") as img_file:
            overlay_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Create prediction result
        prediction_result = PredictionResult(
            probability=float(prediction[0]),
            classification="Malignant" if prediction[0] > 0.5 else "Benign",
            confidence=float(abs(prediction[0] - 0.5) * 2)
        )
        
        # Create image metadata
        image_metadata = ImageMetadata(
            filename=file.filename or "unknown.jpg",
            size=f"{image.width}x{image.height}",
            content_type=file.content_type or "image/jpeg",
            file_size=len(contents)
        )
        
        # Save to database
        history_data = PredictionHistoryCreate(
            prediction=prediction_result,
            image_metadata=image_metadata,
            model_info=ModelInfo(),
            gradcam_path=gradcam_path
        )
        
        saved_history = await PredictionService.create_prediction_history(history_data)
        
        # Prepare response
        result = {
            "id": str(saved_history.id),
            "prediction": {
                "probability": float(prediction[0]),
                "classification": "Malignant" if prediction[0] > 0.5 else "Benign",
                "confidence": float(abs(prediction[0] - 0.5) * 2)
            },
            "metadata": {
                "filename": file.filename,
                "upload_time": datetime.now().isoformat(),
                "model_version": "1.0.0"
            },
            "gradcam_overlay": overlay_base64
        }
        
        # Cleanup temporary files
        try:
            os.remove(overlay_path)
        except:
            pass
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/analyze-and-report")
async def analyze_and_generate_report(file: UploadFile = File(...)):
    """
    Analyze mammogram image and generate comprehensive PDF report
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Process image for prediction
        processed_image = image_processor.preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        
        # Generate Grad-CAM heatmap
        heatmap = grad_cam.generate_heatmap(processed_image)
        
        # Prepare analysis data
        analysis_data = {
            "patient_info": {
                "filename": file.filename,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_size": f"{image.width}x{image.height}"
            },
            "prediction": {
                "probability": float(prediction[0]),
                "classification": "Malignant" if prediction[0] > 0.5 else "Benign",
                "confidence": float(abs(prediction[0] - 0.5) * 2)
            },
            "model_info": {
                "model_version": "1.0.0",
                "architecture": "CNN with Transfer Learning",
                "training_data": "Mammography dataset"
            }
        }
        
        # Generate PDF report
        pdf_path = await pdf_generator.generate_report(
            analysis_data, 
            original_image=image,
            heatmap=heatmap
        )
        
        # Return PDF file
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"breast_cancer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
    except Exception as e:
        logger.error(f"Analysis and report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/gradcam")
async def generate_gradcam(file: UploadFile = File(...)):
    """
    Generate Grad-CAM heatmap for uploaded image
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = image_processor.preprocess_image(image)
        
        # Generate Grad-CAM heatmap
        heatmap = grad_cam.generate_heatmap(processed_image)
        
        # Save heatmap temporarily
        heatmap_path = f"/tmp/gradcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.imsave(heatmap_path, heatmap, cmap='jet')
        
        return FileResponse(
            heatmap_path,
            media_type="image/png",
            filename="gradcam_heatmap.png"
        )
        
    except Exception as e:
        logger.error(f"Grad-CAM generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}")

# Prediction History Endpoints

@app.get("/history", response_model=List[PredictionHistoryResponse])
async def get_prediction_history(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    classification: Optional[str] = Query(None, regex="^(Benign|Malignant)$"),
    patient_id: Optional[str] = Query(None)
):
    """Get prediction history with optional filters"""
    try:
        predictions = await PredictionService.get_prediction_history(
            limit=limit,
            offset=offset,
            classification_filter=classification,
            patient_id_filter=patient_id
        )
        
        return [
            PredictionHistoryResponse(
                id=str(pred.id),
                prediction=pred.prediction,
                image_metadata=pred.image_metadata,
                model_info=pred.model_info,
                created_at=pred.created_at,
                updated_at=pred.updated_at,
                notes=pred.notes,
                patient_id=pred.patient_id,
                gradcam_path=pred.gradcam_path
            )
            for pred in predictions
        ]
    except Exception as e:
        logger.error(f"Error fetching prediction history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.get("/history/{prediction_id}", response_model=PredictionHistoryResponse)
async def get_prediction_by_id(prediction_id: str):
    """Get a specific prediction by ID"""
    try:
        prediction = await PredictionService.get_prediction_by_id(prediction_id)
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        return PredictionHistoryResponse(
            id=str(prediction.id),
            prediction=prediction.prediction,
            image_metadata=prediction.image_metadata,
            model_info=prediction.model_info,
            created_at=prediction.created_at,
            updated_at=prediction.updated_at,
            notes=prediction.notes,
            patient_id=prediction.patient_id,
            gradcam_path=prediction.gradcam_path
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch prediction: {str(e)}")

@app.put("/history/{prediction_id}/notes")
async def update_prediction_notes(prediction_id: str, notes: Dict[str, str]):
    """Update notes for a prediction"""
    try:
        updated_prediction = await PredictionService.update_prediction_notes(
            prediction_id, notes.get("notes", "")
        )
        if not updated_prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        return {"message": "Notes updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating notes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update notes: {str(e)}")

@app.delete("/history/{prediction_id}")
async def delete_prediction(prediction_id: str):
    """Delete a prediction history record"""
    try:
        success = await PredictionService.delete_prediction(prediction_id)
        if not success:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        return {"message": "Prediction deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete prediction: {str(e)}")

@app.get("/statistics")
async def get_statistics():
    """Get prediction statistics"""
    try:
        stats = await PredictionService.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error fetching statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch statistics: {str(e)}")

@app.get("/history/{prediction_id}/gradcam")
async def get_gradcam_image(prediction_id: str):
    """Get Grad-CAM heatmap for a specific prediction"""
    try:
        prediction = await PredictionService.get_prediction_by_id(prediction_id)
        if not prediction or not prediction.gradcam_path:
            raise HTTPException(status_code=404, detail="Grad-CAM image not found")
        
        if os.path.exists(prediction.gradcam_path):
            return FileResponse(
                prediction.gradcam_path,
                media_type="image/png",
                filename=f"gradcam_{prediction_id}.png"
            )
        else:
            raise HTTPException(status_code=404, detail="Grad-CAM image file not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching Grad-CAM image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch Grad-CAM image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)