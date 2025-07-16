from fastapi import FastAPI, File, UploadFile, HTTPException
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
from typing import Dict, Any
import json
import logging

from .models.breast_cancer_model import BreastCancerModel
from .utils.gradcam import GradCAM
from .utils.pdf_generator import PDFReportGenerator
from .utils.image_processor import ImageProcessor

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
    Predict breast cancer from mammogram image
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
        
        # Prepare response
        result = {
            "prediction": {
                "probability": float(prediction[0]),
                "classification": "Malignant" if prediction[0] > 0.5 else "Benign",
                "confidence": float(abs(prediction[0] - 0.5) * 2)
            },
            "metadata": {
                "filename": file.filename,
                "upload_time": datetime.now().isoformat(),
                "model_version": "1.0.0"
            }
        }
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)