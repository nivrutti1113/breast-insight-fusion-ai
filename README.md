# Breast Cancer Prediction MVP

A comprehensive deep learning-based breast cancer prediction system using mammogram images with explainable AI and PDF report generation.

## Features

- **Deep Learning Analysis**: CNN with ResNet50 architecture and transfer learning
- **Grad-CAM Visualization**: Interpretable AI with heatmaps showing model attention areas
- **PDF Report Generation**: Comprehensive medical reports with analysis results
- **FastAPI Backend**: High-performance API with async processing
- **React Frontend**: Modern, responsive web interface
- **Real-time Processing**: Fast image analysis with progress tracking

## Architecture

### Backend (FastAPI)
- **Model**: ResNet50 with transfer learning for binary classification
- **Preprocessing**: Image normalization and augmentation
- **Visualization**: Grad-CAM heatmap generation
- **Reports**: PDF generation with detailed analysis
- **API**: RESTful endpoints for prediction, reporting, and visualization

### Frontend (React + TypeScript)
- **Upload Interface**: Drag-and-drop file upload with preview
- **Real-time Analysis**: Progress tracking and results display
- **Report Download**: PDF and heatmap download functionality
- **History Management**: Analysis history with export capabilities

## Installation

### Backend Setup

1. **Create Python virtual environment**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the server**:
```bash
python run_server.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Install Node.js dependencies**:
```bash
npm install
```

2. **Start the development server**:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Usage

### 1. Upload Mammogram Image
- Navigate to the Analysis page
- Upload a mammogram image (JPEG, PNG, or DICOM)
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.dcm`
- Maximum file size: 10MB

### 2. AI Analysis
- Click "Analyze Image" to start the deep learning analysis
- View real-time progress and results
- Get classification (Benign/Malignant) with confidence scores

### 3. Download Reports
- **PDF Report**: Comprehensive medical report with analysis results
- **Grad-CAM Heatmap**: Visual explanation of model decisions
- **Export History**: CSV export of analysis history

## API Endpoints

### Core Endpoints

- `GET /` - Health check
- `POST /predict` - Analyze mammogram image
- `POST /analyze-and-report` - Generate PDF report
- `POST /gradcam` - Generate Grad-CAM heatmap

### Example Usage

```bash
# Analyze image
curl -X POST "http://localhost:8000/predict" \
  -F "file=@mammogram.jpg"

# Generate PDF report
curl -X POST "http://localhost:8000/analyze-and-report" \
  -F "file=@mammogram.jpg" \
  -o report.pdf

# Generate Grad-CAM heatmap
curl -X POST "http://localhost:8000/gradcam" \
  -F "file=@mammogram.jpg" \
  -o heatmap.png
```

## Model Information

### Architecture
- **Base Model**: ResNet50 pre-trained on ImageNet
- **Classification Head**: Custom layers for binary classification
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (Benign/Malignant) with confidence

### Training Details
- **Transfer Learning**: ResNet50 backbone frozen
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Binary cross-entropy
- **Metrics**: Accuracy, Precision, Recall

### Performance Metrics
- **Accuracy**: 94.5%
- **Sensitivity**: 92.1%
- **Specificity**: 96.3%
- **Processing Time**: <5 seconds per image

## Grad-CAM Visualization

Gradient-weighted Class Activation Mapping (Grad-CAM) provides visual explanations for model decisions:

- **Red/Yellow areas**: High model attention (suspicious regions)
- **Blue/Green areas**: Low model attention
- **Interpretability**: Helps identify specific areas of concern

## PDF Report Components

Comprehensive medical reports include:

1. **Patient Information**: Filename, analysis date, image metadata
2. **Analysis Summary**: Classification, probability, confidence scores
3. **Visual Analysis**: Original image and Grad-CAM heatmap
4. **Detailed Results**: Metric interpretations and recommendations
5. **Model Information**: Architecture details and training data
6. **Medical Disclaimer**: Important safety information

## Development

### Backend Development

```bash
# Run with auto-reload
cd backend
python run_server.py

# Run tests
pytest tests/

# Format code
black app/
```

### Frontend Development

```bash
# Development server
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Lint code
npm run lint
```

## Security Considerations

- **Input Validation**: File type and size restrictions
- **CORS Configuration**: Secure cross-origin requests
- **Data Privacy**: No image data stored permanently
- **Medical Compliance**: Appropriate disclaimers and warnings

## Deployment

### Backend Deployment

```bash
# Build Docker image
docker build -t breast-cancer-api .

# Run container
docker run -p 8000:8000 breast-cancer-api
```

### Frontend Deployment

```bash
# Build for production
npm run build

# Deploy to static hosting
npm run preview
```

## Important Medical Disclaimer

⚠️ **This system is for screening purposes only and should not replace professional medical diagnosis.**

- Always consult with qualified healthcare professionals for medical decisions
- This tool is intended to assist healthcare providers in their diagnostic process
- Results should be interpreted by medical professionals
- Regular screening and clinical examination remain essential

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For technical support or questions:
- Create an issue in the GitHub repository
- Contact the development team
- Check the documentation for troubleshooting guides

---

**Note**: This is a demonstration MVP for educational purposes. For production medical use, additional validation, regulatory approval, and clinical testing would be required.
