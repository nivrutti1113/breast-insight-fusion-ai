# Breast Cancer Detection AI System

A comprehensive full-stack web application for breast cancer detection using deep learning. The system analyzes mammogram images using advanced CNN models and provides predictions with Grad-CAM visualizations for model interpretability.

## 🏗️ Architecture

### Backend (FastAPI + Python)
- **Deep Learning Models**: Ensemble of ResNet50, DenseNet121, and EfficientNet
- **Database**: MongoDB for prediction history storage
- **Visualization**: Grad-CAM heatmaps for model interpretability
- **API**: RESTful endpoints for predictions, history, and model information

### Frontend (React + TypeScript)
- **UI Framework**: React with TypeScript and Tailwind CSS
- **Components**: shadcn/ui for modern, accessible components
- **Features**: Image upload, prediction display, history management, statistics

## 🚀 Features

### Core Functionality
- **Image Upload & Analysis**: Drag-and-drop mammogram image upload
- **AI Predictions**: Binary classification (Benign/Malignant) with confidence scores
- **Grad-CAM Visualization**: Heatmap overlays showing model focus areas
- **Prediction History**: Complete history with search, filter, and management
- **Statistics Dashboard**: Comprehensive analytics and performance metrics

### Advanced Features
- **Multiple Model Types**: ResNet50, DenseNet121, EfficientNet, and Ensemble
- **Data Augmentation**: Advanced preprocessing and augmentation pipeline
- **Model Metrics**: AUC, precision, recall, confusion matrix, classification reports
- **Export Capabilities**: PDF reports and downloadable heatmaps
- **Responsive Design**: Mobile-friendly interface

## 📋 Prerequisites

### Backend Requirements
- Python 3.9+
- TensorFlow 2.15+
- MongoDB (local or cloud)
- 8GB+ RAM (for model training)

### Frontend Requirements
- Node.js 18+
- npm or yarn
- Modern web browser

## 🛠️ Installation & Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd breast-cancer-detection
```

### 2. Backend Setup

#### Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

#### Environment Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=breast_cancer_detection
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

#### Start MongoDB
```bash
# Local MongoDB
mongod

# Or use MongoDB Atlas (cloud)
# Update MONGODB_URL in .env with your connection string
```

#### Run Backend Server
```bash
python run_server.py
```

The API will be available at `http://localhost:8000`

### 3. Frontend Setup

#### Install Dependencies
```bash
cd frontend  # or root directory if React is in root
npm install
```

#### Environment Configuration
```bash
# Create .env.local file
echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local
```

#### Start Development Server
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## 🤖 Model Training

### Quick Start (Synthetic Data)
```bash
cd backend
python train_model.py --model-type ensemble --epochs 10
```

### Advanced Training Options
```bash
# Train specific model type
python train_model.py --model-type resnet50 --epochs 50 --batch-size 32

# Train with custom learning rate
python train_model.py --model-type densenet121 --learning-rate 0.0001

# Train ensemble model
python train_model.py --model-type ensemble --epochs 100 --verbose
```

### Model Types Available
- **ResNet50**: Deep residual network with 50 layers
- **DenseNet121**: Densely connected network with 121 layers
- **EfficientNet**: Efficient scaling of CNN architecture
- **Ensemble**: Combination of all three models for improved accuracy

## 📊 API Documentation

### Core Endpoints

#### Prediction
```http
POST /predict
Content-Type: multipart/form-data

# Upload mammogram image for analysis
# Returns: prediction, confidence, Grad-CAM overlay
```

#### History Management
```http
GET /history                    # Get prediction history
GET /history/{id}              # Get specific prediction
PUT /history/{id}/notes        # Update prediction notes
DELETE /history/{id}           # Delete prediction
```

#### Model Information
```http
GET /model/info               # Model architecture and parameters
GET /model/metrics            # Performance metrics
GET /model/training-history   # Training history
GET /model/summary           # Model summary
```

#### Statistics
```http
GET /statistics              # Prediction statistics and analytics
```

### Response Examples

#### Prediction Response
```json
{
  "id": "64f8a1b2c3d4e5f6g7h8i9j0",
  "prediction": {
    "probability": 0.85,
    "classification": "Malignant",
    "confidence": 0.70
  },
  "metadata": {
    "filename": "mammogram.jpg",
    "upload_time": "2024-01-15T10:30:00Z",
    "model_version": "1.0.0"
  },
  "gradcam_overlay": "base64_encoded_image_data"
}
```

## 🎨 Frontend Components

### Pages
- **HomePage**: Landing page with application overview
- **AnalysisPage**: Image upload and prediction interface
- **HistoryPage**: Prediction history management
- **ResultsPage**: Detailed results and analytics

### Key Components
- **MammogramAnalyzer**: Core analysis component
- **HistoryTable**: Prediction history display
- **StatisticsCards**: Analytics dashboard
- **GradCAMViewer**: Heatmap visualization

## 🚀 Deployment

### Backend Deployment (Render)

1. **Create Render Service**
   - Connect your GitHub repository
   - Use `backend/render.yaml` configuration
   - Set environment variables in Render dashboard

2. **Environment Variables**
   ```
   MONGODB_URL=your_mongodb_connection_string
   DATABASE_NAME=breast_cancer_detection
   ALLOWED_ORIGINS=https://your-frontend-domain.vercel.app
   ```

3. **Deploy**
   - Push changes to main branch
   - Render will automatically deploy

### Frontend Deployment (Vercel)

1. **Create Vercel Project**
   - Import from GitHub
   - Configure build settings:
     - Build Command: `npm run build`
     - Output Directory: `dist`

2. **Environment Variables**
   ```
   VITE_API_BASE_URL=https://your-backend-domain.onrender.com
   ```

3. **Deploy**
   - Push changes to main branch
   - Vercel will automatically deploy

### Docker Deployment

#### Backend
```bash
cd backend
docker build -t breast-cancer-api .
docker run -p 8000:8000 -e MONGODB_URL=your_connection_string breast-cancer-api
```

#### Full Stack with Docker Compose
```bash
docker-compose up -d
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/
```

### Frontend Tests
```bash
npm run test
```

### API Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST -F "file=@sample_mammogram.jpg" http://localhost:8000/predict
```

## 📈 Performance Optimization

### Model Optimization
- **Transfer Learning**: Pre-trained models fine-tuned on mammogram data
- **Ensemble Methods**: Combining multiple models for improved accuracy
- **Data Augmentation**: Rotation, scaling, brightness adjustment
- **Early Stopping**: Prevent overfitting during training

### Application Optimization
- **Caching**: Model caching and result caching
- **Compression**: Image compression for faster uploads
- **CDN**: Static asset delivery via CDN
- **Database Indexing**: Optimized MongoDB queries

## 🔒 Security Considerations

### Backend Security
- **Input Validation**: File type and size validation
- **CORS Configuration**: Restricted origins
- **Rate Limiting**: API rate limiting (implement as needed)
- **Authentication**: Add JWT authentication for production

### Frontend Security
- **XSS Protection**: Content Security Policy headers
- **HTTPS**: SSL/TLS encryption
- **Input Sanitization**: User input validation

## 🐛 Troubleshooting

### Common Issues

#### Backend Issues
```bash
# Model loading error
# Solution: Ensure TensorFlow and model files are properly installed

# MongoDB connection error
# Solution: Check MongoDB service and connection string

# Memory issues during training
# Solution: Reduce batch size or use smaller model
python train_model.py --model-type resnet50 --batch-size 16
```

#### Frontend Issues
```bash
# API connection error
# Solution: Check VITE_API_BASE_URL environment variable

# Build errors
# Solution: Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Performance Issues
- **Slow predictions**: Use GPU acceleration if available
- **High memory usage**: Reduce model complexity or batch size
- **Database queries**: Add appropriate indexes

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Standards
- **Backend**: PEP 8 Python style guide
- **Frontend**: ESLint and Prettier configuration
- **Documentation**: Clear docstrings and comments
- **Testing**: Unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow/Keras**: Deep learning framework
- **FastAPI**: Modern Python web framework
- **React**: Frontend library
- **MongoDB**: Database solution
- **shadcn/ui**: UI component library
- **Tailwind CSS**: Utility-first CSS framework

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review troubleshooting guide

---

**⚠️ Medical Disclaimer**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical decisions.
