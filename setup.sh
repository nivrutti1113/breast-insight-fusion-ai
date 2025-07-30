#!/bin/bash

# Breast Cancer Detection AI System - Quick Setup Script

set -e

echo "🏥 Breast Cancer Detection AI System Setup"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if Node.js is installed for local development
if ! command -v node &> /dev/null; then
    echo "⚠️  Node.js is not installed. Installing via package manager..."
    # Add Node.js installation commands based on OS
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p backend/models
mkdir -p logs

# Copy environment files
echo "🔧 Setting up environment files..."
if [ ! -f backend/.env ]; then
    cp backend/.env.example backend/.env
    echo "✅ Created backend/.env from example"
else
    echo "ℹ️  backend/.env already exists"
fi

if [ ! -f .env.local ]; then
    echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local
    echo "✅ Created .env.local for frontend"
else
    echo "ℹ️  .env.local already exists"
fi

# Setup function
setup_with_docker() {
    echo "🐳 Setting up with Docker..."
    
    # Build and start services
    docker-compose up -d mongodb
    echo "⏳ Waiting for MongoDB to start..."
    sleep 10
    
    docker-compose up -d backend
    echo "⏳ Waiting for backend to start..."
    sleep 15
    
    # Test backend health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is running and healthy"
    else
        echo "❌ Backend health check failed"
        docker-compose logs backend
        exit 1
    fi
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Backend API: http://localhost:8000"
    echo "  2. API Documentation: http://localhost:8000/docs"
    echo "  3. MongoDB: localhost:27017"
    echo ""
    echo "🚀 To start the frontend:"
    echo "  npm install"
    echo "  npm run dev"
    echo ""
    echo "🔧 To train a model:"
    echo "  cd backend"
    echo "  python train_model.py --model-type ensemble"
    echo ""
    echo "📊 To view logs:"
    echo "  docker-compose logs -f backend"
    echo ""
}

setup_without_docker() {
    echo "💻 Setting up for local development..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
        exit 1
    fi
    
    # Setup backend
    echo "🐍 Setting up Python backend..."
    cd backend
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "✅ Created Python virtual environment"
    fi
    
    source venv/bin/activate
    pip install -r requirements.txt
    echo "✅ Installed Python dependencies"
    
    cd ..
    
    # Setup frontend
    echo "⚛️  Setting up React frontend..."
    if [ ! -d "node_modules" ]; then
        npm install
        echo "✅ Installed Node.js dependencies"
    fi
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Start MongoDB: mongod"
    echo "  2. Start backend: cd backend && source venv/bin/activate && python run_server.py"
    echo "  3. Start frontend: npm run dev"
    echo ""
    echo "🔧 To train a model:"
    echo "  cd backend && source venv/bin/activate"
    echo "  python train_model.py --model-type ensemble"
    echo ""
}

# Ask user for setup preference
echo ""
echo "Choose setup method:"
echo "1) Docker (recommended for quick start)"
echo "2) Local development (requires manual MongoDB setup)"
echo ""
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        setup_with_docker
        ;;
    2)
        setup_without_docker
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again and choose 1 or 2."
        exit 1
        ;;
esac

echo ""
echo "🏥 Breast Cancer Detection AI System is ready!"
echo ""
echo "⚠️  Medical Disclaimer: This application is for educational and research"
echo "   purposes only. It should not be used as a substitute for professional"
echo "   medical diagnosis or treatment."
echo ""
echo "📚 For more information, see README.md"