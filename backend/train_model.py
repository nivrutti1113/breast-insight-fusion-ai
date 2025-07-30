#!/usr/bin/env python3
"""
Standalone training script for breast cancer detection model
"""

import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.breast_cancer_model import BreastCancerModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train breast cancer detection model')
    
    parser.add_argument(
        '--model-type', 
        choices=['resnet50', 'densenet121', 'efficientnet', 'ensemble'],
        default='ensemble',
        help='Type of model to train (default: ensemble)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to training data directory (optional, will use synthetic data if not provided)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0001,
        help='Learning rate (default: 0.0001)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for trained models (default: models)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

async def train_model(args):
    """Train the breast cancer detection model"""
    try:
        logger.info(f"Starting training for {args.model_type} model")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize model
        model = BreastCancerModel(model_type=args.model_type)
        
        # Load or create model
        await model.load_model()
        
        logger.info("Model training completed successfully")
        
        # Display model information
        model_info = model.get_model_info()
        logger.info(f"Model Information:")
        logger.info(f"  - Type: {model_info.get('model_type')}")
        logger.info(f"  - Total Parameters: {model_info.get('total_params'):,}")
        logger.info(f"  - Trainable Parameters: {model_info.get('trainable_params'):,}")
        logger.info(f"  - Layers: {model_info.get('layers_count')}")
        
        # Display performance metrics if available
        if model.get_model_metrics():
            metrics = model.get_model_metrics()
            logger.info(f"Performance Metrics:")
            logger.info(f"  - AUC Score: {metrics.get('auc_score', 'N/A'):.4f}")
            
            if 'classification_report' in metrics:
                report = metrics['classification_report']
                logger.info(f"  - Accuracy: {report.get('accuracy', 'N/A'):.4f}")
                logger.info(f"  - Precision (Malignant): {report.get('Malignant', {}).get('precision', 'N/A'):.4f}")
                logger.info(f"  - Recall (Malignant): {report.get('Malignant', {}).get('recall', 'N/A'):.4f}")
                logger.info(f"  - F1-Score (Malignant): {report.get('Malignant', {}).get('f1-score', 'N/A'):.4f}")
        
        return model
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

async def main():
    """Main training function"""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Breast Cancer Detection Model Training")
    logger.info("=" * 50)
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Data Path: {args.data_path or 'Synthetic data'}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info("=" * 50)
    
    try:
        # Train the model
        model = await train_model(args)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {model.model_path}")
        
        if model.history_path and os.path.exists(model.history_path):
            logger.info(f"Training history saved to: {model.history_path}")
        
        if model.metrics_path and os.path.exists(model.metrics_path):
            logger.info(f"Model metrics saved to: {model.metrics_path}")
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())