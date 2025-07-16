import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class BreastCancerModel:
    """
    Breast cancer prediction model using transfer learning with ResNet50
    """
    
    def __init__(self):
        self.model = None
        self.input_shape = (224, 224, 3)
        self.model_path = "models/breast_cancer_model.h5"
        
    async def load_model(self):
        """Load or create the breast cancer prediction model"""
        try:
            if os.path.exists(self.model_path):
                logger.info("Loading existing model...")
                self.model = tf.keras.models.load_model(self.model_path)
            else:
                logger.info("Creating new model...")
                await self._create_model()
                
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            # Fallback to creating new model
            await self._create_model()
    
    async def _create_model(self):
        """Create a new breast cancer prediction model"""
        try:
            # Load pre-trained ResNet50 model
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            
            # Freeze base model layers
            base_model.trainable = False
            
            # Add custom classification layers
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.3)(x)
            predictions = Dense(1, activation='sigmoid')(x)
            
            # Create the model
            self.model = Model(inputs=base_model.input, outputs=predictions)
            
            # Compile the model
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            logger.info("Model created successfully")
            
            # Create dummy training data for demonstration
            # In a real application, you would train on actual mammogram data
            await self._train_demo_model()
            
        except Exception as e:
            logger.error(f"Model creation error: {str(e)}")
            raise e
    
    async def _train_demo_model(self):
        """Train the model with demo data (placeholder for actual training)"""
        try:
            logger.info("Training demo model...")
            
            # Create dummy training data
            # In production, replace with actual mammogram dataset
            X_train = np.random.random((100, 224, 224, 3))
            y_train = np.random.randint(0, 2, (100,))
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                epochs=1,  # Minimal training for demo
                batch_size=16,
                validation_split=0.2,
                verbose=1
            )
            
            # Save the model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            
            logger.info("Demo model training completed")
            
        except Exception as e:
            logger.error(f"Demo training error: {str(e)}")
            # Continue without training for demo purposes
            pass
    
    def predict(self, image):
        """
        Make prediction on preprocessed image
        
        Args:
            image: Preprocessed image array of shape (1, 224, 224, 3)
            
        Returns:
            Prediction probability (0-1)
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            prediction = self.model.predict(image, verbose=0)
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise e
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            return "Model not loaded"
        
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        return "\n".join(summary)
    
    def get_feature_maps(self, image, layer_name=None):
        """
        Get feature maps from intermediate layers
        
        Args:
            image: Input image array
            layer_name: Name of the layer to extract features from
            
        Returns:
            Feature maps from specified layer
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            if layer_name is None:
                # Get the last convolutional layer
                for layer in reversed(self.model.layers):
                    if 'conv' in layer.name.lower():
                        layer_name = layer.name
                        break
            
            # Create intermediate model
            intermediate_model = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer(layer_name).output
            )
            
            # Get feature maps
            feature_maps = intermediate_model.predict(image, verbose=0)
            return feature_maps
            
        except Exception as e:
            logger.error(f"Feature map extraction error: {str(e)}")
            raise e