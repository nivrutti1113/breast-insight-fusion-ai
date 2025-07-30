import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import logging
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class BreastCancerModel:
    """
    Advanced breast cancer prediction model using ensemble of pre-trained networks
    with transfer learning and data augmentation
    """
    
    def __init__(self, model_type='ensemble'):
        self.model = None
        self.input_shape = (224, 224, 3)
        self.model_type = model_type  # 'resnet50', 'densenet121', 'efficientnet', 'ensemble'
        self.model_path = f"models/breast_cancer_{model_type}_model.h5"
        self.history_path = f"models/training_history_{model_type}.json"
        self.metrics_path = f"models/model_metrics_{model_type}.json"
        self.training_history = None
        self.model_metrics = None
        
    async def load_model(self):
        """Load or create the breast cancer prediction model"""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading existing {self.model_type} model...")
                self.model = tf.keras.models.load_model(self.model_path)
                self._load_training_history()
                self._load_model_metrics()
            else:
                logger.info(f"Creating new {self.model_type} model...")
                await self._create_model()
                
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            # Fallback to creating new model
            await self._create_model()
    
    async def _create_model(self):
        """Create a new breast cancer prediction model based on model_type"""
        try:
            if self.model_type == 'ensemble':
                self.model = self._create_ensemble_model()
            elif self.model_type == 'resnet50':
                self.model = self._create_resnet50_model()
            elif self.model_type == 'densenet121':
                self.model = self._create_densenet121_model()
            elif self.model_type == 'efficientnet':
                self.model = self._create_efficientnet_model()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            logger.info(f"{self.model_type} model created successfully")
            
            # Create synthetic training data for demonstration
            # In production, replace with actual mammogram dataset
            await self._train_demo_model()
            
        except Exception as e:
            logger.error(f"Model creation error: {str(e)}")
            raise e
    
    def _create_resnet50_model(self):
        """Create ResNet50-based model"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Fine-tune last few layers
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        return model
    
    def _create_densenet121_model(self):
        """Create DenseNet121-based model"""
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Fine-tune last few layers
        for layer in base_model.layers[:-15]:
            layer.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        return model
    
    def _create_efficientnet_model(self):
        """Create EfficientNet-based model"""
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Fine-tune last few layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        return model
    
    def _create_ensemble_model(self):
        """Create ensemble model combining multiple architectures"""
        # Create base models
        resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        densenet_base = DenseNet121(weights='imagenet', include_top=False, input_shape=self.input_shape)
        efficientnet_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Freeze base layers
        for model in [resnet_base, densenet_base, efficientnet_base]:
            for layer in model.layers:
                layer.trainable = False
        
        # Create feature extractors
        resnet_features = GlobalAveragePooling2D()(resnet_base.output)
        densenet_features = GlobalAveragePooling2D()(densenet_base.output)
        efficientnet_features = GlobalAveragePooling2D()(efficientnet_base.output)
        
        # Combine features
        combined_features = Concatenate()([resnet_features, densenet_features, efficientnet_features])
        
        # Add classification layers
        x = BatchNormalization()(combined_features)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        # Create shared input
        input_layer = tf.keras.Input(shape=self.input_shape)
        
        # Connect inputs to base models
        resnet_out = resnet_base(input_layer)
        densenet_out = densenet_base(input_layer)
        efficientnet_out = efficientnet_base(input_layer)
        
        # Process through feature extraction and classification
        resnet_pooled = GlobalAveragePooling2D()(resnet_out)
        densenet_pooled = GlobalAveragePooling2D()(densenet_out)
        efficientnet_pooled = GlobalAveragePooling2D()(efficientnet_out)
        
        combined = Concatenate()([resnet_pooled, densenet_pooled, efficientnet_pooled])
        
        x = BatchNormalization()(combined)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        final_predictions = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=input_layer, outputs=final_predictions)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        return model
    
    async def _train_demo_model(self):
        """Train the model with synthetic data (placeholder for actual training)"""
        try:
            logger.info("Training model with synthetic data...")
            
            # Create synthetic training data with realistic patterns
            # In production, replace with actual mammogram dataset
            num_samples = 1000
            X_train, y_train, X_val, y_val = self._generate_synthetic_data(num_samples)
            
            # Setup callbacks
            callbacks = self._get_training_callbacks()
            
            # Setup data augmentation
            train_datagen = self._get_data_augmentation()
            
            # Train the model
            history = self.model.fit(
                train_datagen.flow(X_train, y_train, batch_size=32),
                steps_per_epoch=len(X_train) // 32,
                epochs=10,  # Minimal training for demo
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Save training history
            self.training_history = history.history
            self._save_training_history()
            
            # Evaluate model
            self._evaluate_model(X_val, y_val)
            
            # Save the model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            # Continue without training for demo purposes
            pass
    
    def _generate_synthetic_data(self, num_samples):
        """Generate synthetic mammogram-like data for training"""
        np.random.seed(42)  # For reproducibility
        
        # Generate synthetic images with different patterns for benign/malignant
        X_data = []
        y_data = []
        
        for i in range(num_samples):
            # Create base image
            img = np.random.normal(0.5, 0.1, self.input_shape)
            
            # Add pattern based on class
            if i < num_samples // 2:  # Benign cases
                # Add smooth, regular patterns
                img += self._add_benign_pattern(img)
                label = 0
            else:  # Malignant cases
                # Add irregular, suspicious patterns
                img += self._add_malignant_pattern(img)
                label = 1
            
            # Normalize
            img = np.clip(img, 0, 1)
            X_data.append(img)
            y_data.append(label)
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        # Split into train/validation
        split_idx = int(0.8 * len(X_data))
        X_train, X_val = X_data[:split_idx], X_data[split_idx:]
        y_train, y_val = y_data[:split_idx], y_data[split_idx:]
        
        return X_train, y_train, X_val, y_val
    
    def _add_benign_pattern(self, img):
        """Add benign-like patterns to synthetic image"""
        pattern = np.zeros_like(img)
        # Add smooth circular patterns
        center_x, center_y = np.random.randint(50, 174, 2)
        radius = np.random.randint(10, 30)
        y, x = np.ogrid[:224, :224]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        pattern[mask] = np.random.normal(0.1, 0.02, pattern[mask].shape)
        return pattern
    
    def _add_malignant_pattern(self, img):
        """Add malignant-like patterns to synthetic image"""
        pattern = np.zeros_like(img)
        # Add irregular, spiky patterns
        center_x, center_y = np.random.randint(50, 174, 2)
        for _ in range(np.random.randint(3, 8)):
            spike_x = center_x + np.random.randint(-20, 20)
            spike_y = center_y + np.random.randint(-20, 20)
            pattern[max(0, spike_y-5):min(224, spike_y+5), 
                   max(0, spike_x-5):min(224, spike_x+5)] += np.random.normal(0.2, 0.05)
        return pattern
    
    def _get_data_augmentation(self):
        """Setup data augmentation for training"""
        return ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
    
    def _get_training_callbacks(self):
        """Setup training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.model_path.replace('.h5', '_best.h5'),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks
    
    def _evaluate_model(self, X_val, y_val):
        """Evaluate model performance"""
        try:
            # Make predictions
            y_pred_prob = self.model.predict(X_val, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            # Calculate metrics
            auc_score = roc_auc_score(y_val, y_pred_prob)
            
            # Generate classification report
            class_report = classification_report(
                y_val, y_pred, 
                target_names=['Benign', 'Malignant'],
                output_dict=True
            )
            
            # Generate confusion matrix
            conf_matrix = confusion_matrix(y_val, y_pred)
            
            # Store metrics
            self.model_metrics = {
                'auc_score': float(auc_score),
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'evaluation_date': datetime.now().isoformat()
            }
            
            self._save_model_metrics()
            
            logger.info(f"Model evaluation completed. AUC Score: {auc_score:.4f}")
            
        except Exception as e:
            logger.error(f"Model evaluation error: {str(e)}")
    
    def _save_training_history(self):
        """Save training history to file"""
        try:
            if self.training_history:
                os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
                with open(self.history_path, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    history_serializable = {}
                    for key, value in self.training_history.items():
                        if isinstance(value, np.ndarray):
                            history_serializable[key] = value.tolist()
                        else:
                            history_serializable[key] = value
                    json.dump(history_serializable, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving training history: {str(e)}")
    
    def _load_training_history(self):
        """Load training history from file"""
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r') as f:
                    self.training_history = json.load(f)
        except Exception as e:
            logger.error(f"Error loading training history: {str(e)}")
    
    def _save_model_metrics(self):
        """Save model metrics to file"""
        try:
            if self.model_metrics:
                os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
                with open(self.metrics_path, 'w') as f:
                    json.dump(self.model_metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model metrics: {str(e)}")
    
    def _load_model_metrics(self):
        """Load model metrics from file"""
        try:
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'r') as f:
                    self.model_metrics = json.load(f)
        except Exception as e:
            logger.error(f"Error loading model metrics: {str(e)}")
    
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
    
    def get_model_info(self):
        """Get comprehensive model information"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        info = {
            "model_type": self.model_type,
            "input_shape": self.input_shape,
            "total_params": self.model.count_params(),
            "trainable_params": sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
            "non_trainable_params": sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights]),
            "layers_count": len(self.model.layers),
            "optimizer": self.model.optimizer.__class__.__name__,
            "loss_function": self.model.loss,
            "metrics": [m.name if hasattr(m, 'name') else str(m) for m in self.model.metrics]
        }
        
        if self.model_metrics:
            info["performance"] = {
                "auc_score": self.model_metrics.get("auc_score"),
                "evaluation_date": self.model_metrics.get("evaluation_date"),
                "classification_report": self.model_metrics.get("classification_report")
            }
        
        return info
    
    def get_training_history(self):
        """Get training history"""
        return self.training_history
    
    def get_model_metrics(self):
        """Get model performance metrics"""
        return self.model_metrics
    
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