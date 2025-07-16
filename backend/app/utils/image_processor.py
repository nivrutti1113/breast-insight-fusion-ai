import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Image processing utilities for mammogram images
    """
    
    def __init__(self):
        self.target_size = (224, 224)
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
    
    def preprocess_image(self, image):
        """
        Preprocess image for model inference
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed image array of shape (1, 224, 224, 3)
        """
        try:
            # Convert PIL image to numpy array if needed
            if hasattr(image, 'convert'):
                img = np.array(image.convert('RGB'))
            else:
                img = image
            
            # Resize image
            img = cv2.resize(img, self.target_size)
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            img = self.normalize_imagenet(img)
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            raise e
    
    def normalize_imagenet(self, image):
        """
        Apply ImageNet normalization to image
        
        Args:
            image: Image array with values in [0, 1]
            
        Returns:
            Normalized image array
        """
        try:
            # Convert to tensor for easier manipulation
            img_tensor = tf.convert_to_tensor(image)
            
            # Apply normalization
            mean = tf.constant(self.normalize_mean, dtype=tf.float32)
            std = tf.constant(self.normalize_std, dtype=tf.float32)
            
            normalized = (img_tensor - mean) / std
            
            return normalized.numpy()
            
        except Exception as e:
            logger.error(f"ImageNet normalization error: {str(e)}")
            return image
    
    def enhance_mammogram(self, image):
        """
        Apply mammogram-specific image enhancements
        
        Args:
            image: Input mammogram image
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Convert back to RGB
            enhanced_rgb = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
            
            return enhanced_rgb
            
        except Exception as e:
            logger.error(f"Mammogram enhancement error: {str(e)}")
            return image
    
    def detect_roi(self, image):
        """
        Detect region of interest (ROI) in mammogram
        
        Args:
            image: Input mammogram image
            
        Returns:
            Cropped ROI image
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply threshold to create binary image
            _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (assumed to be the breast region)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add some padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                # Crop the ROI
                roi = image[y:y+h, x:x+w]
                return roi
            
            return image
            
        except Exception as e:
            logger.error(f"ROI detection error: {str(e)}")
            return image
    
    def augment_image(self, image):
        """
        Apply data augmentation to image
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        try:
            # Random rotation
            rows, cols = image.shape[:2]
            angle = np.random.uniform(-15, 15)
            rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(image, rotation_matrix, (cols, rows))
            
            # Random horizontal flip
            if np.random.random() > 0.5:
                rotated = cv2.flip(rotated, 1)
            
            # Random brightness adjustment
            brightness = np.random.uniform(0.8, 1.2)
            adjusted = cv2.convertScaleAbs(rotated, alpha=brightness, beta=0)
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Image augmentation error: {str(e)}")
            return image
    
    def postprocess_prediction(self, prediction, threshold=0.5):
        """
        Post-process model prediction
        
        Args:
            prediction: Raw model prediction
            threshold: Classification threshold
            
        Returns:
            Processed prediction with classification and confidence
        """
        try:
            probability = float(prediction[0])
            classification = "Malignant" if probability > threshold else "Benign"
            confidence = float(abs(probability - 0.5) * 2)
            
            return {
                "probability": probability,
                "classification": classification,
                "confidence": confidence,
                "threshold": threshold
            }
            
        except Exception as e:
            logger.error(f"Prediction postprocessing error: {str(e)}")
            return {
                "probability": 0.5,
                "classification": "Unknown",
                "confidence": 0.0,
                "threshold": threshold
            }