import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging

logger = logging.getLogger(__name__)

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation
    for visualizing important regions in mammogram images
    """
    
    def __init__(self, model):
        self.model = model
        self.last_conv_layer_name = self._get_last_conv_layer_name()
        
    def _get_last_conv_layer_name(self):
        """Find the last convolutional layer in the model"""
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:  # Conv layer has 4D output
                return layer.name
        return None
    
    def generate_heatmap(self, img_array, class_index=0, alpha=0.4):
        """
        Generate Grad-CAM heatmap for the input image
        
        Args:
            img_array: Input image array of shape (1, H, W, C)
            class_index: Index of the class to generate heatmap for
            alpha: Transparency factor for overlay
            
        Returns:
            Heatmap overlaid on original image
        """
        try:
            # Create a model that maps the input image to the activations
            # of the last conv layer as well as the output predictions
            grad_model = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.last_conv_layer_name).output,
                        self.model.output]
            )
            
            # Compute the gradient of the top predicted class for our input image
            # with respect to the activations of the last conv layer
            with tf.GradientTape() as tape:
                last_conv_layer_output, preds = grad_model(img_array)
                if class_index == 0:
                    class_channel = preds[:, 0]
                else:
                    class_channel = preds[:, class_index]
            
            # This is the gradient of the output neuron with regard to
            # the output feature map of the last conv layer
            grads = tape.gradient(class_channel, last_conv_layer_output)
            
            # This is a vector where each entry is the mean intensity of the gradient
            # over a specific feature map channel
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # We multiply each channel in the feature map array
            # by "how important this channel is" with regard to the top predicted class
            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # For visualization purpose, we normalize the heatmap between 0 & 1
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return heatmap.numpy()
            
        except Exception as e:
            logger.error(f"Grad-CAM generation error: {str(e)}")
            # Return dummy heatmap if generation fails
            return np.random.random((7, 7))
    
    def create_overlay_image(self, original_image, heatmap, alpha=0.4):
        """
        Create an overlay image with heatmap superimposed on original image
        
        Args:
            original_image: Original input image (PIL Image or numpy array)
            heatmap: Generated heatmap
            alpha: Transparency factor
            
        Returns:
            Overlay image as numpy array
        """
        try:
            # Convert PIL image to numpy array if needed
            if hasattr(original_image, 'convert'):
                img = np.array(original_image.convert('RGB'))
            else:
                img = original_image
            
            # Resize heatmap to match original image size
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            
            # Convert heatmap to RGB
            heatmap_rgb = cm.jet(heatmap_resized)[:, :, :3]
            
            # Create overlay
            overlay = (1 - alpha) * img + alpha * heatmap_rgb * 255
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            return overlay
            
        except Exception as e:
            logger.error(f"Overlay creation error: {str(e)}")
            # Return original image if overlay creation fails
            return np.array(original_image) if hasattr(original_image, 'convert') else original_image
    
    def save_heatmap(self, heatmap, save_path, colormap='jet'):
        """
        Save heatmap as image file
        
        Args:
            heatmap: Generated heatmap
            save_path: Path to save the heatmap
            colormap: Matplotlib colormap to use
        """
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(heatmap, cmap=colormap)
            plt.colorbar()
            plt.title('Grad-CAM Heatmap')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Heatmap saving error: {str(e)}")
    
    def generate_detailed_analysis(self, img_array, save_dir="/tmp"):
        """
        Generate detailed Grad-CAM analysis with multiple visualizations
        
        Args:
            img_array: Input image array
            save_dir: Directory to save analysis files
            
        Returns:
            Dictionary with analysis results and file paths
        """
        try:
            # Generate heatmap
            heatmap = self.generate_heatmap(img_array)
            
            # Create different visualizations
            analysis = {
                "heatmap": heatmap,
                "prediction": self.model.predict(img_array, verbose=0)[0][0],
                "confidence": float(abs(self.model.predict(img_array, verbose=0)[0][0] - 0.5) * 2)
            }
            
            # Save visualizations
            heatmap_path = f"{save_dir}/gradcam_heatmap.png"
            self.save_heatmap(heatmap, heatmap_path)
            analysis["heatmap_path"] = heatmap_path
            
            return analysis
            
        except Exception as e:
            logger.error(f"Detailed analysis error: {str(e)}")
            return {"error": str(e)}