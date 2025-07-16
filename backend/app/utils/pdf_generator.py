from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import logging
from io import BytesIO
from PIL import Image as PILImage
import cv2

logger = logging.getLogger(__name__)

class PDFReportGenerator:
    """
    PDF report generator for breast cancer analysis
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
        self.report_dir = "/tmp/reports"
        os.makedirs(self.report_dir, exist_ok=True)
    
    def _create_custom_styles(self):
        """Create custom paragraph styles for the report"""
        styles = {}
        
        # Title style
        styles['CustomTitle'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        # Section header style
        styles['SectionHeader'] = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=5
        )
        
        # Analysis result style
        styles['AnalysisResult'] = ParagraphStyle(
            'AnalysisResult',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            leftIndent=20
        )
        
        # Warning style
        styles['Warning'] = ParagraphStyle(
            'Warning',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.red,
            spaceAfter=6
        )
        
        # Info style
        styles['Info'] = ParagraphStyle(
            'Info',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            spaceAfter=6
        )
        
        return styles
    
    async def generate_report(self, analysis_data, original_image=None, heatmap=None):
        """
        Generate comprehensive PDF report
        
        Args:
            analysis_data: Dictionary containing analysis results
            original_image: Original mammogram image
            heatmap: Grad-CAM heatmap
            
        Returns:
            Path to generated PDF report
        """
        try:
            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"breast_cancer_report_{timestamp}.pdf"
            filepath = os.path.join(self.report_dir, filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                filepath,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build story (content)
            story = []
            
            # Add title
            story.append(Paragraph("Breast Cancer Analysis Report", self.custom_styles['CustomTitle']))
            story.append(Spacer(1, 12))
            
            # Add patient information
            story.extend(self._create_patient_info_section(analysis_data))
            
            # Add analysis summary
            story.extend(self._create_analysis_summary_section(analysis_data))
            
            # Add images if provided
            if original_image is not None:
                story.extend(self._create_image_section(original_image, heatmap))
            
            # Add detailed results
            story.extend(self._create_detailed_results_section(analysis_data))
            
            # Add model information
            story.extend(self._create_model_info_section(analysis_data))
            
            # Add recommendations
            story.extend(self._create_recommendations_section(analysis_data))
            
            # Add disclaimer
            story.extend(self._create_disclaimer_section())
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"PDF generation error: {str(e)}")
            raise e
    
    def _create_patient_info_section(self, analysis_data):
        """Create patient information section"""
        content = []
        
        content.append(Paragraph("Patient Information", self.custom_styles['SectionHeader']))
        content.append(Spacer(1, 12))
        
        # Create patient info table
        patient_info = analysis_data.get('patient_info', {})
        
        data = [
            ['Filename:', patient_info.get('filename', 'N/A')],
            ['Analysis Date:', patient_info.get('analysis_date', 'N/A')],
            ['Image Size:', patient_info.get('image_size', 'N/A')],
            ['Report ID:', f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"]
        ]
        
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(table)
        content.append(Spacer(1, 20))
        
        return content
    
    def _create_analysis_summary_section(self, analysis_data):
        """Create analysis summary section"""
        content = []
        
        content.append(Paragraph("Analysis Summary", self.custom_styles['SectionHeader']))
        content.append(Spacer(1, 12))
        
        prediction = analysis_data.get('prediction', {})
        classification = prediction.get('classification', 'Unknown')
        probability = prediction.get('probability', 0.0)
        confidence = prediction.get('confidence', 0.0)
        
        # Result summary
        result_style = self.custom_styles['Warning'] if classification == 'Malignant' else self.custom_styles['AnalysisResult']
        
        content.append(Paragraph(f"<b>Classification:</b> {classification}", result_style))
        content.append(Paragraph(f"<b>Probability:</b> {probability:.3f}", self.custom_styles['AnalysisResult']))
        content.append(Paragraph(f"<b>Confidence:</b> {confidence:.3f}", self.custom_styles['AnalysisResult']))
        
        # Interpretation
        if classification == 'Malignant':
            interpretation = "The analysis suggests the presence of potentially malignant tissue. Further medical evaluation is strongly recommended."
        else:
            interpretation = "The analysis suggests the tissue appears benign. However, regular screening is still recommended."
        
        content.append(Spacer(1, 12))
        content.append(Paragraph(f"<b>Interpretation:</b> {interpretation}", self.custom_styles['AnalysisResult']))
        content.append(Spacer(1, 20))
        
        return content
    
    def _create_image_section(self, original_image, heatmap):
        """Create image visualization section"""
        content = []
        
        content.append(Paragraph("Image Analysis", self.custom_styles['SectionHeader']))
        content.append(Spacer(1, 12))
        
        try:
            # Save original image
            orig_img_path = f"/tmp/original_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            if hasattr(original_image, 'save'):
                original_image.save(orig_img_path)
            else:
                PILImage.fromarray(original_image).save(orig_img_path)
            
            # Add original image
            content.append(Paragraph("Original Mammogram:", self.custom_styles['AnalysisResult']))
            img = Image(orig_img_path, width=3*inch, height=3*inch)
            content.append(img)
            content.append(Spacer(1, 12))
            
            # Add heatmap if provided
            if heatmap is not None:
                heatmap_path = f"/tmp/heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.figure(figsize=(6, 6))
                plt.imshow(heatmap, cmap='jet')
                plt.colorbar(label='Activation')
                plt.title('Grad-CAM Heatmap')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                content.append(Paragraph("Grad-CAM Heatmap (Areas of Interest):", self.custom_styles['AnalysisResult']))
                heatmap_img = Image(heatmap_path, width=3*inch, height=3*inch)
                content.append(heatmap_img)
                content.append(Spacer(1, 12))
                
                content.append(Paragraph(
                    "The heatmap shows areas that the AI model focused on during analysis. "
                    "Warmer colors (red/yellow) indicate regions of higher importance.",
                    self.custom_styles['Info']
                ))
            
        except Exception as e:
            logger.error(f"Image section creation error: {str(e)}")
            content.append(Paragraph("Image visualization unavailable", self.custom_styles['Info']))
        
        content.append(Spacer(1, 20))
        return content
    
    def _create_detailed_results_section(self, analysis_data):
        """Create detailed results section"""
        content = []
        
        content.append(Paragraph("Detailed Analysis Results", self.custom_styles['SectionHeader']))
        content.append(Spacer(1, 12))
        
        prediction = analysis_data.get('prediction', {})
        
        # Create detailed results table
        data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Probability Score', f"{prediction.get('probability', 0.0):.4f}", self._interpret_probability(prediction.get('probability', 0.0))],
            ['Confidence Level', f"{prediction.get('confidence', 0.0):.4f}", self._interpret_confidence(prediction.get('confidence', 0.0))],
            ['Classification', prediction.get('classification', 'Unknown'), self._interpret_classification(prediction.get('classification', 'Unknown'))]
        ]
        
        table = Table(data, colWidths=[1.5*inch, 1.5*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(table)
        content.append(Spacer(1, 20))
        
        return content
    
    def _create_model_info_section(self, analysis_data):
        """Create model information section"""
        content = []
        
        content.append(Paragraph("Model Information", self.custom_styles['SectionHeader']))
        content.append(Spacer(1, 12))
        
        model_info = analysis_data.get('model_info', {})
        
        content.append(Paragraph(f"<b>Model Version:</b> {model_info.get('model_version', 'N/A')}", self.custom_styles['AnalysisResult']))
        content.append(Paragraph(f"<b>Architecture:</b> {model_info.get('architecture', 'N/A')}", self.custom_styles['AnalysisResult']))
        content.append(Paragraph(f"<b>Training Data:</b> {model_info.get('training_data', 'N/A')}", self.custom_styles['AnalysisResult']))
        
        content.append(Spacer(1, 12))
        content.append(Paragraph(
            "This analysis was performed using a deep learning model trained on mammography images. "
            "The model uses transfer learning with a pre-trained ResNet50 architecture.",
            self.custom_styles['Info']
        ))
        
        content.append(Spacer(1, 20))
        return content
    
    def _create_recommendations_section(self, analysis_data):
        """Create recommendations section"""
        content = []
        
        content.append(Paragraph("Recommendations", self.custom_styles['SectionHeader']))
        content.append(Spacer(1, 12))
        
        prediction = analysis_data.get('prediction', {})
        classification = prediction.get('classification', 'Unknown')
        probability = prediction.get('probability', 0.0)
        
        if classification == 'Malignant':
            recommendations = [
                "Immediate consultation with a radiologist or oncologist is recommended",
                "Additional imaging studies may be required for confirmation",
                "Biopsy may be recommended based on clinical judgment",
                "Close follow-up and monitoring are essential"
            ]
        else:
            recommendations = [
                "Continue regular mammography screening as recommended",
                "Maintain healthy lifestyle habits",
                "Be aware of any changes in breast tissue",
                "Follow up with healthcare provider as scheduled"
            ]
        
        for i, rec in enumerate(recommendations, 1):
            content.append(Paragraph(f"{i}. {rec}", self.custom_styles['AnalysisResult']))
        
        content.append(Spacer(1, 20))
        return content
    
    def _create_disclaimer_section(self):
        """Create disclaimer section"""
        content = []
        
        content.append(Paragraph("Important Disclaimer", self.custom_styles['SectionHeader']))
        content.append(Spacer(1, 12))
        
        disclaimer_text = """
        This report is generated by an AI-powered analysis system and is intended for screening purposes only. 
        The results should not be used as a substitute for professional medical diagnosis or treatment. 
        This analysis is not a replacement for clinical examination, medical history, or other diagnostic procedures. 
        All medical decisions should be made by qualified healthcare professionals. 
        If you have concerns about your health, please consult with a medical professional immediately.
        """
        
        content.append(Paragraph(disclaimer_text, self.custom_styles['Warning']))
        content.append(Spacer(1, 12))
        
        content.append(Paragraph(
            f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.custom_styles['Info']
        ))
        
        return content
    
    def _interpret_probability(self, probability):
        """Interpret probability score"""
        if probability > 0.8:
            return "High probability of malignancy"
        elif probability > 0.6:
            return "Moderate probability of malignancy"
        elif probability > 0.4:
            return "Low probability of malignancy"
        else:
            return "Very low probability of malignancy"
    
    def _interpret_confidence(self, confidence):
        """Interpret confidence score"""
        if confidence > 0.8:
            return "High confidence in prediction"
        elif confidence > 0.6:
            return "Moderate confidence in prediction"
        else:
            return "Low confidence in prediction"
    
    def _interpret_classification(self, classification):
        """Interpret classification result"""
        if classification == 'Malignant':
            return "Potential malignant tissue detected"
        elif classification == 'Benign':
            return "Tissue appears benign"
        else:
            return "Classification uncertain"