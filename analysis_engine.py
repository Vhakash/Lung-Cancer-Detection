"""
Core analysis engine for medical image processing.
"""
import streamlit as st
import numpy as np
import tempfile
import os
import time
import uuid
from PIL import Image
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

from config import TARGET_IMAGE_SIZE
from logger import logger
from error_handler import handle_errors, validate_file_upload, FileProcessingError
from session_manager import session_manager
from ui_components import ui

from preprocessing import preprocess_image, ensure_color_channels, normalize_dicom_pixel_array
from utils import read_dicom_file, display_dicom_info, calculate_prediction_confidence
from sample_data import get_sample_image
from image_enhancement import apply_enhancement

class AnalysisEngine:
    """Handles medical image analysis workflow."""
    
    @staticmethod
    @handle_errors
    def process_uploaded_file(uploaded_file) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Process uploaded file and return image array and metadata.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (processed_image_array, metadata_dict)
        """
        if not uploaded_file:
            return None, None
        
        # Validate file
        validate_file_upload(uploaded_file)
        
        file_extension = uploaded_file.name.split('.')[-1].lower()
        metadata = {
            'filename': uploaded_file.name,
            'file_type': file_extension,
            'file_size': uploaded_file.size
        }
        
        try:
            if file_extension == 'dcm':
                return AnalysisEngine._process_dicom_file(uploaded_file, metadata)
            else:
                return AnalysisEngine._process_image_file(uploaded_file, metadata)
        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            raise FileProcessingError(f"Failed to process file: {str(e)}")
    
    @staticmethod
    def _process_dicom_file(uploaded_file, metadata: Dict) -> Tuple[np.ndarray, Dict]:
        """Process DICOM file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            # Read DICOM file
            image_data, pixel_array = read_dicom_file(temp_file_path)
            
            # Display DICOM information
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original DICOM Image")
                st.image(pixel_array, caption="Original DICOM Image", use_container_width=True)
                display_dicom_info(image_data)
            
            # Apply windowing/normalization
            with col1:
                st.subheader("Processed DICOM")
                normalized_pixel_array = normalize_dicom_pixel_array(
                    pixel_array,
                    window_percentiles=(5, 95),
                    apply_clahe=True
                )
                st.image(normalized_pixel_array, caption="Windowed/Normalized DICOM", use_container_width=True)
            
            # Convert to format suitable for model
            image_array = ensure_color_channels(normalized_pixel_array)
            processed_image = preprocess_image(image_array, target_size=TARGET_IMAGE_SIZE)
            
            metadata.update({
                'dicom_info': {
                    'patient_id': getattr(image_data, 'PatientID', 'N/A'),
                    'study_date': getattr(image_data, 'StudyDate', 'N/A'),
                    'modality': getattr(image_data, 'Modality', 'N/A')
                }
            })
            
            return processed_image, metadata
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    @staticmethod
    def _process_image_file(uploaded_file, metadata: Dict) -> Tuple[np.ndarray, Dict]:
        """Process regular image file."""
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, caption=f"Original: {uploaded_file.name}", use_container_width=True)
        
        # Convert to numpy array and preprocess
        image_array = np.array(image)
        image_array = ensure_color_channels(image_array)
        processed_image = preprocess_image(image_array, target_size=TARGET_IMAGE_SIZE)
        
        return processed_image, metadata
    
    @staticmethod
    @handle_errors
    def process_sample_image(sample_name: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Process sample image.
        
        Args:
            sample_name: Name of the sample image
            
        Returns:
            Tuple of (processed_image_array, metadata_dict)
        """
        if sample_name == "None":
            return None, None
        
        try:
            sample_image = get_sample_image(sample_name)
            if not sample_image:
                ui.render_error_message(f"Sample image '{sample_name}' not found")
                return None, None
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sample Image")
                st.image(sample_image, caption=f"Sample: {sample_name}", use_container_width=True)
            
            # Process sample image
            image_array = np.array(sample_image)
            image_array = ensure_color_channels(image_array)
            processed_image = preprocess_image(image_array, target_size=TARGET_IMAGE_SIZE)
            
            metadata = {
                'filename': f"{sample_name}.png",
                'file_type': 'sample',
                'sample_name': sample_name
            }
            
            return processed_image, metadata
            
        except Exception as e:
            logger.error(f"Error processing sample image {sample_name}: {str(e)}")
            raise FileProcessingError(f"Failed to process sample image: {str(e)}")
    
    @staticmethod
    @handle_errors
    def apply_image_enhancement(image: np.ndarray, enhancement_type: str, strength: float) -> np.ndarray:
        """
        Apply image enhancement.
        
        Args:
            image: Input image array
            enhancement_type: Type of enhancement to apply
            strength: Enhancement strength
            
        Returns:
            Enhanced image array
        """
        if enhancement_type == "None":
            return image
        
        try:
            with ui.render_loading_spinner(f'Applying {enhancement_type} enhancement...'):
                enhanced_image = apply_enhancement(image, enhancement_type, strength)
                
                # Show enhanced image
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Enhanced Image")
                    st.image(enhanced_image, caption=f"Enhanced with {enhancement_type}", use_container_width=True)
                
                logger.info(f"Applied {enhancement_type} enhancement with strength {strength}")
                return enhanced_image
                
        except Exception as e:
            logger.error(f"Error applying enhancement {enhancement_type}: {str(e)}")
            ui.render_error_message(f"Enhancement failed: {str(e)}")
            return image
    
    @staticmethod
    @handle_errors
    def run_model_prediction(model, image: np.ndarray, sample_name: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Run model prediction on image.
        
        Args:
            model: The ML model to use
            image: Preprocessed image array
            sample_name: Name of sample image if applicable
            
        Returns:
            Tuple of (prediction_array, analysis_results)
        """
        if model is None:
            ui.render_error_message("No model loaded")
            return None, None
        
        try:
            with ui.render_loading_spinner('Analyzing image...'):
                start_time = time.time()
                prediction = model.predict(np.expand_dims(image, axis=0), sample_name=sample_name)
                end_time = time.time()
                
                processing_time = (end_time - start_time) * 1000  # convert to ms
                
                # Calculate results
                label, confidence = calculate_prediction_confidence(prediction[0][0])
                
                analysis_results = {
                    'prediction_value': float(prediction[0][0]),
                    'label': label,
                    'confidence': float(confidence),
                    'processing_time_ms': float(processing_time),
                    'timestamp': datetime.now(),
                    'model_type': getattr(model, 'name', type(model).__name__)
                }
                
                # Add to session history
                session_manager.add_to_analysis_history({
                    'image': image,
                    'model_type': analysis_results['model_type'],
                    'prediction': prediction[0][0],
                    'prediction_label': label,
                    'confidence': confidence,
                    'timestamp': analysis_results['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                    'enhancement': session_manager.get('last_enhancement', None)
                })
                
                logger.info(f"Prediction completed: {label} ({confidence:.2f}% confidence)")
                return prediction, analysis_results
                
        except Exception as e:
            logger.error(f"Error during model prediction: {str(e)}")
            ui.render_error_message(f"Prediction failed: {str(e)}")
            return None, None
    
    @staticmethod
    def save_analysis_to_patient(image: np.ndarray, analysis_results: Dict, metadata: Dict, patient_id: int, notes: str = ""):
        """
        Save analysis results to patient record.
        
        Args:
            image: Processed image array
            analysis_results: Analysis results dictionary
            metadata: Image metadata
            patient_id: Patient ID to save to
            notes: Optional notes
        """
        try:
            # Save image file
            img_to_save = (image * 255).clip(0, 255).astype(np.uint8)
            if img_to_save.ndim == 2:
                img_to_save = np.stack([img_to_save]*3, axis=-1)
            
            save_name = f"scan_{uuid.uuid4().hex[:8]}.png"
            save_path = os.path.join('instance', 'uploads', save_name)
            Image.fromarray(img_to_save).save(save_path)
            
            # Prepare scan data
            scan_data = {
                'file_path': save_path,
                'original_filename': metadata.get('filename', 'unknown'),
                'scan_type': metadata.get('file_type', 'unknown').upper(),
                'prediction': analysis_results['label'],
                'confidence': analysis_results['confidence'],
                'notes': notes,
                'scan_date': analysis_results['timestamp']
            }
            
            # Save to database
            from models import get_db
            db = next(get_db())
            try:
                result = add_scan_to_patient(db, patient_id, scan_data)
                if result:
                    ui.render_success_message("Analysis saved to patient record!")
                    logger.info(f"Saved analysis to patient {patient_id}")
                else:
                    ui.render_error_message("Failed to save analysis")
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error saving analysis to patient: {str(e)}")
            ui.render_error_message(f"Failed to save analysis: {str(e)}")

# Create global analysis engine instance
analysis_engine = AnalysisEngine()