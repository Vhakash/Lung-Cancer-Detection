"""
Data validation utilities for medical imaging and patient data.
"""
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import pydicom
from logger import logger

class MedicalDataValidator:
    """Validate medical imaging data and patient information."""
    
    @staticmethod
    def validate_image_data(image_array: np.ndarray) -> Tuple[bool, str]:
        """Validate medical image data for analysis."""
        try:
            # Check if image is empty
            if image_array.size == 0:
                return False, "Image array is empty"
            
            # Check minimum dimensions
            if len(image_array.shape) < 2:
                return False, "Image must be at least 2-dimensional"
            
            # Check minimum size
            min_size = 32
            if min(image_array.shape[:2]) < min_size:
                return False, f"Image too small (minimum {min_size}x{min_size} pixels)"
            
            # Check maximum size
            max_size = 2048
            if max(image_array.shape[:2]) > max_size:
                return False, f"Image too large (maximum {max_size}x{max_size} pixels)"
            
            # Check data type
            if image_array.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
                return False, f"Unsupported image data type: {image_array.dtype}"
            
            # Check for reasonable contrast
            if image_array.std() < 1e-6:
                return False, "Image has no contrast (all pixels same value)"
            
            # Check for reasonable value range
            if image_array.dtype in [np.float32, np.float64]:
                if image_array.min() < -10 or image_array.max() > 10:
                    logger.warning("Image values outside expected range for float type")
            
            return True, "Image validation passed"
            
        except Exception as e:
            return False, f"Image validation error: {str(e)}"
    
    @staticmethod
    def validate_dicom_file(file_path: str) -> Tuple[bool, str, Optional[Dict]]:
        """Validate DICOM file and extract metadata."""
        try:
            # Read DICOM file
            dicom_data = pydicom.dcmread(file_path)
            
            # Check required DICOM tags
            required_tags = ['PatientID', 'StudyDate', 'Modality']
            missing_tags = []
            
            for tag in required_tags:
                if not hasattr(dicom_data, tag) or not getattr(dicom_data, tag):
                    missing_tags.append(tag)
            
            # Extract metadata
            metadata = {
                'patient_id': getattr(dicom_data, 'PatientID', 'Unknown'),
                'study_date': getattr(dicom_data, 'StudyDate', 'Unknown'),
                'modality': getattr(dicom_data, 'Modality', 'Unknown'),
                'manufacturer': getattr(dicom_data, 'Manufacturer', 'Unknown'),
                'image_size': f"{dicom_data.Rows}x{dicom_data.Columns}",
                'bits_allocated': getattr(dicom_data, 'BitsAllocated', 'Unknown')
            }
            
            # Validate pixel data
            if not hasattr(dicom_data, 'pixel_array'):
                return False, "DICOM file has no pixel data", metadata
            
            pixel_array = dicom_data.pixel_array
            is_valid, message = MedicalDataValidator.validate_image_data(pixel_array)
            
            if not is_valid:
                return False, f"DICOM pixel data invalid: {message}", metadata
            
            warning_msg = ""
            if missing_tags:
                warning_msg = f"Missing optional tags: {', '.join(missing_tags)}"
            
            return True, warning_msg or "DICOM validation passed", metadata
            
        except Exception as e:
            return False, f"DICOM validation error: {str(e)}", None
    
    @staticmethod
    def validate_patient_data(patient_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate patient information."""
        errors = []
        
        # Required fields
        required_fields = ['first_name', 'last_name', 'date_of_birth', 'gender']
        
        for field in required_fields:
            if field not in patient_data or not patient_data[field]:
                errors.append(f"Required field '{field}' is missing or empty")
        
        # Validate names
        for name_field in ['first_name', 'last_name']:
            if name_field in patient_data:
                name = patient_data[name_field]
                if len(name) < 2:
                    errors.append(f"{name_field.replace('_', ' ').title()} too short")
                if len(name) > 50:
                    errors.append(f"{name_field.replace('_', ' ').title()} too long")
                if not name.replace(' ', '').replace('-', '').isalpha():
                    errors.append(f"{name_field.replace('_', ' ').title()} contains invalid characters")
        
        # Validate date of birth
        if 'date_of_birth' in patient_data:
            dob = patient_data['date_of_birth']
            if isinstance(dob, str):
                try:
                    dob = datetime.strptime(dob, '%Y-%m-%d').date()
                except ValueError:
                    errors.append("Invalid date of birth format (use YYYY-MM-DD)")
            
            if isinstance(dob, date):
                today = date.today()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                
                if dob > today:
                    errors.append("Date of birth cannot be in the future")
                elif age > 150:
                    errors.append("Age seems unrealistic (over 150 years)")
                elif age < 0:
                    errors.append("Invalid date of birth")
        
        # Validate gender
        if 'gender' in patient_data:
            valid_genders = ['Male', 'Female', 'Other', 'Unknown']
            if patient_data['gender'] not in valid_genders:
                errors.append(f"Gender must be one of: {', '.join(valid_genders)}")
        
        # Validate email
        if 'email' in patient_data and patient_data['email']:
            email = patient_data['email']
            if '@' not in email or '.' not in email.split('@')[-1]:
                errors.append("Invalid email format")
            if len(email) > 100:
                errors.append("Email address too long")
        
        # Validate phone
        if 'phone' in patient_data and patient_data['phone']:
            phone = patient_data['phone']
            # Remove common formatting characters
            clean_phone = ''.join(c for c in phone if c.isdigit())
            if len(clean_phone) < 10 or len(clean_phone) > 15:
                errors.append("Phone number should be 10-15 digits")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_prediction_result(prediction: np.ndarray) -> Tuple[bool, str]:
        """Validate model prediction results."""
        try:
            if prediction is None:
                return False, "Prediction is None"
            
            if not isinstance(prediction, np.ndarray):
                return False, "Prediction must be numpy array"
            
            if prediction.size == 0:
                return False, "Prediction array is empty"
            
            # Check shape
            if len(prediction.shape) != 2 or prediction.shape[1] != 1:
                return False, f"Expected shape (batch_size, 1), got {prediction.shape}"
            
            # Check value range for binary classification
            if np.any(prediction < 0) or np.any(prediction > 1):
                return False, "Prediction values must be between 0 and 1"
            
            # Check for NaN or infinite values
            if np.any(np.isnan(prediction)) or np.any(np.isinf(prediction)):
                return False, "Prediction contains NaN or infinite values"
            
            return True, "Prediction validation passed"
            
        except Exception as e:
            return False, f"Prediction validation error: {str(e)}"

# Global validator instance
medical_validator = MedicalDataValidator()