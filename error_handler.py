"""
Error handling utilities for the application.
"""
import streamlit as st
import traceback
from functools import wraps
from typing import Callable, Any
from logger import logger

class AppError(Exception):
    """Base exception class for application errors."""
    pass

class ValidationError(AppError):
    """Raised when input validation fails."""
    pass

class ModelError(AppError):
    """Raised when model operations fail."""
    pass

class DatabaseError(AppError):
    """Raised when database operations fail."""
    pass

class FileProcessingError(AppError):
    """Raised when file processing fails."""
    pass

def handle_errors(func: Callable) -> Callable:
    """
    Decorator to handle errors gracefully in Streamlit functions.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            st.error(f"Input validation error: {str(e)}")
            logger.warning(f"Validation error in {func.__name__}: {str(e)}")
        except ModelError as e:
            st.error(f"Model processing error: {str(e)}")
            logger.error(f"Model error in {func.__name__}: {str(e)}")
        except DatabaseError as e:
            st.error(f"Database error: {str(e)}")
            logger.error(f"Database error in {func.__name__}: {str(e)}")
        except FileProcessingError as e:
            st.error(f"File processing error: {str(e)}")
            logger.error(f"File processing error in {func.__name__}: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        return None
    return wrapper

def validate_file_upload(uploaded_file) -> bool:
    """
    Validate uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        True if valid, raises ValidationError if not
        
    Raises:
        ValidationError: If file validation fails
    """
    if uploaded_file is None:
        raise ValidationError("No file uploaded")
    
    # Check file size
    if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
        raise ValidationError("File size exceeds 50MB limit")
    
    # Check file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    allowed_extensions = ['jpg', 'jpeg', 'png', 'dcm']
    
    if file_extension not in allowed_extensions:
        raise ValidationError(f"File type '{file_extension}' not supported. Allowed types: {', '.join(allowed_extensions)}")
    
    return True

def validate_patient_data(patient_data: dict) -> bool:
    """
    Validate patient data.
    
    Args:
        patient_data: Dictionary containing patient information
        
    Returns:
        True if valid, raises ValidationError if not
        
    Raises:
        ValidationError: If patient data validation fails
    """
    required_fields = ['first_name', 'last_name', 'date_of_birth', 'gender']
    
    for field in required_fields:
        if not patient_data.get(field):
            raise ValidationError(f"Required field '{field}' is missing or empty")
    
    # Validate email format if provided
    email = patient_data.get('email')
    if email and '@' not in email:
        raise ValidationError("Invalid email format")
    
    return True

def safe_execute(func: Callable, *args, **kwargs) -> tuple[bool, Any]:
    """
    Safely execute a function and return success status with result.
    
    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (success: bool, result: Any)
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {str(e)}")
        return False, str(e)