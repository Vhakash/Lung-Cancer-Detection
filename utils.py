import streamlit as st
import numpy as np
import pydicom
from datetime import datetime
import pandas as pd
import random

def read_dicom_file(filepath):
    """Read a DICOM file and return both the file data and pixel array.
    
    Args:
        filepath (str): Path to the DICOM file
        
    Returns:
        tuple: (dicom_data, pixel_array)
    """
    # Read the DICOM file
    dicom_data = pydicom.dcmread(filepath)
    
    # Extract the pixel array
    pixel_array = dicom_data.pixel_array
    
    # Convert to appropriate format for display
    # DICOM images can have varying bit depths
    if dicom_data.BitsStored > 8:
        # Scale to 8-bit for display
        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()
        # Avoid division by zero
        if pixel_max != pixel_min:
            pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
        else:
            pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
    
    return dicom_data, pixel_array

def display_dicom_info(dicom_data):
    """Display relevant DICOM file information.
    
    Args:
        dicom_data (pydicom.dataset.FileDataset): DICOM dataset
    """
    # Create an expandable section for DICOM metadata
    with st.expander("DICOM Metadata"):
        # Extract relevant attributes that are commonly present
        metadata = {
            "Patient ID": getattr(dicom_data, "PatientID", "N/A"),
            "Patient Name": str(getattr(dicom_data, "PatientName", "N/A")),
            "Study Date": getattr(dicom_data, "StudyDate", "N/A"),
            "Modality": getattr(dicom_data, "Modality", "N/A"),
            "Manufacturer": getattr(dicom_data, "Manufacturer", "N/A"),
            "Pixel Spacing": getattr(dicom_data, "PixelSpacing", "N/A"),
            "Image Size": f"{dicom_data.Rows} x {dicom_data.Columns}",
            "Bits Allocated": getattr(dicom_data, "BitsAllocated", "N/A"),
            "Bits Stored": getattr(dicom_data, "BitsStored", "N/A")
        }
        
        # Display as a table
        df = pd.DataFrame(list(metadata.items()), columns=["Attribute", "Value"])
        st.table(df)

def calculate_prediction_confidence(prediction_value):
    """Calculate prediction label and confidence percentage.
    
    Args:
        prediction_value (float): Raw prediction value from the model (0-1)
        
    Returns:
        tuple: (label, confidence_percentage)
    """
    # Determine class label based on threshold of 0.5
    label = "Cancer" if prediction_value >= 0.5 else "Healthy"
    
    # Calculate confidence percentage
    confidence = prediction_value * 100 if prediction_value >= 0.5 else (1 - prediction_value) * 100
    
    return label, confidence

def initialize_analysis_history():
    """Initialize the analysis history in session state if it doesn't exist."""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

def add_to_history(image, model_type, prediction, enhancement=None):
    """Add the current analysis to history.
    
    Args:
        image (numpy.ndarray): The processed image
        model_type (str): Type of model used
        prediction (numpy.ndarray): Prediction result
        enhancement (str, optional): Enhancement type if applied
    """
    # Ensure history exists in session state
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Calculate prediction label and confidence
    label, confidence = calculate_prediction_confidence(prediction[0][0])
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add to history (limited to last 10 entries to avoid memory issues)
    st.session_state.analysis_history.append({
        'timestamp': timestamp,
        'image': image,
        'model_type': model_type,
        'prediction': prediction[0][0],
        'prediction_label': label,
        'confidence': confidence,
        'enhancement': enhancement
    })
    
    # Keep only the most recent 10 analyses
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history.pop(0)

def get_analysis_history():
    """Get the analysis history from session state.
    
    Returns:
        list: List of analysis history entries
    """
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    return st.session_state.analysis_history

def clear_analysis_history():
    """Clear the analysis history in session state."""
    st.session_state.analysis_history = []

def compare_model_performances():
    """Compare performance metrics of different models.
    
    Returns:
        pandas.DataFrame: DataFrame with model performance metrics
    """
    # In a real application, this would retrieve actual metrics
    # Here we'll create simulated metrics for demonstration
    
    models = ["Basic CNN", "Transfer Learning"]
    
    # Create metrics with transfer learning being slightly better
    metrics = {
        'Model Type': models,
        'Accuracy': [0.85, 0.92],
        'Precision': [0.82, 0.91],
        'Recall': [0.84, 0.89],
        'F1 Score': [0.83, 0.90],
        'Processing Time (ms)': [45, 120]  # Transfer learning is typically slower
    }
    
    # Create a DataFrame
    df = pd.DataFrame(metrics)
    
    return df
