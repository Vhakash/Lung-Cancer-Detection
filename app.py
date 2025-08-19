"""
Lung Cancer Detection AI - Main Application
A Streamlit-based medical imaging analysis tool for lung cancer detection.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from PIL import Image
import pydicom
import io
import time
import zipfile
from datetime import datetime
import uuid

# Import configuration and utilities
from config import APP_TITLE, APP_ICON, PAGE_LAYOUT, ensure_directories
from logger import logger
from error_handler import handle_errors, validate_file_upload, ValidationError
from session_manager import session_manager
from ui_components import ui

# Import patient management
from patient_ui import show_patient_form, show_patient_list, show_patient_details
from models import get_db, Patient, Scan
from patient_utils import create_patient, get_patients, add_scan_to_patient

# Import core functionality
from model import create_model, load_pretrained_model
from preprocessing import preprocess_image, ensure_color_channels, normalize_dicom_pixel_array
from visualization import visualize_prediction, visualize_model_performance, visualize_activation_maps, visualize_feature_maps, visualize_grad_cam
from utils import read_dicom_file, display_dicom_info, calculate_prediction_confidence, compare_model_performances
from sample_data import get_sample_image, get_sample_image_names
from image_enhancement import apply_enhancement

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT
)

# Ensure required directories exist
ensure_directories()

# Initialize session state
session_manager.initialize_session_state()

logger.info("Application started")

# Render application header
ui.render_header()

# Render sidebar navigation
nav_choice = ui.render_sidebar_navigation()

# Handle navigation
@handle_errors
def handle_navigation(choice):
    """Handle navigation based on user selection."""
    if choice == "Analyze":
        session_manager.navigate_to('analyze')
    elif choice == "Patients":
        session_manager.navigate_to('patients')
    elif choice == "History":
        session_manager.navigate_to('history')
    elif choice == "Compare Models":
        session_manager.navigate_to('compare')

handle_navigation(nav_choice)

# Render model selector
model_option = ui.render_model_selector()

# Handle model loading
@handle_errors
def load_model(model_type):
    """Load the selected model."""
    try:
        if model_type == "Basic CNN":
            return create_model()
        else:
            return load_pretrained_model()
    except Exception as e:
        logger.error(f"Failed to load model {model_type}: {str(e)}")
        ui.render_error_message(f"Failed to load model: {str(e)}")
        return None

# Initialize or update model
if session_manager.get('model') is None or session_manager.get('model_option') != model_option:
    with ui.render_loading_spinner("Loading model..."):
        model = load_model(model_option)
        if model:
            session_manager.set('model', model)
            session_manager.set('model_option', model_option)
            logger.info(f"Loaded model: {model_option}")

# Show active model info
if session_manager.get('model'):
    active_model_name = getattr(session_manager.get('model'), 'name', type(session_manager.get('model')).__name__)
    st.sidebar.caption(f"Active model: {active_model_name}")

# Render visualization selector
visualization_option, gradcam_last_conv = ui.render_visualization_selector()

# Render enhancement selector
enhancement_option, enhancement_strength = ui.render_enhancement_selector()

# Render sample selector
sample_option = ui.render_sample_selector()
session_manager.set('sample_option', sample_option)

# Store UI selections in session state for other components
session_manager.set('visualization_option', visualization_option)
session_manager.set('gradcam_layer', gradcam_last_conv)
session_manager.set('enhancement_option', enhancement_option)
session_manager.set('enhancement_strength', enhancement_strength)

# Quick action buttons
if st.sidebar.button("Compare Models", key="compare_models_button"):
    session_manager.navigate_to('compare')
    st.rerun()

if st.sidebar.button("View Analysis History", key="view_history_button"):
    session_manager.navigate_to('history')
    st.rerun()

if st.sidebar.button("Clear History", key="clear_history_button"):
    session_manager.clear_analysis_history()
    ui.render_success_message("Analysis history cleared!")

# Render patient management quick actions
ui.render_patient_quick_actions()

# Import views and analysis interface
from views import views
from analysis_interface import analysis_interface

# Main content routing based on current page
current_page = session_manager.get('current_page', 'analyze')

if current_page == 'compare' or session_manager.get('show_model_comparison', False):
    views.render_model_comparison()
elif current_page == 'history' or session_manager.get('show_history', False):
    views.render_analysis_history()
elif current_page == 'patients' or session_manager.get('show_patient_list', False):
    views.render_patient_management()
elif nav_choice == "About":
    views.render_about()
else:
    # Default to analysis interface
    analysis_interface.render_main_analysis()

logger.info("Application session completed")