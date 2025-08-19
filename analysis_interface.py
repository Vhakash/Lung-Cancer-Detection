"""
Main analysis interface for medical image processing.
"""
import streamlit as st
import numpy as np
import io
import zipfile
import pandas as pd
from datetime import datetime
from PIL import Image
from typing import Optional, Dict, Any

from config import TARGET_IMAGE_SIZE
from logger import logger
from error_handler import handle_errors
from session_manager import session_manager
from ui_components import ui
from analysis_engine import analysis_engine

from visualization import visualize_prediction, visualize_model_performance, visualize_activation_maps, visualize_feature_maps, visualize_grad_cam
from models import get_db
from patient_utils import get_patients, add_scan_to_patient

class AnalysisInterface:
    """Main interface for medical image analysis."""
    
    @staticmethod
    @handle_errors
    def render_main_analysis():
        """Render the main analysis interface."""
        st.header("🔬 Analyze Medical Image")
        
        # Input method tabs
        tab1, tab2 = st.tabs(["📁 Upload Image", "🔬 Use Sample Image"])
        
        processed_image = None
        metadata = None
        use_sample = False
        
        # Tab 1: Upload Image
        with tab1:
            uploaded_file = ui.render_file_uploader(
                "Choose a lung CT scan or X-ray image file",
                ["jpg", "jpeg", "png", "dcm"]
            )
            
            if uploaded_file:
                processed_image, metadata = analysis_engine.process_uploaded_file(uploaded_file)
        
        # Tab 2: Sample Image
        with tab2:
            sample_option = session_manager.get('sample_option', 'None')
            if sample_option != "None":
                st.write(f"Selected sample: **{sample_option}**")
                processed_image, metadata = analysis_engine.process_sample_image(sample_option)
                use_sample = True
        
        # Process the image if we have one
        if processed_image is not None:
            AnalysisInterface._process_and_analyze_image(processed_image, metadata, use_sample)
    
    @staticmethod
    def _process_and_analyze_image(image: np.ndarray, metadata: Dict, use_sample: bool):
        """Process and analyze the image."""
        # Apply enhancement if selected
        enhancement_option = session_manager.get('enhancement_option', 'None')
        enhancement_strength = session_manager.get('enhancement_strength', 1.0)
        
        if enhancement_option != "None":
            session_manager.set('last_enhancement', enhancement_option)
            final_image = analysis_engine.apply_image_enhancement(
                image, enhancement_option, enhancement_strength
            )
        else:
            session_manager.set('last_enhancement', None)
            final_image = image
        
        # Get model and run prediction
        model = session_manager.get('model')
        if model is None:
            ui.render_error_message("No model loaded. Please select a model from the sidebar.")
            return
        
        sample_name = metadata.get('sample_name') if use_sample else None
        prediction, analysis_results = analysis_engine.run_model_prediction(
            model, final_image, sample_name
        )
        
        if prediction is not None and analysis_results is not None:
            AnalysisInterface._render_analysis_results(
                final_image, prediction, analysis_results, metadata
            )
    
    @staticmethod
    def _render_analysis_results(image: np.ndarray, prediction: np.ndarray, 
                                analysis_results: Dict, metadata: Dict):
        """Render analysis results in tabs."""
        col1, col2 = st.columns(2)
        
        with col2:
            results_tab, viz_tab, save_tab = st.tabs(["📊 Results", "🎨 Visualizations", "💾 Save"])
            
            with results_tab:
                AnalysisInterface._render_results_tab(analysis_results)
            
            with viz_tab:
                AnalysisInterface._render_visualization_tab(image, analysis_results)
            
            with save_tab:
                AnalysisInterface._render_save_tab(image, analysis_results, metadata)
    
    @staticmethod
    def _render_results_tab(analysis_results: Dict):
        """Render the results tab."""
        st.subheader("🎯 Analysis Results")
        
        label = analysis_results['label']
        confidence = analysis_results['confidence']
        processing_time = analysis_results['processing_time_ms']
        
        # Display prediction with appropriate styling
        if label == "Cancer":
            st.error(f"**🔴 Prediction: {label}** (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"**🟢 Prediction: {label}** (Confidence: {confidence:.2f}%)")
        
        # Display metrics
        metrics = {
            "Processing Time": f"{processing_time:.2f} ms",
            "Model": analysis_results['model_type'],
            "Timestamp": analysis_results['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        }
        
        ui.render_metrics_cards(metrics)
        
        # Visualize prediction confidence
        visualize_prediction(analysis_results['prediction_value'])
        
        # Download section
        st.markdown("---")
        st.subheader("📥 Download Report")
        
        AnalysisInterface._render_download_options(analysis_results)
    
    @staticmethod
    def _render_download_options(analysis_results: Dict):
        """Render download options for analysis results."""
        timestamp = analysis_results['timestamp'].strftime("%Y%m%d_%H%M%S")
        
        # Prepare report data
        report_data = {
            'timestamp': timestamp,
            'model': analysis_results['model_type'],
            'prediction': analysis_results['label'],
            'confidence_percent': round(analysis_results['confidence'], 2),
            'processing_time_ms': round(analysis_results['processing_time_ms'], 2),
            'enhancement': session_manager.get('last_enhancement', 'None'),
        }
        
        # Create CSV
        try:
            csv_data = pd.DataFrame([report_data]).to_csv(index=False).encode('utf-8')
        except Exception:
            # Fallback CSV creation
            csv_header = ",".join(report_data.keys())
            csv_values = ",".join(str(v) for v in report_data.values())
            csv_data = f"{csv_header}\n{csv_values}\n".encode('utf-8')
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download CSV Report",
                data=csv_data,
                file_name=f"analysis_report_{timestamp}.csv",
                mime="text/csv",
                key="dl_csv_report"
            )
        
        with col2:
            # Create ZIP bundle (placeholder - would need image data)
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr(f"analysis_report_{timestamp}.csv", csv_data)
            
            st.download_button(
                label="📦 Download ZIP Bundle",
                data=zip_buffer.getvalue(),
                file_name=f"analysis_{timestamp}.zip",
                mime="application/zip",
                key="dl_zip_bundle"
            )
    
    @staticmethod
    def _render_visualization_tab(image: np.ndarray, analysis_results: Dict):
        """Render the visualization tab."""
        st.subheader("🎨 Visualizations")
        
        visualization_option = session_manager.get('visualization_option', 'Prediction Confidence')
        model = session_manager.get('model')
        
        if visualization_option == "Prediction Confidence":
            visualize_prediction(analysis_results['prediction_value'])
        elif visualization_option == "Class Activation Maps":
            visualize_activation_maps(image, model)
        elif visualization_option == "Feature Maps":
            visualize_feature_maps(image, model)
        elif visualization_option == "Grad-CAM":
            gradcam_layer = session_manager.get('gradcam_layer')
            visualize_grad_cam(image, model, last_conv_layer_name=gradcam_layer)
        
        st.subheader("📈 Model Performance")
        model_type = session_manager.get('model_option', 'Basic CNN')
        visualize_model_performance(model_type)
    
    @staticmethod
    def _render_save_tab(image: np.ndarray, analysis_results: Dict, metadata: Dict):
        """Render the save tab for patient records."""
        st.subheader("💾 Save Analysis")
        
        # Get available patients
        db = next(get_db())
        try:
            patients = get_patients(db)
            patient_options = [(None, "-- Select patient --")]
            patient_options.extend([
                (p.id, f"{p.last_name}, {p.first_name} (MRN: {p.medical_record_number})")
                for p in patients
            ])
        except Exception as e:
            logger.error(f"Error fetching patients: {str(e)}")
            patients = []
            patient_options = [(None, "-- Select patient --")]
        finally:
            db.close()
        
        if not patients:
            ui.render_info_message("No patients found. Create a patient to save this analysis.")
            AnalysisInterface._render_quick_patient_form()
        else:
            AnalysisInterface._render_patient_selection(
                patient_options, image, analysis_results, metadata
            )
    
    @staticmethod
    def _render_quick_patient_form():
        """Render a quick patient creation form."""
        with st.expander("➕ Create New Patient", expanded=True):
            with st.form(key="quick_patient_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    first_name = st.text_input("First Name")
                    dob = st.date_input("Date of Birth")
                    email = st.text_input("Email", placeholder="name@example.com")
                
                with col2:
                    last_name = st.text_input("Last Name")
                    gender = st.selectbox("Gender", ["", "Male", "Female", "Other"], index=0)
                    phone = st.text_input("Phone", placeholder="+1-555-123-4567")
                
                if st.form_submit_button("Create Patient"):
                    if not all([first_name, last_name, dob, gender]):
                        ui.render_warning_message("Please fill all required fields.")
                    else:
                        try:
                            from patient_utils import create_patient
                            db = next(get_db())
                            try:
                                create_patient(db, {
                                    'first_name': first_name,
                                    'last_name': last_name,
                                    'date_of_birth': dob,
                                    'gender': gender,
                                    'email': email,
                                    'phone': phone
                                })
                                ui.render_success_message("Patient created successfully!")
                                st.rerun()
                            finally:
                                db.close()
                        except Exception as e:
                            ui.render_error_message(f"Failed to create patient: {str(e)}")
    
    @staticmethod
    def _render_patient_selection(patient_options, image: np.ndarray, 
                                 analysis_results: Dict, metadata: Dict):
        """Render patient selection and save functionality."""
        # Patient selection
        selected_label = st.selectbox(
            "👤 Assign to patient",
            options=[label for _, label in patient_options],
            index=0,
            key="assign_patient_select"
        )
        
        # Find selected patient ID
        selected_patient_id = None
        for pid, label in patient_options:
            if label == selected_label:
                selected_patient_id = pid
                break
        
        # Notes input
        notes = st.text_area("📝 Notes", placeholder="Optional notes about this analysis")
        
        # Save button
        if st.button("💾 Save to Patient Record", key="save_scan_btn"):
            if not selected_patient_id:
                ui.render_warning_message("Please select a patient to save the analysis.")
            else:
                analysis_engine.save_analysis_to_patient(
                    image, analysis_results, metadata, selected_patient_id, notes
                )

# Create global analysis interface instance
analysis_interface = AnalysisInterface()