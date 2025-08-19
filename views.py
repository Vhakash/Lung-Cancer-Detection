"""
Different view components for the application.
"""
import streamlit as st
import matplotlib.pyplot as plt
import os
from typing import Optional

from logger import logger
from error_handler import handle_errors
from session_manager import session_manager
from ui_components import ui
from models import get_db, Patient, Scan
from patient_ui import show_patient_form, show_patient_list, show_patient_details
from patient_utils import create_patient
from utils import compare_model_performances

class Views:
    """Collection of different application views."""
    
    @staticmethod
    @handle_errors
    def render_model_comparison():
        """Render the model comparison view."""
        st.subheader("🔬 Model Performance Comparison")
        
        # Get performance metrics
        metrics_df = compare_model_performances()
        
        # Display the metrics table
        ui.render_data_table(metrics_df, "Performance Metrics")
        
        # Create a bar chart for visual comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get metrics without the model names and processing time for bar chart
        plot_metrics = metrics_df.drop(columns=['Model Type', 'Processing Time (ms)'])
        
        # Plot
        plot_metrics.plot(kind='bar', ax=ax)
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        
        st.pyplot(fig)
        
        # Add explanatory text
        st.markdown("""
        ### 📊 Comparison Insights
        - **Transfer Learning Model** generally shows better performance across all metrics
        - The improved performance comes at a cost of slightly increased processing time
        - For critical diagnostic applications, the Transfer Learning model would be preferred
        - For faster screening applications where speed is important, the Basic CNN may be sufficient
        """)
        
        # Navigation button
        if st.button("🔙 Back to Analysis", key="close_compare_view"):
            session_manager.navigate_to('analyze')
            st.rerun()
    
    @staticmethod
    @handle_errors
    def render_analysis_history():
        """Render the analysis history view."""
        st.subheader("📋 Analysis History")
        
        # Filters
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            name_filter = st.text_input("🔍 Search by patient name")
        with col2:
            pred_filter = st.selectbox("Prediction Filter", ["All", "Cancer", "Healthy"], index=0)
        with col3:
            limit = st.selectbox("Show Results", [20, 50, 100], index=0)
        
        # Get scans from database
        db = next(get_db())
        try:
            query = db.query(Scan).order_by(Scan.scan_date.desc()).limit(limit)
            scans = query.all()
        except Exception as e:
            ui.render_error_message(f"Failed to load history: {str(e)}")
            logger.error(f"Database error in history view: {str(e)}")
            scans = []
        finally:
            db.close()
        
        # Apply filters
        filtered_rows = Views._filter_scan_results(scans, name_filter, pred_filter)
        
        if not filtered_rows:
            ui.render_info_message("No analyses found matching your criteria.")
        else:
            Views._render_scan_results(filtered_rows)
        
        # Navigation button
        if st.button("🔙 Back to Analysis", key="close_history_view"):
            session_manager.navigate_to('analyze')
            st.rerun()
    
    @staticmethod
    def _filter_scan_results(scans, name_filter: str, pred_filter: str):
        """Filter scan results based on criteria."""
        def match_name(patient):
            if not name_filter or not patient:
                return True
            full_name = f"{patient.last_name}, {patient.first_name}".lower()
            return name_filter.lower() in full_name
        
        filtered_rows = []
        if scans:
            db = next(get_db())
            try:
                for scan in scans:
                    patient = None
                    if scan.patient_id:
                        patient = db.query(Patient).filter(Patient.id == scan.patient_id).first()
                    
                    # Apply prediction filter
                    scan_label = scan.prediction or "N/A"
                    if pred_filter != "All":
                        filter_label = "Cancer" if pred_filter == "Cancer" else "Healthy"
                        if scan_label != filter_label:
                            continue
                    
                    # Apply name filter
                    if not match_name(patient):
                        continue
                    
                    filtered_rows.append((scan, patient))
            finally:
                db.close()
        
        return filtered_rows
    
    @staticmethod
    def _render_scan_results(rows):
        """Render scan results in a grid layout."""
        for scan, patient in rows:
            with st.container():
                col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                
                with col1:
                    # Thumbnail
                    if scan.file_path and os.path.exists(scan.file_path):
                        st.image(scan.file_path, use_container_width=True)
                    else:
                        st.caption("📷 No image")
                
                with col2:
                    patient_name = f"{patient.first_name} {patient.last_name}" if patient else "Unknown"
                    st.write(f"**👤 Patient:** {patient_name}")
                    scan_date = scan.scan_date.strftime('%Y-%m-%d %H:%M') if scan.scan_date else 'N/A'
                    st.write(f"**📅 Date:** {scan_date}")
                
                with col3:
                    prediction = scan.prediction or 'N/A'
                    confidence = f'{scan.confidence:.2f}%' if scan.confidence is not None else 'N/A'
                    
                    # Color code predictions
                    if prediction == "Cancer":
                        st.error(f"**🔴 Prediction:** {prediction}")
                    elif prediction == "Healthy":
                        st.success(f"**🟢 Prediction:** {prediction}")
                    else:
                        st.write(f"**❓ Prediction:** {prediction}")
                    
                    st.write(f"**📊 Confidence:** {confidence}")
                    
                    if scan.notes:
                        st.caption(f"📝 {scan.notes}")
                
                with col4:
                    if patient and st.button("👁️ View", key=f"view_patient_{scan.id}"):
                        session_manager.set('selected_patient_id', patient.id)
                        session_manager.navigate_to('patients')
                        st.rerun()
                
                st.divider()
    
    @staticmethod
    @handle_errors
    def render_patient_management():
        """Render patient management views."""
        # Check if we're showing patient form
        if session_manager.get('show_patient_form', False):
            Views._render_patient_form()
        # Check if we're showing specific patient details
        elif session_manager.get('selected_patient_id'):
            Views._render_patient_details()
        # Default to patient list
        else:
            Views._render_patient_list()
    
    @staticmethod
    def _render_patient_form():
        """Render the add/edit patient form."""
        st.header("➕ Add New Patient")
        
        form_data = show_patient_form()
        if form_data:
            db = next(get_db())
            try:
                create_patient(db, form_data)
                db.commit()
                ui.render_success_message("Patient created successfully!")
                session_manager.set('show_patient_form', False)
                session_manager.navigate_to('patients')
                st.rerun()
            except Exception as e:
                db.rollback()
                ui.render_error_message(f"Error creating patient: {str(e)}")
                logger.error(f"Error creating patient: {str(e)}")
            finally:
                db.close()
        
        # Navigation button
        if st.button("🔙 Back to Patient List", key="back_to_patients"):
            session_manager.set('show_patient_form', False)
            st.rerun()
    
    @staticmethod
    def _render_patient_details():
        """Render detailed patient view."""
        patient_id = session_manager.get('selected_patient_id')
        if patient_id:
            db = next(get_db())
            try:
                show_patient_details(db, patient_id)
            finally:
                db.close()
    
    @staticmethod
    def _render_patient_list():
        """Render the patient list view."""
        db = next(get_db())
        try:
            show_patient_list(db)
        finally:
            db.close()
        
        # Navigation button
        if st.button("🔙 Back to Analysis", key="back_to_analysis"):
            session_manager.navigate_to('analyze')
            st.rerun()
    
    @staticmethod
    def render_about():
        """Render the about page."""
        st.subheader("ℹ️ About Lung Cancer Detection AI")
        
        st.markdown("""
        ### 🎯 Purpose
        This application is designed to assist medical professionals in the early detection 
        of lung cancer through AI-powered analysis of medical images.
        
        ### 🔬 Technology
        - **Deep Learning Models**: CNN and Transfer Learning approaches
        - **Image Processing**: Advanced preprocessing and enhancement techniques
        - **Visualization**: Multiple visualization methods for model interpretation
        - **Patient Management**: Comprehensive patient record system
        
        ### ⚠️ Important Disclaimer
        This tool is for **educational and research purposes only**. It should not be used 
        as a substitute for professional medical diagnosis. Always consult with qualified 
        healthcare professionals for medical decisions.
        
        ### 🛠️ Features
        - 🔍 AI-powered image analysis
        - 📊 Multiple visualization options
        - 👥 Patient record management
        - 📈 Model performance comparison
        - 📋 Analysis history tracking
        
        ### 📞 Support
        For technical support or questions, please refer to the documentation or 
        contact the development team.
        """)
        
        # Navigation button
        if st.button("🔙 Back to Analysis", key="back_from_about"):
            session_manager.navigate_to('analyze')
            st.rerun()

# Create global views instance
views = Views()