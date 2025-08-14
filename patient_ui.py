import streamlit as st
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from models import Patient, Scan, get_db
from patient_utils import *
import os

def show_patient_form(patient: Optional[Patient] = None) -> Dict[str, Any]:
    """Render patient form and return form data"""
    is_edit = patient is not None
    form_data = {}
    
    with st.form(key='patient_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            form_data['first_name'] = st.text_input("First Name", value=patient.first_name if is_edit else "")
            form_data['last_name'] = st.text_input("Last Name", value=patient.last_name if is_edit else "")
            form_data['date_of_birth'] = st.date_input(
                "Date of Birth", 
                value=patient.date_of_birth if is_edit else datetime(1990, 1, 1).date(),
                max_value=datetime.now().date()
            )
            form_data['gender'] = st.selectbox(
                "Gender", 
                ["Male", "Female", "Other", "Unknown"],
                index=0 if not is_edit else ["Male", "Female", "Other", "Unknown"].index(patient.gender)
            )
        
        with col2:
            form_data['email'] = st.text_input("Email", value=patient.email if is_edit else "")
            form_data['phone'] = st.text_input("Phone", value=patient.phone if is_edit else "")
            form_data['address'] = st.text_area("Address", value=patient.address if is_edit else "")
        
        if st.form_submit_button("Save Patient"):
            return form_data
    return None

def show_patient_list(db: Session):
    """Show list of patients with search and pagination"""
    st.subheader("Patient Records")
    
    # Search bar
    search_term = st.text_input("Search patients")
    
    if search_term:
        patients = search_patients(db, search_term)
    else:
        patients = get_patients(db)
    
    # Display patients in a table
    if patients:
        for patient in patients:
            with st.expander(f"{patient.last_name}, {patient.first_name} - {patient.medical_record_number}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**DOB:** {patient.date_of_birth}")
                    st.write(f"**Email:** {patient.email or 'N/A'}")
                    st.write(f"**Phone:** {patient.phone or 'N/A'}")
                with col2:
                    if st.button("View Details", key=f"view_{patient.id}"):
                        st.session_state['selected_patient_id'] = patient.id
                    if st.button("Delete", key=f"del_{patient.id}"):
                        if delete_patient(db, patient.id):
                            st.success("Patient deleted successfully")
                            st.rerun()
    else:
        st.info("No patients found")

def show_patient_details(db: Session, patient_id: int):
    """Show detailed view of a single patient"""
    patient = get_patient(db, patient_id)
    if not patient:
        st.error("Patient not found")
        return
    
    st.subheader(f"{patient.first_name} {patient.last_name}")
    
    # Patient info
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Medical Record #:** {patient.medical_record_number}")
            st.write(f"**Date of Birth:** {patient.date_of_birth}")
            st.write(f"**Gender:** {patient.gender}")
        with col2:
            st.write(f"**Email:** {patient.email or 'N/A'}")
            st.write(f"**Phone:** {patient.phone or 'N/A'}")
    
    # Scan history
    st.subheader("Scan History")
    scans = get_patient_scans(db, patient_id)
    
    if scans:
        for scan in scans:
            with st.expander(f"{scan.scan_date.strftime('%Y-%m-%d %H:%M')} - {scan.scan_type or 'Unknown'}"):
                # Show image thumbnail if available
                if scan.file_path and os.path.exists(scan.file_path):
                    st.image(scan.file_path, caption=os.path.basename(scan.file_path), use_container_width=True)
                st.write(f"**Prediction:** {scan.prediction or 'N/A'}")
                st.write(f"**Confidence:** {f'{scan.confidence:.2f}%' if scan.confidence else 'N/A'}")
                st.write(f"**Notes:** {scan.notes or 'No notes'}")
    else:
        st.info("No scan history available")
    
    if st.button("Back to Patient List"):
        if 'selected_patient_id' in st.session_state:
            del st.session_state['selected_patient_id']
        st.rerun()
