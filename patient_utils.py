from sqlalchemy.orm import Session
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from models import Patient, Scan, get_db
import uuid

def generate_medical_record_number() -> str:
    """Generate a unique medical record number"""
    return f"MRN-{str(uuid.uuid4().hex[:8]).upper()}"

def create_patient(db: Session, patient_data: Dict[str, Any]) -> Patient:
    """Create a new patient record"""
    if 'date_of_birth' in patient_data and isinstance(patient_data['date_of_birth'], str):
        patient_data['date_of_birth'] = datetime.strptime(patient_data['date_of_birth'], '%Y-%m-%d').date()
    
    if 'medical_record_number' not in patient_data or not patient_data['medical_record_number']:
        patient_data['medical_record_number'] = generate_medical_record_number()
    
    db_patient = Patient(**patient_data)
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

def get_patient(db: Session, patient_id: int) -> Optional[Patient]:
    """Retrieve a patient by ID"""
    return db.query(Patient).filter(Patient.id == patient_id).first()

def get_patients(db: Session, skip: int = 0, limit: int = 100) -> List[Patient]:
    """Retrieve a list of patients with pagination"""
    return db.query(Patient).offset(skip).limit(limit).all()

def search_patients(db: Session, search_term: str) -> List[Patient]:
    """Search patients by name or medical record number"""
    return db.query(Patient).filter(
        (Patient.first_name.ilike(f"%{search_term}%")) |
        (Patient.last_name.ilike(f"%{search_term}%")) |
        (Patient.medical_record_number.ilike(f"%{search_term}%"))
    ).all()

def update_patient(db: Session, patient_id: int, patient_data: Dict[str, Any]) -> Optional[Patient]:
    """Update an existing patient record"""
    db_patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not db_patient:
        return None
    
    for key, value in patient_data.items():
        if key == 'date_of_birth' and isinstance(value, str):
            value = datetime.strptime(value, '%Y-%m-%d').date()
        setattr(db_patient, key, value)
    
    db.commit()
    db.refresh(db_patient)
    return db_patient

def delete_patient(db: Session, patient_id: int) -> bool:
    """Delete a patient record and all associated scans"""
    db_patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not db_patient:
        return False
    
    db.delete(db_patient)
    db.commit()
    return True

def add_scan_to_patient(db: Session, patient_id: int, scan_data: Dict[str, Any]) -> Optional[Scan]:
    """Add a new scan record to a patient"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        return None
    
    scan_data['patient_id'] = patient_id
    db_scan = Scan(**scan_data)
    db.add(db_scan)
    db.commit()
    db.refresh(db_scan)
    return db_scan

def get_patient_scans(db: Session, patient_id: int) -> List[Scan]:
    """Get all scans for a specific patient"""
    return db.query(Scan).filter(Scan.patient_id == patient_id).order_by(Scan.scan_date.desc()).all()

def get_scan(db: Session, scan_id: int) -> Optional[Scan]:
    """Get a specific scan by ID"""
    return db.query(Scan).filter(Scan.id == scan_id).first()

def update_scan_results(db: Session, scan_id: int, results: Dict[str, Any]) -> Optional[Scan]:
    """Update scan analysis results"""
    db_scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not db_scan:
        return None
    
    for key, value in results.items():
        setattr(db_scan, key, value)
    
    db.commit()
    db.refresh(db_scan)
    return db_scan
