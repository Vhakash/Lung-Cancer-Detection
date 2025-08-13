from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Float, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from typing import Optional

# Create SQLite database in the instance folder
os.makedirs('instance', exist_ok=True)
DATABASE_URL = 'sqlite:///instance/patient_records.db'

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency to get DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Patient(Base):
    """Patient information model"""
    __tablename__ = 'patients'
    
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    gender = Column(String(10), nullable=False)  # 'M', 'F', 'Other', 'Unknown'
    email = Column(String(120), unique=True, nullable=True)
    phone = Column(String(20), nullable=True)
    address = Column(Text, nullable=True)
    medical_record_number = Column(String(50), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to scans/analyses
    scans = relationship("Scan", back_populates="patient", cascade="all, delete-orphan")

class Scan(Base):
    """Medical scan information model"""
    __tablename__ = 'scans'
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False)
    scan_date = Column(DateTime, default=datetime.utcnow)
    scan_type = Column(String(50))  # CT, X-Ray, etc.
    file_path = Column(String(500), nullable=False)
    original_filename = Column(String(255), nullable=False)
    notes = Column(Text, nullable=True)
    
    # Analysis results
    prediction = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    findings = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="scans")

def init_db():
    """Initialize the database with tables"""
    Base.metadata.create_all(bind=engine)

# Initialize the database when this module is imported
init_db()
