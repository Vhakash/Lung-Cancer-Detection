"""
Security utilities for medical data protection and HIPAA compliance.
"""
import hashlib
import secrets
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from cryptography.fernet import Fernet
from logger import logger

class SecurityManager:
    """Manage security features for medical data protection."""
    
    def __init__(self):
        self.encryption_key = self._get_or_create_key()
        self.cipher_suite = Fernet(self.encryption_key)
    
    def _get_or_create_key(self) -> bytes:
        """Get existing encryption key or create new one."""
        key_file = "instance/encryption.key"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Create new key
            key = Fernet.generate_key()
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            logger.info("Created new encryption key")
            return key
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive patient data."""
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return encrypted_data.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive patient data."""
        try:
            decrypted_data = self.cipher_suite.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    @staticmethod
    def hash_patient_id(patient_id: str, salt: Optional[str] = None) -> str:
        """Create secure hash of patient ID for anonymization."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        combined = f"{patient_id}{salt}"
        hashed = hashlib.sha256(combined.encode()).hexdigest()
        return f"{salt}:{hashed}"
    
    @staticmethod
    def verify_patient_id_hash(patient_id: str, hashed_value: str) -> bool:
        """Verify patient ID against its hash."""
        try:
            salt, expected_hash = hashed_value.split(':', 1)
            combined = f"{patient_id}{salt}"
            actual_hash = hashlib.sha256(combined.encode()).hexdigest()
            return actual_hash == expected_hash
        except Exception:
            return False
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks."""
        # Remove path separators and dangerous characters
        dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
        sanitized = filename
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Limit length
        if len(sanitized) > 100:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:95] + ext
        
        return sanitized
    
    @staticmethod
    def validate_file_upload(file_content: bytes, allowed_extensions: List[str]) -> Tuple[bool, str]:
        """Validate uploaded file for security."""
        # Check file size (max 50MB)
        max_size = 50 * 1024 * 1024
        if len(file_content) > max_size:
            return False, "File size exceeds 50MB limit"
        
        # Check for malicious patterns
        malicious_patterns = [
            b'<script',
            b'javascript:',
            b'<?php',
            b'<%',
            b'exec(',
            b'system(',
            b'shell_exec('
        ]
        
        content_lower = file_content.lower()
        for pattern in malicious_patterns:
            if pattern in content_lower:
                return False, "File contains potentially malicious content"
        
        return True, "File validation passed"
    
    @staticmethod
    def log_access_attempt(user_info: Dict, resource: str, success: bool):
        """Log access attempts for audit trail."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_info': user_info,
            'resource': resource,
            'success': success,
            'ip_address': user_info.get('ip_address', 'unknown')
        }
        
        if success:
            logger.info(f"Access granted: {log_entry}")
        else:
            logger.warning(f"Access denied: {log_entry}")
    
    @staticmethod
    def generate_session_token() -> str:
        """Generate secure session token."""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def is_token_expired(token_timestamp: datetime, expiry_hours: int = 24) -> bool:
        """Check if session token is expired."""
        expiry_time = token_timestamp + timedelta(hours=expiry_hours)
        return datetime.now() > expiry_time

class HIPAACompliance:
    """HIPAA compliance utilities for medical data handling."""
    
    @staticmethod
    def anonymize_patient_data(patient_data: Dict) -> Dict:
        """Anonymize patient data for research purposes."""
        anonymized = patient_data.copy()
        
        # Remove direct identifiers
        identifiers_to_remove = [
            'first_name', 'last_name', 'email', 'phone', 
            'address', 'medical_record_number'
        ]
        
        for identifier in identifiers_to_remove:
            if identifier in anonymized:
                del anonymized[identifier]
        
        # Generalize date of birth to age ranges
        if 'date_of_birth' in anonymized:
            dob = anonymized['date_of_birth']
            if isinstance(dob, str):
                dob = datetime.strptime(dob, '%Y-%m-%d').date()
            
            age = datetime.now().date().year - dob.year
            
            # Age ranges for anonymization
            if age < 18:
                age_range = "Under 18"
            elif age < 30:
                age_range = "18-29"
            elif age < 50:
                age_range = "30-49"
            elif age < 70:
                age_range = "50-69"
            else:
                age_range = "70+"
            
            anonymized['age_range'] = age_range
            del anonymized['date_of_birth']
        
        # Add anonymization timestamp
        anonymized['anonymized_at'] = datetime.now().isoformat()
        
        return anonymized
    
    @staticmethod
    def create_audit_log_entry(action: str, user_id: str, patient_id: str, details: Dict) -> Dict:
        """Create standardized audit log entry."""
        return {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'user_id': user_id,
            'patient_id': SecurityManager.hash_patient_id(patient_id),
            'details': details,
            'compliance_version': '1.0'
        }
    
    @staticmethod
    def get_data_retention_policy() -> Dict:
        """Get data retention policy for different data types."""
        return {
            'patient_records': {'retention_years': 7, 'description': 'Medical records retention'},
            'audit_logs': {'retention_years': 6, 'description': 'HIPAA audit trail'},
            'research_data': {'retention_years': 10, 'description': 'Anonymized research data'},
            'system_logs': {'retention_years': 1, 'description': 'Application logs'}
        }

# Global security manager instance
security_manager = SecurityManager()
hipaa_compliance = HIPAACompliance()