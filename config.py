"""
Configuration settings for the Lung Cancer Detection application.
"""
import os
from pathlib import Path

# Application Settings
APP_TITLE = "Lung Cancer Detection AI"
APP_ICON = "🫁"
PAGE_LAYOUT = "wide"

# Model Settings
DEFAULT_MODEL = "Basic CNN"
AVAILABLE_MODELS = ["Basic CNN", "InceptionV3 Transfer Learning"]
TARGET_IMAGE_SIZE = (224, 224)
PREDICTION_THRESHOLD = 0.5

# Database Settings
DATABASE_DIR = Path("instance")
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/patient_records.db"
UPLOADS_DIR = DATABASE_DIR / "uploads"

# Image Processing Settings
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "dcm"]
MAX_FILE_SIZE_MB = 50
DICOM_WINDOW_PERCENTILES = (5, 95)

# UI Settings
MAX_HISTORY_ENTRIES = 10
ITEMS_PER_PAGE = 20
SEARCH_RESULTS_LIMIT = 100

# Enhancement Settings
ENHANCEMENT_STRENGTH_RANGE = (0.5, 1.5)
ENHANCEMENT_DEFAULT = 1.0

# Visualization Settings
AVAILABLE_VISUALIZATIONS = [
    "Prediction Confidence", 
    "Class Activation Maps", 
    "Feature Maps", 
    "Grad-CAM"
]

# Security Settings
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.dcm'}
MAX_UPLOAD_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes

# Logging Settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance Settings
CACHE_TTL = 3600  # 1 hour in seconds

# Ensure required directories exist
def ensure_directories():
    """Create required directories if they don't exist."""
    DATABASE_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)

# Initialize directories when config is imported
ensure_directories()