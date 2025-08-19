"""
Test script to verify the improvements work correctly.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test that all new modules can be imported."""
    try:
        from config import APP_TITLE, DATABASE_URL
        from logger import logger
        from error_handler import handle_errors, ValidationError
        from session_manager import SessionManager
        from ui_components import UIComponents
        from analysis_engine import AnalysisEngine
        from views import Views
        from analysis_interface import AnalysisInterface
        
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration system."""
    try:
        from config import APP_TITLE, ensure_directories
        
        assert APP_TITLE == "Lung Cancer Detection AI"
        ensure_directories()  # Should create directories without error
        
        print("✅ Configuration system working!")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_logger():
    """Test logging system."""
    try:
        from logger import logger
        
        logger.info("Test log message")
        print("✅ Logging system working!")
        return True
    except Exception as e:
        print(f"❌ Logging error: {e}")
        return False

def test_error_handling():
    """Test error handling system."""
    try:
        from error_handler import ValidationError, validate_file_upload
        
        # Test validation error
        try:
            validate_file_upload(None)
            print("❌ Validation should have failed")
            return False
        except ValidationError:
            print("✅ Error handling system working!")
            return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_session_manager():
    """Test session manager."""
    try:
        from session_manager import SessionManager
        
        # Test basic functionality (without Streamlit context)
        sm = SessionManager()
        print("✅ Session manager created successfully!")
        return True
    except Exception as e:
        print(f"❌ Session manager error: {e}")
        return False

def test_real_model():
    """Test the new real CNN model."""
    try:
        from model import LungCancerCNN, create_model
        import numpy as np
        
        # Test model creation
        model = LungCancerCNN(model_type="basic")
        print("✅ Real CNN model created successfully!")
        
        # Test prediction with dummy data
        dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
        prediction = model.predict(dummy_image)
        
        if prediction is not None and prediction.shape == (1, 1):
            print("✅ Model prediction working!")
        else:
            print("❌ Model prediction failed!")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Real model test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("🧪 Running improvement tests...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Logger Test", test_logger),
        ("Error Handling Test", test_error_handling),
        ("Session Manager Test", test_session_manager),
        ("Real Model Test", test_real_model),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your improvements are working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)