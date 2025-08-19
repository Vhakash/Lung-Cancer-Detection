# 🫁 Lung Cancer Detection AI

An advanced AI-powered tool for detecting potential signs of lung cancer from medical images, featuring a modern modular architecture and comprehensive patient management system.

## ✨ Key Features

### 🔬 AI-Powered Analysis
- **Real CNN Models**: Professional-grade Basic CNN (656K params) and EfficientNetB0 Transfer Learning (4.8M params)
- **Advanced Preprocessing**: Medical-grade DICOM processing with windowing and CLAHE
- **Image Enhancement**: Multiple enhancement techniques for better analysis
- **Training Capabilities**: Train models on real medical data or synthetic data for demonstration
- **Real-time Predictions**: Fast, accurate analysis with confidence scoring

### 📊 Visualization & Interpretation
- **Prediction Confidence**: Interactive gauge charts and metrics
- **Class Activation Maps**: Highlight regions of interest
- **Feature Maps**: Visualize what the model sees
- **Grad-CAM**: Advanced gradient-based visualization (with TensorFlow)

### 👥 Patient Management
- **Complete Patient Records**: Demographics, contact info, medical record numbers
- **Scan History Tracking**: Full audit trail of all analyses
- **Search & Filter**: Find patients and scans quickly
- **Data Export**: Download reports in CSV and ZIP formats

### 🏗️ Modern Architecture
- **Modular Design**: Clean separation of concerns
- **Error Handling**: Comprehensive error management and logging
- **Configuration Management**: Centralized settings and constants
- **Session Management**: Robust state management
- **UI Components**: Reusable interface elements

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone and navigate to the project:**
   ```bash
   git clone <your-repo-url>
   cd "Lung Scan"
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser to:** `http://localhost:8501`

## 🏗️ Architecture Overview

### Core Modules

```
📁 Lung Scan/
├── 🎯 app.py                    # Main application entry point
├── ⚙️ config.py                 # Configuration management
├── 📝 logger.py                 # Centralized logging
├── 🛡️ error_handler.py          # Error handling utilities
├── 🔄 session_manager.py        # Session state management
├── 🎨 ui_components.py          # Reusable UI components
├── 🔬 analysis_engine.py        # Core analysis logic
├── 📱 analysis_interface.py     # Main analysis UI
├── 👁️ views.py                  # Different application views
├── 🤖 model.py                  # AI model management
├── 🖼️ preprocessing.py          # Image processing
├── 📊 visualization.py          # Data visualization
├── 👥 patient_*.py              # Patient management
└── 🧪 test_improvements.py      # Test suite
```

### Key Improvements Made

#### 1. **Modular Architecture**
- Separated concerns into focused modules
- Reduced main app.py from 1000+ lines to ~50 lines
- Improved maintainability and testability

#### 2. **Error Handling & Logging**
- Comprehensive error handling with custom exceptions
- Centralized logging system with file and console output
- User-friendly error messages with technical logging

#### 3. **Configuration Management**
- Centralized configuration in `config.py`
- Environment-specific settings
- Easy customization without code changes

#### 4. **Session State Management**
- Robust session state handling
- Type-safe session operations
- Automatic cleanup and initialization

#### 5. **UI/UX Improvements**
- Reusable UI components
- Consistent styling and messaging
- Better loading indicators and feedback
- Improved navigation and user flow

## 🔧 Configuration

### Environment Setup
The application automatically creates required directories and uses SQLite for data storage. Key settings can be modified in `config.py`:

```python
# Model Settings
DEFAULT_MODEL = "Basic CNN"
TARGET_IMAGE_SIZE = (224, 224)

# Database Settings
DATABASE_URL = "sqlite:///instance/patient_records.db"

# UI Settings
MAX_HISTORY_ENTRIES = 10
```

### Logging
Logs are written to both console and `instance/app.log`. Adjust log level in `config.py`:

```python
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
```

## 🤖 Training Your Own Models

### Quick Demo Training
Train models on synthetic data for demonstration:
```bash
python train_model.py --demo
```

### Training on Real Medical Data

1. **Prepare your dataset structure:**
   ```bash
   python train_model.py --create_structure
   ```

2. **Organize your data:**
   ```
   sample_dataset/
   ├── train/
   │   ├── normal/     # Normal lung images
   │   └── cancer/     # Cancer lung images
   └── validation/
       ├── normal/     # Normal validation images
       └── cancer/     # Cancer validation images
   ```

3. **Train models:**
   ```bash
   # Train basic CNN
   python train_model.py --data_path sample_dataset --model_type basic --epochs 20
   
   # Train transfer learning model
   python train_model.py --data_path sample_dataset --model_type transfer --epochs 15
   
   # Train both models
   python train_model.py --data_path sample_dataset --model_type both --epochs 20
   ```

### Model Architecture Details

**Basic CNN (656,577 parameters):**
- 4 Convolutional blocks with BatchNorm and Dropout
- Global Average Pooling
- 2 Dense layers with regularization
- Optimized for medical imaging

**Transfer Learning (4,845,220 parameters):**
- EfficientNetB0 backbone (ImageNet pretrained)
- Custom classification head
- Fine-tuning capabilities
- Superior performance on complex cases

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
python test_improvements.py
```

## 📊 Usage Guide

### 1. **Analyze Images**
- Upload DICOM files or standard images (JPG, PNG)
- Try sample images for demonstration
- Apply image enhancements
- View detailed analysis results

### 2. **Manage Patients**
- Create patient records with demographics
- View patient scan history
- Search and filter patients
- Export patient data

### 3. **Compare Models**
- View performance metrics
- Compare different AI models
- Understand model strengths and limitations

### 4. **Review History**
- Browse all past analyses
- Filter by patient or prediction
- Jump to patient details

## 🛠️ Development

### Adding New Features

1. **Create feature branch:**
   ```bash
   git checkout -b feature/new-feature
   ```

2. **Follow the modular pattern:**
   - Add configuration to `config.py`
   - Create reusable components in appropriate modules
   - Add error handling and logging
   - Update tests

3. **Test your changes:**
   ```bash
   python test_improvements.py
   streamlit run app.py
   ```

### Code Style
- Use type hints where possible
- Follow PEP 8 conventions
- Add docstrings to functions
- Handle errors gracefully
- Log important operations

### Database Schema
The application uses SQLAlchemy with SQLite:
- **Patients**: Demographics and contact information
- **Scans**: Image analyses linked to patients

## ⚠️ Important Disclaimers

- **For Educational/Research Use Only**: Not for clinical diagnosis
- **Mock Models**: Current models are for demonstration
- **Data Privacy**: Ensure HIPAA compliance for real patient data
- **Medical Supervision**: Always consult healthcare professionals

## 🔮 Future Enhancements

### Planned Features
- [x] **Real trained models integration** ✅ **COMPLETED**
- [ ] 3D DICOM series support
- [ ] PDF report generation
- [ ] User authentication system
- [ ] Audit logging for compliance
- [ ] Batch processing capabilities
- [ ] API endpoints for integration
- [ ] Docker containerization
- [ ] Model fine-tuning on custom datasets
- [ ] Ensemble model predictions

### Performance Optimizations
- [ ] Model caching and optimization
- [ ] Database indexing improvements
- [ ] Image processing acceleration
- [ ] Memory usage optimization

## 📞 Support

For technical support or questions:
- Check the test suite: `python test_improvements.py`
- Review logs in `instance/app.log`
- Ensure all dependencies are installed
- Verify Python version compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ using Streamlit, TensorFlow, and SQLAlchemy
- Thanks to the open-source community for excellent tools
- Medical imaging community for guidance on best practices

---

**Ready to analyze? Run `streamlit run app.py` and start detecting! 🚀**
