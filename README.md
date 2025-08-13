# Lung Cancer Detection AI

An AI-powered tool for detecting potential signs of lung cancer from medical images.

## ✨ Features

- **AI-Powered Analysis**: Utilizes deep learning models to analyze lung CT scans and X-rays
- **Multiple Models**: Choose between Basic CNN and InceptionV3 Transfer Learning
- **Image Enhancement**: Apply various image enhancement techniques for better analysis
- **Visualization Tools**: View prediction confidence, activation maps, and feature maps
- **Patient Management**: Track patient records and their scan history
- **Model Comparison**: Compare performance metrics between different models

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lungcancerdetection.git
   cd lungcancerdetection/Lung Scan
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

## 🏥 Patient Management System

The application now includes a comprehensive patient management system that allows you to:

- Create and manage patient records
- Track patient scan history
- View detailed analysis of each scan
- Search and filter patient records

### Key Components

- **Patient Records**: Store patient information including name, date of birth, contact details, and medical record number
- **Scan History**: Track all scans performed for each patient
- **Analysis Results**: View detailed analysis including predictions and confidence scores

## 🛠️ Development

### Project Structure

- `app.py`: Main application file with Streamlit UI
- `models.py`: Database models for patients and scans
- `patient_ui.py`: UI components for patient management
- `patient_utils.py`: Utility functions for patient operations
- `model.py`: AI model definitions and loading
- `preprocessing.py`: Image preprocessing utilities
- `visualization.py`: Visualization components
- `utils.py`: General utility functions
- `sample_data.py`: Sample medical images for testing
- `image_enhancement.py`: Image enhancement utilities

### Adding New Features

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test thoroughly

3. Commit your changes:
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```

4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Open a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all contributors who have helped improve this project
- Built with ❤️ using Streamlit and TensorFlow
