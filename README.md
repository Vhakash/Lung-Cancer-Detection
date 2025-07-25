# 🫁 Lung Cancer Detection AI

This project is a Streamlit-based web application designed to assist in the early detection of lung cancer by analyzing medical images such as CT scans and X-rays. The application uses AI models and various image enhancement techniques to provide predictions and visualizations that can aid in diagnosis.

---

## Features

- **Medical Image Analysis**: Upload lung CT scans or X-ray images for analysis.
- **AI-Powered Predictions**: Detect potential signs of lung cancer with confidence scores.
- **Visualization Tools**:
  - Prediction Confidence
  - Class Activation Maps
  - Feature Maps
- **Image Enhancement**:
  - Contrast Enhancement
  - Histogram Equalization
  - Adaptive Histogram Equalization
  - Gaussian Smoothing
  - Edge Enhancement
  - Sharpening
- **Sample Images**: Test the app using preloaded sample medical images.
- **Model Comparison**: Compare the performance of different AI models.
- **Analysis History**: View and manage the history of analyzed images.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Required Python libraries:
  - `streamlit`
  - `numpy`
  - `matplotlib`
  - `pillow`
  - `pydicom`
  - `opencv-python`
  - `seaborn`

#### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Lung-Cancer-Detection.git
   cd Lung-Cancer-Detection
   ```
2. Install the required Python libraries:
   ```bash
   pip install streamlit numpy matplotlib pillow pydicom opencv-python seaborn
   ```
3. (Optional) Install additional libraries for enhanced image processing:
   ```bash
   pip install scikit-image
   ```

---
### Install dependencies
```bash
pip install -r requirements.txt
```
### Run the application
```bash
streamlit run app.py
```

## Usage

1. Launch the Streamlit web application:
    ```bash
    streamlit run app.py
    ```
2. Open your web browser and go to `http://localhost:8501`.
3. Upload your lung CT scan or X-ray image.
4. Explore the AI-powered predictions and visualizations.
5. (Optional) Compare different AI models and view analysis history.


## Usage
1. Upload an Image: Upload a lung CT scan or X-ray image, or select a sample image from the sidebar.
2. Select Model and Enhancements:
        Choose an AI model (e.g., Basic CNN or InceptionV3 Transfer Learning).
        Apply optional image enhancement techniques.
3. Analyze the Image:
        View predictions with confidence scores.
        Explore visualizations such as activation maps and feature maps.
4. Compare Models: Use the "Compare Models" button to evaluate different AI models.
5. Manage History: View or clear the analysis history from the sidebar.
---

## Project Structure

```
Lung-Cancer-Detection/
├── app.py                     # Main Streamlit application
├── model.py                   # AI model-related functions
├── preprocessing.py           # Image preprocessing utilities
├── visualization.py           # Visualization functions
├── utils.py                   # Utility functions
├── sample_data.py             # Sample image-related functions
├── image_enhancement.py       # Image enhancement techniques
└── requirements.txt           # Python dependencies
```
## Key Files
app.py: The main application file that integrates all functionalities.

image_enhancement.py: Contains various image enhancement techniques such as contrast adjustment, histogram equalization, and sharpening.

visualization.py: Provides tools for visualizing predictions, activation maps, and feature maps.
model.py: Defines AI models and their loading mechanisms.

preprocessing.py: Handles image preprocessing tasks like normalization and color channel adjustments.

utils.py: Includes utility functions for DICOM file handling, history management, and prediction confidence calculation.
---

##Image Enhancement Techniques
The following enhancement techniques are available in the app:

1. Contrast Enhancement: Improves the contrast of the image.
2. Histogram Equalization: Enhances the image by redistributing pixel intensities.
3. Adaptive Histogram Equalization: Applies localized histogram equalization for better contrast.
4. Gaussian Smoothing: Reduces noise and smoothens the image.
5. Edge Enhancement: Highlights edges in the image for better feature visibility.
6. Sharpening: Enhances the sharpness of the image.

## License

This project is licensed under the MIT License.

## Disclaimer

This tool is for research and educational purposes only. It is not intended for clinical use.
