import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from PIL import Image
import pydicom
import io
import time
import random

# Import our functionality from module files
from model import MockModel, create_model, load_pretrained_model
from preprocessing import preprocess_image, ensure_color_channels
from visualization import visualize_prediction, visualize_model_performance, visualize_activation_maps, visualize_feature_maps
from utils import read_dicom_file, display_dicom_info, calculate_prediction_confidence, add_to_history, get_analysis_history, clear_analysis_history, compare_model_performances, initialize_analysis_history
from sample_data import get_sample_image, get_sample_image_names
from image_enhancement import apply_enhancement, get_available_enhancements

# Initialize session state for tracking analysis history
initialize_analysis_history()

# Set page configuration
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🫁",
    layout="wide"
)

# Application title and description
st.title("Lung Cancer Detection using CNN")
st.markdown("""
This application uses a Convolutional Neural Network (CNN) to detect the presence of lung cancer in medical images.
Upload a lung CT scan or X-ray image to get a prediction.
""")

# Sidebar with options
st.sidebar.title("Settings")

# Model options
st.sidebar.subheader("Model Options")
model_option = st.sidebar.selectbox(
    "Select Model",
    ["Basic CNN", "InceptionV3 Transfer Learning"],
    index=0
)

# Visualization options
st.sidebar.subheader("Visualization Options")
visualization_option = st.sidebar.selectbox(
    "Visualization Type",
    ["Prediction Confidence", "Class Activation Maps", "Feature Maps"],
    index=0
)

# Image Enhancement options
st.sidebar.subheader("Image Enhancement")
enhancement_option = st.sidebar.selectbox(
    "Enhancement Type",
    ["None"] + get_available_enhancements(),
    index=0
)

enhancement_strength = st.sidebar.slider(
    "Enhancement Strength",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.1,
    disabled=(enhancement_option == "None")
)

# Sample Images
st.sidebar.subheader("Sample Images")
sample_option = st.sidebar.selectbox(
    "Select Sample Image",
    ["None"] + get_sample_image_names(),
    index=0
)

# Model Comparison
if st.sidebar.button("Compare Models"):
    st.session_state.show_model_comparison = True
else:
    if 'show_model_comparison' not in st.session_state:
        st.session_state.show_model_comparison = False

# Analysis History
if st.sidebar.button("View Analysis History"):
    st.session_state.show_history = True
else:
    if 'show_history' not in st.session_state:
        st.session_state.show_history = False
        
# Clear History
if st.sidebar.button("Clear History"):
    clear_analysis_history()
    st.sidebar.success("Analysis history cleared!")
    if 'show_history' in st.session_state:
        st.session_state.show_history = False

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    if model_option == "Basic CNN":
        st.session_state.model = create_model()
    else:
        st.session_state.model = load_pretrained_model()

# Check if model changed
if 'model_option' not in st.session_state or st.session_state.model_option != model_option:
    st.session_state.model_option = model_option
    if model_option == "Basic CNN":
        st.session_state.model = create_model()
    else:
        st.session_state.model = load_pretrained_model()

# Model Comparison View
if st.session_state.show_model_comparison:
    st.subheader("Model Comparison")
    
    # Get performance metrics
    metrics_df = compare_model_performances()
    
    # Display the metrics table
    st.dataframe(metrics_df)
    
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
    ### Comparison Insights
    - **Transfer Learning Model** generally shows better performance across all metrics.
    - The improved performance comes at a cost of slightly increased processing time.
    - For critical diagnostic applications, the Transfer Learning model would be preferred.
    - For faster screening applications where speed is important, the Basic CNN may be sufficient.
    """)
    
    # Button to close the comparison view
    if st.button("Close Comparison View"):
        st.session_state.show_model_comparison = False
        st.rerun()

# History View
if st.session_state.show_history:
    st.subheader("Analysis History")
    
    # Get history
    history = get_analysis_history()
    
    if not history:
        st.info("No analysis history available. Analyze some images to build history.")
    else:
        # Create tabs for each history entry
        tab_labels = [f"Analysis {i+1} - {h['timestamp']}" for i, h in enumerate(history)]
        tabs = st.tabs(tab_labels)
        
        for i, (tab, entry) in enumerate(zip(tabs, history)):
            with tab:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(entry['image'], caption=f"Image {i+1}", use_column_width=True)
                    
                with col2:
                    st.write(f"**Timestamp:** {entry['timestamp']}")
                    st.write(f"**Model:** {entry['model_type']}")
                    
                    # Display enhancement info if available
                    if entry['enhancement']:
                        st.write(f"**Enhancement:** {entry['enhancement']}")
                    
                    # Display prediction result
                    if entry['prediction_label'] == 'Cancer':
                        st.error(f"**Prediction:** {entry['prediction_label']} (Confidence: {entry['confidence']:.2f}%)")
                    else:
                        st.success(f"**Prediction:** {entry['prediction_label']} (Confidence: {entry['confidence']:.2f}%)")
        
        # Button to close history view
        if st.button("Close History View"):
            st.session_state.show_history = False
            st.rerun()
            
# If we're not in a special view, show the main interface
if not (st.session_state.show_model_comparison or st.session_state.show_history):
    # Create header section for main interface
    st.header("Analyze Medical Image")
    
    # Input method tabs
    tab1, tab2 = st.tabs(["Upload Image", "Use Sample Image"])
    
    use_sample = False
    
    # Tab 1: Upload Image
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose a lung CT scan or X-ray image file", 
            type=["jpg", "jpeg", "png", "dcm"]
        )
    
    # Tab 2: Sample Image
    with tab2:
        if sample_option != "None":
            st.write(f"Selected sample: **{sample_option}**")
            sample_image = get_sample_image(sample_option)
            if sample_image:
                st.image(sample_image, caption=f"Sample Image: {sample_option}", use_column_width=True)
                use_sample = True
    
    # Process images (either uploaded or sample)
    if uploaded_file is not None or use_sample:
        # Creating columns for display
        col1, col2 = st.columns(2)
        
        try:
            # Process uploaded file
            if uploaded_file is not None:
                # Determine file type and read accordingly
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'dcm':
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name
                    
                    # Read DICOM file
                    image_data, pixel_array = read_dicom_file(temp_file_path)
                    
                    # Display DICOM information
                    with col1:
                        st.subheader("Original Image")
                        st.image(pixel_array, caption="Original DICOM Image", use_column_width=True)
                        display_dicom_info(image_data)
                    
                    # Convert to format suitable for model
                    processed_image = preprocess_image(pixel_array)
                    
                    # Clean up the temp file
                    os.unlink(temp_file_path)
                    
                else:
                    # For other image formats
                    image = Image.open(uploaded_file)
                    
                    with col1:
                        st.subheader("Original Image")
                        st.image(image, caption=f"Original Image: {uploaded_file.name}", use_column_width=True)
                    
                    # Convert to numpy array and preprocess
                    image_array = np.array(image)
                    # Ensure image has proper color channels
                    image_array = ensure_color_channels(image_array)
                    processed_image = preprocess_image(image_array)
            
            # Process sample image
            elif use_sample:
                image_array = np.array(sample_image)
                
                with col1:
                    st.subheader("Original Sample Image")
                    st.image(sample_image, caption=f"Sample: {sample_option}", use_column_width=True)
                
                # Ensure image has proper color channels
                image_array = ensure_color_channels(image_array)
                processed_image = preprocess_image(image_array)
            
            # Apply enhancement if selected
            if enhancement_option != "None":
                with st.spinner(f'Applying {enhancement_option} enhancement...'):
                    # Apply enhancement
                    enhanced_image = apply_enhancement(
                        processed_image,
                        enhancement_option,
                        enhancement_strength
                    )
                    
                    # Show the enhanced image
                    with col1:
                        st.subheader("Enhanced Image")
                        st.image(enhanced_image, caption=f"Enhanced with {enhancement_option}", use_column_width=True)
                    
                    # Use enhanced image for prediction
                    final_image = enhanced_image
            else:
                # Use original processed image
                final_image = processed_image
            
            # Make prediction
            with st.spinner('Analyzing image...'):
                start_time = time.time()
                prediction = st.session_state.model.predict(np.expand_dims(final_image, axis=0))
                end_time = time.time()
                
                # Calculate processing time
                processing_time = (end_time - start_time) * 1000  # convert to ms
                
                # Add to history
                add_to_history(
                    final_image,
                    "Transfer Learning" if model_option == "InceptionV3 Transfer Learning" else "Basic CNN",
                    prediction,
                    enhancement_option if enhancement_option != "None" else None
                )
                
                # Display results
                with col2:
                    st.subheader("Analysis Results")
                    
                    # Classification result
                    label, confidence = calculate_prediction_confidence(prediction[0][0])
                    
                    # Style the result display
                    if label == "Cancer":
                        st.error(f"**Prediction: {label}** (Confidence: {confidence:.2f}%)")
                    else:
                        st.success(f"**Prediction: {label}** (Confidence: {confidence:.2f}%)")
                    
                    st.write(f"Processing time: {processing_time:.2f} ms")
                    
                    # Display visualizations based on selected option
                    if visualization_option == "Prediction Confidence":
                        visualize_prediction(prediction[0][0])
                    elif visualization_option == "Class Activation Maps":
                        visualize_activation_maps(st.session_state.model, final_image)
                    elif visualization_option == "Feature Maps":
                        # Feature maps visualization
                        st.subheader("Feature Maps")
                        visualize_feature_maps(st.session_state.model, final_image)
                    
                    # Add disclaimer
                    st.info("""
                    **Disclaimer**: This tool is for educational purposes only and should not be used for medical diagnosis.
                    Always consult a healthcare professional for medical advice.
                    """)
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Display information about the model and approach
with st.expander("About the Model"):
    st.markdown("""
    ### CNN Architecture
    This application uses a Convolutional Neural Network (CNN) for detecting lung cancer in medical images.
    
    #### Basic CNN Model Architecture:
    - Input layer: Accepts medical images (224x224x3)
    - Convolutional layers: Extract features from the images
    - Pooling layers: Reduce dimensionality
    - Dropout layers: Prevent overfitting
    - Dense layers: Final classification
    - Output: Binary classification (cancer/non-cancer)
    
    #### Transfer Learning Option:
    When selecting the InceptionV3 transfer learning option, we use a pre-trained InceptionV3 model 
    fine-tuned on medical images for better accuracy.
    
    ### Image Preprocessing
    - Resizing to 224x224 pixels
    - Normalization (pixel values scaled to 0-1)
    - Channel adjustments for consistency
    
    ### Performance Metrics
    The model is evaluated using:
    - Accuracy: Overall correctness
    - Sensitivity: Ability to correctly identify cancer cases
    - Specificity: Ability to correctly identify non-cancer cases
    """)

# Footer
st.markdown("---")
st.markdown("© 2023 Lung Cancer Detection App | Not for clinical use")
