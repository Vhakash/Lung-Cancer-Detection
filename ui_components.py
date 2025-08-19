"""
Reusable UI components for the application.
"""
import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any
from config import APP_TITLE, APP_ICON
from error_handler import handle_errors
from logger import logger

class UIComponents:
    """Collection of reusable UI components."""
    
    @staticmethod
    def render_header():
        """Render the application header."""
        st.markdown(f"""
        # {APP_ICON} {APP_TITLE}
        """)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### Early detection saves lives
            This AI-powered tool analyzes medical images to detect potential signs of lung cancer.
            Simply upload a lung CT scan or X-ray image, or try one of our sample images.
            """)
        with col2:
            # Placeholder for logo or additional info
            st.info("🔬 Advanced AI Analysis\n📊 Detailed Visualizations\n👥 Patient Management")
        
        st.divider()
    
    @staticmethod
    def render_sidebar_navigation():
        """
        Render sidebar navigation and return selected option.
        
        Returns:
            Selected navigation option
        """
        st.sidebar.markdown("## 🔧 Navigation")
        
        nav_options = ["Analyze", "Patients", "History", "Compare Models", "About"]
        selected = st.sidebar.radio(
            "Choose Section",
            nav_options,
            index=0,
            key="nav_radio"
        )
        
        return selected
    
    @staticmethod
    def render_model_selector():
        """
        Render model selection sidebar component.
        
        Returns:
            Selected model option
        """
        st.sidebar.markdown("### 🧠 Model Selection")
        model_option = st.sidebar.selectbox(
            "Choose AI Model",
            ["Basic CNN", "InceptionV3 Transfer Learning"],
            index=0,
            key="model_select"
        )
        
        return model_option
    
    @staticmethod
    def render_visualization_selector():
        """
        Render visualization options sidebar component.
        
        Returns:
            Tuple of (visualization_option, gradcam_layer)
        """
        st.sidebar.markdown("### 📊 Visualization Tools")
        visualization_option = st.sidebar.selectbox(
            "Choose Visualization",
            ["Prediction Confidence", "Class Activation Maps", "Feature Maps", "Grad-CAM"],
            index=0,
            key="viz_select"
        )
        
        gradcam_layer = None
        if visualization_option == "Grad-CAM":
            gradcam_layer = st.sidebar.text_input(
                "Last conv layer (optional)", 
                value="", 
                help="Name of the last convolutional layer in your Keras model. Leave blank to auto-detect."
            ) or None
        
        return visualization_option, gradcam_layer
    
    @staticmethod
    def render_enhancement_selector():
        """
        Render image enhancement sidebar component.
        
        Returns:
            Tuple of (enhancement_option, enhancement_strength)
        """
        from image_enhancement import get_available_enhancements
        
        st.sidebar.markdown("### 🔍 Image Enhancement")
        enhancement_option = st.sidebar.selectbox(
            "Enhancement Technique",
            ["None"] + get_available_enhancements(),
            index=0,
            key="enhancement_select"
        )
        
        enhancement_strength = st.sidebar.slider(
            "Enhancement Strength",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.1,
            disabled=(enhancement_option == "None"),
            key="enhancement_slider"
        )
        
        return enhancement_option, enhancement_strength
    
    @staticmethod
    def render_sample_selector():
        """
        Render sample image selector.
        
        Returns:
            Selected sample option
        """
        from sample_data import get_sample_image_names
        
        st.sidebar.markdown("### 🔬 Sample Medical Images")
        sample_option = st.sidebar.selectbox(
            "Select Sample Case",
            ["None"] + get_sample_image_names(),
            index=0,
            key="sample_select"
        )
        
        return sample_option
    
    @staticmethod
    def render_loading_spinner(message: str = "Processing..."):
        """
        Render a loading spinner with message.
        
        Args:
            message: Loading message to display
        """
        return st.spinner(message)
    
    @staticmethod
    def render_success_message(message: str):
        """
        Render a success message.
        
        Args:
            message: Success message to display
        """
        st.success(f"✅ {message}")
        logger.info(f"Success: {message}")
    
    @staticmethod
    def render_error_message(message: str):
        """
        Render an error message.
        
        Args:
            message: Error message to display
        """
        st.error(f"❌ {message}")
        logger.error(f"Error displayed: {message}")
    
    @staticmethod
    def render_warning_message(message: str):
        """
        Render a warning message.
        
        Args:
            message: Warning message to display
        """
        st.warning(f"⚠️ {message}")
        logger.warning(f"Warning displayed: {message}")
    
    @staticmethod
    def render_info_message(message: str):
        """
        Render an info message.
        
        Args:
            message: Info message to display
        """
        st.info(f"ℹ️ {message}")
    
    @staticmethod
    def render_metrics_cards(metrics: Dict[str, Any]):
        """
        Render metrics in card format.
        
        Args:
            metrics: Dictionary of metric name to value
        """
        cols = st.columns(len(metrics))
        
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(
                    label=metric_name,
                    value=metric_value
                )
    
    @staticmethod
    def render_data_table(data: pd.DataFrame, title: Optional[str] = None):
        """
        Render a data table with optional title.
        
        Args:
            data: DataFrame to display
            title: Optional table title
        """
        if title:
            st.subheader(title)
        
        st.dataframe(data, use_container_width=True)
    
    @staticmethod
    def render_file_uploader(label: str = "Choose a file", file_types: List[str] = None):
        """
        Render a file uploader with validation.
        
        Args:
            label: Uploader label
            file_types: List of allowed file types
            
        Returns:
            Uploaded file object or None
        """
        if file_types is None:
            file_types = ["jpg", "jpeg", "png", "dcm"]
        
        uploaded_file = st.file_uploader(
            label,
            type=file_types,
            key="file_uploader"
        )
        
        return uploaded_file
    
    @staticmethod
    def render_download_buttons(data_dict: Dict[str, bytes], timestamp: str):
        """
        Render download buttons for multiple file types.
        
        Args:
            data_dict: Dictionary mapping file type to file data
            timestamp: Timestamp for filename
        """
        cols = st.columns(len(data_dict))
        
        for i, (file_type, file_data) in enumerate(data_dict.items()):
            with cols[i]:
                st.download_button(
                    label=f"Download {file_type.upper()}",
                    data=file_data,
                    file_name=f"analysis_{timestamp}.{file_type.lower()}",
                    mime=f"application/{file_type.lower()}",
                    key=f"dl_{file_type.lower()}"
                )
    
    @staticmethod
    @handle_errors
    def render_patient_quick_actions():
        """Render quick action buttons for patient management."""
        st.sidebar.markdown("## 👥 Patient Management")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("View Patients", key="view_patients_btn"):
                from session_manager import session_manager
                session_manager.navigate_to('patients')
                st.rerun()
        
        with col2:
            if st.button("Add Patient", key="add_patient_btn"):
                from session_manager import session_manager
                session_manager.set('show_patient_form', True)
                session_manager.navigate_to('patients')
                st.rerun()

# Create global UI components instance
ui = UIComponents()