import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
import io
from PIL import Image

def visualize_prediction(prediction_value):
    """Visualize the prediction confidence as a gauge chart.
    
    Args:
        prediction_value (float): The prediction value (0-1)
    """
    # Calculate confidence percentage
    confidence_percent = prediction_value * 100 if prediction_value >= 0.5 else (1 - prediction_value) * 100
    
    # Create a gauge chart
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'polar': True})
    
    # Calculate angle from prediction
    if prediction_value >= 0.5:
        angle = (prediction_value - 0.5) * 2 * np.pi
        color = 'red'
    else:
        angle = (0.5 - prediction_value) * 2 * np.pi
        color = 'green'
    
    # Draw the gauge
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set the plot limits
    ax.set_rlim(0, 1)
    
    # Draw background
    ax.fill_between(np.linspace(0, np.pi, 100), 0, 1, color='lightgray', alpha=0.5)
    ax.fill_between(np.linspace(np.pi, 2*np.pi, 100), 0, 1, color='lightcoral', alpha=0.5)
    
    # Draw the prediction needle
    ax.plot([0, angle], [0, 0.8], color=color, linewidth=3)
    ax.scatter(angle, 0.8, color=color, s=100)
    
    # Add labels
    ax.text(-np.pi/4, 0.4, 'Healthy', ha='center', va='center', fontsize=12)
    ax.text(np.pi + np.pi/4, 0.4, 'Cancer', ha='center', va='center', fontsize=12)
    ax.text(np.pi/2, 0.2, f"{confidence_percent:.1f}% confidence", ha='center', va='center', 
            fontsize=15, fontweight='bold', color=color)
    
    # Remove axis ticks and labels
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines.clear()
    
    st.pyplot(fig)

def visualize_model_performance(model_type):
    """Visualize model performance metrics.
    
    Args:
        model_type (str): Type of the model (Basic CNN or Transfer Learning)
    """
    # Create mock performance metrics based on model type
    # In a real application, these would come from model validation
    if model_type == "InceptionV3 Transfer Learning":
        accuracy = 0.92
        precision = 0.91
        recall = 0.89
        f1_score = 0.90
    else:  # Basic CNN
        accuracy = 0.85
        precision = 0.82
        recall = 0.84
        f1_score = 0.83
    
    # Create a dataframe for visualization
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1_score]
    
    df = pd.DataFrame({
        'Metric': metrics,
        'Value': values
    })
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(df['Metric'], df['Value'], color=sns.color_palette("viridis", len(metrics)))
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Set chart properties
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title(f'Performance Metrics for {model_type}')
    
    # Add descriptive text
    if model_type == "InceptionV3 Transfer Learning":
        st.write("""
        **Transfer Learning Model Performance:**
        - Higher overall accuracy and precision
        - Better at detecting subtle features
        - More robust against variations
        """)
    else:
        st.write("""
        **Basic CNN Model Performance:**
        - Good baseline performance
        - Faster processing time
        - Simpler architecture
        """)
    
    st.pyplot(fig)

def visualize_activation_maps(image, model):
    """Visualize class activation maps to highlight regions of interest.
    
    Args:
        image (numpy.ndarray): Input image (preprocessed)
        model (object): The CNN model (with get_activation_map method)
    """
    # Get activation map from the model
    activation_map = model.get_activation_map(image)
    
    # Create a figure for visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot activation map
    ax2.imshow(activation_map, cmap='jet')
    ax2.set_title('Activation Map')
    ax2.axis('off')
    
    # Plot overlay
    ax3.imshow(image)
    overlay = ax3.imshow(activation_map, cmap='jet', alpha=0.6)
    ax3.set_title('Overlay')
    ax3.axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(overlay, ax=ax3)
    cbar.set_label('Activation Intensity')
    
    st.pyplot(fig)
    
    # Add explanation
    st.write("""
    **Class Activation Map Interpretation:**
    - Bright red/yellow areas indicate regions most influential for the model's decision.
    - These highlighted areas often correspond to abnormal tissue patterns.
    - The overlay shows how these regions align with the original image features.
    """)

def visualize_feature_maps(image, model, layer_index=1):
    """Visualize feature maps from a specific CNN layer.
    
    Args:
        image (numpy.ndarray): Input image (preprocessed)
        model (object): The CNN model (with get_feature_maps method)
        layer_index (int): Index of the layer to visualize
    """
    # Get feature maps from the model
    feature_maps = model.get_feature_maps(image, layer_index)
    
    # Select a subset of feature maps to display (max 8)
    num_maps = min(8, feature_maps.shape[2])
    
    # Create figure for visualization
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    # Plot each feature map
    for i in range(num_maps):
        feature_map = feature_maps[:, :, i]
        
        # Normalize for better visualization
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        axes[i].imshow(feature_map, cmap='viridis')
        axes[i].set_title(f'Feature Map {i+1}')
        axes[i].axis('off')
    
    # Hide empty subplots if any
    for i in range(num_maps, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add explanation
    st.write("""
    **Feature Map Interpretation:**
    - Each feature map shows different patterns detected by the CNN filters.
    - Earlier layers detect simple features (edges, textures).
    - Deeper layers identify more complex patterns (tissue abnormalities, structures).
    - These visualizations help understand what the model is looking for in the image.
    """)
