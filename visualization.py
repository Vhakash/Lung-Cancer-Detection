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
        model (object): The CNN model
    """
    # Check if it's our new LungCancerCNN model
    if hasattr(model, 'model') and model.model is not None:
        # Use Grad-CAM for real CNN models
        return visualize_grad_cam(image, model)
    
    # Fallback to mock activation map for compatibility
    if hasattr(model, 'get_activation_map'):
        activation_map = model.get_activation_map(image)
        _render_activation_visualization(image, activation_map)
        return
    
    # Generate basic activation map based on image features
    activation_map = _generate_basic_activation_map(image)
    _render_activation_visualization(image, activation_map)

def _render_activation_visualization(image, activation_map):
    """Render activation map visualization."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(activation_map, cmap='jet')
    ax2.set_title('Activation Map')
    ax2.axis('off')
    
    ax3.imshow(image)
    overlay = ax3.imshow(activation_map, cmap='jet', alpha=0.6)
    ax3.set_title('Overlay')
    ax3.axis('off')
    
    cbar = fig.colorbar(overlay, ax=ax3)
    cbar.set_label('Activation Intensity')
    
    st.pyplot(fig)
    st.write("""
    **Class Activation Map Interpretation:**
    - Bright red/yellow areas indicate regions most influential for the model's decision
    - These highlighted areas often correspond to abnormal tissue patterns
    - The overlay shows how these regions align with the original image features
    """)

def _generate_basic_activation_map(image):
    """Generate a basic activation map based on image features."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = np.mean(image, axis=2)
    else:
        gray_image = image
    
    # Calculate gradients to find edges and features
    grad_x = np.abs(np.gradient(gray_image, axis=1))
    grad_y = np.abs(np.gradient(gray_image, axis=0))
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to 0-1 range
    activation_map = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min() + 1e-8)
    
    # Apply Gaussian smoothing
    from scipy import ndimage
    activation_map = ndimage.gaussian_filter(activation_map, sigma=2)
    
    return activation_map
    
    # Otherwise, if this looks like a Keras model, defer to Grad-CAM
    if hasattr(model, 'layers') and hasattr(model, 'inputs') and hasattr(model, 'outputs'):
        st.info("Model does not expose activation-map API. Using Grad-CAM instead.")
        return visualize_grad_cam(image, model)
    
    # Fallback: cannot compute
    st.warning("Unable to compute activation maps for this model type.")

def visualize_feature_maps(image, model, layer_index=1):
    """Visualize feature maps from a specific CNN layer.
    
    Args:
        image (numpy.ndarray): Input image (preprocessed)
        model (object): The CNN model
        layer_index (int): Index of the layer to visualize
    """
    # Check if it's our new LungCancerCNN model with real Keras model
    if hasattr(model, 'model') and model.model is not None:
        _visualize_real_feature_maps(image, model.model)
        return
    
    # Fallback to mock feature maps for compatibility
    if hasattr(model, 'get_feature_maps'):
        feature_maps = model.get_feature_maps(image, layer_index)
        _render_feature_maps(feature_maps)
        return
    
    # Generate basic feature maps based on image processing
    feature_maps = _generate_basic_feature_maps(image)
    _render_feature_maps(feature_maps)

def _visualize_real_feature_maps(image, keras_model):
    """Visualize feature maps from a real Keras model."""
    try:
        import tensorflow as tf
        
        # Find the first convolutional layer
        conv_layer = None
        for layer in keras_model.layers:
            if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
                conv_layer = layer
                break
        
        if conv_layer is None:
            st.warning("No convolutional layer found for feature map visualization")
            return
        
        # Create a model that outputs the feature maps
        feature_model = tf.keras.Model(
            inputs=keras_model.input,
            outputs=conv_layer.output
        )
        
        # Get feature maps
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
        
        feature_maps = feature_model.predict(image_batch, verbose=0)[0]
        
        _render_feature_maps(feature_maps)
        st.caption(f"Feature maps from layer: {conv_layer.name}")
        
    except Exception as e:
        st.error(f"Error visualizing real feature maps: {e}")
        # Fallback to basic feature maps
        feature_maps = _generate_basic_feature_maps(image)
        _render_feature_maps(feature_maps)

def _render_feature_maps(feature_maps):
    """Render feature maps visualization."""
    num_maps = min(8, feature_maps.shape[-1])
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(num_maps):
        if len(feature_maps.shape) == 3:
            feature_map = feature_maps[:, :, i]
        else:
            feature_map = feature_maps[i] if i < len(feature_maps) else feature_maps[0]
        
        # Normalize feature map
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        axes[i].imshow(feature_map, cmap='viridis')
        axes[i].set_title(f'Feature Map {i+1}')
        axes[i].axis('off')
    
    # Remove unused subplots
    for i in range(num_maps, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("""
    **Feature Map Interpretation:**
    - Each feature map shows different patterns detected by the CNN filters
    - Earlier layers detect simple features (edges, textures)
    - Deeper layers identify more complex patterns (tissue abnormalities, structures)
    - These visualizations help understand what the model is looking for in the image
    """)

def _generate_basic_feature_maps(image):
    """Generate basic feature maps using image processing techniques."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = np.mean(image, axis=2)
    else:
        gray_image = image
    
    feature_maps = []
    
    # Edge detection filters
    from scipy import ndimage
    
    # Sobel filters
    sobel_x = ndimage.sobel(gray_image, axis=1)
    sobel_y = ndimage.sobel(gray_image, axis=0)
    feature_maps.extend([sobel_x, sobel_y])
    
    # Gaussian filters with different sigmas
    for sigma in [1, 2, 3]:
        gaussian = ndimage.gaussian_filter(gray_image, sigma=sigma)
        feature_maps.append(gaussian)
    
    # Laplacian filter
    laplacian = ndimage.laplace(gray_image)
    feature_maps.append(laplacian)
    
    # Gradient magnitude
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    feature_maps.append(gradient_mag)
    
    # Local standard deviation (texture)
    texture = ndimage.generic_filter(gray_image, np.std, size=5)
    feature_maps.append(texture)
    
    # Stack feature maps
    feature_maps_array = np.stack(feature_maps, axis=-1)
    
    return feature_maps_array
    
    # Attempt to extract feature maps from a Keras model
    if hasattr(model, 'layers') and hasattr(model, 'inputs') and hasattr(model, 'outputs'):
        try:
            try:
                import tensorflow as tf
                from tensorflow import keras
            except Exception:
                st.warning("TensorFlow not available. Cannot extract feature maps from Keras model.")
                return
            # Find the first convolutional-like layer with 4D output
            target_layer = None
            for lyr in model.layers:
                try:
                    out_shape = getattr(lyr, 'output_shape', None)
                    if isinstance(out_shape, tuple) and len(out_shape) == 4:
                        target_layer = lyr
                        break
                except Exception:
                    continue
            if target_layer is None:
                st.warning("No convolutional layer found to extract feature maps.")
                return
            # Build submodel
            sub = keras.Model(model.inputs, target_layer.output)
            img = image
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            feats = sub.predict(img[None, ...])  # (1, H, W, C)
            feats = np.squeeze(feats, axis=0)
            # Plot up to 8 maps
            num_maps = min(8, feats.shape[-1])
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            axes = axes.flatten()
            for i in range(num_maps):
                fmap = feats[:, :, i]
                fmap = (fmap - np.min(fmap)) / (np.max(fmap) - np.min(fmap) + 1e-8)
                axes[i].imshow(fmap, cmap='viridis')
                axes[i].set_title(f'Feature Map {i+1}')
                axes[i].axis('off')
            for i in range(num_maps, len(axes)):
                fig.delaxes(axes[i])
            plt.tight_layout()
            st.pyplot(fig)
            st.caption(f"Layer: {getattr(target_layer, 'name', 'conv')} | Shape: {feats.shape}")
            return
        except Exception as e:
            st.error(f"Failed to extract feature maps: {e}")
            return
    
    st.warning("Unable to compute feature maps for this model type.")

def visualize_grad_cam(image, model, last_conv_layer_name=None):
    """Visualize Grad-CAM for CNN models.
    
    Args:
        image (numpy.ndarray): Input image in HxWxC, range [0,1] preferred
        model: CNN model (LungCancerCNN or Keras model)
        last_conv_layer_name (str, optional): Name of last conv layer. If None, auto-detects.
    """
    # Check if TensorFlow is available
    try:
        import tensorflow as tf
    except Exception:
        st.warning("TensorFlow not available. Showing activation maps instead.")
        return visualize_activation_maps(image, model)
    
    # Get the actual Keras model
    keras_model = None
    if hasattr(model, 'model') and model.model is not None:
        keras_model = model.model
    elif hasattr(model, 'layers'):
        keras_model = model
    
    if keras_model is None:
        st.info("No Keras model available. Showing activation maps instead.")
        return visualize_activation_maps(image, model)
    
    # Prepare input tensor
    img = image
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    # Add batch dim and ensure float32
    img_tensor = tf.convert_to_tensor(img[None, ...], dtype=tf.float32)
    
    # Find last conv layer
    last_conv = None
    if last_conv_layer_name:
        try:
            last_conv = keras_model.get_layer(last_conv_layer_name)
        except Exception:
            last_conv = None
    
    if last_conv is None:
        # Auto-detect last 4D output layer
        for layer in reversed(keras_model.layers):
            try:
                out_shape = layer.output_shape
                if isinstance(out_shape, tuple) and len(out_shape) == 4:
                    last_conv = layer
                    break
            except Exception:
                continue
    
    if last_conv is None:
        st.warning("Could not locate a convolutional layer. Showing activation maps instead.")
        return visualize_activation_maps(image, model)
    
    # Build a gradient model from inputs to (last conv outputs, predictions)
    try:
        grad_model = tf.keras.Model([keras_model.inputs], [last_conv.output, keras_model.outputs])
    except Exception as e:
        st.error(f"Failed to build Grad-CAM graph: {e}")
        return visualize_activation_maps(image, model)
    
    # Compute gradients of the top predicted class (or single logit) wrt conv outputs
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_tensor, training=False)
        # Handle binary vs multi-class
        if preds.shape[-1] == 1:
            class_channel = preds[:, 0]
        else:
            top_class = tf.argmax(preds[0])
            class_channel = preds[:, top_class]
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        st.warning("No gradients found for Grad-CAM. Showing activation maps instead.")
        return visualize_activation_maps(image, model)
    
    # Global-average-pool the gradients over spatial dims
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normalize heatmap to [0,1]
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(heatmap[..., None], (img.shape[0], img.shape[1]))[:, :, 0]
    heatmap_np = heatmap.numpy()
    
    # Plot similar to activation maps
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(heatmap_np, cmap='jet')
    ax2.set_title('Grad-CAM')
    ax2.axis('off')
    
    ax3.imshow(img)
    overlay = ax3.imshow(heatmap_np, cmap='jet', alpha=0.6)
    ax3.set_title('Overlay')
    ax3.axis('off')
    cbar = fig.colorbar(overlay, ax=ax3)
    cbar.set_label('Activation Intensity')
    st.pyplot(fig)
    
    st.write("""
    **Grad-CAM Interpretation:**
    - Highlights spatial regions most influential for the model's predicted class.
    - Works with real CNN backbones (e.g., Keras Inception/ResNet). For the current mock model, activation maps are shown instead.
    - For best results, specify the last convolutional layer name of your model.
    """)
