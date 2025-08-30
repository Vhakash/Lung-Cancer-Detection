import tensorflow as tf
import keras
from keras.applications import EfficientNetB0
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast
from keras.models import Sequential, Model
import matplotlib.pyplot
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


def load_lung_cancer_data():
    """
    Load and prepare the lung cancer data for transfer learning

    Returns: X_train, X_val, X_test, y_train, y_val, y_test, class_weights
    """

    print("=" * 50)
    print("LOADING THE LUNG CANCER DETECTION DATA")
    print("=" * 50)

    #Load the preprocessed data
    print("📂 Loading the preprocess data")

    with np.load('preprocessed_data.npz') as data:
        X = data['images']
        y = data['labels']

    # Load class names
    with open('class_names.json', 'r') as f:
        class_names_dict = json.load(f)
    
    print(f"✅ Preprocess data loaded successfully: {X.shape}, {y.shape}")
    print(f"📊 Classes: {class_names_dict}")

    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📈 Original class distribution:")
    for i, count in zip(unique, counts):
        class_name = [k for k, v in class_names_dict.items() if v == i][0]
        print(f"   • Class {i} ({class_name}): {count} images")

    #Data preprocessing
    print("\n🔧 Preprocessing for transfer learning...")
    if X.max() > 1.0:
        X = X.astype('float32') / 255.0
        print("   • Normalized to [0,1] range")
    else:
        X = X.astype('float32')
        print("   • Already Normalized")

    # Calculate Class Weights BEFORE converting to categorical
    print("\n⚖️ Calculating class weights for imbalance...")
    class_labels = np.unique(y)
    y_indices = y  # Keep original integer labels for class weights
    weights = compute_class_weight(
        class_weight='balanced',
        classes=class_labels,
        y=y_indices
    )
    class_weights = dict(zip(class_labels, weights))
    print(f"   • Class weights calculated: {class_weights}")

    #convert to categorical 
    y_categorical = keras.utils.to_categorical(y, num_classes = 3)
    print("   • Converted labels to categorical")

    # Split data
    print(f"\n📊 Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, 
        stratify=y_indices  # Use original labels for stratification
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42,
        stratify=np.argmax(y_temp, axis=1)  # Use categorical labels for stratification
    )
    
    print(f"   • Training: {X_train.shape}")
    print(f"   • Validation: {X_val.shape}")
    print(f"   • Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights


# Load the data
X_train, X_val, X_test, y_train, y_val, y_test, class_weights = load_lung_cancer_data()




# -------------------------------- STEP 2  : D A T A  A U G M E N T A T I O N ------------------------------------
#to tackle the issue of overfitting and improve model generalization and avoid class imbalance
data_augmentation = keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomContrast(0.1)
])





# -----------------------------STEP - 3 : B U I L D  T R A N S F E R  L E A R N I N G  M O D E L ------------------------------------

